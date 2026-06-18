"""RoboTwin env adapter for DBPO×EXPO-FT online RL (双臂 aloha-agilex 仿真).

镜像 `client/envs/droid_env.py` 的接口（reset / get_observation / step(action)→{executed_action}
/ get_info_for_step()→(done, success, reward, mask)），把已改造的 RoboTwin 任务 env 包成
EXPO-FT 的 env server 可驱动的对象。**零污染 DROID client**：本文件在独立包 `client_robotwin/`，
不 import 也不改 `client/` 内任何文件。

RoboTwin 侧 API（精读 `/home/xukainan/RoboTwin/envs/_base_task.py` + `script/eval_policy.py`
+ `policy/pi05/deploy_policy.py` 得到）：
  - 任务 env = `envs.{task_name}` 模块里的同名类（继承 Base_Task），含 `play_once`/`check_success`。
  - `setup_demo(now_ep_num, seed, is_test=True, **task_config)`：按 seed 布置一个 episode 场景。
  - `get_obs()` → 嵌套 dict：`observation[cam]["rgb"]`（head/left/right_camera）+ `joint_action["vector"]`(14)。
  - `take_action(action, action_type="qpos")`：**逐 action 执行（方式2，逐帧 TOPP，变帧数）**；
    `take_chunk_action(chunk,...)` 是方式1（整段 TOPPRA），`_apply_k_skip` 是方式3（固定帧）。
    本适配器按 env_client 的「每步一个 action」协议，**固定走方式2 的 `take_action`**。
  - `check_success()` → bool；`set_instruction`/`get_instruction`；`close_env()`；`exec_backend` 属性。

奖励：稀疏 0/1 终局（success→1，否则 0），无 shaping。macro-action 聚合在 learner 端做。
动作时序：policy 出 H=50 chunk，env 每次 step 执行其中一个 action，learner 执行前 H_e(=25) 步后 replan。
"""

import importlib
import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml


class RoboTwinEnv:
    """Wrap one RoboTwin task env with the EXPO-FT env-server interface."""

    def __init__(
        self,
        task_name: str = None,
        robotwin_task_config: Optional[dict] = None,
        *,
        instruction_type: str = "unseen",
        exec_backend: str = "per_action",   # 方式2（逐 action TOPP）；与每步一个 action 协议匹配
        n_real_dims: int = 14,              # 双臂 2×(6关节+1夹爪)
        max_decision_steps: int = 200,      # 每 episode 最多 env-step（与 RLinf max_episode_steps 一致）
        seed: int = 0,
        is_eval: bool = False,
        video_dir: Optional[str] = None,
        **kwargs,                           # 吸收 config-as-kwargs（env/env_name/control_hz/... 被忽略）
    ):
        assert task_name is not None, "robotwin task config 必须含 `task_name`（RoboTwin envs.{task_name}）"
        self.task_name = task_name
        self.task_config = dict(robotwin_task_config or {})
        self.instruction_type = instruction_type
        self.exec_backend = exec_backend
        self.n_real_dims = int(n_real_dims)
        self.max_decision_steps = int(max_decision_steps)
        self.is_eval = is_eval
        self.video_dir = video_dir

        self._seed = int(seed)
        self._ep_count = 0
        self._steps_since_reset = 0
        self._success_once = False
        self._instruction = ""
        self.env = self._make_env()
        # 解析 RoboTwin setup_demo 所需的完整 args（yaml + 相机/embodiment 二次解析），缓存复用。
        self._setup_kwargs = self._resolve_setup_kwargs()

    # ---- RoboTwin env 构造（等价 eval_policy.class_decorator）----
    def _make_env(self):
        module = importlib.import_module(f"envs.{self.task_name}")
        # RoboTwin 约定：任务类名 = 文件名（snake_case），如 envs/adjust_bottle.py -> class adjust_bottle
        cls = getattr(module, self.task_name, None)
        if cls is None:  # 兜底：取模块里第一个看起来像任务类的对象
            cands = [getattr(module, n) for n in dir(module)
                     if n.lower() == self.task_name.lower().replace("_", "")]
            cls = cands[0] if cands else None
        if cls is None:
            raise RuntimeError(f"找不到 RoboTwin 任务类 envs.{self.task_name}.{self.task_name}")
        env = cls()
        # 后端选择（RoboTwin deploy 读取 TASK_ENV.exec_backend；本适配器逐 action 走 take_action）。
        env.exec_backend = self.exec_backend
        return env

    # ---- 复刻 RoboTwin script/eval_policy.py 的 config 组装（加载 yaml + 解析相机/embodiment）----
    def _resolve_setup_kwargs(self) -> Dict[str, Any]:
        """把 robotwin_task_config 解析成可直接 `setup_demo(**kwargs)` 的完整 dict。

        镜像 `script/eval_policy.py::main`(line 78-117) + `eval_policy()`(line 229 设 eval_mode=True)：
          ① 按 `task_config` 名加载 RoboTwin `task_config/{name}.yml`；② 用户覆盖项盖其上；
          ③ 由 camera type 解析 head_camera_h/w；④ 由 embodiment 解析 robot file + embodiment config。
        相对路径假定 cwd=RoboTwin 根（run_robotwin_client 启动时已 `os.chdir(robotwin_root)`）。
        """
        cfg = dict(self.task_config)
        yaml_stem = cfg.pop("task_config", None)
        args: Dict[str, Any] = {}
        if yaml_stem is not None:
            with open(f"./task_config/{yaml_stem}.yml", "r", encoding="utf-8") as f:
                args = yaml.load(f.read(), Loader=yaml.FullLoader)
        args.update(cfg)                  # 用户覆盖项（eval_mode/render_freq/camera 等）盖在 yaml 之上
        args["task_name"] = self.task_name

        assert "camera" in args and "embodiment" in args, (
            "robotwin_task_config 需提供 RoboTwin 场景配置：请设 `task_config`=task_config/ 下的 yaml 名"
            "（如 'demo_clean'，含 camera/embodiment/data_type/domain_randomization）")

        # 相机 h/w（eval_policy.py:100-102）
        with open("./task_config/_camera_config.yml", "r", encoding="utf-8") as f:
            cam_cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        head_type = args["camera"]["head_camera_type"]
        args["head_camera_h"] = cam_cfg[head_type]["h"]
        args["head_camera_w"] = cam_cfg[head_type]["w"]

        # embodiment（eval_policy.py:85-117）：解析 robot file + 加载各自 config.yml
        with open("./task_config/_embodiment_config.yml", "r", encoding="utf-8") as f:
            emb_types = yaml.load(f.read(), Loader=yaml.FullLoader)

        def _emb_file(t: str) -> str:
            rf = emb_types[t]["file_path"]
            if rf is None:
                raise RuntimeError(f"embodiment {t} 缺 file_path")
            return rf

        def _emb_config(robot_file: str) -> dict:
            with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
                return yaml.load(f.read(), Loader=yaml.FullLoader)

        emb = args["embodiment"]
        if len(emb) == 1:                 # 单 embodiment（双臂同体，如 aloha-agilex）
            args["left_robot_file"] = _emb_file(emb[0])
            args["right_robot_file"] = _emb_file(emb[0])
            args["dual_arm_embodied"] = True
        elif len(emb) == 3:               # 左右异体 + 间距
            args["left_robot_file"] = _emb_file(emb[0])
            args["right_robot_file"] = _emb_file(emb[1])
            args["embodiment_dis"] = emb[2]
            args["dual_arm_embodied"] = False
        else:
            raise ValueError("embodiment items should be 1 or 3 (见 eval_policy.py:104-114)")
        args["left_embodiment_config"] = _emb_config(args["left_robot_file"])
        args["right_embodiment_config"] = _emb_config(args["right_robot_file"])

        args.setdefault("eval_mode", True)  # 对齐 RoboTwin eval（eval_policy.py:229）；可在 config 覆盖为 False（seen 纹理）
        return args

    # ---- 与 droid_env 同名接口 ----
    def reset(self) -> Dict[str, Any]:
        """新 episode：按递增 seed 布置场景，返回首帧 obs。"""
        seed = self._seed + self._ep_count
        # 完全对齐 RoboTwin eval（script/eval_policy.py:237）：setup_demo(now_ep_num, seed,
        # is_test=True, **解析后的 args)；args 已含 eval_mode / camera h-w / embodiment config 等。
        self.env.setup_demo(now_ep_num=self._ep_count, seed=seed,
                            is_test=True, **self._setup_kwargs)
        self._instruction = self._resolve_instruction()
        self.env.set_instruction(instruction=self._instruction)
        self._steps_since_reset = 0
        self._success_once = False
        self._ep_count += 1
        return self.get_observation()

    def get_observation(self) -> Dict[str, Any]:
        """RoboTwin get_obs → openpi `pi05_aloha_robotwin` RepackTransform 期望的扁平 obs dict。

        映射依据（核对自 `openpi .../training/config.py::_robotwin_drifting_config` 的 RepackTransform
        + `policies/aloha_policy.py::AlohaInputs` + RoboTwin `process_data.py`）：
          - 键 = LeRobot 扁平列名 `observation.images.{cam_high,cam_left_wrist,cam_right_wrist}` / `observation.state`。
          - cam_high=head_camera（主图），cam_left/right_wrist=left/right_camera。
          - 图像 **CHW uint8**（RoboTwin get_rgb 返回 HWC，需转置）；AlohaInputs 要求 [C,H,W]。
          - state=joint_action.vector(14) = [左臂6,左夹爪1,右臂6,右夹爪1]。
        """
        raw = self.env.get_obs()
        cams = raw["observation"]
        state = np.asarray(raw["joint_action"]["vector"], dtype=np.float32)  # (14,)

        def _chw(rgb):
            a = np.asarray(rgb)
            if a.ndim == 3 and a.shape[-1] == 3:   # HWC -> CHW
                a = np.transpose(a, (2, 0, 1))
            return np.ascontiguousarray(a.astype(np.uint8))

        return {
            "observation.images.cam_high": _chw(cams["head_camera"]["rgb"]),
            "observation.images.cam_left_wrist": _chw(cams["left_camera"]["rgb"]),
            "observation.images.cam_right_wrist": _chw(cams["right_camera"]["rgb"]),
            "observation.state": state,
            "prompt": self._instruction,
        }

    def step(self, action) -> Dict[str, Any]:
        """执行**一个** action（方式2 逐 action TOPP）。返回 {executed_action}。"""
        a = np.asarray(action, dtype=np.float64).reshape(-1)[: self.n_real_dims]
        self.env.take_action(a, action_type="qpos")
        self._steps_since_reset += 1
        # RoboTwin take_action 不返回改写后的动作；executed = 下发的 qpos 目标。
        return {"executed_action": a}

    def get_info_for_step(self, raw_obs=None) -> Tuple[bool, bool, float, float]:
        """评测终止：(done, success, reward, mask)。稀疏 0/1 终局，无 shaping。"""
        success = bool(self.env.check_success())
        self._success_once = self._success_once or success
        time_up = self._steps_since_reset >= self.max_decision_steps
        done = bool(self._success_once or time_up)
        reward = 1.0 if success else 0.0     # 稀疏：仅成功步给 1（macro 聚合在 learner）
        mask = 0.0 if done else 1.0
        if done:
            logging.info("[RoboTwin] episode done: success=%s time_up=%s steps=%d",
                         self._success_once, time_up, self._steps_since_reset)
        return done, success, reward, mask

    def get_info_for_step_dict(self) -> Dict[str, Any]:
        done, success, reward, mask = self.get_info_for_step()
        return {"done": done, "success": success, "reward": reward, "mask": mask}

    @property
    def steps_since_reset(self) -> int:
        return self._steps_since_reset

    def _resolve_instruction(self) -> str:
        try:
            instr = self.env.get_instruction()
            if isinstance(instr, (list, tuple)) and instr:
                return str(np.random.choice(instr))
            if isinstance(instr, str) and instr:
                return instr
        except Exception:
            pass
        return self.task_name.replace("_", " ")

    def close(self):
        try:
            self.env.close_env()
        except Exception:
            logging.exception("[RoboTwin] close_env failed")
