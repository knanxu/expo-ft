"""RoboTwin env adapter for EXPO-FT online RL (双臂 aloha-agilex 仿真).

镜像 `client/envs/droid_env.py` 的接口（reset / get_observation / step(action)→{executed_action}
/ get_info_for_step()→(done, success, reward, mask)），把已改造的 RoboTwin 任务 env 包成
EXPO-FT 的 env server 可驱动的对象。**零污染 DROID client**：本文件在独立包 `client_robotwin/`，
不 import 也不改 `client/` 内任何文件。

RoboTwin 侧 API（精读 `/home/xukainan/RoboTwin/envs/_base_task.py` + `script/eval_policy.py`
+ `policy/pi05/deploy_policy.py` 得到）：
  - 任务 env = `envs.{task_name}` 模块里的同名类（继承 Base_Task），含 `play_once`/`check_success`。
  - `setup_demo(now_ep_num, seed, is_test=True, **task_config)`：按 seed 布置一个 episode 场景。
  - `get_obs()` → 嵌套 dict：`observation[cam]["rgb"]`（head/left/right_camera）+ `joint_action["vector"]`(14)。
  - `take_action(action, action_type="qpos")`：逐 action 执行；`take_chunk_action(chunk,...)`
    是整段 TOPPRA；`take_chunk_action_streaming` 是 fixed-time。
    本适配器按 env_client 的「每步一个 action」协议，**固定走方式2 的 `take_action`**。
  - `check_success()` → bool；`set_instruction`/`get_instruction`；`close_env()`；`exec_backend` 属性。

奖励：稀疏 0/1 终局（success→1，否则 0），无 shaping。macro-action 聚合在 learner 端做。
动作时序：policy 出 H=50 chunk，env 每次 step 执行其中一个 action，learner 执行前 H_e(=25) 步后 replan。
"""

import importlib
import inspect
import logging
import os
import subprocess
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml

from client_robotwin.envs.expert_takeover import (
    should_takeover,
    record_expert_tape,
    TapeReplayer,
)


class RoboTwinEnv:
    """Wrap one RoboTwin task env with the EXPO-FT env-server interface."""

    def __init__(
        self,
        task_name: str = None,
        robotwin_task_config: Optional[dict] = None,
        *,
        instruction_type: str = "seen",     # 模板池（"seen"/"unseen"）；务必与 gate-4 eval 的 --instruction_type 一致
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
        # SpeedTune 整段执行参数（经 config-as-kwargs / create_env 传入；step_chunk 用）。
        self._stream_hold_steps = int(kwargs.get("stream_hold_steps", 15) or 15)
        _ks = kwargs.get("k_skip", None)
        self._k_skip = int(_ks) if _ks not in (None, "", 0) else None
        self._eval_vsf = int(kwargs.get("eval_video_save_freq", 25) or 25)  # eval 视频每 N 物理步采 1 帧
        # SpeedTune 执行层力矩底座（真机 τ_max）。None=∞=原行为（EXPO/BC 零侵入）。
        # 可传标量（所有 arm 关节同）或 per-joint list（长度=单臂 arm dof）。
        _fl = kwargs.get("force_limit", None)
        self._force_limit = _fl if _fl not in (None, "", 0) else None
        # 自动专家介入（替代 DROID 人类在环）：用到 takeover_step_frac·step_budget 预算仍未 success →
        # 从当前状态调 play_once 录专家带逐 step 回放（action_type="human"）。EXPO/BC 通用，零侵入。
        self._takeover_enable = bool(kwargs.get("takeover_enable", True))
        self._takeover_step_frac = float(kwargs.get("takeover_step_frac", 0.5) or 0.5)
        self._takeover_save_freq = int(kwargs.get("takeover_save_freq", 15) or 15)
        self._takeover_active = False
        self._takeover_attempted = False
        self._replayer = None

        self._seed = int(seed)
        self._ep_count = 0
        self._steps_since_reset = 0
        self._success_once = False
        self._instruction = ""
        # episode 提示词回退：优先 config.language_instruction（经 config-as-kwargs 传入），
        # 否则任务名（RoboTwin get_instruction 在无 play_once 时返回 None）。
        self._lang_fallback = kwargs.get("language_instruction") or task_name.replace("_", " ")
        self.env = self._make_env()
        # 解析 RoboTwin setup_demo 所需的完整 args（yaml + 相机/embodiment 二次解析），缓存复用。
        self._setup_kwargs = self._resolve_setup_kwargs()
        # episode 预算：默认用 RoboTwin 自己的 step_lim（首个 reset 后从 env 读到，见 reset）；
        # 在那之前用 max_decision_steps 兜底。
        self._step_budget = int(self.max_decision_steps)
        # sapien 资源释放周期（对齐 RoboTwin eval 的 clear_cache_freq，demo_clean.yml=5）。
        self._clear_cache_freq = int(self._setup_kwargs.get("clear_cache_freq", 5) or 5)
        self._max_reset_retries = 20        # setup_demo 不稳定 seed 的最大换 seed 次数
        # RLinf 对齐（vector_env.py:107/115）：缓存一次 RoboTwin episode_info（占位符 {A}/{B}/{a}/{b}），
        # 供每集随机生成模板指令。None=未取；{}=已尝试但失败（走 fallback，不再重试）。
        self._episode_info = None

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
        """新 episode：按递增 seed 布置场景，返回首帧 obs。

        对齐 RoboTwin eval（script/eval_policy.py:396-434）：
          ① 每个新 episode **前**先 `close_env`（周期性 clear_cache，防 sapien 资源/显存累积——
             RoboTwin 的 setup_scene 每次新建 engine/renderer/scene，不释放会漏）；
          ② setup_demo 可能因布局不稳定抛 `UnStableError`（或其它异常）——像 eval_policy 那样
             换下一个 seed 重试，而不是让整轮 rollout 崩。
        """
        try:
            from envs.utils.create_actor import UnStableError  # RoboTwin venv 内可用
        except Exception:
            UnStableError = ()  # 退化：仅靠下面的通用 Exception 兜底

        # RLinf 对齐：首集前一次性取并缓存 episode_info（供模板指令生成）。
        self._ensure_episode_info()

        # episode 间释放上一幕（首个 episode 无上一幕）。close_env 内部 self.close() 为 no-op，
        # 真正作用是 clear_cache=True 时的 sapien_clear_cache（对齐 eval_policy 的 clear_cache_freq）。
        if self._ep_count > 0:
            try:
                self.env.close_env(clear_cache=(self._ep_count % self._clear_cache_freq == 0))
            except Exception:
                logging.exception("[RoboTwin] close_env between episodes failed")

        last_err = None
        for _ in range(self._max_reset_retries):
            seed = self._seed + self._ep_count
            try:
                # 完全对齐 RoboTwin eval（script/eval_policy.py:434）：setup_demo(now_ep_num, seed,
                # is_test=True, **解析后的 args)；args 已含 eval_mode / camera h-w / embodiment config 等。
                self.env.setup_demo(now_ep_num=self._ep_count, seed=seed,
                                    is_test=True, **self._setup_kwargs)
                break
            except UnStableError as e:           # 该 seed 布局不稳 → 换下一个 seed
                last_err = e
                logging.warning("[RoboTwin] UnStableError at seed=%d; trying next seed.", seed)
            except Exception as e:               # 其它 setup 失败同样换 seed 重试（对齐 eval_policy）
                last_err = e
                logging.warning("[RoboTwin] setup_demo failed at seed=%d (%s); trying next seed.", seed, e)
            try:
                self.env.close_env(clear_cache=False)
            except Exception:
                pass
            self._ep_count += 1
        else:
            raise RuntimeError(
                f"[RoboTwin] setup_demo failed after {self._max_reset_retries} seeds") from last_err

        # episode 预算 = RoboTwin 自己的 step_lim（eval_mode 下由 _eval_step_limit.yml 决定，
        # 如 stack_blocks_two=800 个 take_action）；缺失（eval_mode=False）才回退 max_decision_steps。
        self._step_budget = int(getattr(self.env, "step_lim", None) or self.max_decision_steps)

        # RLinf 对齐：每集从缓存 episode_info 随机生成模板指令（与 eval_policy_client 同源），
        # 而非固定 fallback "stack the two blocks"——否则语言条件与训练/eval 不符 → rollout 必失败。
        self._instruction = self._create_instruction()
        self.env.set_instruction(instruction=self._instruction)
        self._apply_force_limit()   # SpeedTune 力矩底座（None=不施加，EXPO/BC 零侵入）
        self._steps_since_reset = 0
        self._success_once = False
        self._takeover_active = False
        self._takeover_attempted = False
        self._replayer = None
        self._ep_count += 1
        return self.get_observation()

    def reseed(self, seed: Optional[int] = None) -> None:
        """重置 episode 计数器，使下次 reset 从 seed（默认沿用构造 seed）重跑**同一批布局**。

        供固定系数 sweep（scripts/bench_fixed_speed.py）在同批布局上对比多个加速配置——用它代替
        「重建 env」做控制变量，避免累积 sapien scene 显存（同卡 server+bench 时尤其关键）。
        主动 close 上一配置遗留的 scene + clear_cache 释放显存；保留已缓存的 episode_info（不重跑
        play_once）。纯新增，不改既有 reset/step 协议。
        """
        try:
            self.env.close_env(clear_cache=True)
        except Exception:
            logging.exception("[RoboTwin] reseed close_env failed (继续)")
        if seed is not None:
            self._seed = int(seed)
        self._ep_count = 0
        self._steps_since_reset = 0
        self._success_once = False
        self._takeover_active = False
        self._takeover_attempted = False
        self._replayer = None

    def get_observation(self) -> Dict[str, Any]:
        """RoboTwin get_obs → openpi `pi05_aloha_robotwin` RepackTransform 期望的扁平 obs dict。

        映射依据（核对自 `openpi .../training/config.py::_robotwin_drifting_config` 的 RepackTransform
        + `policies/aloha_policy.py::AlohaInputs` + RoboTwin `process_data.py`）：
          - 键 = LeRobot 扁平列名 `observation.images.{cam_high,cam_left_wrist,cam_right_wrist}` / `observation.state`。
          - cam_high=head_camera（主图），cam_left/right_wrist=left/right_camera。
          - 图像 **CHW uint8**（RoboTwin get_rgb 返回 HWC，需转置）；AlohaInputs 要求 [C,H,W]。
          - state=joint_action.vector(14) = [左臂6,左夹爪1,右臂6,右夹爪1]。
        """
        if self._takeover_active and self._replayer is not None:
            return self._replayer.current_obs()
        return self._flatten_obs(self.env.get_obs())

    def _flatten_obs(self, raw) -> Dict[str, Any]:
        """RoboTwin get_obs → openpi 扁平 obs dict（在线 rollout 与专家录播共用）。"""
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

    def _should_takeover(self) -> bool:
        return (not self._takeover_attempted) and should_takeover(
            steps_since_reset=self._steps_since_reset,
            step_budget=self._step_budget,
            success_once=self._success_once,
            step_frac=self._takeover_step_frac,
            enabled=self._takeover_enable,
        )

    def _start_takeover(self) -> bool:
        """从当前状态录专家带；成功→进入回放，失败→放弃接管（episode 走到超时自然失败）。"""
        self._takeover_attempted = True
        frames = record_expert_tape(
            self.env, self._flatten_obs,
            save_freq=self._takeover_save_freq, n_real_dims=self.n_real_dims,
        )
        if frames is None:
            logging.warning("[RoboTwin] expert takeover recording failed; episode runs to timeout")
            return False
        self._replayer = TapeReplayer(frames)
        self._takeover_active = True
        logging.info("[RoboTwin] expert takeover started: %d frames at step %d",
                     len(frames), self._steps_since_reset)
        return True

    def step(self, action) -> Dict[str, Any]:
        """逐 action 执行。接管期回放 expert 动作(action_type=human)；否则正常 take_action(policy)。"""
        if not self._takeover_active and self._should_takeover():
            self._start_takeover()
        if self._takeover_active and self._replayer is not None:
            qpos = np.asarray(self._replayer.next_action(), dtype=np.float64).reshape(-1)[: self.n_real_dims]
            self._steps_since_reset += 1
            return {"executed_action": qpos, "action_type": "human"}
        a = np.asarray(action, dtype=np.float64).reshape(-1)[: self.n_real_dims]
        self.env.take_action(a, action_type="qpos")
        self._steps_since_reset += 1
        # RoboTwin take_action 不返回改写后的动作；executed = 下发的 qpos 目标。
        return {"executed_action": a, "action_type": "policy"}

    def _apply_force_limit(self):
        """对左右臂 arm 关节施加 PD force_limit（真机 τ_max 物理底座）。

        None → 不施加（默认 ∞，EXPO/BC 零侵入）。不改 robot.py：在 env 层重设
        set_drive_property，覆盖 force_limit，stiffness/damping 沿用 robot 已设值。
        每次 reset 后调用（setup_demo 重建 robot 会重置 drive property）。
        """
        if self._force_limit is None:
            return
        robot = getattr(self.env, "robot", None)
        if robot is None:
            return
        try:
            arms = [(robot.left_arm_joints, robot.left_joint_stiffness, robot.left_joint_damping),
                    (robot.right_arm_joints, robot.right_joint_stiffness, robot.right_joint_damping)]
            for joints, stiff, damp in arms:
                fl = self._force_limit
                fl_list = fl if isinstance(fl, (list, tuple)) else [float(fl)] * len(joints)
                for j, jf in zip(joints, fl_list):
                    j.set_drive_property(stiffness=stiff, damping=damp, force_limit=float(jf))
        except Exception:
            logging.exception("[RoboTwin] apply force_limit failed (继续不限力矩)")

    # ---- SpeedTune 整段执行（新增，不改既有 per-action step）---------------
    # exec_backend 名 → RoboTwin 三后端（envs/_base_task.py）方法选择。
    _RT_BACKEND = {
        "fixed_time": "streaming",          # 论文式固定时长（take_chunk_action_streaming）
        "per_action_toppra": "per_action",  # 逐 action TOPP（take_chunk_action_per_action）
        "chunk_toppra": "whole_chunk",      # 整段 TOPPRA（take_chunk_action）
    }

    @staticmethod
    def _call_backend(fn, chunk, **kwargs):
        """只传 fn 实际接受的 kwargs，适配不同 RoboTwin 版本的后端签名。

        不同 RoboTwin 分支的 take_chunk_action_* 签名可能不同（例如有/无 ``max_actions``、
        ``video_save_freq``）。按 fn 的真实签名过滤，避免 ``unexpected keyword argument``；
        若 fn 接受 ``**kwargs``（VAR_KEYWORD）则原样全传。
        """
        params = inspect.signature(fn).parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return fn(chunk, **kwargs)
        accepted = {k: val for k, val in kwargs.items() if k in params}
        return fn(chunk, **accepted)

    def step_chunk(self, chunk, speed_params=None, exec_backend=None) -> Dict[str, Any]:
        """整段执行一个 action chunk + 速度控制参数（SpeedTune 决策粒度）。

        直接对接 RoboTwin 三种执行后端。fixed-time/per-action 保留 ``v`` 重构；whole-chunk
        先截取 ``execution_steps``，仅透传 ``vel_limit``，加速度由执行层按 ``4V²`` 派生。

        Args:
          chunk:        ``[H, n_real_dims]`` 绝对 qpos 目标序列（原始，不压缩）。
          speed_params: backend 对应的离散参数；whole-chunk 仅含 ``vel_limit``。
          exec_backend: "fixed_time"/"per_action_toppra"/"chunk_toppra"；缺省用 self.exec_backend。
        Returns:
          {"executed_action": 最后一帧 qpos, "n_exec_steps": 实际仿真帧数(dense_steps),
           "duration": TOPPRA 时长(s), "exec_status": "success"/"topp_fallback"/"truncated"}。
          reward/done 仍由 ``get_info_for_step`` 给出（稀疏 0/1；speed reward 在 learner 端 success-gated）。
        """
        speed_params = dict(speed_params or {})
        backend = exec_backend or self.exec_backend
        rt = self._RT_BACKEND.get(backend, "whole_chunk")
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim == 1:
            chunk = chunk[None]
        chunk = chunk[:, : self.n_real_dims]
        v = float(speed_params.get("v", 1.0))
        vel_limit = float(speed_params.get("vel_limit", 5.0))
        acc_limit = float(speed_params.get("acc_limit", 10.0))
        vsf = self._eval_vsf if getattr(self.env, "eval_video_path", None) else -1  # eval 视频开着时 >0

        if rt == "streaming":
            info = self._call_backend(self.env.take_chunk_action_streaming, chunk,
                                      v=v, hold_steps=self._stream_hold_steps,
                                      max_actions=self._k_skip, video_save_freq=vsf)
        elif rt == "per_action":
            info = self._call_backend(self.env.take_chunk_action_per_action, chunk,
                                      vel_limit=vel_limit, acc_limit=acc_limit, v=v,
                                      max_actions=self._k_skip, video_save_freq=vsf)
        else:  # whole_chunk：先取 execution_steps 帧，再 spline/TOPPRA；acc=4*vel² 在执行层派生
            info = self._call_backend(self.env.take_chunk_action, chunk,
                                      vel_limit=vel_limit, execution_steps=self._k_skip,
                                      video_save_freq=vsf)
        info = info or {}
        # episode 预算按消耗的 action 数累加（与 RoboTwin step_lim 同语义；后端内部也自查 step_lim）。
        self._steps_since_reset += int(info.get("take_action_cnt_delta", 0) or 0)
        lg, rg, lc, rc = self._contact_info()
        result = {
            "executed_action": chunk[-1],
            "n_exec_steps": int(info.get("dense_steps", 0) or 0),
            "duration": float(info.get("duration", 0.0) or 0.0),
            "exec_status": str(info.get("status", "success")),
            "left_gripper": lg, "right_gripper": rg,      # ∈[0,1] 0=闭合/抓取
            "left_contact": lc, "right_contact": rc,      # sapien 物理接触（夹爪↔物体）
            "execution_steps": int(info.get("execution_steps", info.get("take_action_cnt_delta", 0)) or 0),
            "planned_cruise_fraction": float(info.get("planned_cruise_fraction", 0.0) or 0.0),
            "fixed_time_speed_violation": bool(info.get("fixed_time_speed_violation", False)),
            "max_planned_qvel": float(info.get("max_planned_qvel", 0.0) or 0.0),
        }
        if "vel_limit" in info:
            result["vel_limit"] = float(info["vel_limit"])
        if "acc_limit" in info:
            result["acc_limit"] = float(info["acc_limit"])
        return result

    # ---- eval 用：接触信号 + 视频录制（train 路径不触发）-------------------
    # 物体判定：entity 名不含这些机器人/场景关键词即视为可抓物体。
    _ROBOT_KW = ("link", "arm", "gripper", "finger", "wrist", "base", "joint",
                 "ground", "wall", "table", "mount", "camera", "head", "body", "torso", "panda")

    def _is_object(self, name: str) -> bool:
        low = str(name).lower()
        return not any(kw in low for kw in self._ROBOT_KW)

    def _contact_info(self):
        """(left_gripper_val, right_gripper_val, left_contact, right_contact)。

        gripper val ∈[0,1]（0=闭合/抓取，1=张开）来自 robot；contact 用 sapien
        ``scene.get_contacts()`` 检测含 gripper/finger 的 link 与"物体"的物理接触。
        失败兜底 (0,0,False,False)，绝不影响主流程。
        """
        robot = getattr(self.env, "robot", None)
        try:
            lg = float(robot.get_left_gripper_val())
            rg = float(robot.get_right_gripper_val())
        except Exception:
            lg = rg = 0.0
        lc = rc = False
        try:
            for c in self.env.scene.get_contacts():
                if not getattr(c, "points", None):
                    continue
                n0, n1 = c.bodies[0].entity.name, c.bodies[1].entity.name
                for ga, gb in ((n0, n1), (n1, n0)):
                    la = ga.lower()
                    if ("gripper" in la or "finger" in la) and self._is_object(gb):
                        if "left" in la or la.startswith("fl") or "_l" in la:
                            lc = True
                        elif "right" in la or la.startswith("fr") or "_r" in la:
                            rc = True
                        else:
                            lc = rc = True
        except Exception:
            pass
        return lg, rg, lc, rc

    def start_eval_video(self, episode_id: int = 0):
        """开 eval 视频（is_eval + video_dir 时）：ffmpeg 录 head_camera，对齐 RoboTwin eval_policy。"""
        if not self.is_eval or not self.video_dir:
            return
        self.stop_eval_video()  # 收尾上一个（若有）
        try:
            os.makedirs(self.video_dir, exist_ok=True)
            h = int(self._setup_kwargs.get("head_camera_h", 480) or 480)
            w = int(self._setup_kwargs.get("head_camera_w", 640) or 640)
            vsf, phys_hz = int(self._eval_vsf), 250
            out = os.path.join(self.video_dir, f"episode{int(episode_id)}.mp4")
            ff = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                 "-pixel_format", "rgb24", "-video_size", f"{w}x{h}",
                 "-framerate", f"{phys_hz}/{vsf}", "-i", "-", "-pix_fmt", "yuv420p",
                 "-vcodec", "libx264", "-crf", "23", out], stdin=subprocess.PIPE)
            self.env.eval_video_path = self.video_dir   # 让 _tick_eval_video 生效
            self.env._set_eval_video_ffmpeg(ff, fps=phys_hz / vsf, phys_hz=phys_hz, video_save_freq=vsf)
            logging.info("[RoboTwin] eval video → %s", out)
        except Exception as e:
            logging.warning("[RoboTwin] 起 eval video 失败（继续不录）：%s", e)
            try:
                self.env.eval_video_path = None
            except Exception:
                pass

    def stop_eval_video(self):
        """收尾 eval 视频（ffmpeg finalize）。"""
        try:
            if getattr(self.env, "eval_video_ffmpeg", None):
                self.env._del_eval_video_ffmpeg()
        except Exception:
            pass
        try:
            self.env.eval_video_path = None
        except Exception:
            pass

    def get_info_for_step(self, raw_obs=None) -> Tuple[bool, bool, float, float]:
        """评测终止：(done, success, reward, mask)。接管期严格走 tape（play_once 已把物理推到末态，
        真实 check_success 会从回放第一帧就 True → tape 没放完就 done）；否则稀疏 0/1 终局。"""
        if self._takeover_active and self._replayer is not None:
            done, success, reward, mask = self._replayer.current_info()
            self._success_once = self._success_once or success
            if done:
                logging.info("[RoboTwin] takeover episode done via tape: success=%s steps=%d",
                             success, self._steps_since_reset)
            return done, success, reward, mask
        success = bool(self.env.check_success())
        self._success_once = self._success_once or success
        # 用 RoboTwin 自己的 step_lim 作 episode 预算（reset 时读到），而非过小的固定 200——
        # 否则 stack_blocks_two（step_lim=800）会在 ~1/4 预算处被截断，任务做不完→success 恒 0→无 RL 信号。
        time_up = self._steps_since_reset >= self._step_budget
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

    def _ensure_episode_info(self) -> None:
        """RLinf 对齐：一次性获取并缓存 RoboTwin episode_info（占位符 {A}/{B}/{a}/{b}）。

        RLinf 用 `task.get_info()`（廉价、不动机器人）；本仓 RoboTwin **没有 get_info**，
        info["info"] 只在 `play_once` 里填，故启动时跑一次专家 play_once 取之（仅一次，之后复用——
        与 RLinf「episode_info 只取一次」一致）。play_once 需 curobo（gate-4 的 eval_policy_client 同样用它）。
        失败则 self._episode_info={}，每集指令降级到 fallback（不再重试）。
        """
        if self._episode_info is not None:
            return
        try:
            from envs.utils.create_actor import UnStableError
        except Exception:
            UnStableError = ()
        for k in range(self._max_reset_retries):
            seed = self._seed + k
            try:
                self.env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **self._setup_kwargs)
                info = self.env.play_once()                       # 填 self.env.info["info"]
                got = info.get("info") if isinstance(info, dict) else None
                self._episode_info = dict(got) if got else dict(
                    (getattr(self.env, "info", {}) or {}).get("info", {}) or {})
            except UnStableError:
                self._episode_info = None
            except Exception:
                logging.exception("[RoboTwin] play_once for episode_info failed at seed=%d", seed)
                self._episode_info = None
            try:
                self.env.close_env(clear_cache=False)
            except Exception:
                pass
            if self._episode_info:
                logging.info("[RoboTwin] cached episode_info for instructions: %s", self._episode_info)
                return
        self._episode_info = {}    # 已尝试但失败：标记避免每集重试，指令走 fallback
        logging.warning("[RoboTwin] could not capture episode_info; instructions fall back to '%s'",
                        self._lang_fallback)

    def _create_instruction(self) -> str:
        """RLinf 对齐（vector_env.py:117 create_instruction）：用缓存 episode_info 经
        `generate_episode_descriptions` 生成模板指令，每集 `np.random.choice` 随机选一条。
        生成函数在 RoboTwin/description/utils（cwd=robotwin_root，见 run_robotwin_client 的 os.chdir）。
        """
        if not self._episode_info:
            return self._resolve_instruction()
        try:
            import sys
            if "./description/utils" not in sys.path:
                sys.path.append("./description/utils")
            from generate_episode_instructions import generate_episode_descriptions
            results = generate_episode_descriptions(self.task_name, [self._episode_info], 100)
            pool = results[0].get(self.instruction_type) or results[0].get("seen") or []
            if len(pool) > 0:
                return str(np.random.choice(pool))
        except Exception:
            logging.exception("[RoboTwin] generate_episode_descriptions failed; using fallback instruction")
        return self._resolve_instruction()

    def _resolve_instruction(self) -> str:
        try:
            instr = self.env.get_instruction()
            if isinstance(instr, (list, tuple)) and instr:
                return str(np.random.choice(instr))
            if isinstance(instr, str) and instr:
                return instr
        except Exception:
            pass
        return self._lang_fallback

    def close(self):
        try:
            self.env.close_env()
        except Exception:
            logging.exception("[RoboTwin] close_env failed")
