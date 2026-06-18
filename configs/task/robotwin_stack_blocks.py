"""RoboTwin stack-two-blocks task config (双侧通用：env-server + learner)。

仿 `configs/task/pick.py` 的依赖隔离模式：RoboTwinEnv 的 import 用 try/except 守护，
所以 learner（无 RoboTwin/sapien 依赖）也能 import 本 config 取参数；只有 env-server
（`client_robotwin/run_robotwin_client.py`，在 RoboTwin venv）会真正用到 `config.env`。

**不改** `pick.py`/`real_base.py` 等既有 config（CLAUDE.md 零侵入）。
"""

import numpy as np
import ml_collections

# 依赖隔离：仅 env-server（RoboTwin venv）能成功 import；learner 侧 except 跳过。
try:
    from client_robotwin.envs.robotwin_env import RoboTwinEnv
except Exception:
    RoboTwinEnv = None
    print("Not importing robotwin env [module] (learner 侧正常)")


def get_config():
    config = ml_collections.ConfigDict()

    # --- env-server 用（create_env 读取）---
    if RoboTwinEnv is not None:
        config.env = RoboTwinEnv
    config.env_name = "robotwin_stack_blocks"
    config.language_instruction = "stack the two blocks"

    # --- learner 用（train_pi_robo_dbpo 读取）---
    config.env_type = "sim"               # 走 train_pi_robo 的 ('droid','sim') 分支
    config.control_hz = 25                # RoboTwin 控制频率（按你的设置确认）
    config.residual_action_xyzg = False   # RoboTwin 用 14 维关节动作，非 DROID 的 xyzg 残差
    config.example_action = np.zeros((1, 14), dtype=np.float32)  # 双臂 14 维占位

    # --- RoboTwinEnv 构造 kwargs（经 config-as-kwargs 传入；RoboTwinEnv 按名提取，余者 **kwargs 吸收）---
    config.task_name = "stack_blocks_two"     # ⚠️ 改成你 RoboTwin 里 stack-two-blocks 的真实任务名（envs/{task_name}.py）
    config.exec_backend = "per_action"        # 方式2（逐 action TOPP）；与 per-action step 协议匹配
    config.n_real_dims = 14                   # 双臂 2×(6关节+1夹爪)
    config.max_decision_steps = 200           # 每 episode 最多 env-step（对齐 RLinf max_episode_steps）
    config.seed = 0
    # RoboTwin 场景配置（透传给 setup_demo(**robotwin_task_config)）：相机/渲染/data_type 等。
    # ⚠️ 必须启用 rgb(head/left/right_camera) + qpos，否则 get_obs 取不到图像/14 维 state。
    # 建议直接加载 RoboTwin 的 task_config yaml（如 /home/xukainan/RoboTwin/task_config/<task>.yml）。
    config.robotwin_task_config = ml_collections.ConfigDict({
        "data_type": {"rgb": True, "qpos": True, "endpose": False},
        # TODO(用户): 按 RoboTwin task_config yaml 补全 camera / render_freq / domain_randomization 等。
    })

    return config
