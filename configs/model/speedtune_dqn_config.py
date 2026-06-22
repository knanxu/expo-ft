"""Config for SpeedTuneLearner: branching Rainbow-DQN 加速模块（冻结 VLA 之上，两阶段解耦）。

零侵入增量（与 EXPO/BC config 平级，不改它们）。三种执行方式用**同一份 config**、
靠 ``exec_backend`` 切换（三个独立 run）：

    python train_speedtune_async.py --config configs/model/speedtune_dqn_config.py \
        --config.exec_backend per_action_toppra ...

VLA 字段复用 expo_ft_pi_drift_config 的 pi0.5 加载约定；但 VLA **全程冻结**（不训），故无
``actor_trainable_regex``。动作空间/reward 的 grid·α·β 见
``expo_ft/speedtune/exec_backends.py``（默认已通过防-reward-hacking 验证）。
"""

import os

import ml_collections
from ml_collections.config_dict import config_dict


def get_config():
    config = ml_collections.ConfigDict()
    config.model_cls = "SpeedTuneLearner"

    # --- 冻结 pi0.5 drift VLA（只前向产 suffix_feat，不训练）---
    config.use_pi05 = True
    config.pi05_config_name = "pi05_aloha_robotwin_drifting_stack_blocks_two"
    config.pi05_resize_size = 224
    # 冻结 VLA ckpt = 微调后 drift action head 的 pi0.5（云端训练产物）。用环境变量传，
    # 便于三个 exec_backend run 复用同一 ckpt；也可直接改成绝对路径。
    config.pi05_weight_loader_path = os.environ.get("SPEEDTUNE_VLA_CKPT", "")
    config.pi05_assets_dir = os.environ.get("SPEEDTUNE_VLA_ASSETS", "")     # drift norm_stats 所在 assets 目录
    config.pi05_asset_id = os.environ.get("SPEEDTUNE_VLA_ASSET_ID", "")     # 如 "robotwin"

    # --- 执行方式（三选一，对应 exec_backends；决定 DQN head 数与动作空间）---
    # "fixed_time"(1 head) / "per_action_toppra"(3 head) / "chunk_toppra"(3 head)
    config.exec_backend = "per_action_toppra"
    # 论文式 frame skip：streaming/per_action 每决策只执行 reconstruct(v) 后前 k_skip 个 action 即重推
    # VLA（闭环），缩短 MDP horizon。whole_chunk（整段 TOPPRA）忽略此项。None/0=整段。
    config.k_skip = 10
    config.stream_hold_steps = 15        # fixed_time(streaming) 每目标 hold 物理步（250/15≈16.7Hz，对齐采集）

    # --- 动作时序：drift 双臂 chunk H=50；SpeedTune 决策粒度 = 一整段 chunk ---
    config.action_horizon = 50           # H（pi0.5 chunk 长度）
    config.n_real_dims = 14              # 双臂真实维 2×(6关节+1夹爪)

    # --- branching Rainbow-DQN ---
    config.n_atoms = 51                  # C51 原子数
    config.v_min = -1.0                  # value support 下限（按 reward 量级；见下）
    config.v_max = 11.0                  # 上限（留余量：reward≈r_task + Σαᵢvᵢ^βᵢ，n-step 折扣累积）
    config.dueling = True
    config.q_hidden_dims = (256, 256)
    config.detach_q_input = True         # stop_gradient suffix_feat（VLA 冻结，再加一道保险）
    config.q_lr = 1e-4
    config.max_grad_norm = 10.0
    config.tau = 0.005                   # target 软更新
    config.gamma = 0.99                  # 折扣（n-step 聚合用）
    config.n_step = 3                    # n-step return

    # --- PER ---
    config.per_alpha = 0.6
    config.per_beta = 0.4                # 训练中可线性退火到 1.0（train 脚本控制）
    config.per_beta_end = 1.0
    config.per_eps = 1e-6

    # --- 探索（epsilon-greedy，每 head 独立）---
    config.epsilon_start = 1.0
    config.epsilon_end = 0.05
    config.epsilon_decay_steps = 20000   # 决策步数

    # --- replay / 训练循环 ---
    config.buffer_capacity = 100000
    config.batch_size = 64
    config.learning_starts = 1000        # buffer 攒够多少决策步才开始训
    config.updates_per_step = 1          # 每个决策步训几次 DQN
    config.target_update_period = 1      # 软更新每步做；>1 时改周期硬更新（train 脚本支持）

    # --- reward 统一覆盖钮（None=用 exec_backends 各变量默认 α/β）---
    # 细调单个变量请改 exec_backends._default_specs；这里是一键统一缩放。
    config.reward_alpha = config_dict.placeholder(float)
    config.reward_beta = config_dict.placeholder(float)

    # --- 杂项 ---
    config.max_iters = 100000            # 总决策步预算
    config.seed = 0

    return config
