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

    # --- SpeedTune 执行层力矩底座（真机 ARX5 τ_max）。per-joint 单臂(6), N·m；空串=不施加(∞)=旧行为。
    #     仅 SpeedTune 读此字段；EXPO/BC config 不含 → 零侵入。learner 端 parse_force_limit 解析后经
    #     env_creation_request 透传，env 每次 reset 后施加（见 robotwin_env._apply_force_limit）。---
    config.force_limit = os.environ.get("SPEEDTUNE_FORCE_LIMIT", "30,40,30,15,10,10")

    # --- 执行方式（三选一，对应 exec_backends；决定 DQN head 数与动作空间）---
    # "fixed_time"(1 head) / "per_action_toppra"(3 head) / "chunk_toppra"(1 head)
    config.exec_backend = "per_action_toppra"
    # Backend-specific frame skip：fixed_time 必须满足 k_skip*v_max<=H；whole-chunk
    # 不做 reconstruct(v)，使用更长窗口提高单次 TOPPRA 的有效路径长度。
    config.k_skip = 10                    # per_action 兼容默认
    config.fixed_time_k_skip = 10
    config.chunk_toppra_k_skip = 40
    config.stream_hold_steps = 15        # fixed_time(streaming) 每目标 hold 物理步（250/15≈16.7Hz，对齐采集）

    # --- 动作时序：drift 双臂 chunk H=50；SpeedTune 决策粒度 = 一整段 chunk ---
    config.action_horizon = 50           # H（pi0.5 chunk 长度）
    config.n_real_dims = 14              # 双臂真实维 2×(6关节+1夹爪)

    # --- branching Rainbow-DQN ---
    config.n_atoms = 100                 # C51 原子数
    config.v_min = -1.0                  # per_action 兼容 support
    config.v_max = 11.0
    # r_max=4²=16, gamma=.99, step_lim=800：Qmin=0；fixed 最多80决策，
    # Qmax=16*(1-.99^80)/(1-.99)=883.963；whole 最多40决策，Qmax=529.645。
    config.fixed_time_support = (0.0, 900.0)
    config.chunk_toppra_support = (0.0, 550.0)
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
    config.epsilon_decay_steps = 4000    # 决策步数；实机/仿真小样本实验不宜 20k 慢退火

    # --- 单速度head课程（fixed_time / chunk_toppra）---
    config.curriculum_enabled = True
    config.curriculum_window_size = 20
    config.curriculum_success_threshold = 0.7

    # --- replay / 训练循环 ---
    config.buffer_capacity = 100000
    config.batch_size = 64
    config.learning_starts = 1000        # 旧 async checkpoint 配置兼容；新 trainer 不再用它作启动门槛
    config.warmup_episodes = 10          # 对齐 train_pi_robo：至少完成 10 个 episode 后开始更新
    # P 方式 replay（值据 expo-ft 论文超参确定）：每 episode 末做 update_per_episode 次 update 调用，
    # 每次 utd_ratio 个梯度步 → 每 episode 共 update_per_episode×utd_ratio 个梯度步。
    config.utd_ratio = 20                # 每次 update 调用的梯度步数（P: 循环 utd_ratio 次 learner.update）
    config.update_per_episode = 6        # 每 episode 的 update 调用次数
    config.target_update_period = 1      # 软更新每步做；>1 时改周期硬更新（train 脚本支持）

    # --- reward 统一覆盖钮（None=用 exec_backends 各变量默认 α/β）---
    # success_gated: 现有防 reward hacking reward，失败 episode 速度项为 0。
    # paper_speedtuning: 复现 SpeedTuning 论文 r=α*v^β+r_task，速度项不 success-gated。
    config.reward_mode = "success_gated"
    config.reward_alpha = 1.0
    config.reward_beta = 2.0
    # Paper mode 的 C51 support 动态估计用高层 action step 数，不用物理仿真 dense step。
    config.paper_speedtuning_episode_steps = 800
    config.paper_speedtuning_task_reward_max = 1.0

    # --- 杂项 ---
    # 终止预算对齐 train_pi_robo 的环境交互步：一次 step = 一次 VLA+DQN 决策和 env.step_chunk；
    # 不按 gradient update 数或 chunk 内 250Hz dense simulation steps 计数。
    config.max_iters = 40000
    config.seed = 0

    return config
