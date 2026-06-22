"""EXPO-FT (EXPOLearner) on a **drift** pi0.5 actor + RoboTwin 双臂 aloha env.

与 `expo_ft_pi_config.py` 的唯一区别 = 把 pi0.5 actor 从 DROID flow-matching 切到
**RoboTwin drift**（CLAUDE.md 零侵入：新增本文件，**不改** `expo_ft_pi_config.py`）。

为什么只是 config 切换、算法零改动：
  - 合并后的 openpi `pi0.py` 里 `sample_actions`(:347) / `compute_loss`(:254) 已按 `use_drifting_loss`
    自动分发到 `_sample_actions_drifting` / `_compute_loss_drifting`。EXPO 的采样（`_jitted_infer`）与
    actor 更新（`train_step`）天然吃到 drift，`expo_ft.py` 一行不用改。
  - RoboTwin obs/action 走 openpi `pi05_aloha_robotwin_drifting_*` 同一套 transforms（AlohaInputs/
    RepackTransform/Normalize），replay buffer 的 3 路图像槽 base+left/right_wrist 对上 aloha 三相机；
    EXPO `sample_actions`(:532) 已含 `process_transformed_outputs` 反归一化。

配套（算法无关，EXPO/speedtune 共用）：`configs/task/robotwin_stack_blocks.py`（env_type="sim"）。
训练入口直接复用现成 `train_pi_robo_async.py`（`EXPOLearner` 分支），无需新入口。
"""

from configs.model import sac_config


def get_config():
    config = sac_config.get_config()

    # ===== 与 expo_ft_pi_config.py 完全一致的 EXPO 超参（critic / residual / 采样）=====
    config.model_cls = "EXPOLearner"

    config.num_qs = 10
    config.num_min_qs = 2
    config.critic_layer_norm = True

    config.N = 8
    config.n_edit_samples = 8

    config.adjust_target_entropy = False
    config.entropy_scale = 1.0
    config.edit_scale = 0.2
    config.actor_drop = 0.0
    config.actor_lr = 3e-4
    config.critic_lr = 3e-4

    config.latent_dim_image = 512
    config.latent_dim_state = 64
    config.include_state = True
    config.encoder_stage_sizes = (3, 4, 6, 3)
    config.encoder_num_filters = 64
    config.hidden_dims = (256, 256, 256)

    config.encode_batch_split = 1
    config.batch_split = 1

    config.use_full_augmentation = True

    # ===== 唯一的实质改动：pi0.5 actor 指向 RoboTwin drift =====
    config.use_pi05 = True
    # 双臂 RoboTwin drifting openpi config（use_drifting_loss=True，AlohaInputs，14-D 关节空间，H=50）。
    # 若换任务，改成对应的 pi05_aloha_robotwin_drifting_{task_name}。
    config.pi05_config_name = "pi05_aloha_robotwin_drifting_stack_blocks_two"
    config.pi05_resize_size = 224
    config.freeze_pi05_encoder = True
    config.freeze_critic_encoder = False

    # 🔴 必填：RoboTwin stack-two-blocks 的 **DBP Stage-1 drift checkpoint**。
    # 留空则 actor 从 base flow 权重初始化，drift 单步生成无效（action_out_proj 出的是 velocity）。
    config.pi05_weight_loader_path = ""
    # 🔴 必填：RoboTwin DBP 训练得到的 norm_stats（assets_dir/asset_id），replay buffer 的
    # `_build_transform_pipeline` 强制要 norm_stats，否则 raise。留空则用 openpi config 默认
    # （pi05_base/assets 的 trossen），与 RoboTwin 14-D 关节统计很可能不匹配——务必指向你的 DBP norm_stats。
    config.pi05_assets_dir = ""
    config.pi05_asset_id = ""

    # RoboTwin 是 14-D 关节空间，残差**不做** xyzg 屏蔽（DROID 专用）。与 robotwin task config 一致。
    config.residual_action_xyzg = False
    # 离线 demo 作 success-only actor 更新 / critic 暖启。RoboTwinEnv 有 check_success → success 标注可用。
    config.actor_success_only = True

    # ===== rollout 执行步数（replan_steps）建议 =====
    # ⚠️ EXPO 的 replan_steps 不在本 config，而是 `train_pi_robo_async.py` 的 CLI flag（默认 8）：
    #       python train_pi_robo_async.py --config configs/model/expo_ft_pi_drift_config.py \
    #              --config_task configs/task/robotwin_stack_blocks.py --replan_steps 8 ...
    # drift 仍预测整段 H=50，replan_steps 只是「执行前多少步再 replan」。
    # **首跑用 replan_steps=8**：EXPO 里 Q/residual 维度 = replan_steps × action_dim(14)：
    #   8→112 维（可控），25→350 维（≈DROID 56 的 6×，单 env 下 SAC residual+Q 难收敛）。
    # 25/50 控制比例虽与 DROID 8/16 同（50%），但 8 给更多决策点(800/8≈100)与更密 critic 信号；
    # 基线学起来后再逐步加大。

    return config
