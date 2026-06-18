"""Config for DBPOLearner: on-policy (PPO/BPO) RL on a single-step drift pi0.5 actor.

超参分两组：
  [A] 严格对齐 DBPO 原实现（github.com/YuxuanGao0822/DBPO，见 docs/DBPO_REFERENCE_VS_OURS.md）。
  [B] 迁到 pi0.5 VLA（双臂 H=50, d_a=14）必须偏离 DBPO 原设计的项（见文件末 NOTE）。

DBPO 原值来源：`configs/finetune.yaml` + `dbpo/methods/dbpo/ppo_adapter.py` 默认。
"""

import math

import ml_collections
from ml_collections.config_dict import config_dict


def get_config():
    config = ml_collections.ConfigDict()
    config.model_cls = "DBPOLearner"

    # --- pi0.5 drift actor ---
    config.use_pi05 = True
    # RoboTwin 双臂 drifting 配置（openpi `_robotwin_drifting_config`：use_drifting_loss=True，
    # AlohaInputs/RepackTransform，action 14-D 关节空间）。task_name 改成你的真实任务名。
    config.pi05_config_name = "pi05_aloha_robotwin_drifting_stack_blocks_two"
    config.pi05_resize_size = 224
    config.pi05_weight_loader_path = ""  # RoboTwin stack-two-blocks DBP (Stage-1) drift ckpt
    config.pi05_assets_dir = ""
    config.pi05_asset_id = ""
    # [B] DBPO trains the full small U-Net; pi0.5 是 3B，必须冻结 VLM/SigLIP 只训 action expert/投影
    # （否则 adam≈24GB）。这是迁 VLA 的必要偏离（非超参）。
    config.actor_trainable_regex = ".*(action_.*proj|state_proj|time_mlp|action_time_mlp|llm.*_1).*"

    # ========== [A] 严格对齐 DBPO 原实现 ==========
    # --- surrogate：DBPO 原版 = 纯 PPO；BPO 是本项目扩展 ---
    config.rl_surrogate = "ppo"   # DBPO-faithful base；切 "bpo" 时见下 clip_eps 注释
    # PPO clip ε：DBPO clip_ploss_coef=0.02。⚠️ 切 BPO 时改成 0.3（BPO repo clip_ratio）。
    config.clip_eps = 0.02
    config.bpo_lambda = 1e-3      # BPO repo lambda_（温度）
    # ⭐ DBPO 默认 normalize_act_space_dimension=True：executed-prefix logp 按 N=H_e·d_a 归一化
    # （per-coordinate mean log-prob）。这是 DBPO 控制高维 ratio 的核心（Var[log r]~1/N）。
    # 你之前漏了它（raw sum）→ 是 700 维不稳的真正根因。迁 pi0.5 必须保持 True。
    config.normalize_dims = True
    config.gae_lambda = 0.95
    config.discount = 0.99        # γ：直接用在决策步上（DBPO 不做 γ^H_e）
    config.c_value = 0.5          # vf_coef
    config.c_entropy = 0.01       # ent_coef（DBPO=0.01；归一化后与 logp 同尺度，可转移）
    config.lambda_anchor = 1.0    # anchor_loss_coeff（见 NOTE：anchor 归约需对齐 / VLA 宜增强）
    config.normalize_adv = True
    config.value_clip_eps = config_dict.placeholder(float)  # DBPO clip_vloss_coef 默认 None

    # --- log-std 头 g_psi：DBPO clamp 到 [log0.03, log0.10] 的窄带（强 ratio 稳定器）---
    config.logstd_hidden_dims = (256, 256)
    config.logstd_init = math.log(0.03)   # ≈ -3.51（DBPO init_logprob_std=0.03）
    config.logstd_min = math.log(0.03)    # ≈ -3.51（DBPO min_logprob_std）
    config.logstd_max = math.log(0.10)    # ≈ -2.30（DBPO max_logprob_std）

    # --- DBPO 采样/数值稳定器（Phase-4b train loop / sampler 接入）---
    config.min_sampling_std = 0.03        # 采样 std 下限（探索）
    config.randn_clip_value = 3.0         # 采样截断到 mean ± 3σ（限每坐标 (a-μ)/σ ≤ 3）
    config.logprob_min = -2.0             # logp clamp 下限
    config.target_kl = 0.02               # approx_kl 早停阈值
    config.n_critic_warmup_iters = 5      # 前 N iter 只训 critic
    config.max_grad_norm = 0.5            # 梯度裁剪

    # --- value 头 V_phi ---
    config.value_hidden_dims = (256, 256)

    # --- 学习率（DBPO actor_lr=1e-5, critic_lr=1e-4；logstd 头在 DBPO 属 actor，故用 actor_lr）---
    config.actor_lr = 1e-5
    config.logstd_lr = 1e-5               # 对齐 DBPO（logstd 在 actor_ft，actor_lr）；可酌情上调
    config.value_lr = 1e-4               # DBPO critic_lr

    # ========== [B] pi0.5 / 高维必要设置 ==========
    # 动作时序：drift 双臂 chunk H=50。H_e=replan_steps（≡env replan 间隔/RoboTwin pi0_step）。
    # 决策（2026-06-18）：预测 50、**执行前 25 步**（receding-horizon，H_e=25）。normalize_dims=True
    # 下 ratio 已稳；H_e=25 给每 episode ~8 决策步（200/25），信用分配比整段 50（仅 4 步）更细。
    config.replan_steps = 25             # H_e（执行前缀 25；env pi0_step 必须也=25）
    config.n_real_dims = 14              # 双臂真实维 2×(6关节+1夹爪)，其余 padding 由 dim_mask 屏蔽

    # on-policy rollout / update（[B] 单 env：远小于 DBPO 的 n_steps=500×n_envs=40 / batch=50000）
    config.rollout_size = 256            # 单 env 每轮决策步
    config.ppo_epochs = 5               # DBPO update_epochs=5（之前 10 来自 BPO 建议，已对齐 DBPO）
    config.num_minibatches = 8

    return config
