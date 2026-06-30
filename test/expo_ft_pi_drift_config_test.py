"""configs.model.expo_ft_pi_drift_config 契约测试（纯 CPU：ml_collections，**不依赖** jax/torch）。

锁住三件事：
  ① dispatch 关键字段（误改 → 整个 pipeline 走错分支）；
  ② 三个必填占位留空的契约（污染检测：若有人把本地占位路径写进 config，此测试 fail 提醒
     这三个该由云端 env var VLA_CKPT/VLA_ASSETS_DIR/VLA_ASSET_ID 经 CLI 覆盖填入）；
  ③ 与 DROID baseline 的 EXPO 超参逐字段对齐 —— 把项目核心论断「drift 对 EXPO 是纯 config
     切换、算法零改动」变成可执行断言。

⚠️ 本地不跑 pytest；云端 expo-ft venv：
    python -m pytest test/expo_ft_pi_drift_config_test.py -v
"""

from configs.model import expo_ft_pi_config, expo_ft_pi_drift_config


# drift config 注释声称「与 expo_ft_pi_config 完全一致」的 EXPO 算法超参（不含有意差异字段）。
_CORE_EXPO_HPARAMS = (
    "model_cls",
    "num_qs",
    "num_min_qs",
    "critic_layer_norm",
    "N",
    "n_edit_samples",
    "adjust_target_entropy",
    "entropy_scale",
    "edit_scale",
    "actor_drop",
    "actor_lr",
    "critic_lr",
    "latent_dim_image",
    "latent_dim_state",
    "include_state",
    "encoder_stage_sizes",
    "encoder_num_filters",
    "hidden_dims",
    "encode_batch_split",
    "batch_split",
    "use_pi05",
    "pi05_resize_size",
    "freeze_pi05_encoder",
    "freeze_critic_encoder",
    "actor_success_only",
    "use_full_augmentation",
    "pi05_weight_loader_path",
    "pi05_assets_dir",
    "pi05_asset_id",
)


def test_get_config_dispatches_to_expo_drift_robotwin():
    c = expo_ft_pi_drift_config.get_config()
    assert c.model_cls == "EXPOLearner"
    assert c.use_pi05 is True
    assert c.pi05_config_name == "pi05_aloha_robotwin_drifting_stack_blocks_two"
    assert c.residual_action_xyzg is False  # RoboTwin 14-D 关节空间，不做 DROID 的 xyzg 屏蔽
    assert c.actor_success_only is True


def test_required_cloud_fill_fields_left_blank():
    # 必填占位必须留空（交云端 env var 经 --config.pi05_* 覆盖）。误填本地路径 = 污染 → 此处报警。
    c = expo_ft_pi_drift_config.get_config()
    assert c.pi05_weight_loader_path == ""
    assert c.pi05_assets_dir == ""
    assert c.pi05_asset_id == ""


def test_drift_matches_droid_expo_hparams_except_actor_pointer():
    # 除 pi05_config_name(指向 RoboTwin drift) / residual_action_xyzg(14-D) 两处有意差异外，
    # drift 与 DROID 的 EXPO 算法超参应逐字段相等 —— 任何漂移说明误改了既有算法配置。
    drift = expo_ft_pi_drift_config.get_config()
    droid = expo_ft_pi_config.get_config()
    for k in _CORE_EXPO_HPARAMS:
        assert drift[k] == droid[k], (
            f"EXPO hparam '{k}' drifted from DROID baseline: {drift[k]!r} != {droid[k]!r}"
        )
