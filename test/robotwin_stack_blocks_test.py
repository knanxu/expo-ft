"""configs.task.robotwin_stack_blocks 契约测试（纯 CPU：ml_collections/numpy）。

import 时 RoboTwinEnv 走 try/except（learner 侧无 sapien → None）；本测试**不访问** config.env，
故 learner venv 与 RoboTwin venv 两种环境都应通过。

⚠️ 本地不跑 pytest；云端：
    python -m pytest test/robotwin_stack_blocks_test.py -v
"""

from configs.task import robotwin_stack_blocks


def test_dispatch_fields_route_to_robotwin_sim():
    c = robotwin_stack_blocks.get_config()
    assert c.env_type == "sim"  # 进 train_pi_robo_async 的 ('droid','sim') 分支
    assert c.dataset_loader == "robotwin"  # 走 process_robotwin_dataset 而非 DROID loader
    assert c.env_name == "robotwin_stack_blocks"


def test_action_space_is_14d_bimanual():
    c = robotwin_stack_blocks.get_config()
    assert tuple(c.example_action.shape) == (1, 14)  # 双臂 2×(6关节+1夹爪)
    assert c.n_real_dims == 14
    assert c.residual_action_xyzg is False  # 与 model config 一致


def test_takeover_and_instruction_defaults():
    c = robotwin_stack_blocks.get_config()
    assert c.takeover_enable is True  # 自动专家介入默认开（打破 rollout 0% 死锁）
    assert 0.0 < c.takeover_step_frac <= 1.0
    assert c.instruction_type in ("seen", "unseen")
    assert c.task_name == "stack_blocks_two"
    assert isinstance(c.language_instruction, str) and c.language_instruction  # fallback 指令非空
