from pathlib import Path

from configs.model.speedtune_dqn_config import get_config


ROOT = Path(__file__).parents[1]


def test_config_enables_twenty_episode_seventy_percent_curriculum():
    config = get_config()
    assert config.curriculum_enabled is True
    assert config.curriculum_window_size == 20
    assert config.curriculum_success_threshold == 0.7


def test_sync_trainer_masks_actions_updates_and_checkpoints_curriculum():
    source = (ROOT / "train_speedtune_sync.py").read_text()
    assert "build_speed_curriculum" in source
    assert "max_action_idxs=episode_action_limits" in source
    assert "curriculum.record_action(idxs)" in source
    assert "curriculum.record_episode(" in source
    assert '"curriculum_state": (' in source
    assert "curriculum.state_dict() if curriculum is not None else None" in source
    assert "if not force_terminal:" in source


def test_async_tokens_carry_episode_action_limit():
    source = (ROOT / "train_speedtune_async.py").read_text()
    assert "build_speed_curriculum" in source
    assert "episode_step, episode_action_limits = update_token" in source
    assert "_episode_updates.put((_step[0], completed_action_limits))" in source
    assert "max_action_idxs=episode_action_limits" in source
    assert "curriculum.record_action(idxs)" in source
    assert "curriculum.record_episode(" in source
    assert '"curriculum_state": (' in source
    assert "curriculum.state_dict() if curriculum is not None else None" in source


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} SpeedTune trainer curriculum tests passed")


if __name__ == "__main__":
    main()
