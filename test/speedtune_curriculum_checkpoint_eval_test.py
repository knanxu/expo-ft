from configs.model.speedtune_dqn_config import get_config
from expo_ft.speedtune.checkpoint_contract import (
    build_checkpoint_metadata,
    curriculum_action_limits,
)
from expo_ft.speedtune.curriculum import SpeedCurriculum
from expo_ft.speedtune.exec_backends import build_backend


def test_checkpoint_contract_records_static_curriculum_configuration():
    config = get_config()
    metadata = build_checkpoint_metadata(config, build_backend("fixed_time"))
    assert metadata["version"] == 2
    assert metadata["curriculum_config"] == {
        "enabled": True,
        "window_size": 20,
        "success_threshold": 0.7,
    }


def test_eval_action_limit_is_restored_from_curriculum_state():
    config = get_config()
    backend = build_backend("chunk_toppra")
    metadata = build_checkpoint_metadata(config, backend)
    curriculum = SpeedCurriculum(
        speed_grid=backend.vars[0].grid,
        window_size=20,
        success_threshold=0.7,
    )
    curriculum.max_unlocked_idx = 3
    metadata["curriculum_state"] = curriculum.state_dict()

    assert curriculum_action_limits(metadata, backend.head_sizes) == (3,)


def test_missing_enabled_curriculum_state_is_rejected():
    config = get_config()
    backend = build_backend("fixed_time")
    metadata = build_checkpoint_metadata(config, backend)
    try:
        curriculum_action_limits(metadata, backend.head_sizes)
    except ValueError as error:
        assert "curriculum_state" in str(error)
    else:
        raise AssertionError("missing curriculum state was accepted")


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} curriculum checkpoint/eval tests passed")


if __name__ == "__main__":
    main()
