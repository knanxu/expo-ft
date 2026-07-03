from expo_ft.speedtune.paired_eval import (
    noise_seed,
    paired_metrics,
    select_diagnostic_episode_ids,
)


def test_noise_seed_is_deterministic_per_episode_and_decision():
    assert noise_seed(42, 3, 7) == noise_seed(42, 3, 7)
    assert noise_seed(42, 3, 7) != noise_seed(42, 3, 8)
    assert noise_seed(42, 3, 7) != noise_seed(42, 4, 7)


def test_paired_metrics_use_only_jointly_successful_episode_indices():
    fixed = [
        {"ep": 0, "task_success": True, "n_decision_steps": 20, "total_dense_steps": 200, "end_to_end_wall_s": 4.0},
        {"ep": 1, "task_success": True, "n_decision_steps": 30, "total_dense_steps": 300, "end_to_end_wall_s": 6.0},
        {"ep": 2, "task_success": False, "n_decision_steps": 5, "total_dense_steps": 50, "end_to_end_wall_s": 1.0},
    ]
    chunk = [
        {"ep": 0, "task_success": True, "n_decision_steps": 10, "total_dense_steps": 100, "end_to_end_wall_s": 2.0},
        {"ep": 1, "task_success": False, "n_decision_steps": 10, "total_dense_steps": 100, "end_to_end_wall_s": 2.0},
        {"ep": 2, "task_success": True, "n_decision_steps": 10, "total_dense_steps": 100, "end_to_end_wall_s": 2.0},
    ]
    result = paired_metrics(fixed, chunk)
    assert result["paired_success_count"] == 1
    assert result["paired_episode_ids"] == [0]
    assert result["physics_speedup"]["median"] == 2.0
    assert result["wall_speedup"]["median"] == 2.0
    assert result["decision_step_ratio"]["median"] == 2.0


def test_paired_metrics_return_none_statistics_without_joint_success():
    result = paired_metrics(
        [{"ep": 0, "task_success": True, "n_decision_steps": 1, "total_dense_steps": 1, "end_to_end_wall_s": 1}],
        [{"ep": 0, "task_success": False, "n_decision_steps": 1, "total_dense_steps": 1, "end_to_end_wall_s": 1}],
    )
    assert result["paired_success_count"] == 0
    assert result["physics_speedup"]["median"] is None


def test_paired_metrics_can_require_safe_reward_success():
    fixed = [{"ep": 0, "task_success": True, "reward_success": False,
              "n_decision_steps": 2, "total_dense_steps": 2, "end_to_end_wall_s": 2}]
    chunk = [{"ep": 0, "task_success": True, "reward_success": True,
              "n_decision_steps": 1, "total_dense_steps": 1, "end_to_end_wall_s": 1}]
    assert paired_metrics(fixed, chunk)["paired_success_count"] == 1
    assert paired_metrics(fixed, chunk, success_key="reward_success")["paired_success_count"] == 0


def _episode(ep, *, task=True, reward=True):
    return {
        "ep": ep,
        "task_success": task,
        "reward_success": reward,
        "n_decision_steps": 1,
        "total_dense_steps": 1,
        "end_to_end_wall_s": 1.0,
    }


def test_video_selection_prioritizes_task_failures_then_safety_then_success():
    fixed = [
        _episode(0),
        _episode(1, task=False, reward=False),
        _episode(2),
        _episode(3, reward=False),
        _episode(4),
        _episode(5),
        _episode(6, task=False, reward=False),
    ]
    chunk = [
        _episode(0),
        _episode(1),
        _episode(2, task=False, reward=False),
        _episode(3),
        _episode(4),
        _episode(5),
        _episode(6),
    ]

    selected = select_diagnostic_episode_ids(fixed, chunk, limit=5)

    assert selected == [1, 2, 6, 3, 0]


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} paired eval tests passed")
