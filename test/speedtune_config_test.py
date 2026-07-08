from configs.model.speedtune_dqn_config import get_config
from expo_ft.speedtune.runtime_config import (
    backend_k_skip,
    backend_support,
    episode_reward_success,
    finite_horizon_q_max,
)


def test_speedtune_config_uses_one_hundred_c51_atoms():
    config = get_config()
    assert config.n_atoms == 100
    assert config.reward_alpha == 1.0
    assert config.reward_beta == 2.0
    assert config.reward_mode == "success_gated"
    assert config.paper_speedtuning_episode_steps == 800


def test_backend_specific_k_skip_and_support():
    config = get_config()
    assert backend_k_skip(config, "fixed_time") == 10
    assert backend_k_skip(config, "chunk_toppra") == 40
    assert backend_k_skip(config, "per_action_toppra") == 10
    assert backend_support(config, "fixed_time") == (0.0, 900.0)
    assert backend_support(config, "chunk_toppra") == (0.0, 550.0)
    assert backend_support(config, "per_action_toppra") == (-1.0, 11.0)


def test_support_covers_finite_horizon_raw_speed_squared_q_range():
    config = get_config()
    fixed_q_max = finite_horizon_q_max(
        4.0 ** 2, config.gamma, 800 // config.fixed_time_k_skip
    )
    chunk_q_max = finite_horizon_q_max(
        4.0 ** 2, config.gamma, 800 // config.chunk_toppra_k_skip
    )
    assert abs(fixed_q_max - 883.9628579779023) < 1e-9
    assert abs(chunk_q_max - 291.3488998444305) < 1e-9
    fixed_support = backend_support(config, "fixed_time")
    chunk_support = backend_support(config, "chunk_toppra")
    assert fixed_support == (0.0, 900.0)
    assert chunk_support == (0.0, 550.0)
    assert fixed_support[1] >= fixed_q_max
    assert chunk_support[1] >= chunk_q_max


def test_paper_speedtuning_support_scales_with_alpha():
    config = get_config()
    config.reward_mode = "paper_speedtuning"
    config.reward_alpha = 1e-4
    config.reward_beta = 2.0
    fixed_support = backend_support(config, "fixed_time")
    chunk_support = backend_support(config, "chunk_toppra")
    expected_fixed_max = 1.0 + finite_horizon_q_max(
        1e-4 * 4.0 ** 2, config.gamma, 800 // config.fixed_time_k_skip
    )
    expected_chunk_max = 1.0 + finite_horizon_q_max(
        1e-4 * 4.0 ** 2, config.gamma, 800 // config.chunk_toppra_k_skip
    )
    assert fixed_support[0] == 0.0
    assert chunk_support[0] == 0.0
    assert abs(fixed_support[1] - expected_fixed_max) < 1e-9
    assert abs(chunk_support[1] - expected_chunk_max) < 1e-9
    assert fixed_support[1] > chunk_support[1]


def test_episode_reward_success_gates_task_failure_and_speed_violation():
    assert episode_reward_success("fixed_time", True, False) is True
    assert episode_reward_success("fixed_time", False, False) is False
    assert episode_reward_success("fixed_time", True, True) is False
    assert episode_reward_success("chunk_toppra", True, True) is True
    assert episode_reward_success("chunk_toppra", False, False) is False


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} speedtune config tests passed")
