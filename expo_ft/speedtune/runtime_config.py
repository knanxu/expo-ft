"""Backend-specific SpeedTune runtime hyperparameters."""


def finite_horizon_q_max(max_reward: float, gamma: float, max_decisions: int) -> float:
    """Constant max-reward policy's finite-horizon discounted return."""
    if max_decisions <= 0:
        return 0.0
    if gamma == 1.0:
        return float(max_reward) * int(max_decisions)
    return float(max_reward) * (1.0 - float(gamma) ** int(max_decisions)) / (1.0 - float(gamma))


def backend_k_skip(config, backend_name: str):
    if backend_name == "fixed_time":
        return int(config.fixed_time_k_skip)
    if backend_name == "chunk_toppra":
        return int(config.chunk_toppra_k_skip)
    value = config.get("k_skip", None)
    return None if value in (None, 0) else int(value)


def backend_support(config, backend_name: str):
    reward_mode = str(config.get("reward_mode", "success_gated"))
    if reward_mode == "paper_speedtuning":
        reward_alpha = config.get("reward_alpha", 1.0)
        reward_beta = config.get("reward_beta", 2.0)
        reward_alpha = 1.0 if reward_alpha is None else float(reward_alpha)
        reward_beta = 2.0 if reward_beta is None else float(reward_beta)
        max_speed_reward = reward_alpha * (4.0 ** reward_beta)
        support_max = float(config.get("paper_speedtuning_task_reward_max", 1.0))
        support_max += finite_horizon_q_max(
            max_speed_reward,
            float(config.gamma),
            int(config.get("paper_speedtuning_episode_steps", 800)),
        )
        return 0.0, float(support_max)

    if backend_name == "fixed_time":
        values = config.fixed_time_support
    elif backend_name == "chunk_toppra":
        values = config.chunk_toppra_support
    else:
        values = (config.v_min, config.v_max)
    return float(values[0]), float(values[1])


def episode_reward_success(
    backend_name: str, task_success: bool, fixed_time_speed_violation: bool
) -> bool:
    """任务成功且（仅 fixed-time）未规划超速时，episode 才获得速度奖励。"""
    violation = bool(fixed_time_speed_violation) if backend_name == "fixed_time" else False
    return bool(task_success and not violation)
