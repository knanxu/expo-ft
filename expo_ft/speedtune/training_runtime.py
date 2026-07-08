"""Shared episode-level runtime helpers for SpeedTune trainers."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from expo_ft.speedtune.runtime_config import episode_reward_success


def paper_speedtuning_reward(
    v_list: Sequence[float],
    r_task: float,
    *,
    alpha: float,
    beta: float,
) -> float:
    """Paper SpeedTuning reward for one chunk-level DQN transition."""
    speed = sum(max(0.0, float(value)) ** float(beta) for value in v_list)
    return float(float(alpha) * speed + float(r_task))


def update_ready(
    *,
    completed_episodes: int,
    replay_size: int,
    batch_size: int,
    warmup_episodes: int = 10,
) -> bool:
    """Match ``train_pi_robo.py``: wait for episodes and one full batch."""
    return (
        int(completed_episodes) >= int(warmup_episodes)
        and int(replay_size) >= int(batch_size)
    )


def run_episode_updates(
    learner,
    buffer,
    *,
    batch_size: int,
    beta: float,
    update_groups: int,
    utd_ratio: int,
    max_action_idxs=None,
    on_group_end: Optional[Callable] = None,
):
    """Run the exact per-episode update schedule and aggregate scalar metrics."""
    metrics_history = []
    n_updates = 0
    for _ in range(int(update_groups)):
        for _ in range(int(utd_ratio)):
            batch = buffer.sample(int(batch_size), beta=float(beta))
            learner, td, metrics = learner.update(
                batch, max_action_idxs=max_action_idxs
            )
            buffer.update_priorities(batch["tree_indices"], np.asarray(td))
            metrics_history.append(metrics)
            n_updates += 1
        if on_group_end is not None:
            on_group_end(learner)

    aggregate = {}
    if metrics_history:
        scalar_keys = [
            key
            for key, value in metrics_history[0].items()
            if np.isscalar(value) or getattr(value, "ndim", 1) == 0
        ]
        aggregate = {
            key: float(np.mean([float(metrics[key]) for metrics in metrics_history]))
            for key in scalar_keys
        }
    return learner, aggregate, n_updates


def flush_pending_episode(
    pending,
    buffer,
    backend,
    *,
    task_success: bool,
    speed_violation: bool,
    force_terminal: bool = False,
    reward_mode: str = "success_gated",
    reward_alpha: float = 1.0,
    reward_beta: float = 2.0,
    gamma: float = 0.99,
):
    """Relabel and insert one episode after its final outcome is known.

    ``force_terminal`` is used only when the environment-decision budget ends in the
    middle of an episode. It prevents n-step state leaking while assigning no success
    reward to the incomplete trajectory.
    """
    if not pending:
        return None

    effective_success = bool(task_success) and not bool(force_terminal)
    reward_mode = str(reward_mode)
    if reward_mode == "paper_speedtuning":
        reward_success = effective_success
    elif reward_mode == "success_gated":
        reward_success = episode_reward_success(
            backend.name, effective_success, bool(speed_violation)
        )
    else:
        raise ValueError(
            f"unknown SpeedTune reward_mode {reward_mode!r}; "
            "expected 'success_gated' or 'paper_speedtuning'"
        )

    last_index = len(pending) - 1
    execution_steps_total = 0
    for index, transition in enumerate(pending):
        next_feat = (
            pending[index + 1]["feat"]
            if index < last_index
            else transition["feat"]
        )
        done = bool(transition["done"]) or (
            bool(force_terminal) and index == last_index
        )
        execution_steps = max(int(transition.get("execution_steps", 1) or 1), 1)
        execution_steps_total += execution_steps
        if reward_mode == "paper_speedtuning":
            r_task = 0.0 if bool(force_terminal) else float(transition["r_task"])
            reward = paper_speedtuning_reward(
                transition["v_list"],
                r_task,
                alpha=float(reward_alpha),
                beta=float(reward_beta),
            )
            buffer.insert(
                transition["feat"],
                transition["action_idxs"],
                reward,
                next_feat,
                done,
                discount=float(gamma),
            )
        else:
            reward = backend.total_reward(
                transition["r_task"], reward_success, transition["v_list"]
            )
            buffer.insert(
                transition["feat"],
                transition["action_idxs"],
                reward,
                next_feat,
                done,
                discount=float(gamma),
            )

    return {
        "task_success": effective_success,
        "reward_success": bool(reward_success),
        "speed_violation": bool(speed_violation),
        "length": len(pending),
        "exec_time_s": float(sum(item["duration"] for item in pending)),
        "dense_steps": int(sum(item["n_exec"] for item in pending)),
        "execution_steps": int(execution_steps_total),
        "truncated": bool(force_terminal),
    }
