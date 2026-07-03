"""Shared episode-level runtime helpers for SpeedTune trainers."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from expo_ft.speedtune.runtime_config import episode_reward_success


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
    on_group_end: Optional[Callable] = None,
):
    """Run the exact per-episode update schedule and aggregate scalar metrics."""
    metrics_history = []
    n_updates = 0
    for _ in range(int(update_groups)):
        for _ in range(int(utd_ratio)):
            batch = buffer.sample(int(batch_size), beta=float(beta))
            learner, td, metrics = learner.update(batch)
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
):
    """Relabel and insert one episode after its final outcome is known.

    ``force_terminal`` is used only when the environment-decision budget ends in the
    middle of an episode. It prevents n-step state leaking while assigning no success
    reward to the incomplete trajectory.
    """
    if not pending:
        return None

    effective_success = bool(task_success) and not bool(force_terminal)
    reward_success = episode_reward_success(
        backend.name, effective_success, bool(speed_violation)
    )
    last_index = len(pending) - 1
    for index, transition in enumerate(pending):
        next_feat = (
            pending[index + 1]["feat"]
            if index < last_index
            else transition["feat"]
        )
        done = bool(transition["done"]) or (
            bool(force_terminal) and index == last_index
        )
        reward = backend.total_reward(
            transition["r_task"], reward_success, transition["v_list"]
        )
        buffer.insert(
            transition["feat"],
            transition["action_idxs"],
            reward,
            next_feat,
            done,
        )

    return {
        "task_success": effective_success,
        "reward_success": bool(reward_success),
        "length": len(pending),
        "exec_time_s": float(sum(item["duration"] for item in pending)),
        "dense_steps": int(sum(item["n_exec"] for item in pending)),
        "truncated": bool(force_terminal),
    }
