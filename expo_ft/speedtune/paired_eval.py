"""Pure helpers for reproducible, paired SpeedTune backend evaluation."""

import numpy as np


def noise_seed(global_seed: int, episode: int, decision_step: int) -> int:
    state = np.random.SeedSequence(
        [int(global_seed), int(episode), int(decision_step)]
    ).generate_state(1, dtype=np.uint32)
    return int(state[0])


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {key: None for key in ("mean", "median", "p25", "p75")}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
    }


def paired_metrics(fixed_episodes, chunk_episodes, success_key="task_success"):
    fixed = {int(item["ep"]): item for item in fixed_episodes}
    chunk = {int(item["ep"]): item for item in chunk_episodes}
    paired_ids = [
        ep for ep in sorted(set(fixed) & set(chunk))
        if fixed[ep][success_key] and chunk[ep][success_key]
    ]
    physics_ratios = []
    wall_ratios = []
    decision_ratios = []
    for ep in paired_ids:
        fixed_steps = float(fixed[ep]["total_dense_steps"])
        chunk_steps = float(chunk[ep]["total_dense_steps"])
        fixed_wall = float(fixed[ep]["end_to_end_wall_s"])
        chunk_wall = float(chunk[ep]["end_to_end_wall_s"])
        if chunk_steps > 0:
            physics_ratios.append(fixed_steps / chunk_steps)
        if chunk_wall > 0:
            wall_ratios.append(fixed_wall / chunk_wall)
        fixed_decisions = float(fixed[ep].get("n_decision_steps", 0))
        chunk_decisions = float(chunk[ep].get("n_decision_steps", 0))
        if chunk_decisions > 0:
            decision_ratios.append(fixed_decisions / chunk_decisions)
    return {
        "paired_success_count": len(paired_ids),
        "success_key": success_key,
        "paired_episode_ids": paired_ids,
        "physics_speedup": _stats(physics_ratios),
        "wall_speedup": _stats(wall_ratios),
        "decision_step_ratio": _stats(decision_ratios),
        "physics_speedup_per_episode": physics_ratios,
        "wall_speedup_per_episode": wall_ratios,
        "decision_step_ratio_per_episode": decision_ratios,
    }


def diagnostic_episode_reason(first, second, episode):
    """Explain why an episode is selected for diagnostic video replay."""
    a = first[int(episode)]
    b = second[int(episode)]
    if not a.get("task_success", False) or not b.get("task_success", False):
        return "task_failure"
    if not a.get("reward_success", False) or not b.get("reward_success", False):
        return "safety_reward_failure"
    return "joint_success_reference"


def select_diagnostic_episode_ids(first_episodes, second_episodes=None, *, limit=5):
    """Select stable paired ids: task failures, safety failures, then successes."""
    first = {int(item["ep"]): item for item in first_episodes}
    if second_episodes is None:
        second = first
    else:
        second = {int(item["ep"]): item for item in second_episodes}
    common = sorted(set(first) & set(second))
    buckets = {"task_failure": [], "safety_reward_failure": [],
               "joint_success_reference": []}
    for episode in common:
        reason = diagnostic_episode_reason(first, second, episode)
        buckets[reason].append(episode)
    ordered = (
        buckets["task_failure"]
        + buckets["safety_reward_failure"]
        + buckets["joint_success_reference"]
    )
    return ordered[:max(int(limit), 0)]
