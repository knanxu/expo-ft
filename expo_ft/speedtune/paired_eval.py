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
    for ep in paired_ids:
        fixed_steps = float(fixed[ep]["total_dense_steps"])
        chunk_steps = float(chunk[ep]["total_dense_steps"])
        fixed_wall = float(fixed[ep]["end_to_end_wall_s"])
        chunk_wall = float(chunk[ep]["end_to_end_wall_s"])
        if chunk_steps > 0:
            physics_ratios.append(fixed_steps / chunk_steps)
        if chunk_wall > 0:
            wall_ratios.append(fixed_wall / chunk_wall)
    return {
        "paired_success_count": len(paired_ids),
        "success_key": success_key,
        "paired_episode_ids": paired_ids,
        "physics_speedup": _stats(physics_ratios),
        "wall_speedup": _stats(wall_ratios),
        "physics_speedup_per_episode": physics_ratios,
        "wall_speedup_per_episode": wall_ratios,
    }
