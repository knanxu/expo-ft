"""Pure helpers for fixed raw-speed SpeedTune evaluation sweeps."""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Iterable, Mapping, Sequence

import numpy as np


DEFAULT_SPEEDS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
SUPPORTED_BACKENDS = ("fixed_time", "chunk_toppra")


def parse_speeds(spec: str | Iterable[float]) -> tuple[float, ...]:
    """Parse an ordered subset of the supported 0.5-step raw speed grid."""
    if isinstance(spec, str):
        values = tuple(float(item.strip()) for item in spec.split(",") if item.strip())
    else:
        values = tuple(float(item) for item in spec)
    if not values:
        raise ValueError("speed grid cannot be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"speed grid contains duplicate values: {values}")
    if tuple(sorted(values)) != values:
        raise ValueError(f"speed grid must be increasing: {values}")
    unsupported = [value for value in values if value not in DEFAULT_SPEEDS]
    if unsupported:
        raise ValueError(
            f"unsupported speed values {unsupported}; supported grid is {DEFAULT_SPEEDS}"
        )
    return values


class ConstantSpeedLearner:
    """Minimal greedy-controller interface consumed by run_backend_episodes."""

    def __init__(self, backend, speed: float):
        if backend.name not in SUPPORTED_BACKENDS or backend.n_heads != 1:
            raise ValueError(
                f"fixed-speed sweep supports only single-head {SUPPORTED_BACKENDS}, got {backend.name}"
            )
        matches = [
            index for index, value in enumerate(backend.vars[0].grid)
            if math.isclose(float(value), float(speed), rel_tol=0.0, abs_tol=1e-9)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"speed {speed} is not an action value for {backend.name}: {backend.vars[0].grid}"
            )
        self.action_idx = int(matches[0])

    def greedy_action_idxs(self, feat, *, max_action_idxs=None):
        del max_action_idxs
        batch_size = int(np.asarray(feat).shape[0])
        return np.full((batch_size, 1), self.action_idx, dtype=np.int32)


def result_row(result: Mapping, speed: float) -> dict:
    """Convert one run_backend_episodes aggregate into one sweep table row."""
    backend = str(result["exec_backend"])
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"unsupported sweep backend: {backend}")
    expected_param = "v" if backend == "fixed_time" else "vel_limit"
    if result["speed_param_name"] != expected_param:
        raise ValueError(
            f"{backend} expected raw parameter {expected_param}, got {result['speed_param_name']}"
        )
    mean_dense_success = float(result["mean_dense_steps_success"])
    return {
        "backend": backend,
        "speed_param_name": expected_param,
        "speed": float(speed),
        "k_skip": int(result["k_skip"]),
        "n_episodes": int(result["n_episodes"]),
        "task_success": int(result["success"]),
        "success_rate": float(result["success_rate"]),
        "reward_success": int(result["reward_success"]),
        "reward_success_rate": float(result["reward_success_rate"]),
        "fixed_time_speed_violation_rate": float(result["fixed_time_speed_violation_rate"]),
        "fallback_rate": float(result["fallback_rate"]),
        "mean_decision_steps": float(result["mean_decision_steps"]),
        "mean_decision_steps_success": float(result["mean_decision_steps_success"]),
        "mean_dense_steps": float(result["mean_dense_steps"]),
        "mean_dense_steps_success": mean_dense_success,
        "mean_sim_time_s": float(result["mean_sim_time_s"]),
        "mean_sim_time_s_success": mean_dense_success / 250.0,
        "max_planned_qvel": float(result["max_planned_qvel"]),
        "mean_episode_max_planned_qvel": float(result["mean_episode_max_planned_qvel"]),
    }


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump(_json_safe(payload), file, indent=2)


def write_aggregate_artifacts(
    output_dir: str,
    rows: Sequence[Mapping],
    metadata: Mapping,
    *,
    make_plots: bool = True,
) -> None:
    """Write machine-readable and human-readable sweep aggregates."""
    os.makedirs(output_dir, exist_ok=True)
    backend_order = {name: index for index, name in enumerate(SUPPORTED_BACKENDS)}
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (backend_order.get(row["backend"], 99), float(row["speed"])),
    )
    write_json(
        os.path.join(output_dir, "fixed_speed_summary.json"),
        {"metadata": dict(metadata), "results": ordered},
    )

    fields = list(ordered[0].keys()) if ordered else []
    with open(os.path.join(output_dir, "fixed_speed_summary.csv"), "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)

    lines = [
        "# SpeedTune fixed raw-speed sweep",
        "",
        f"- episodes per backend/speed: {metadata.get('n_episodes_per_config')}",
        f"- speeds: {metadata.get('speeds')}",
        f"- k_skip: {metadata.get('k_skip')}",
        "",
        "| backend | raw parameter | speed | k_skip | task success | reward success | mean physics steps | success-only steps | max planned qvel |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ordered:
        lines.append(
            f"| {row['backend']} | {row['speed_param_name']} | {row['speed']:.1f} | "
            f"{row['k_skip']} | {row['task_success']}/{row['n_episodes']} "
            f"({100 * row['success_rate']:.1f}%) | {row['reward_success']}/{row['n_episodes']} "
            f"({100 * row['reward_success_rate']:.1f}%) | {row['mean_dense_steps']:.1f} | "
            f"{row['mean_dense_steps_success']:.1f} | {row['max_planned_qvel']:.3f} |"
        )
    with open(os.path.join(output_dir, "fixed_speed_report.md"), "w") as file:
        file.write("\n".join(lines) + "\n")

    if make_plots and ordered:
        _plot_aggregate(output_dir, ordered)


def _plot_aggregate(output_dir: str, rows: Sequence[Mapping]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    colors = {"fixed_time": "tab:orange", "chunk_toppra": "tab:blue"}
    for backend in SUPPORTED_BACKENDS:
        selected = [row for row in rows if row["backend"] == backend]
        if not selected:
            continue
        speeds = [row["speed"] for row in selected]
        axes[0].plot(speeds, [100 * row["success_rate"] for row in selected],
                     "-o", label=backend, color=colors[backend])
        axes[1].plot(speeds, [row["mean_dense_steps"] for row in selected],
                     "-o", label=f"{backend} all", color=colors[backend])
        axes[1].plot(speeds, [row["mean_dense_steps_success"] for row in selected],
                     "--o", label=f"{backend} success", color=colors[backend], alpha=0.7)
    fixed = [row for row in rows if row["backend"] == "fixed_time"]
    if fixed:
        axes[2].plot([row["speed"] for row in fixed],
                     [row["max_planned_qvel"] for row in fixed],
                     "-o", color=colors["fixed_time"], label="overall max")
        axes[2].plot([row["speed"] for row in fixed],
                     [row["mean_episode_max_planned_qvel"] for row in fixed],
                     "--o", color="tab:green", label="mean episode max")
        axes[2].axhline(4.0, color="red", linestyle=":", label="4 rad/s gate")
    axes[0].set_ylabel("task success rate (%)")
    axes[1].set_ylabel("250 Hz physical steps")
    axes[2].set_ylabel("planned max |qvel| (rad/s)")
    for axis in axes:
        axis.set_xlabel("raw v / vel_limit")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fixed_speed_sweep.png"), dpi=160)
    plt.close(fig)
