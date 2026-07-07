"""Pure helpers for whole-chunk TOPPRA vel_limit × k_skip sweeps."""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Iterable, Mapping, Sequence


DEFAULT_K_SKIPS = (10, 20, 40, 50)


def parse_k_skips(spec: str | Iterable[int], *, max_k: int = 50) -> tuple[int, ...]:
    """Parse an increasing list of whole-chunk execution_steps values."""
    if isinstance(spec, str):
        values = tuple(int(item.strip()) for item in spec.split(",") if item.strip())
    else:
        values = tuple(int(item) for item in spec)
    if not values:
        raise ValueError("k_skip grid cannot be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"k_skip grid contains duplicate values: {values}")
    if tuple(sorted(values)) != values:
        raise ValueError(f"k_skip grid must be increasing: {values}")
    invalid = [value for value in values if value <= 0 or value > int(max_k)]
    if invalid:
        raise ValueError(f"k_skip values {invalid} are outside valid range 1..{max_k}")
    return values


def result_row(result: Mapping, *, vel_limit: float, k_skip: int) -> dict:
    """Convert one run_backend_episodes result into a sweep table row."""
    if result["exec_backend"] != "chunk_toppra":
        raise ValueError(f"expected chunk_toppra result, got {result['exec_backend']}")
    if result["speed_param_name"] != "vel_limit":
        raise ValueError(f"expected vel_limit speed parameter, got {result['speed_param_name']}")
    mean_dense_success = float(result["mean_dense_steps_success"])
    mean_time_success = (
        mean_dense_success / 250.0 if math.isfinite(mean_dense_success) else float("nan")
    )
    return {
        "backend": "chunk_toppra",
        "vel_limit": float(vel_limit),
        "k_skip": int(k_skip),
        "n_episodes": int(result["n_episodes"]),
        "task_success": int(result["success"]),
        "success_rate": float(result["success_rate"]),
        "reward_success": int(result["reward_success"]),
        "reward_success_rate": float(result["reward_success_rate"]),
        "fallback_rate": float(result["fallback_rate"]),
        "mean_decision_steps": float(result["mean_decision_steps"]),
        "mean_decision_steps_success": float(result["mean_decision_steps_success"]),
        "mean_dense_steps": float(result["mean_dense_steps"]),
        "mean_sim_time_s": float(result["mean_sim_time_s"]),
        "mean_dense_steps_success": mean_dense_success,
        "mean_sim_time_s_success": mean_time_success,
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
    """Write JSON/CSV/Markdown and an optional heatmap-style plot."""
    os.makedirs(output_dir, exist_ok=True)
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (int(row["k_skip"]), float(row["vel_limit"])),
    )
    write_json(
        os.path.join(output_dir, "wholechunk_kskip_vel_summary.json"),
        {"metadata": dict(metadata), "results": ordered},
    )
    fields = list(ordered[0].keys()) if ordered else []
    with open(os.path.join(output_dir, "wholechunk_kskip_vel_summary.csv"), "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)

    lines = [
        "# Whole-chunk TOPPRA vel_limit × k_skip sweep",
        "",
        f"- episodes per config: {metadata.get('n_episodes_per_config')}",
        f"- vel_limits: {metadata.get('vel_limits')}",
        f"- k_skips: {metadata.get('k_skips')}",
        "",
        "| k_skip | vel_limit | task success | success rate | success mean sim time (s) | success mean dense steps | fallback episode rate |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ordered:
        success_time = row["mean_sim_time_s_success"]
        success_time_text = "nan" if math.isnan(success_time) else f"{success_time:.3f}"
        dense_success = row["mean_dense_steps_success"]
        dense_success_text = "nan" if math.isnan(dense_success) else f"{dense_success:.1f}"
        lines.append(
            f"| {row['k_skip']} | {row['vel_limit']:.1f} | "
            f"{row['task_success']}/{row['n_episodes']} | "
            f"{100 * row['success_rate']:.1f}% | {success_time_text} | "
            f"{dense_success_text} | {100 * row['fallback_rate']:.1f}% |"
        )
    with open(os.path.join(output_dir, "wholechunk_kskip_vel_report.md"), "w") as file:
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

    k_values = sorted({int(row["k_skip"]) for row in rows})
    speeds = sorted({float(row["vel_limit"]) for row in rows})
    by_key = {(int(row["k_skip"]), float(row["vel_limit"])): row for row in rows}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for k in k_values:
        selected = [by_key[(k, speed)] for speed in speeds if (k, speed) in by_key]
        x = [row["vel_limit"] for row in selected]
        axes[0].plot(x, [100 * row["success_rate"] for row in selected], "-o", label=f"K={k}")
        axes[1].plot(x, [row["mean_sim_time_s_success"] for row in selected], "-o", label=f"K={k}")
    axes[0].set_ylabel("task success rate (%)")
    axes[1].set_ylabel("success-only mean sim time (s)")
    for axis in axes:
        axis.set_xlabel("vel_limit")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "wholechunk_kskip_vel_sweep.png"), dpi=160)
    plt.close(fig)
