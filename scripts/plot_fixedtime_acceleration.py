"""Plot measured fixedtime acceleration from the recorded episode-0 traces."""

import argparse
import csv
from pathlib import Path

import numpy as np


PHYS_HZ = 250.0
REFERENCE_ACC_LIMIT = 10.0
JOINTS = tuple([f"L{i}" for i in range(1, 7)] + [f"R{i}" for i in range(1, 7)])


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace_dir",
        default="/home/xukainan/expo-ft/scripts/bench_out/fixedtime_velocity",
    )
    return parser.parse_args()


def _load(trace_dir, v):
    path = trace_dir / f"episode0_v{v}_force_limited_trace.npz"
    data = np.load(path)
    qvel = np.asarray(data["qvel"], dtype=np.float64)
    stored_qacc = np.asarray(data["qacc"], dtype=np.float64)
    qacc = np.diff(qvel, axis=0) * PHYS_HZ
    if qvel.ndim != 2 or qvel.shape[1] != 12:
        raise RuntimeError(f"unexpected qvel shape in {path}: {qvel.shape}")
    if stored_qacc.shape != qacc.shape or not np.allclose(stored_qacc, qacc):
        raise RuntimeError(f"stored qacc is inconsistent with measured qvel in {path}")
    return qvel, qacc


def main():
    args = _parse_args()
    trace_dir = Path(args.trace_dir)
    traces = {v: _load(trace_dir, v) for v in (1, 4)}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    for axis, v in zip(axes, (1, 4)):
        _, qacc = traces[v]
        times = np.arange(2, len(qacc) + 2, dtype=np.float64) / PHYS_HZ
        for joint_id in range(6):
            axis.plot(times, qacc[:, joint_id], linewidth=0.7, label=JOINTS[joint_id])
        axis.axhline(
            REFERENCE_ACC_LIMIT,
            color="black",
            linestyle="--",
            linewidth=1,
            label="±10 rad/s² reference",
        )
        axis.axhline(-REFERENCE_ACC_LIMIT, color="black", linestyle="--", linewidth=1)
        peak = float(np.max(np.abs(qacc)))
        axis.set_title(f"fixed_time v={v}, measured qacc, peak={peak:.3f} rad/s²")
        axis.set_ylabel("rad/s²")
        axis.grid(alpha=0.2)
        axis.legend(ncol=7, fontsize=8)
    axes[-1].set_xlabel("simulation time (s)")
    figure_path = trace_dir / "episode0_fixedtime_acceleration_v1_vs_v4.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    rows = []
    for v, (_, qacc) in traces.items():
        for joint_id, joint in enumerate(JOINTS):
            values = np.abs(qacc[:, joint_id])
            rows.append(
                {
                    "v": v,
                    "joint": joint,
                    "abs_qacc_p50_rad_s2": float(np.percentile(values, 50)),
                    "abs_qacc_p95_rad_s2": float(np.percentile(values, 95)),
                    "abs_qacc_p99_rad_s2": float(np.percentile(values, 99)),
                    "abs_qacc_p99_9_rad_s2": float(np.percentile(values, 99.9)),
                    "abs_qacc_max_rad_s2": float(np.max(values)),
                    "fraction_over_10_rad_s2": float(
                        np.mean(values > REFERENCE_ACC_LIMIT)
                    ),
                }
            )
    csv_path = trace_dir / "episode0_fixedtime_acceleration_per_joint.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {figure_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
