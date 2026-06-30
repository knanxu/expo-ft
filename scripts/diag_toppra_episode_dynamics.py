"""Compare per-action and whole-chunk TOPPRA dynamics on one expert episode.

Run from the RoboTwin repository root. Expert replay always uses v=1 and no
k_skip. After every 250 Hz physics step this records measured arm qpos/qvel,
finite-difference qacc, planned drive velocity, drive position, and tracking
error for all 12 arm joints.
"""

import argparse
import collections
import csv
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import bench_exec_backends as bench


PHYS_HZ = 250.0
DT = 1.0 / PHYS_HZ
CHUNK = 50
LIMIT_GRID = ((1.0, 1.0), (2.0, 2.0), (3.0, 4.0), (5.0, 8.0))
BACKENDS = ("per_action", "whole_chunk")
JOINTS = tuple([f"L{i}" for i in range(1, 7)] + [f"R{i}" for i in range(1, 7)])
RATED_VEL = np.asarray([5.0, 5.0, 5.5, 5.5, 5.0, 5.0] * 2, dtype=np.float64)


class SceneProbe:
    """Transparent scene proxy recording arm state after every physics step."""

    def __init__(self, scene, robot):
        self._scene = scene
        self.robot = robot
        self.qpos = []
        self.qvel = []
        self.drive_qpos = []
        self.plan_qvel = []

    def __getattr__(self, name):
        return getattr(self._scene, name)

    def _measured(self, side, velocity):
        entity = getattr(self.robot, f"{side}_entity")
        joints = getattr(self.robot, f"{side}_arm_joints")
        active = entity.get_active_joints()
        raw = entity.get_qvel() if velocity else entity.get_qpos()
        return [float(raw[active.index(joint)]) for joint in joints]

    def step(self):
        result = self._scene.step()
        self.qpos.append(
            self._measured("left", False) + self._measured("right", False)
        )
        self.qvel.append(
            self._measured("left", True) + self._measured("right", True)
        )
        joints = list(self.robot.left_arm_joints) + list(self.robot.right_arm_joints)
        self.drive_qpos.append(
            [float(joint.get_drive_target()[0]) for joint in joints]
        )
        self.plan_qvel.append(
            [float(joint.get_drive_velocity_target()[0]) for joint in joints]
        )
        return result


class RetimeCounter:
    """Count TOPPRA solves/fallbacks without changing their results."""

    def __init__(self):
        self.calls = 0
        self.fallbacks = 0
        self.reasons = collections.Counter()
        self._module = None
        self._original = None

    def __enter__(self):
        import envs.robot.toppra_chunk_executor as tce

        self._module = tce
        self._original = tce.retime_chunk

        def wrapped(*args, **kwargs):
            result = self._original(*args, **kwargs)
            self.calls += 1
            if result.get("status") != "success":
                self.fallbacks += 1
                self.reasons[str(result.get("fallback_reason"))] += 1
            return result

        tce.retime_chunk = wrapped
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._module.retime_chunk = self._original


def _as_trace(probe):
    qpos = np.asarray(probe.qpos, dtype=np.float64)
    qvel = np.asarray(probe.qvel, dtype=np.float64)
    drive_qpos = np.asarray(probe.drive_qpos, dtype=np.float64)
    plan_qvel = np.asarray(probe.plan_qvel, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 12:
        raise RuntimeError(f"unexpected recorded qpos shape: {qpos.shape}")
    qacc = np.full_like(qvel, np.nan)
    if len(qvel) > 1:
        qacc[1:] = np.diff(qvel, axis=0) * PHYS_HZ
    return {
        "qpos": qpos,
        "qvel": qvel,
        "qacc": qacc,
        "plan_qvel": plan_qvel,
        "drive_qpos": drive_qpos,
        "tracking_error": drive_qpos - qpos,
    }


def _abs_percentiles(values):
    finite = np.abs(np.asarray(values)[np.isfinite(values)])
    if finite.size == 0:
        return {str(p): None for p in (50, 95, 99, 100)}
    return {str(p): float(np.percentile(finite, p)) for p in (50, 95, 99, 100)}


def _per_joint_rows(tag, backend, vel_limit, acc_limit, trace):
    rows = []
    for joint_id, joint in enumerate(JOINTS):
        qvel = np.abs(trace["qvel"][:, joint_id])
        qacc = np.abs(trace["qacc"][:, joint_id])
        plan = np.abs(trace["plan_qvel"][:, joint_id])
        error = np.abs(trace["tracking_error"][:, joint_id])

        def stats(values):
            values = values[np.isfinite(values)]
            return [float(np.percentile(values, p)) for p in (50, 95, 99, 100)]

        rows.append(
            {
                "tag": tag,
                "backend": backend,
                "vel_limit": vel_limit,
                "acc_limit": acc_limit,
                "joint": joint,
                **dict(zip(("qvel_p50", "qvel_p95", "qvel_p99", "qvel_max"), stats(qvel))),
                **dict(zip(("qacc_p50", "qacc_p95", "qacc_p99", "qacc_max"), stats(qacc))),
                **dict(zip(("plan_qvel_p50", "plan_qvel_p95", "plan_qvel_p99", "plan_qvel_max"), stats(plan))),
                **dict(zip(("error_p50", "error_p95", "error_p99", "error_max"), stats(error))),
                "fraction_qvel_over_config_limit": float(np.mean(qvel > vel_limit)),
                "fraction_qvel_over_rated_limit": float(np.mean(qvel > RATED_VEL[joint_id])),
                "fraction_qacc_over_config_limit": float(np.mean(qacc[1:] > acc_limit)),
            }
        )
    return rows


def _tag(backend, vel_limit, acc_limit):
    return f"{backend}_vel{vel_limit:g}_acc{acc_limit:g}"


def _write_trace(output_dir, tag, trace, summary):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"{tag}.npz",
        **trace,
        dt=np.asarray(DT),
        joint_names=np.asarray(JOINTS),
    )
    fields = ("qpos", "qvel", "qacc", "plan_qvel", "drive_qpos", "tracking_error")
    header = ["step", "time_s"] + [
        f"{field}_{joint}" for field in fields for joint in JOINTS
    ]
    with open(output_dir / f"{tag}.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for index in range(len(trace["qpos"])):
            row = [index + 1, (index + 1) * DT]
            for field in fields:
                row.extend(trace[field][index].tolist())
            writer.writerow(row)
    with open(output_dir / f"{tag}.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def _run_one(env, env_args, actions, backend, vel_limit, acc_limit):
    setup_kwargs = dict(env_args)
    seed = setup_kwargs.pop("_diag_seed")
    env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **setup_kwargs)
    bench.apply_real_params(env.robot)
    probe = SceneProbe(env.scene, env.robot)
    env.scene = probe
    dense_steps = 0
    durations = 0.0
    chunk_statuses = []
    started = time.perf_counter()
    try:
        with RetimeCounter() as retime:
            for start in range(0, len(actions), CHUNK):
                chunk = actions[start : start + CHUNK]
                if len(chunk) < 2:
                    break
                if backend == "per_action":
                    info = env.take_chunk_action_per_action(
                        chunk,
                        vel_limit=vel_limit,
                        acc_limit=acc_limit,
                        v=1.0,
                        max_actions=None,
                        video_save_freq=-1,
                    )
                else:
                    info = env.take_chunk_action(
                        chunk,
                        vel_limit=vel_limit,
                        acc_limit=acc_limit,
                        v=1.0,
                        video_save_freq=-1,
                    )
                dense_steps += int(info.get("dense_steps", 0) or 0)
                durations += float(info.get("duration", 0.0) or 0.0)
                chunk_statuses.append(str(info.get("status", "success")))
                if info.get("status") == "truncated" or env.eval_success:
                    break
            trace = _as_trace(probe)
            success = bool(env.eval_success)
            retime_calls = retime.calls
            fallback_count = retime.fallbacks
            fallback_reasons = dict(retime.reasons)
    finally:
        env.scene = probe._scene

    if len(trace["qpos"]) != dense_steps:
        raise RuntimeError(
            f"recorded steps {len(trace['qpos'])} != backend dense_steps {dense_steps}"
        )
    nonfinite = any(
        not np.all(np.isfinite(values))
        for key, values in trace.items()
        if key != "qacc"
    ) or not np.all(np.isfinite(trace["qacc"][1:]))
    tag = _tag(backend, vel_limit, acc_limit)
    summary = {
        "tag": tag,
        "backend": backend,
        "vel_limit": vel_limit,
        "acc_limit": acc_limit,
        "v": 1.0,
        "k_skip": None,
        "dense_steps": dense_steps,
        "recorded_steps": len(trace["qpos"]),
        "sim_duration_s": len(trace["qpos"]) * DT,
        "planned_duration_s": durations,
        "wall_time_s": time.perf_counter() - started,
        "success": success,
        "chunk_statuses": chunk_statuses,
        "retime_calls": retime_calls,
        "fallback_count": fallback_count,
        "fallback_reasons": fallback_reasons,
        "nonfinite": nonfinite,
        "actual_abs_qvel_rad_s": _abs_percentiles(trace["qvel"]),
        "actual_abs_qacc_rad_s2": _abs_percentiles(trace["qacc"]),
        "planned_abs_qvel_rad_s": _abs_percentiles(trace["plan_qvel"]),
        "tracking_abs_error_rad": _abs_percentiles(trace["tracking_error"]),
        "fraction_steps_any_joint_over_config_vel": float(
            np.mean(np.any(np.abs(trace["qvel"]) > vel_limit, axis=1))
        ),
        "fraction_steps_any_joint_over_rated_vel": float(
            np.mean(np.any(np.abs(trace["qvel"]) > RATED_VEL, axis=1))
        ),
        "fraction_steps_any_joint_over_config_acc": float(
            np.mean(np.any(np.abs(trace["qacc"][1:]) > acc_limit, axis=1))
        ),
    }
    return tag, trace, summary


def _write_per_joint_summary(output_dir, rows):
    if not rows:
        return
    with open(output_dir / "per_joint_summary.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_timeseries(output_dir, vel_limit, acc_limit, pair):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    limits_tag = f"vel{vel_limit:g}_acc{acc_limit:g}"
    for field, limit, unit, filename in (
        ("qvel", vel_limit, "rad/s", f"velocity_timeseries_{limits_tag}.png"),
        ("qacc", acc_limit, "rad/s²", f"acceleration_timeseries_{limits_tag}.png"),
    ):
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
        for axis, backend in zip(axes, BACKENDS):
            trace = pair[backend]["trace"]
            values = trace[field]
            times = np.arange(1, len(values) + 1) * DT
            for joint_id in range(6):
                axis.plot(times, values[:, joint_id], linewidth=0.7, label=JOINTS[joint_id])
            axis.axhline(limit, color="black", linestyle="--", linewidth=1)
            axis.axhline(-limit, color="black", linestyle="--", linewidth=1)
            peak = float(np.nanmax(np.abs(values)))
            axis.set_title(f"{backend}: {field}, peak={peak:.3f} {unit}")
            axis.set_ylabel(unit)
            axis.grid(alpha=0.2)
            axis.legend(ncol=6, fontsize=8)
        axes[-1].set_xlabel("simulation time (s)")
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)

    x = np.arange(len(JOINTS))
    width = 0.36
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    for offset, backend in zip((-width / 2, width / 2), BACKENDS):
        trace = pair[backend]["trace"]
        axes[0].bar(
            x + offset, np.nanmax(np.abs(trace["qvel"]), axis=0), width,
            label=backend,
        )
        axes[1].bar(
            x + offset, np.nanmax(np.abs(trace["qacc"]), axis=0), width,
            label=backend,
        )
    axes[0].axhline(vel_limit, color="black", linestyle="--", linewidth=1)
    axes[1].axhline(acc_limit, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("max |qvel| (rad/s)")
    axes[1].set_ylabel("max |qacc| (rad/s²)")
    for axis in axes:
        axis.set_xticks(x, JOINTS)
        axis.grid(axis="y", alpha=0.2)
        axis.legend()
    fig.savefig(output_dir / f"peak_by_joint_{limits_tag}.png", dpi=160)
    plt.close(fig)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="stack_blocks_two")
    parser.add_argument("--rollout", default="demo_clean")
    parser.add_argument("--task_config", default="demo_clean")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        default="/home/xukainan/expo-ft/scripts/bench_out/toppra_episode_dynamics",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path("data") / args.task / args.rollout / "data" / f"episode{args.episode}.hdf5"
    with h5py.File(data_path) as file:
        actions = np.asarray(file["joint_action"]["vector"][...], dtype=np.float64)

    env, env_args = bench.build_env(args.task, args.task_config)
    env_args["_diag_seed"] = args.seed
    rows = []
    run_summaries = []
    try:
        for vel_limit, acc_limit in LIMIT_GRID:
            pair = {}
            stop_after_pair = False
            for backend in BACKENDS:
                env_args["_diag_seed"] = args.seed
                print(
                    f"[run] backend={backend} vel={vel_limit:g} acc={acc_limit:g}",
                    flush=True,
                )
                tag, trace, summary = _run_one(
                    env, env_args, actions, backend, vel_limit, acc_limit
                )
                _write_trace(output_dir, tag, trace, summary)
                rows.extend(
                    _per_joint_rows(tag, backend, vel_limit, acc_limit, trace)
                )
                _write_per_joint_summary(output_dir, rows)
                run_summaries.append(summary)
                with open(output_dir / "run_summaries.json", "w", encoding="utf-8") as file:
                    json.dump(run_summaries, file, indent=2)
                pair[backend] = {"trace": trace, "summary": summary}
                print(
                    f"[done] {tag}: dense={summary['dense_steps']} "
                    f"sim={summary['sim_duration_s']:.3f}s success={summary['success']} "
                    f"fallback={summary['fallback_count']} "
                    f"qvel_max={summary['actual_abs_qvel_rad_s']['100']:.3f} "
                    f"qacc_max={summary['actual_abs_qacc_rad_s2']['100']:.3f}",
                    flush=True,
                )
                stop_after_pair |= summary["nonfinite"] or summary["fallback_count"] > 0
                try:
                    env.close_env(clear_cache=False)
                except TypeError:
                    env.close_env()
            if set(pair) == set(BACKENDS):
                _plot_timeseries(output_dir, vel_limit, acc_limit, pair)
            if stop_after_pair:
                print("[stop] non-finite state or TOPPRA fallback; higher limits skipped", flush=True)
                break
    finally:
        try:
            env.close_env(clear_cache=True)
        except Exception:
            pass

    print(f"[complete] outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
