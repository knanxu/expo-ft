#! /usr/bin/env python
"""SpeedTune 双 backend 加速对比 eval：同一冻结 VLA + 两个训练好的 DQN(各自 backend)，
在 RoboTwin 上各跑 N episode，对比执行速度 + 加速控制参数。

核心用途：测 whole-chunk TOPPRA vs fixed-time 的执行速度。
  - 加速指标用 **dense_steps（物理步数, ÷250=仿真秒）**——这是跨 backend 唯一硬可比的执行时间基准
    （两者都统计 250Hz 物理步）。加速比 = A 步数 / B 步数。
  - 每个 backend 各 episode 录视频 + 画原始 v/vel_limit 曲线（复用 eval_speedtune.run_backend_episodes）。
  - 额外产出对比图(compare_speedup.png / compare_knob.png) + 对比表(compare_summary.json)。

VLA(3B) **只加载一次**两 backend 共用（_build_vla），避免双倍显存。两个 backend 各连自己的 env server
（不同 port，由 scripts/run_eval_compare.sh 起），故不做严格「同物理状态并排」——用同一组 seed 各跑 N
episode 比聚合指标（eval 惯例）。

用法（云端，learner venv；先用 scripts/run_eval_compare.sh 起两个 env server）：
    SPEEDTUNE_VLA_CKPT=<drift ckpt> SPEEDTUNE_VLA_ASSETS=<...> SPEEDTUNE_VLA_ASSET_ID=<...> \
    uv run python eval_speedtune_compare.py \
        --config configs/model/speedtune_dqn_config.py \
        --config_task configs/task/robotwin_stack_blocks.py \
        --backend_a fixed_time          --ckpt_a logs/.../fixed_time/checkpoints/update_<N>        --port_a 8103 \
        --backend_b chunk_toppra        --ckpt_b logs/.../chunk_toppra/checkpoints/update_<N>      --port_b 8102 \
        --n_episodes 30 --output_dir logs/speedtune_compare
"""

import os
import json
import logging

import numpy as np
from absl import app, flags

import jax
import etils.epath as epath

import openpi.training.sharding as openpi_sharding
from expo_ft.env.env_client import EnvClientWrapper
from expo_ft.speedtune.exec_backends import parse_force_limit
from expo_ft.speedtune.paired_eval import (
    diagnostic_episode_reason,
    paired_metrics,
    select_diagnostic_episode_ids,
)
from expo_ft.utils.train_utils import init_logging

# 复用 eval_speedtune 的可复用单元（import 会触发其 flags 定义，共享
# config/config_task/seed/n_episodes/max_decision_steps/record_video/resume/fsdp_devices/
# client_host/output_dir 这些 flags；eval_speedtune 的 dqn_ckpt/client_port 本脚本不用）。
import eval_speedtune as ev

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

FLAGS = flags.FLAGS

# 双 backend 各自的 flags（新名，不与 eval_speedtune 已定义的冲突）。
flags.DEFINE_string("ckpt_a", None, "backend A 的 DQN checkpoint 目录（含 q_net/，即 update_<N>）。")
flags.DEFINE_string("backend_a", "fixed_time", "backend A 名（基准，默认 fixed_time）。")
flags.DEFINE_integer("port_a", 8102, "backend A 的 env server port。")
flags.DEFINE_string("ckpt_b", None, "backend B 的 DQN checkpoint 目录（含 q_net/，即 update_<N>）。")
flags.DEFINE_string("backend_b", "chunk_toppra", "backend B 名（被测，默认 chunk_toppra）。")
flags.DEFINE_integer("port_b", 8103, "backend B 的 env server port。")

def _make_env(out_sub, exec_backend, port, config, config_task):
    """为一个 backend 建 eval env（连其专属 port 的 env server），视频存到 out_sub/videos。"""
    example_action = np.asarray(
        config_task.get("example_action", np.zeros((1, int(config.n_real_dims)), dtype=np.float32)),
        dtype=np.float32)
    return EnvClientWrapper(
        env_creation_request={
            "example_action": example_action, "env_usage": "eval",
            "video_dir": os.path.join(out_sub, "videos"),
            "exec_backend": exec_backend,
            "k_skip": ev.backend_k_skip(config, exec_backend),
            "stream_hold_steps": int(config.get("stream_hold_steps", 15)),
            "eval_video_save_freq": 25,
            "force_limit": parse_force_limit(config.get("force_limit", "")),
        },
        host=FLAGS.client_host, port=port,
    )


def _representative_episode(episodes):
    """取代表 episode 画参数曲线：优先首个成功，否则首个。"""
    succ = [e for e in episodes if e["success"]]
    if succ:
        return succ[0]
    return episodes[0] if episodes else None


def _param_keys(episode):
    """从代表 episode 反推后端实际控制的原始速度参数（v 或 vel_limit）。"""
    if not episode or not episode["recs"]:
        return []
    return [
        key for key in ("v", "vel_limit")
        if key in episode["recs"][0]
    ]


def _ratio(x, y):
    """x/y，处理 nan/0：分母无效返回 None。"""
    x, y = float(x), float(y)
    if np.isnan(x) or np.isnan(y) or y == 0.0:
        return None
    return x / y


def _save_compare_summary(out_dir, ra, rb):
    """写 compare_summary.json + 打印加速比（dense_steps 为硬可比基准）。"""
    na, nb = ra["exec_backend"], rb["exec_backend"]
    keep = ("n_episodes", "success", "success_rate", "reward_success", "reward_success_rate",
            "mean_decision_steps", "mean_decision_steps_success",
            "mean_dense_steps", "mean_dense_steps_success", "mean_sim_time_s",
            "mean_end_to_end_wall_s", "mean_vla_wall_s", "mean_dqn_wall_s",
            "mean_env_rpc_wall_s", "fixed_time_speed_violation_rate", "fallback_rate",
            "k_skip", "max_action_idxs", "max_unlocked_speed",
            "speed_action_counts", "speed_param_name",
            "speed_in_contact_mean", "speed_free_mean", "slows_near_contact",
            "max_planned_qvel", "mean_episode_max_planned_qvel")
    paired = paired_metrics(ra["episodes"], rb["episodes"])
    paired_safe = paired_metrics(
        ra["episodes"], rb["episodes"], success_key="reward_success"
    )
    summary = {
        na: {k: ra[k] for k in keep},
        nb: {k: rb[k] for k in keep},
        f"paired_speedup ({na}/{nb})": paired,
        f"paired_safe_speedup ({na}/{nb})": paired_safe,
        "note": (f"主指标只使用同一 seed 下 {na}/{nb} 都 task-success 的 episode pair；"
                 f"physics speedup=dense_steps({na})/dense_steps({nb})，wall speedup 同理，"
                 f">1 表示 {nb} 更快。"),
    }
    with open(os.path.join(out_dir, "compare_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logging.info("==== SpeedTune 加速对比 (%s vs %s) ====", na, nb)
    logging.info("%s: success=%d/%d 平均执行步数=%.0f (成功 episode=%.0f)",
                 na, ra["success"], ra["n_episodes"], ra["mean_dense_steps"], ra["mean_dense_steps_success"])
    logging.info("%s: success=%d/%d 平均执行步数=%.0f (成功 episode=%.0f)",
                 nb, rb["success"], rb["n_episodes"], rb["mean_dense_steps"], rb["mean_dense_steps_success"])
    sp_median = paired["physics_speedup"]["median"]
    if sp_median is not None:
        logging.info("严格配对加速比 (%s/%s, n=%d, median)=%.2fx → %s %s",
                     na, nb, paired["paired_success_count"], sp_median, nb,
                     "更快 ✓" if sp_median > 1.0 else "更慢/持平")
    else:
        logging.info("严格配对加速比无法计算（没有共同成功 episode）")
    decision_median = paired["decision_step_ratio"]["median"]
    if decision_median is not None:
        logging.info("共同成功 episode decision 数比 (%s/%s, median)=%.2f",
                     na, nb, decision_median)


def _run_paired_video_replay(
    env_a, env_b, vla, dqn_a, dqn_b, ra, rb, *,
    backend_a, backend_b, max_decision_steps, seed, out_a, out_b, limit,
    k_skip_a, k_skip_b, out_dir,
):
    selected = select_diagnostic_episode_ids(
        ra["episodes"], rb["episodes"], limit=limit
    )
    if not selected:
        return []
    logging.info("Paired diagnostic video replay episode ids: %s", selected)
    replay_a = ev.run_backend_episodes(
        env_a, vla, dqn_a["backend"], dqn_a["learner"], backend_a,
        n_episodes=max(selected) + 1, max_decision_steps=max_decision_steps,
        seed=seed, out_dir=out_a, record_video=True,
        video_episode_ids=set(selected), save_episode_artifacts=False,
        k_skip=k_skip_a, max_action_idxs=dqn_a["max_action_idxs"],
    )
    replay_b = ev.run_backend_episodes(
        env_b, vla, dqn_b["backend"], dqn_b["learner"], backend_b,
        n_episodes=max(selected) + 1, max_decision_steps=max_decision_steps,
        seed=seed, out_dir=out_b, record_video=True,
        video_episode_ids=set(selected), save_episode_artifacts=False,
        k_skip=k_skip_b, max_action_idxs=dqn_b["max_action_idxs"],
    )
    primary_a = {int(item["ep"]): item for item in ra["episodes"]}
    primary_b = {int(item["ep"]): item for item in rb["episodes"]}
    repeated_a = {int(item["ep"]): item for item in replay_a["episodes"]}
    repeated_b = {int(item["ep"]): item for item in replay_b["episodes"]}
    manifest = []
    for episode in selected:
        a_tag = "SUCCESS" if repeated_a[episode]["task_success"] else "FAIL"
        b_tag = "SUCCESS" if repeated_b[episode]["task_success"] else "FAIL"
        manifest.append({
            "episode": episode,
            "seed": primary_a[episode]["seed"],
            "reason": diagnostic_episode_reason(primary_a, primary_b, episode),
            backend_a: {
                "primary_task_success": primary_a[episode]["task_success"],
                "primary_reward_success": primary_a[episode]["reward_success"],
                "replay_task_success": repeated_a[episode]["task_success"],
                "video": os.path.join(backend_a, "videos", f"episode{episode}_{a_tag}.mp4"),
            },
            backend_b: {
                "primary_task_success": primary_b[episode]["task_success"],
                "primary_reward_success": primary_b[episode]["reward_success"],
                "replay_task_success": repeated_b[episode]["task_success"],
                "video": os.path.join(backend_b, "videos", f"episode{episode}_{b_tag}.mp4"),
            },
        })
    with open(os.path.join(out_dir, "video_manifest.json"), "w") as file:
        json.dump({"selected_episode_ids": selected, "episodes": manifest}, file, indent=2)
    return manifest


def _plot_compare(out_dir, ra, rb):
    """执行步数与后端原始速度参数对比图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logging.warning("matplotlib 不可用，跳过对比图（json 已存）：%s", e)
        return

    na, nb = ra["exec_backend"], rb["exec_backend"]
    epa, epb = ra["episodes"], rb["episodes"]

    # ---- 图1 compare_speedup.png：per-episode 执行步数 + 代表 episode 原始速度 ----
    n = max(len(epa), len(epb), 1)
    x = np.arange(n)
    w = 0.38
    da = [e["total_dense_steps"] for e in epa] + [0] * (n - len(epa))
    db = [e["total_dense_steps"] for e in epb] + [0] * (n - len(epb))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].bar(x - w / 2, da, w, label=na, color="tab:orange")
    axes[0].bar(x + w / 2, db, w, label=nb, color="tab:blue")
    for i, e in enumerate(epa):
        mk, col = ("S", "green") if e["success"] else ("F", "red")
        axes[0].text(x[i] - w / 2, da[i], mk, ha="center", va="bottom", color=col,
                     fontsize=12, fontweight="bold")
    for i, e in enumerate(epb):
        mk, col = ("S", "green") if e["success"] else ("F", "red")
        axes[0].text(x[i] + w / 2, db[i], mk, ha="center", va="bottom", color=col,
                     fontsize=12, fontweight="bold")
    axes[0].axhline(ra["mean_dense_steps"], ls="--", color="tab:orange", alpha=0.5)
    axes[0].axhline(rb["mean_dense_steps"], ls="--", color="tab:blue", alpha=0.5)
    axes[0].set_xlabel("episode")
    axes[0].set_ylabel("execution dense_steps (÷250 = sim sec)")
    axes[0].set_title("Per-episode exec steps (lower=faster; S=success F=fail)")
    axes[0].set_xticks(x)
    axes[0].legend(fontsize=9)

    rea, reb = _representative_episode(epa), _representative_episode(epb)
    if rea:
        key = ra["speed_param_name"]
        axes[1].plot([r["step"] for r in rea["recs"]], [r[key] for r in rea["recs"]],
                     "-o", color="tab:orange", ms=3, label=f"{na} ep{rea['ep']}")
    if reb:
        key = rb["speed_param_name"]
        axes[1].plot([r["step"] for r in reb["recs"]], [r[key] for r in reb["recs"]],
                     "-o", color="tab:blue", ms=3, label=f"{nb} ep{reb['ep']}")
    axes[1].set_xlabel("decision step")
    axes[1].set_ylabel("raw speed parameter")
    axes[1].set_title("Raw v / vel_limit vs decision step")
    axes[1].legend(fontsize=9)

    sp = paired_metrics(epa, epb)["physics_speedup"]["median"]
    sp_txt = f"  |  speedup({nb}/{na}, success)={sp:.2f}x" if sp is not None else ""
    fig.suptitle(f"SpeedTune backend comparison: {na} vs {nb}{sp_txt}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "compare_speedup.png"), dpi=120)
    plt.close(fig)

    # ---- 图2 compare_knob.png：各 backend 原始速度参数随决策步 ----
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5), sharey=False)
    for ax, rep, name in ((axes2[0], rea, na), (axes2[1], reb, nb)):
        if not rep:
            ax.set_title(f"{name}: no episode")
            continue
        ev._overlay_contact_grasp(ax, rep["recs"], x_key="step")  # 接触方块阴影 + 抓取竖线
        steps = [r["step"] for r in rep["recs"]]
        for k in _param_keys(rep):
            ax.plot(steps, [r.get(k, np.nan) for r in rep["recs"]], "-o", ms=3, label=k)
        ax.set_xlabel("decision step")
        ax.set_ylabel("raw speed parameter")
        _tag = " [SUCCESS]" if rep["success"] else " [FAILED]"
        ax.set_title(f"{name} ep{rep['ep']}{_tag}: speed params + grasp/contact")
        ax.legend(fontsize=8)
    fig2.suptitle("Raw speed parameter issued per backend, vs decision step")
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "compare_knob.png"), dpi=120)
    plt.close(fig2)


def main(_):
    init_logging()
    config, config_task = FLAGS.config, FLAGS.config_task
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
    jax.config.update("jax_default_matmul_precision", "highest")

    if not FLAGS.ckpt_a or not FLAGS.ckpt_b:
        raise ValueError("--ckpt_a 和 --ckpt_b 都必填（两个 backend 各自训练好的 DQN checkpoint update_<N>）")

    mesh = openpi_sharding.make_mesh(FLAGS.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    shardings = (data_sharding, replicated_sharding)

    out_dir = FLAGS.output_dir
    os.makedirs(out_dir, exist_ok=True)
    out_a = os.path.join(out_dir, FLAGS.backend_a)
    out_b = os.path.join(out_dir, FLAGS.backend_b)
    os.makedirs(out_a, exist_ok=True)
    os.makedirs(out_b, exist_ok=True)

    logging.info("创建两个 RoboTwin eval env：A=%s(port %d) / B=%s(port %d) ...",
                 FLAGS.backend_a, FLAGS.port_a, FLAGS.backend_b, FLAGS.port_b)
    env_a = _make_env(out_a, FLAGS.backend_a, FLAGS.port_a, config, config_task)
    env_b = _make_env(out_b, FLAGS.backend_b, FLAGS.port_b, config, config_task)
    env_a.reset()
    env_b.reset()

    # VLA 只加载一次（用 env_a 取 example obs；obs 格式与 backend 无关），两 backend 共用。
    vla = ev._build_vla(config, config_task, FLAGS.seed, mesh, shardings, env_a, FLAGS.resume)
    dqn_a = ev._build_dqn(config, FLAGS.seed, FLAGS.backend_a, FLAGS.ckpt_a, vla["feat_dim"], vla["vfeat0"])
    dqn_b = ev._build_dqn(config, FLAGS.seed, FLAGS.backend_b, FLAGS.ckpt_b, vla["feat_dim"], vla["vfeat0"])

    logging.info("==== 跑 backend A = %s（基准）====", FLAGS.backend_a)
    ra = ev.run_backend_episodes(
        env_a, vla, dqn_a["backend"], dqn_a["learner"], FLAGS.backend_a,
        n_episodes=FLAGS.n_episodes, max_decision_steps=FLAGS.max_decision_steps,
        seed=FLAGS.seed, out_dir=out_a, record_video=False,
        k_skip=ev.backend_k_skip(config, FLAGS.backend_a),
        max_action_idxs=dqn_a["max_action_idxs"])

    logging.info("==== 跑 backend B = %s（被测）====", FLAGS.backend_b)
    rb = ev.run_backend_episodes(
        env_b, vla, dqn_b["backend"], dqn_b["learner"], FLAGS.backend_b,
        n_episodes=FLAGS.n_episodes, max_decision_steps=FLAGS.max_decision_steps,
        seed=FLAGS.seed, out_dir=out_b, record_video=False,
        k_skip=ev.backend_k_skip(config, FLAGS.backend_b),
        max_action_idxs=dqn_b["max_action_idxs"])

    _save_compare_summary(out_dir, ra, rb)
    _plot_compare(out_dir, ra, rb)
    if FLAGS.record_video and FLAGS.video_episodes > 0:
        _run_paired_video_replay(
            env_a, env_b, vla, dqn_a, dqn_b, ra, rb,
            backend_a=FLAGS.backend_a, backend_b=FLAGS.backend_b,
            max_decision_steps=FLAGS.max_decision_steps, seed=FLAGS.seed,
            out_a=out_a, out_b=out_b, limit=FLAGS.video_episodes,
            k_skip_a=ev.backend_k_skip(config, FLAGS.backend_a),
            k_skip_b=ev.backend_k_skip(config, FLAGS.backend_b),
            out_dir=out_dir,
        )
    logging.info("对比输出: %s（compare_summary.json + compare_speedup.png + compare_knob.png；"
                 "各 backend 子目录含 videos/ + episode*_speed.{json,png}）", out_dir)


if __name__ == "__main__":
    app.run(main)
