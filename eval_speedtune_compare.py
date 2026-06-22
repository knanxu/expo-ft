#! /usr/bin/env python
"""SpeedTune 双 backend 加速对比 eval：同一冻结 VLA + 两个训练好的 DQN(各自 backend)，
在 RoboTwin 上各跑 N episode，对比执行速度 + 加速控制参数。

核心用途：测「现在更新执行方式后的 per_action_toppra」vs「fixed_time」的执行速度。
  - 加速指标用 **dense_steps（物理步数, ÷250=仿真秒）**——这是跨 backend 唯一硬可比的执行时间基准
    （fixed_time=M×hold_steps；per_action=Σ各段步数；都 250Hz 仿真）。加速比 = A 步数 / B 步数。
  - 每个 backend 各 episode 录视频 + 画激进度/参数曲线（复用 eval_speedtune.run_backend_episodes）。
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
        --backend_b per_action_toppra   --ckpt_b logs/.../per_action_toppra/checkpoints/update_<N> --port_b 8102 \
        --n_episodes 5 --output_dir logs/speedtune_compare
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
flags.DEFINE_integer("port_a", 8103, "backend A 的 env server port。")
flags.DEFINE_string("ckpt_b", None, "backend B 的 DQN checkpoint 目录（含 q_net/，即 update_<N>）。")
flags.DEFINE_string("backend_b", "per_action_toppra", "backend B 名（被测，默认 per_action_toppra）。")
flags.DEFINE_integer("port_b", 8102, "backend B 的 env server port。")

# rec 的固定字段（非 speed_params）；用于从 rec 反推该 backend 的实际加速参数 keys。
_FIXED_REC_KEYS = {
    "step", "t_sim", "dense_steps", "aggr_mean", "aggr",
    "left_gripper", "right_gripper", "left_contact", "right_contact",
    "exec_status", "success",
}


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
            "k_skip": config.get("k_skip", None),
            "stream_hold_steps": int(config.get("stream_hold_steps", 15)),
            "eval_video_save_freq": 25,
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
    """从代表 episode 的首个 rec 反推该 backend 实际下发的加速参数 keys（v/vel_scale/acc_scale）。"""
    if not episode or not episode["recs"]:
        return []
    return [k for k in episode["recs"][0].keys() if k not in _FIXED_REC_KEYS]


def _ratio(x, y):
    """x/y，处理 nan/0：分母无效返回 None。"""
    x, y = float(x), float(y)
    if np.isnan(x) or np.isnan(y) or y == 0.0:
        return None
    return x / y


def _save_compare_summary(out_dir, ra, rb):
    """写 compare_summary.json + 打印加速比（dense_steps 为硬可比基准）。"""
    na, nb = ra["exec_backend"], rb["exec_backend"]
    keep = ("success", "success_rate", "mean_dense_steps", "mean_dense_steps_success",
            "mean_sim_time_s", "aggr_in_contact_mean", "aggr_free_mean")
    sp_all = _ratio(ra["mean_dense_steps"], rb["mean_dense_steps"])             # >1 → B 比 A 快
    sp_succ = _ratio(ra["mean_dense_steps_success"], rb["mean_dense_steps_success"])
    summary = {
        na: {k: ra[k] for k in keep},
        nb: {k: rb[k] for k in keep},
        f"speedup_all_episodes ({na}/{nb})": sp_all,
        f"speedup_success_only ({na}/{nb})": sp_succ,
        "note": (f"dense_steps=物理步数(÷250=仿真秒)，跨 backend 唯一硬可比的执行时间基准；"
                 f"speedup>1 表示 {nb} 比 {na} 快（执行步数更少）。success_only 只在双方都成功的 "
                 f"episode 上比，避免失败早停拉低步数。"),
    }
    with open(os.path.join(out_dir, "compare_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logging.info("==== SpeedTune 加速对比 (%s vs %s) ====", na, nb)
    logging.info("%s: success=%d/%d 平均执行步数=%.0f (成功 episode=%.0f)",
                 na, ra["success"], ra["n_episodes"], ra["mean_dense_steps"], ra["mean_dense_steps_success"])
    logging.info("%s: success=%d/%d 平均执行步数=%.0f (成功 episode=%.0f)",
                 nb, rb["success"], rb["n_episodes"], rb["mean_dense_steps"], rb["mean_dense_steps_success"])
    if sp_succ is not None:
        logging.info("加速比 (%s/%s, 成功 episode)=%.2fx → %s %s",
                     na, nb, sp_succ, nb, "更快 ✓" if sp_succ > 1.0 else "更慢/持平")
    else:
        logging.info("加速比无法计算（某 backend 无成功 episode）；看 all_episodes 比值=%s", sp_all)


def _plot_compare(out_dir, ra, rb):
    """compare_speedup.png（执行步数 bar + 激进度对比）+ compare_knob.png（各 backend 实际参数曲线）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logging.warning("matplotlib 不可用，跳过对比图（json 已存）：%s", e)
        return

    na, nb = ra["exec_backend"], rb["exec_backend"]
    epa, epb = ra["episodes"], rb["episodes"]

    # ---- 图1 compare_speedup.png：per-episode 执行步数 bar + 代表 episode 激进度对比 ----
    n = max(len(epa), len(epb), 1)
    x = np.arange(n)
    w = 0.38
    da = [e["total_dense_steps"] for e in epa] + [0] * (n - len(epa))
    db = [e["total_dense_steps"] for e in epb] + [0] * (n - len(epb))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].bar(x - w / 2, da, w, label=na, color="tab:orange")
    axes[0].bar(x + w / 2, db, w, label=nb, color="tab:blue")
    for i, e in enumerate(epa):
        if not e["success"]:
            axes[0].text(x[i] - w / 2, da[i], "✗", ha="center", va="bottom", color="red", fontsize=9)
    for i, e in enumerate(epb):
        if not e["success"]:
            axes[0].text(x[i] + w / 2, db[i], "✗", ha="center", va="bottom", color="red", fontsize=9)
    axes[0].axhline(ra["mean_dense_steps"], ls="--", color="tab:orange", alpha=0.5)
    axes[0].axhline(rb["mean_dense_steps"], ls="--", color="tab:blue", alpha=0.5)
    axes[0].set_xlabel("episode")
    axes[0].set_ylabel("execution dense_steps (÷250 = sim sec)")
    axes[0].set_title("per-episode 执行步数（越低越快; ✗=失败 episode）")
    axes[0].set_xticks(x)
    axes[0].legend(fontsize=9)

    rea, reb = _representative_episode(epa), _representative_episode(epb)
    if rea:
        axes[1].plot([r["step"] for r in rea["recs"]], [r["aggr_mean"] for r in rea["recs"]],
                     "-o", color="tab:orange", ms=3, label=f"{na} ep{rea['ep']}")
    if reb:
        axes[1].plot([r["step"] for r in reb["recs"]], [r["aggr_mean"] for r in reb["recs"]],
                     "-o", color="tab:blue", ms=3, label=f"{nb} ep{reb['ep']}")
    axes[1].set_xlabel("decision step")
    axes[1].set_ylabel("aggressiveness (0慢~1快)")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("加速激进度（归一化）随决策步")
    axes[1].legend(fontsize=9)

    sp = _ratio(ra["mean_dense_steps_success"], rb["mean_dense_steps_success"])
    sp_txt = f"  |  speedup({nb}/{na}, success)={sp:.2f}x" if sp is not None else ""
    fig.suptitle(f"SpeedTune backend 对比: {na} vs {nb}{sp_txt}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "compare_speedup.png"), dpi=120)
    plt.close(fig)

    # ---- 图2 compare_knob.png：各 backend 实际加速参数 (v/vel_scale/acc_scale) 随决策步 ----
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5), sharey=False)
    for ax, rep, name in ((axes2[0], rea, na), (axes2[1], reb, nb)):
        if not rep:
            ax.set_title(f"{name}: 无 episode")
            continue
        steps = [r["step"] for r in rep["recs"]]
        for k in _param_keys(rep):
            ax.plot(steps, [r.get(k, np.nan) for r in rep["recs"]], "-o", ms=3, label=k)
        ax.set_xlabel("decision step")
        ax.set_ylabel("speed param value")
        ax.set_title(f"{name} ep{rep['ep']}: 实际加速参数")
        ax.legend(fontsize=9)
    fig2.suptitle("各 backend 实际下发的加速控制参数随决策步")
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
        seed=FLAGS.seed, out_dir=out_a, record_video=FLAGS.record_video)

    logging.info("==== 跑 backend B = %s（被测）====", FLAGS.backend_b)
    rb = ev.run_backend_episodes(
        env_b, vla, dqn_b["backend"], dqn_b["learner"], FLAGS.backend_b,
        n_episodes=FLAGS.n_episodes, max_decision_steps=FLAGS.max_decision_steps,
        seed=FLAGS.seed, out_dir=out_b, record_video=FLAGS.record_video)

    _save_compare_summary(out_dir, ra, rb)
    _plot_compare(out_dir, ra, rb)
    logging.info("对比输出: %s（compare_summary.json + compare_speedup.png + compare_knob.png；"
                 "各 backend 子目录含 videos/ + episode*_speed.{json,png}）", out_dir)


if __name__ == "__main__":
    app.run(main)
