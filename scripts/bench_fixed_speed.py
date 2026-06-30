#! /usr/bin/env python
"""固定加速系数 sweep：在冻结 VLA(pi0.5 drift) **之后**外接一个**固定速度控制系数**的加速模块
（替代 SpeedTune 的 branching Rainbow-DQN 动态选档），测各档位下的任务成功率。

设计见 docs/superpowers/specs/2026-06-30-fixed-speed-sweep-design.md。要点：

  - 加速对象 = VLA 实时推理出的 action chunk（经 step_chunk 的固定 v/vel_limit/acc_limit），
    位置与 SpeedTune 完全相同，只是把 DQN 选档换成固定常数。**不针对专家数据**：全程走
    step_chunk，自动专家介入(takeover/play_once)只在 per-action step() 触发，故绝不发生——
    成功率纯粹来自「VLA 动作 + 固定加速系数」。
  - 复用冻结 VLA（只加载一次）；不加载 DQN、不调 backend.decode，直接构造固定 speed_params dict。
  - 同批布局（控制变量）：每个配置开始前重建 env（create_env 覆盖，_ep_count 回 0），所有配置
    都跑 seed 0..N-1 的同一批布局，成功率差异只来自加速系数。
  - sweep：fixed_time 扫 v(7)；per_action_toppra 分别扫 v(7)/vel_limit(4)/acc_limit(9)，
    一个变量变化时其余两个固定 1.0（去重后共 26 个配置）。

用法（云端，learner venv；server 端先起 client_robotwin.run_robotwin_client，与 train 相同）：
    SPEEDTUNE_VLA_CKPT=<drift ckpt> SPEEDTUNE_VLA_ASSETS=<...> SPEEDTUNE_VLA_ASSET_ID=trossen \
    uv run python scripts/bench_fixed_speed.py \
        --config configs/model/speedtune_dqn_config.py \
        --config_task configs/task/robotwin_stack_blocks.py \
        --n_episodes 50 --seed 0 --max_decision_steps 400 \
        --backends both --config_subset 0/1 \
        --client_port 8102 --output_dir logs/bench_fixed_speed

并行（~1300 ep 单卡约 10-22h）：起多个 server（不同卡/端口），各进程用不同 --config_subset i/N
+ 对应 --client_port，最后人工合并各自的 bench_summary.csv。
"""

import os
import csv
import json
import logging

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

import jax
import etils.epath as epath

import openpi.training.sharding as openpi_sharding
from expo_ft.env.env_client import EnvClientWrapper
from expo_ft.utils.train_utils import init_logging
from expo_ft.speedtune.exec_backends import parse_force_limit

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

FLAGS = flags.FLAGS

flags.DEFINE_integer("n_episodes", 50, "每个配置跑多少 episode 统计成功率。")
flags.DEFINE_integer("seed", 0, "Random seed（同批布局：每配置都从该 seed 起跑相同 N 个布局）。")
flags.DEFINE_integer("max_decision_steps", 400, "每 episode 决策步上限（env 内部 step_lim 也会终止）。")
flags.DEFINE_string("backends", "both", "测哪些后端：both | fixed_time | per_action。")
flags.DEFINE_string("config_subset", "0/1", "配置分片 i/N（并行用，round-robin 切 configs[i::N]）。")
flags.DEFINE_string(
    "force_limit", None,
    "执行层力矩底座 per-joint N·m。不传=继承 config.force_limit（默认 30,40,30,15,10,10）；"
    "'none'/'off'/'inf'=关掉(∞，纯看速度系数)；或具体 '30,40,30,15,10,10'。")
flags.DEFINE_boolean("record_video", False, "是否录每 episode 视频（sweep 默认 False，量大不录）。")
flags.DEFINE_string("output_dir", "./logs/bench_fixed_speed", "输出目录（csv/json/png）。")
flags.DEFINE_string(
    "plot_from_csv", "",
    "若设：从 csv glob 合并各分片结果、写合并 csv/json + 画完整图（不跑 VLA/env，可 "
    "JAX_PLATFORMS=cpu 跑）。例：'logs/bench_xxx/shard_*/bench_summary.csv'。")
flags.DEFINE_string("client_host", "localhost", "Env server host.")
flags.DEFINE_integer("client_port", 8102, "Env server port.")
flags.DEFINE_boolean("resume", False, "build_pi05 resume（一般 False）。")
flags.DEFINE_integer("fsdp_devices", 1, "FSDP devices（VLA 冻结，单卡即可）。")

config_flags.DEFINE_config_file(
    "config", "configs/model/speedtune_dqn_config.py", "SpeedTune DQN config（取 VLA/执行参数）。",
    lock_config=False)
config_flags.DEFINE_config_file(
    "config_task", "configs/task/robotwin_stack_blocks.py", "Task config.", lock_config=False)


# --- sweep 档位（用户指定；一个变量变化时其余两个固定 1.0）---
V_GRID = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
VEL_GRID = (1.0, 2.0, 3.0, 4.0)
ACC_GRID = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)

PER_ACTION = "per_action_toppra"
FIXED_TIME = "fixed_time"


def build_frozen_vla(config, config_task, seed, mesh, shardings, env, resume):
    """加载冻结 VLA + drift 前向 / delta 反归一化（逻辑同 eval_speedtune._build_vla，去掉 DQN 相关）。

    返回 dict: drift_forward / unnormalize / preprocess / H / A。VLA 全程冻结（只前向）。
    """
    from expo_ft.agents.vla.pi05 import build_pi05
    from expo_ft.agents.vla.drift_adapters import make_drift_apply_fn

    data_sharding, replicated_sharding = shardings
    n_real_dims = int(config.n_real_dims)
    resize_size = int(config.pi05_resize_size)

    actor, actor_train_state, *_ = build_pi05(
        config, seed, mesh, data_sharding, replicated_sharding,
        resume, config_task.language_instruction,
    )
    H = int(actor.model_config.action_horizon)
    A = int(actor.model_config.action_dim)
    actor.action_dim = n_real_dims
    actor.state_dim = n_real_dims

    def preprocess(raw_obs):
        raw = dict(raw_obs)
        raw.setdefault("action", np.zeros((H, n_real_dims), dtype=np.float32))
        return actor.process_raw_inputs(raw, action_dim=n_real_dims, resize_size=resize_size)

    drift_apply_fn = make_drift_apply_fn(actor_train_state.model_def)
    vla_params = actor_train_state.params

    @jax.jit
    def _forward(params, obs, z):
        return drift_apply_fn(params, obs, z, None)

    def drift_forward(obs_m, z):
        return _forward(vla_params, obs_m, z)  # (mean, cond_emb, value_feat)

    def unnormalize(mean, obs_m):
        """model-space mean[1,H,A] → 真实 qpos chunk[H,n_real]（delta 反归一化，加回当前 state）。"""
        unpad = actor._unpad_actions(mean)
        real = actor.process_transformed_outputs(np.asarray(unpad), state=np.asarray(obs_m["state"]))
        return np.asarray(real[0])

    # 探针一次前向（确认 H/A/feat 维度，并触发 jit 编译）。
    example_obs = preprocess(env.get_observation())
    example_z = jax.random.normal(jax.random.PRNGKey(seed + 7), (1, H, A))
    _m0, _c0, vfeat0 = drift_forward(example_obs, example_z)
    logging.info("bench VLA: H=%d A=%d feat_dim=%d", H, A, int(np.asarray(vfeat0).shape[-1]))

    return dict(drift_forward=drift_forward, unnormalize=unnormalize, preprocess=preprocess, H=H, A=A)


def build_configs(which):
    """生成 sweep 配置列表（按 (backend,v,vel,acc) 去重；per_action 基线 (1,1,1) 只跑一次）。

    每个 config: dict(backend, v, vel_limit, acc_limit, speed_params)。fixed_time 只含 v
    （streaming 不吃 vel/acc）；per_action 含 v/vel_limit/acc_limit 三者。
    """
    out = {}

    def add(backend, v, vel, acc, speed_params):
        key = (backend, round(float(v), 3),
               None if vel is None else round(float(vel), 3),
               None if acc is None else round(float(acc), 3))
        if key not in out:
            out[key] = dict(backend=backend, v=float(v),
                            vel_limit=None if vel is None else float(vel),
                            acc_limit=None if acc is None else float(acc),
                            speed_params=speed_params)

    if which in ("both", "fixed_time"):
        for v in V_GRID:
            add(FIXED_TIME, v, None, None, {"v": float(v)})

    if which in ("both", "per_action"):
        for v in V_GRID:                       # v sweep（vel=1, acc=1 固定）
            add(PER_ACTION, v, 1.0, 1.0, {"v": float(v), "vel_limit": 1.0, "acc_limit": 1.0})
        for vel in VEL_GRID:                   # vel_limit sweep（v=1, acc=1 固定）
            add(PER_ACTION, 1.0, vel, 1.0, {"v": 1.0, "vel_limit": float(vel), "acc_limit": 1.0})
        for acc in ACC_GRID:                   # acc_limit sweep（v=1, vel=1 固定）
            add(PER_ACTION, 1.0, 1.0, acc, {"v": 1.0, "vel_limit": 1.0, "acc_limit": float(acc)})

    return list(out.values())


def shard_configs(configs, subset):
    """按 'i/N' round-robin 切片（configs[i::N]），并行分摊用。"""
    try:
        i_s, n_s = str(subset).split("/")
        i, n = int(i_s), int(n_s)
    except Exception:
        raise ValueError(f"--config_subset 格式应为 i/N（如 0/4），得到 {subset!r}")
    if not (0 <= i < n):
        raise ValueError(f"--config_subset i={i} 应 ∈ [0,{n})")
    return configs[i::n]


def resolve_force_limit(flag_val, config):
    """force_limit flag → per-joint list 或 None(∞)。不传=继承 config；'none/off/inf'=关掉。"""
    if flag_val is None:
        return parse_force_limit(config.get("force_limit", ""))
    s = str(flag_val).strip().lower()
    if s in ("none", "off", "inf", "infinite", ""):
        return None
    return parse_force_limit(flag_val)


def run_fixed_config(env, vla, cfg, n_episodes, max_decision_steps, seed, record_video):
    """对一个固定 speed_params 配置跑 n_episodes，返回聚合 result dict。

    每决策步：冻结 VLA 出 chunk → env.step_chunk(chunk, 固定 speed_params, backend) → get_info_for_step。
    """
    drift_forward = vla["drift_forward"]
    unnormalize = vla["unnormalize"]
    preprocess = vla["preprocess"]
    H, A = vla["H"], vla["A"]
    backend = cfg["backend"]
    speed_params = cfg["speed_params"]

    rng = np.random.default_rng(seed)   # 固定 seed → 各配置同样的 z 序列（同布局起点尽量可比）
    episodes = []
    n_succ = 0

    for ep in range(n_episodes):
        obs = env.reset()
        if record_video:
            env.start_video(ep)
        ep_success = False
        dense = 0
        sim_t = 0.0
        n_steps = 0
        n_fallback = 0

        for _step in tqdm.tqdm(range(max_decision_steps),
                               desc=f"{backend} v={cfg['v']} vel={cfg['vel_limit']} acc={cfg['acc_limit']} ep{ep}",
                               disable=False):
            obs_m = preprocess(obs)
            z = jax.random.normal(jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1))), (1, H, A))
            mean, _cond, _vfeat = drift_forward(obs_m, z)
            real_chunk = unnormalize(mean, obs_m)

            _executed, info = env.step_chunk(real_chunk, speed_params, backend)
            done, success, _r, _mask = env.get_info_for_step()
            ep_success = ep_success or bool(success)
            dense += int(info.get("n_exec_steps", 0))
            sim_t += float(info.get("duration", 0.0))
            n_steps += 1
            if info.get("exec_status") == "topp_fallback":
                n_fallback += 1
            if done:
                break
            obs = env.get_observation()

        if record_video:
            env.stop_video()
        n_succ += int(ep_success)
        episodes.append(dict(ep=ep, success=bool(ep_success), decision_steps=n_steps,
                             dense_steps=int(dense), sim_time_s=round(sim_t, 4),
                             n_fallback=int(n_fallback)))
        logging.info("[%s v=%s vel=%s acc=%s] ep%d: success=%s dec_steps=%d dense=%d sim=%.2fs",
                     backend, cfg["v"], cfg["vel_limit"], cfg["acc_limit"],
                     ep, ep_success, n_steps, dense, sim_t)

    total_steps = sum(e["decision_steps"] for e in episodes)
    total_fb = sum(e["n_fallback"] for e in episodes)
    return dict(
        backend=backend, v=cfg["v"], vel_limit=cfg["vel_limit"], acc_limit=cfg["acc_limit"],
        n_episodes=n_episodes, n_success=n_succ, success_rate=n_succ / max(n_episodes, 1),
        mean_dense_steps=float(np.mean([e["dense_steps"] for e in episodes])) if episodes else 0.0,
        mean_sim_time_s=float(np.mean([e["sim_time_s"] for e in episodes])) if episodes else 0.0,
        fallback_rate=total_fb / max(total_steps, 1),
        episodes=episodes,
    )


CSV_FIELDS = ["backend", "sweep_var", "v", "vel_limit", "acc_limit", "n_episodes", "n_success",
              "success_rate", "mean_dense_steps", "mean_sim_time_s", "fallback_rate"]


def _sweep_var_of(r):
    """标注该配置主要变化的变量（便于人读 csv；画图用筛选条件、不依赖此标签）。"""
    if r["backend"] == FIXED_TIME:
        return "v"
    v, vel, acc = r["v"], r["vel_limit"], r["acc_limit"]
    if _is(vel, 1.0) and _is(acc, 1.0) and not _is(v, 1.0):
        return "v"
    if _is(v, 1.0) and _is(acc, 1.0) and not _is(vel, 1.0):
        return "vel_limit"
    if _is(v, 1.0) and _is(vel, 1.0) and not _is(acc, 1.0):
        return "acc_limit"
    return "baseline"   # (1,1,1)：三条 sweep 共用的基线点


def write_outputs(out_dir, results):
    """增量写盘（每配置完成后重写全量，防长跑中途崩溃丢结果）。"""
    with open(os.path.join(out_dir, "bench_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in results:
            row = {k: r.get(k) for k in CSV_FIELDS}
            row["sweep_var"] = _sweep_var_of(r)
            w.writerow(row)
    with open(os.path.join(out_dir, "bench_summary.json"), "w") as f:
        json.dump(results, f, indent=2)


def _is(x, val):
    return x is not None and abs(float(x) - float(val)) < 1e-6


def plot_results(out_dir, results, subset):
    """画 fixed_time(v) + per_action(v/vel/acc) 折线：success_rate(左轴) + mean_dense_steps(右轴)。

    分片(subset != 'i/1')时本进程结果不完整，仅画本进程有的点并 warning；完整图请合并 csv 后另画。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logging.warning("matplotlib 不可用，跳过画图（csv/json 已存）：%s", e)
        return

    try:
        _, n = str(subset).split("/")
        if int(n) != 1:
            logging.warning("分片模式(subset=%s)：本进程结果不完整，图仅含本片点；完整图请合并各片 csv 后另画。",
                            subset)
    except Exception:
        pass

    def line(points, xkey, title, fname, xlabel):
        if not points:
            return
        xs = [p[xkey] for p in points]
        sr = [100.0 * p["success_rate"] for p in points]
        ds = [p["mean_dense_steps"] for p in points]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xs, sr, "-o", color="tab:blue", label="success rate (%)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("success rate (%)", color="tab:blue")
        ax.set_ylim(-5, 105)
        ax2 = ax.twinx()
        ax2.plot(xs, ds, "--s", color="tab:red", alpha=0.6, label="mean dense steps (exec cost)")
        ax2.set_ylabel("mean dense steps (lower=faster)", color="tab:red")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=120)
        plt.close(fig)

    ft = sorted([r for r in results if r["backend"] == FIXED_TIME], key=lambda r: r["v"])
    line(ft, "v", "fixed_time: success vs v", "fixed_time_v.png", "v (compress ratio)")

    pa = [r for r in results if r["backend"] == PER_ACTION]
    line(sorted([r for r in pa if _is(r["vel_limit"], 1.0) and _is(r["acc_limit"], 1.0)],
                key=lambda r: r["v"]),
         "v", "per_action: success vs v (vel=1, acc=1)", "per_action_v.png", "v (compress ratio)")
    line(sorted([r for r in pa if _is(r["v"], 1.0) and _is(r["acc_limit"], 1.0)],
                key=lambda r: r["vel_limit"]),
         "vel_limit", "per_action: success vs vel_limit (v=1, acc=1)", "per_action_vel.png",
         "vel_limit (rad/s)")
    line(sorted([r for r in pa if _is(r["v"], 1.0) and _is(r["vel_limit"], 1.0)],
                key=lambda r: r["acc_limit"]),
         "acc_limit", "per_action: success vs acc_limit (v=1, vel=1)", "per_action_acc.png",
         "acc_limit (rad/s^2)")


def _reseed_env(env, seed):
    """让 server 端 RoboTwinEnv 重置 episode 计数器（下次 reset 从 seed 重跑同一批布局）。

    用底层 EnvClient._call_operation 发 'reseed' op（client_robotwin 新增）；EnvClientWrapper
    保持不动（CLAUDE.md：env_client.py 通用、不动）。
    """
    env.client._call_operation("reseed", {"env_id": env.env_id, "seed": int(seed)})


def _load_results_from_csv(glob_pat):
    """从 csv glob（各分片 bench_summary.csv）合并 results（去重 (backend,v,vel,acc)），供 plot_from_csv。"""
    import glob as _glob

    def _num(x):
        return None if x in ("", "None", None) else float(x)

    out, seen = [], set()
    for path in sorted(_glob.glob(glob_pat)):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                v, vel, acc = _num(row.get("v")), _num(row.get("vel_limit")), _num(row.get("acc_limit"))
                key = (row["backend"], v, vel, acc)
                if key in seen:
                    continue
                seen.add(key)
                out.append(dict(
                    backend=row["backend"], v=v, vel_limit=vel, acc_limit=acc,
                    n_episodes=int(float(row["n_episodes"])), n_success=int(float(row["n_success"])),
                    success_rate=float(row["success_rate"]),
                    mean_dense_steps=float(row["mean_dense_steps"]),
                    mean_sim_time_s=float(row["mean_sim_time_s"]),
                    fallback_rate=float(row["fallback_rate"]),
                ))
    logging.info("loaded %d unique configs from csv glob %r", len(out), glob_pat)
    return out


def main(_):
    init_logging()

    # 合并画图模式（4 卡分片跑完的收尾）：纯 csv + matplotlib，不跑 VLA/env。
    if FLAGS.plot_from_csv:
        results = _load_results_from_csv(FLAGS.plot_from_csv)
        os.makedirs(FLAGS.output_dir, exist_ok=True)
        write_outputs(FLAGS.output_dir, results)
        plot_results(FLAGS.output_dir, results, "0/1")
        logging.info("merged %d configs → %s（bench_summary.csv/json + *.png）",
                     len(results), FLAGS.output_dir)
        return

    config, config_task = FLAGS.config, FLAGS.config_task
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
    jax.config.update("jax_default_matmul_precision", "highest")

    mesh = openpi_sharding.make_mesh(FLAGS.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    out_dir = FLAGS.output_dir
    os.makedirs(out_dir, exist_ok=True)

    force_limit = resolve_force_limit(FLAGS.force_limit, config)
    logging.info("bench force_limit=%s（None=∞ 不施加）", force_limit)

    example_action = np.asarray(
        config_task.get("example_action", np.zeros((1, int(config.n_real_dims)), dtype=np.float32)),
        dtype=np.float32)

    def make_request():
        # env 创建即设执行参数；exec_backend 初始值无所谓（step_chunk 每次显式传具体 backend）。
        return {
            "example_action": example_action, "env_usage": "eval",
            "video_dir": os.path.join(out_dir, "videos"),
            "exec_backend": str(config.exec_backend),
            "k_skip": config.get("k_skip", None),
            "stream_hold_steps": int(config.get("stream_hold_steps", 15)),
            "eval_video_save_freq": 25,
            "force_limit": force_limit,
        }

    # 初始 env：仅用于 VLA 探针（一次前向编译）；之后每配置重建 env 从 seed 0 跑同批布局。
    logging.info("Creating RoboTwin env via env_client (probe) ...")
    env = EnvClientWrapper(env_creation_request=make_request(),
                           host=FLAGS.client_host, port=FLAGS.client_port)
    env.reset()
    vla = build_frozen_vla(config, config_task, FLAGS.seed, mesh,
                           (data_sharding, replicated_sharding), env, FLAGS.resume)

    configs = build_configs(FLAGS.backends)
    configs = shard_configs(configs, FLAGS.config_subset)
    logging.info("bench: %d configs (backends=%s subset=%s), %d episodes each (~%d ep total)",
                 len(configs), FLAGS.backends, FLAGS.config_subset, FLAGS.n_episodes,
                 len(configs) * FLAGS.n_episodes)

    results = []
    for ci, cfg in enumerate(configs):
        logging.info("==== config %d/%d: backend=%s v=%s vel=%s acc=%s ====",
                     ci + 1, len(configs), cfg["backend"], cfg["v"], cfg["vel_limit"], cfg["acc_limit"])
        # 同批布局：reseed 重置 server 端 _ep_count（+ 释放上一配置 scene），下个配置从 seed 0
        # 跑相同 N 个布局。不重建 env → 全程单个 sapien 实例，无显存累积（同卡 server+bench 安全）。
        _reseed_env(env, FLAGS.seed)
        res = run_fixed_config(env, vla, cfg, FLAGS.n_episodes, FLAGS.max_decision_steps,
                               FLAGS.seed, FLAGS.record_video)
        results.append(res)
        write_outputs(out_dir, results)   # 增量写盘
        logging.info("---- success=%.1f%% (%d/%d) | mean_dense=%.0f sim=%.2fs | fallback=%.0f%% ----",
                     100 * res["success_rate"], res["n_success"], res["n_episodes"],
                     res["mean_dense_steps"], res["mean_sim_time_s"], 100 * res["fallback_rate"])

    plot_results(out_dir, results, FLAGS.config_subset)
    logging.info("==== bench DONE ==== 输出: %s（bench_summary.csv/json + *.png）", out_dir)
    for r in results:
        logging.info("  %-18s v=%-4s vel=%-4s acc=%-4s : success=%.0f%% dense=%.0f fallback=%.0f%%",
                     r["backend"], r["v"], r["vel_limit"], r["acc_limit"],
                     100 * r["success_rate"], r["mean_dense_steps"], 100 * r["fallback_rate"])


if __name__ == "__main__":
    app.run(main)
