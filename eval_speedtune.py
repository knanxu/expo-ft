#! /usr/bin/env python
"""SpeedTune eval：冻结 VLA(base policy) + 训练好的 DQN(速度模块)，在 RoboTwin 跑 N episode。

每个 episode：① 录制 head_camera 视频(server 端 ffmpeg)；② 每个决策步记录速度控制参数
(v/vel_scale/acc_scale + 归一化激进度) 与接触信号(夹爪值 + sapien 物理接触) + 累计仿真时间；
③ 存 json + 画「激进度 vs 时间(接触时段阴影)」图。结尾汇总**接触时段 vs 非接触时段的平均激进度**
——直接回答「是否在接触附近减速、其他时候加速」。

DQN 用 greedy(无探索)。架构同 train：client-learner 分离(VLA/DQN 在 JAX venv，RoboTwin 在 sim venv)，
经 client_robotwin 的 step_chunk(回传接触) + start/stop_video(录像) 协议。

用法（云端，learner venv）：
    SPEEDTUNE_VLA_CKPT=<drift ckpt> SPEEDTUNE_VLA_ASSETS=<...> SPEEDTUNE_VLA_ASSET_ID=<...> \
    uv run python eval_speedtune.py \
        --config configs/model/speedtune_dqn_config.py --config.exec_backend per_action_toppra \
        --config_task configs/task/robotwin_stack_blocks.py \
        --dqn_ckpt logs/.../checkpoints/update_<N> --n_episodes 5 \
        --client_port 8102 --output_dir logs/speedtune_eval
（server 端起 client_robotwin.run_robotwin_client，与 train 相同。）
"""

import os
import json
import logging

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

import jax
import jax.numpy as jnp
import etils.epath as epath

import openpi.training.sharding as openpi_sharding
from expo_ft.env.env_client import EnvClientWrapper
from expo_ft.utils.train_utils import init_logging

from expo_ft.agents.alg.speedtune_dqn import SpeedTuneLearner
from expo_ft.speedtune.exec_backends import build_backend

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

FLAGS = flags.FLAGS

flags.DEFINE_string("dqn_ckpt", None, "训练好的 speedtune DQN checkpoint 目录（含 q_net/，即 update_<N>）。")
flags.DEFINE_integer("n_episodes", 5, "eval episode 数。")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("max_decision_steps", 400, "每 episode 决策步上限（防卡死；env 内部 step_lim 也会终止）。")
flags.DEFINE_string("output_dir", "./logs/speedtune_eval", "输出目录（视频/json/png）。")
flags.DEFINE_string("client_host", "localhost", "Env server host.")
flags.DEFINE_integer("client_port", 8102, "Env server port.")
flags.DEFINE_boolean("record_video", True, "是否录制 episode 视频（server 端 ffmpeg）。")
flags.DEFINE_boolean("resume", False, "build_pi05 resume（一般 False）。")
flags.DEFINE_integer("fsdp_devices", 1, "FSDP devices for sharding（eval VLA 冻结，单卡即可）。")

config_flags.DEFINE_config_file(
    "config", "configs/model/speedtune_dqn_config.py", "SpeedTune DQN config.", lock_config=False)
config_flags.DEFINE_config_file(
    "config_task", "configs/task/robotwin_stack_blocks.py", "Task config.", lock_config=False)


def _build_eval(config, config_task, seed, mesh, shardings, env):
    """加载冻结 VLA + drift 前向/反归一化 + exec_backend + DQN（结构同 train 的 _setup_speedtune）。"""
    from expo_ft.agents.vla.pi05 import build_pi05
    from expo_ft.agents.vla.drift_adapters import make_drift_apply_fn

    data_sharding, replicated_sharding = shardings
    n_real_dims = int(config.n_real_dims)
    resize_size = int(config.pi05_resize_size)

    actor, actor_train_state, _t, _ak, _md = build_pi05(
        config, seed, mesh, data_sharding, replicated_sharding,
        FLAGS.resume, config_task.language_instruction,
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
        return _forward(vla_params, obs_m, z)

    def unnormalize(mean, obs_m):
        unpad = actor._unpad_actions(mean)
        real = actor.process_transformed_outputs(np.asarray(unpad), state=np.asarray(obs_m["state"]))
        return np.asarray(real[0])

    example_obs = preprocess(env.get_observation())
    example_z = jax.random.normal(jax.random.PRNGKey(seed + 7), (1, H, A))
    _m0, _c0, vfeat0 = drift_forward(example_obs, example_z)
    feat_dim = int(np.asarray(vfeat0).shape[-1])

    backend = build_backend(str(config.exec_backend))
    logging.info("SpeedTune eval: feat_dim=%d H=%d A=%d backend=%s head_sizes=%s",
                 feat_dim, H, A, backend.name, backend.head_sizes)

    dqn = SpeedTuneLearner.create(
        rng=jax.random.PRNGKey(seed), feat_dim=feat_dim, head_sizes=backend.head_sizes,
        n_atoms=int(config.n_atoms), v_min=float(config.v_min), v_max=float(config.v_max),
        hidden_dims=tuple(config.q_hidden_dims), dueling=bool(config.dueling),
        detach_input=bool(config.detach_q_input), epsilon=0.0,   # greedy
        example_feat=jnp.asarray(vfeat0),
    )
    dqn = _restore_dqn(dqn, FLAGS.dqn_ckpt)
    return dict(drift_forward=drift_forward, unnormalize=unnormalize, preprocess=preprocess,
                backend=backend, learner=dqn, H=H, A=A)


def _restore_dqn(learner, ckpt_dir):
    """orbax 恢复训练好的 q_net 参数到 learner（target 不需要，eval 只用 q_net greedy）。"""
    import orbax.checkpoint as ocp
    if not ckpt_dir:
        raise ValueError("--dqn_ckpt 必填：训练好的 speedtune DQN checkpoint 目录（update_<N>）")
    qdir = epath.Path(ckpt_dir).resolve() / "q_net"
    if not qdir.exists():
        raise FileNotFoundError(f"找不到 {qdir}（--dqn_ckpt 应指向含 q_net/ 的 update_<N> 目录）")
    with ocp.StandardCheckpointer() as c:
        try:
            q_params = c.restore(qdir, learner.q_net.params)        # 带模板（shape/dtype 对齐）
        except Exception:
            q_params = c.restore(qdir)                              # 退化：无模板
    logging.info("restored DQN q_net from %s", qdir)
    return learner.replace(q_net=learner.q_net.replace(params=q_params))


def _save_and_plot(out_dir, ep, recs):
    """存 episode 速度/接触时间序列 json + 画「激进度 vs 时间(接触阴影)」png。"""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"episode{ep}_speed.json"), "w") as f:
        json.dump(recs, f, indent=2)
    if not recs:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logging.warning("matplotlib 不可用，跳过画图（json 已存）：%s", e)
        return

    t = [r["t_sim"] for r in recs]
    aggr = [r["aggr_mean"] for r in recs]
    lg = [r["left_gripper"] for r in recs]
    rg = [r["right_gripper"] for r in recs]
    contact = [bool(r["left_contact"] or r["right_contact"]) for r in recs]

    fig, ax = plt.subplots(figsize=(11, 5))
    # 接触时段阴影（每个决策步是 [t_{i-1}, t_i] 一段）
    for i, c in enumerate(contact):
        if c:
            t0 = t[i - 1] if i > 0 else 0.0
            ax.axvspan(t0, t[i], color="red", alpha=0.12,
                       label="contact" if i == contact.index(True) else None)
    ax.plot(t, aggr, "-o", color="tab:blue", label="aggressiveness (0慢~1快)")
    ax.plot(t, lg, "--", color="tab:green", alpha=0.6, label="left gripper (0闭~1开)")
    ax.plot(t, rg, "--", color="tab:olive", alpha=0.6, label="right gripper")
    ax.set_xlabel("sim time (s)")
    ax.set_ylabel("aggressiveness / gripper")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"episode {ep}: speed knob vs contact")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"episode{ep}_speed.png"), dpi=120)
    plt.close(fig)


def main(_):
    init_logging()
    config, config_task = FLAGS.config, FLAGS.config_task
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
    jax.config.update("jax_default_matmul_precision", "highest")

    mesh = openpi_sharding.make_mesh(FLAGS.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    out_dir = FLAGS.output_dir
    os.makedirs(out_dir, exist_ok=True)
    exec_backend = str(config.exec_backend)
    example_action = np.asarray(
        config_task.get("example_action", np.zeros((1, int(config.n_real_dims)), dtype=np.float32)),
        dtype=np.float32)

    logging.info("Creating RoboTwin eval env (is_eval, 录视频) via env_client ...")
    env = EnvClientWrapper(
        env_creation_request={
            "example_action": example_action, "env_usage": "eval",
            "video_dir": os.path.join(out_dir, "videos"),
            "exec_backend": exec_backend,
            "k_skip": config.get("k_skip", None),
            "stream_hold_steps": int(config.get("stream_hold_steps", 15)),
            "eval_video_save_freq": 25,
        },
        host=FLAGS.client_host, port=FLAGS.client_port,
    )
    env.reset()

    setup = _build_eval(config, config_task, FLAGS.seed, mesh,
                        (data_sharding, replicated_sharding), env)
    drift_forward, unnormalize, preprocess = setup["drift_forward"], setup["unnormalize"], setup["preprocess"]
    backend, learner, H, A = setup["backend"], setup["learner"], setup["H"], setup["A"]

    rng = np.random.default_rng(FLAGS.seed)
    n_succ = 0
    all_contact_aggr, all_free_aggr = [], []   # 跨 episode 汇总：接触 vs 非接触时段激进度

    for ep in range(FLAGS.n_episodes):
        obs = env.reset()
        if FLAGS.record_video:
            env.start_video(ep)
        recs, t_acc, ep_success = [], 0.0, False

        for step_i in tqdm.tqdm(range(FLAGS.max_decision_steps), desc=f"ep{ep}", disable=False):
            obs_m = preprocess(obs)
            z = jax.random.normal(jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1))), (1, H, A))
            mean, _cond, value_feat = drift_forward(obs_m, z)
            feat = np.asarray(value_feat[0])
            real_chunk = unnormalize(mean, obs_m)

            idxs = np.asarray(learner.greedy_action_idxs(feat[None])[0], dtype=np.int32)  # greedy 无探索
            speed_params, v_list = backend.decode(idxs)

            executed, info = env.step_chunk(real_chunk, speed_params, exec_backend)
            done, success, _r, _mask = env.get_info_for_step()
            ep_success = ep_success or bool(success)
            t_acc += float(info.get("duration", 0.0))

            lc, rc = bool(info.get("left_contact", False)), bool(info.get("right_contact", False))
            aggr_mean = float(np.mean(v_list)) if len(v_list) else 0.0
            rec = dict(
                step=step_i, t_sim=round(t_acc, 4),
                dense_steps=int(info.get("n_exec_steps", 0)),
                aggr_mean=aggr_mean, aggr=[float(x) for x in v_list],
                left_gripper=float(info.get("left_gripper", 0.0)),
                right_gripper=float(info.get("right_gripper", 0.0)),
                left_contact=lc, right_contact=rc,
                exec_status=info.get("exec_status", "success"), success=bool(success),
            )
            rec.update({k: float(v) for k, v in speed_params.items()})  # v / vel_scale / acc_scale
            recs.append(rec)
            (all_contact_aggr if (lc or rc) else all_free_aggr).append(aggr_mean)

            if done:
                break
            obs = env.get_observation()

        if FLAGS.record_video:
            env.stop_video()
        n_succ += int(ep_success)
        _save_and_plot(out_dir, ep, recs)
        logging.info("episode %d: success=%s steps=%d sim_time=%.2fs (video+json+png 已存)",
                     ep, ep_success, len(recs), t_acc)

    # ---- 汇总：接触 vs 非接触时段的平均激进度（回答「接触附近是否减速」）----
    c_mean = float(np.mean(all_contact_aggr)) if all_contact_aggr else float("nan")
    f_mean = float(np.mean(all_free_aggr)) if all_free_aggr else float("nan")
    summary = {
        "n_episodes": FLAGS.n_episodes, "success": n_succ,
        "success_rate": n_succ / max(FLAGS.n_episodes, 1),
        "aggr_in_contact_mean": c_mean,       # 接触时段平均激进度（越低=接触附近越减速）
        "aggr_free_mean": f_mean,             # 非接触时段平均激进度
        "decel_near_contact": bool(c_mean < f_mean) if (all_contact_aggr and all_free_aggr) else None,
        "n_contact_steps": len(all_contact_aggr), "n_free_steps": len(all_free_aggr),
    }
    with open(os.path.join(out_dir, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logging.info("==== SpeedTune eval 汇总 ====")
    logging.info("success: %d/%d (%.0f%%)", n_succ, FLAGS.n_episodes, 100 * summary["success_rate"])
    logging.info("接触时段平均激进度 = %.3f | 非接触时段 = %.3f → 接触附近%s",
                 c_mean, f_mean,
                 "减速 ✓" if summary["decel_near_contact"] else
                 ("未减速" if summary["decel_near_contact"] is False else "(数据不足)"))
    logging.info("输出目录: %s（videos/ + episode*_speed.{json,png} + eval_summary.json）", out_dir)


if __name__ == "__main__":
    app.run(main)
