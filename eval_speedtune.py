#! /usr/bin/env python
"""SpeedTune eval：冻结 VLA(base policy) + 训练好的 DQN(速度模块)，在 RoboTwin 跑 N episode。

每个 episode：① 录制 head_camera 视频(server 端 ffmpeg)；② 每个决策步记录速度控制参数
(v/vel_limit/acc_limit + 归一化激进度) 与接触信号(夹爪值 + sapien 物理接触) + 累计仿真时间；
③ 存 json + 画「激进度 vs 时间(接触时段阴影)」图。结尾汇总**接触时段 vs 非接触时段的平均激进度**
——直接回答「是否在接触附近减速、其他时候加速」。

DQN 用 greedy(无探索)。架构同 train：client-learner 分离(VLA/DQN 在 JAX venv，RoboTwin 在 sim venv)，
经 client_robotwin 的 step_chunk(回传接触) + start/stop_video(录像) 协议。

本模块的 ``_build_vla`` / ``_build_dqn`` / ``run_backend_episodes`` 抽成可复用单元，供
``eval_speedtune_compare.py``（双 backend 加速对比）共用——VLA(3B) 只加载一次，两个 backend 复用。

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
import time

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
from expo_ft.speedtune.exec_backends import build_backend, parse_force_limit
from expo_ft.speedtune.runtime_config import (
    backend_k_skip,
    backend_support,
    episode_reward_success,
)
from expo_ft.speedtune.paired_eval import noise_seed
from expo_ft.speedtune.checkpoint_contract import (
    build_checkpoint_metadata,
    validate_checkpoint_metadata,
    WHOLE_CHUNK_ACC_RULE,
)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

FLAGS = flags.FLAGS

flags.DEFINE_string("dqn_ckpt", None, "训练好的 speedtune DQN checkpoint 目录（含 q_net/，即 update_<N>）。")
flags.DEFINE_integer("n_episodes", 30, "eval episode 数。")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("max_decision_steps", 400, "每 episode 决策步上限（防卡死；env 内部 step_lim 也会终止）。")
flags.DEFINE_string("output_dir", "./logs/speedtune_eval", "输出目录（视频/json/png）。")
flags.DEFINE_string("client_host", "localhost", "Env server host.")
flags.DEFINE_integer("client_port", 8102, "Env server port.")
flags.DEFINE_boolean("record_video", False, "是否录制 episode 视频；严格墙钟计时默认关闭。")
flags.DEFINE_boolean("resume", False, "build_pi05 resume（一般 False）。")
flags.DEFINE_integer("fsdp_devices", 1, "FSDP devices for sharding（eval VLA 冻结，单卡即可）。")

config_flags.DEFINE_config_file(
    "config", "configs/model/speedtune_dqn_config.py", "SpeedTune DQN config.", lock_config=False)
config_flags.DEFINE_config_file(
    "config_task", "configs/task/robotwin_stack_blocks.py", "Task config.", lock_config=False)


def _build_vla(config, config_task, seed, mesh, shardings, env, resume):
    """加载冻结 VLA + drift 前向/反归一化（**与 exec_backend / DQN 无关**，可被多 backend 复用）。

    返回 dict: drift_forward / unnormalize / preprocess / H / A / feat_dim / vfeat0。
    feat_dim、vfeat0 仅由 VLA 决定（与 backend 无关），供 _build_dqn 构造 DQN。
    """
    from expo_ft.agents.vla.pi05 import build_pi05
    from expo_ft.agents.vla.drift_adapters import make_drift_apply_fn

    data_sharding, replicated_sharding = shardings
    n_real_dims = int(config.n_real_dims)
    resize_size = int(config.pi05_resize_size)

    actor, actor_train_state, _t, _ak, _md = build_pi05(
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
        return _forward(vla_params, obs_m, z)

    def unnormalize(mean, obs_m):
        unpad = actor._unpad_actions(mean)
        real = actor.process_transformed_outputs(np.asarray(unpad), state=np.asarray(obs_m["state"]))
        return np.asarray(real[0])

    example_obs = preprocess(env.get_observation())
    example_z = jax.random.normal(jax.random.PRNGKey(seed + 7), (1, H, A))
    _m0, _c0, vfeat0 = drift_forward(example_obs, example_z)
    feat_dim = int(np.asarray(vfeat0).shape[-1])
    logging.info("SpeedTune eval VLA: feat_dim=%d H=%d A=%d", feat_dim, H, A)

    return dict(drift_forward=drift_forward, unnormalize=unnormalize, preprocess=preprocess,
                H=H, A=A, feat_dim=feat_dim, vfeat0=vfeat0)


def _build_dqn(config, seed, exec_backend, dqn_ckpt, feat_dim, vfeat0):
    """构造 exec_backend + 从 ckpt 恢复 DQN（greedy，epsilon=0）。返回 dict: backend / learner。"""
    backend = build_backend(str(exec_backend))
    logging.info("SpeedTune eval DQN: backend=%s head_sizes=%s ckpt=%s",
                 backend.name, backend.head_sizes, dqn_ckpt)

    expected_metadata = build_checkpoint_metadata(config, backend)
    validate_checkpoint_metadata(dqn_ckpt, expected_metadata)
    v_min, v_max = backend_support(config, str(exec_backend))
    dqn = SpeedTuneLearner.create(
        rng=jax.random.PRNGKey(seed), feat_dim=feat_dim, head_sizes=backend.head_sizes,
        n_atoms=int(config.n_atoms), v_min=v_min, v_max=v_max,
        hidden_dims=tuple(config.q_hidden_dims), dueling=bool(config.dueling),
        detach_input=bool(config.detach_q_input), epsilon=0.0,   # greedy
        example_feat=jnp.asarray(vfeat0),
    )
    dqn = _restore_dqn(dqn, dqn_ckpt)
    # Exclude one-time JAX compilation from deployment wall-time metrics.
    np.asarray(dqn.greedy_action_idxs(jnp.asarray(vfeat0)))
    return dict(backend=backend, learner=dqn)


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


def _overlay_contact_grasp(ax, recs, x_key="t_sim", grasp_thresh=0.5):
    """在 ax 上叠加：① 接触方块时段(红色阴影, left/right_contact=夹爪↔物体接触)；
    ② 抓取事件(夹爪 开→闭 穿越阈值的紫色竖线, gripper∈[0,1] 0=闭/抓)。

    recs 需含 left_contact/right_contact、left_gripper/right_gripper、x_key。
    返回是否画了任何标注（供调用方决定图例）。
    """
    if not recs:
        return
    xs = [r[x_key] for r in recs]
    _cl = True
    for i, r in enumerate(recs):
        if r.get("left_contact") or r.get("right_contact"):
            x0 = xs[i - 1] if i > 0 else xs[0]
            ax.axvspan(x0, xs[i], color="red", alpha=0.10,
                       label="contact block" if _cl else None)
            _cl = False
    _gl = True
    for i in range(1, len(recs)):
        for key in ("left_gripper", "right_gripper"):
            if recs[i - 1].get(key, 1.0) >= grasp_thresh > recs[i].get(key, 1.0):  # 开→闭 = 抓取
                ax.axvline(xs[i], color="purple", ls=":", alpha=0.75,
                           label="grasp (gripper close)" if _gl else None)
                _gl = False
                break  # 同一步左右都闭只画一条竖线


def _save_and_plot(out_dir, ep, recs, ep_success=None):
    """存 episode 速度/接触时间序列 json + 画 png（标题标 success + 接触方块阴影 + 抓取竖线）。"""
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

    fig, ax = plt.subplots(figsize=(11, 5))
    _overlay_contact_grasp(ax, recs, x_key="t_sim")   # 接触方块阴影 + 抓取竖线
    ax.plot(t, aggr, "-o", color="tab:blue", label="aggressiveness (0=slow ~ 1=fast)")
    ax.plot(t, lg, "--", color="tab:green", alpha=0.6, label="left gripper (0=closed ~ 1=open)")
    ax.plot(t, rg, "--", color="tab:olive", alpha=0.6, label="right gripper")
    ax.set_xlabel("sim time (s)")
    ax.set_ylabel("aggressiveness / gripper")
    ax.set_ylim(-0.05, 1.05)
    _tag = "" if ep_success is None else (" [SUCCESS]" if ep_success else " [FAILED]")
    ax.set_title(f"episode {ep}{_tag}: speed knob vs grasp/contact")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"episode{ep}_speed.png"), dpi=120)
    plt.close(fig)


def run_backend_episodes(env, vla, backend, learner, exec_backend, *,
                         n_episodes, max_decision_steps, seed, out_dir, record_video,
                         k_skip=None):
    """跑 n_episodes（录视频 + 逐 chunk 记录 + 每 episode _save_and_plot），返回聚合 result dict。

    Args:
      env:     已 reset 的 EnvClientWrapper（连对应 backend 的 env server）。
      vla:     _build_vla 返回的 dict（drift_forward/unnormalize/preprocess/H/A）。
      backend: _build_dqn 返回的 ExecBackend（decode 用）。
      learner: _build_dqn 返回的 greedy DQN。
      exec_backend: backend 名字（传 env.step_chunk + 记录/标题）。

    Returns:
      result dict：exec_backend / n_episodes / success / success_rate /
        mean_dense_steps / mean_sim_time_s / mean_dense_steps_success /
        aggr_in_contact_mean / aggr_free_mean / decel_near_contact /
        n_contact_steps / n_free_steps / episodes(每 episode 的 success/步数/recs)。
      其中 **dense_steps（物理步数, ÷250=仿真秒）是跨 backend 唯一硬可比的执行时间基准**。
    """
    drift_forward, unnormalize, preprocess = vla["drift_forward"], vla["unnormalize"], vla["preprocess"]
    H, A = vla["H"], vla["A"]

    n_succ = 0
    n_reward_succ = 0
    all_contact_aggr, all_free_aggr = [], []   # 跨 episode 汇总：接触 vs 非接触时段激进度
    episodes = []

    env.reseed(seed)
    for ep in range(n_episodes):
        obs = env.reset()
        if record_video:
            env.start_video(ep)
        recs, t_acc, planned_time, ep_success = [], 0.0, 0.0, False
        ep_speed_violation = False
        vla_wall = dqn_wall = env_rpc_wall = obs_rpc_wall = 0.0
        episode_wall_start = time.perf_counter()

        for step_i in tqdm.tqdm(range(max_decision_steps), desc=f"{exec_backend} ep{ep}", disable=False):
            stage_start = time.perf_counter()
            obs_m = preprocess(obs)
            z = jax.random.normal(
                jax.random.PRNGKey(noise_seed(seed, ep, step_i)), (1, H, A)
            )
            mean, _cond, value_feat = drift_forward(obs_m, z)
            feat = np.asarray(value_feat[0])
            real_chunk = unnormalize(mean, obs_m)
            vla_step_wall = time.perf_counter() - stage_start
            vla_wall += vla_step_wall

            stage_start = time.perf_counter()
            idxs = np.asarray(learner.greedy_action_idxs(feat[None])[0], dtype=np.int32)  # greedy 无探索
            speed_params, reward_values = backend.decode(idxs)
            aggr_values = [
                var.aggressiveness(int(idx)) for var, idx in zip(backend.vars, idxs)
            ]
            dqn_step_wall = time.perf_counter() - stage_start
            dqn_wall += dqn_step_wall

            stage_start = time.perf_counter()
            executed, info = env.step_chunk(real_chunk, speed_params, exec_backend)
            done, success, _r, _mask = env.get_info_for_step()
            env_step_wall = time.perf_counter() - stage_start
            env_rpc_wall += env_step_wall
            ep_success = ep_success or bool(success)
            ep_speed_violation |= bool(info.get("fixed_time_speed_violation", False))
            dense_steps = int(info.get("n_exec_steps", 0))
            t_acc += dense_steps / 250.0
            planned_time += float(info.get("duration", 0.0))

            lc, rc = bool(info.get("left_contact", False)), bool(info.get("right_contact", False))
            aggr_mean = float(np.mean(aggr_values)) if aggr_values else 0.0
            rec = dict(
                step=step_i, t_sim=round(t_acc, 4),
                dense_steps=dense_steps,
                aggr_mean=aggr_mean, aggr=[float(x) for x in aggr_values],
                reward_values=[float(x) for x in reward_values],
                left_gripper=float(info.get("left_gripper", 0.0)),
                right_gripper=float(info.get("right_gripper", 0.0)),
                left_contact=lc, right_contact=rc,
                exec_status=info.get("exec_status", "success"), success=bool(success),
                execution_steps=int(info.get("execution_steps", 0)),
                planned_cruise_fraction=float(info.get("planned_cruise_fraction", 0.0)),
                fixed_time_speed_violation=bool(info.get("fixed_time_speed_violation", False)),
                max_planned_qvel=float(info.get("max_planned_qvel", 0.0)),
                vla_wall_s=vla_step_wall, dqn_wall_s=dqn_step_wall,
                env_rpc_wall_s=env_step_wall,
            )
            rec.update({k: float(v) for k, v in speed_params.items()})  # v / vel_limit / acc_limit
            if "acc_limit" in info and "acc_limit" not in rec:
                rec["derived_acc_limit"] = float(info["acc_limit"])
            recs.append(rec)
            (all_contact_aggr if (lc or rc) else all_free_aggr).append(aggr_mean)

            if done:
                break
            stage_start = time.perf_counter()
            obs = env.get_observation()
            obs_step_wall = time.perf_counter() - stage_start
            obs_rpc_wall += obs_step_wall
            rec["observation_rpc_wall_s"] = obs_step_wall

        end_to_end_wall = time.perf_counter() - episode_wall_start
        ep_reward_success = episode_reward_success(
            exec_backend, ep_success, ep_speed_violation
        )

        if record_video:
            env.stop_video()
            _tag = "SUCCESS" if ep_success else "FAIL"
            try:  # 视频名加 success/fail 后缀（同机共享 video_dir；失败仅警告不中断）
                _vp = os.path.join(out_dir, "videos", f"episode{ep}.mp4")
                if os.path.exists(_vp):
                    os.replace(_vp, os.path.join(out_dir, "videos", f"episode{ep}_{_tag}.mp4"))
            except Exception as _e:
                logging.warning("[%s] episode%d video rename 失败(忽略): %s", exec_backend, ep, _e)
        n_succ += int(ep_success)
        n_reward_succ += int(ep_reward_success)
        _save_and_plot(out_dir, ep, recs, ep_success)
        ep_dense = int(sum(int(r["dense_steps"]) for r in recs))
        fallback_count = sum(r["exec_status"] == "topp_fallback" for r in recs)
        cruise_fraction = (
            float(sum(r["planned_cruise_fraction"] * r["dense_steps"] for r in recs) / ep_dense)
            if ep_dense > 0 and exec_backend == "chunk_toppra" else 0.0
        )
        episodes.append(dict(
            ep=ep, seed=int(seed + ep), success=bool(ep_success),
            task_success=bool(ep_success), reward_success=bool(ep_reward_success),
            fixed_time_speed_violation=bool(ep_speed_violation),
            n_decision_steps=len(recs), k_skip=k_skip,
            acc_rule=(WHOLE_CHUNK_ACC_RULE if exec_backend == "chunk_toppra" else None),
            fallback_count=int(fallback_count),
            planned_cruise_fraction=cruise_fraction,
            total_dense_steps=ep_dense, sim_time_s=round(ep_dense / 250.0, 4),
            planned_duration_s=planned_time,
            vla_wall_s=vla_wall, dqn_wall_s=dqn_wall,
            env_rpc_wall_s=env_rpc_wall, observation_rpc_wall_s=obs_rpc_wall,
            end_to_end_wall_s=end_to_end_wall, recs=recs,
        ))
        logging.info("[%s] episode %d: success=%s decision_steps=%d dense_steps=%d sim_time=%.2fs",
                     exec_backend, ep, ep_success, len(recs), ep_dense, t_acc)

    # ---- 汇总：接触 vs 非接触时段的平均激进度 + 加速指标（dense_steps / sim_time）----
    c_mean = float(np.mean(all_contact_aggr)) if all_contact_aggr else float("nan")
    f_mean = float(np.mean(all_free_aggr)) if all_free_aggr else float("nan")
    succ_eps = [e for e in episodes if e["success"]]
    action_counts = {}
    for var in backend.vars:
        counts = {}
        for episode in episodes:
            for rec in episode["recs"]:
                value = str(rec[var.name])
                counts[value] = counts.get(value, 0) + 1
        action_counts[var.name] = counts
    result = dict(
        exec_backend=exec_backend, n_episodes=n_episodes, success=n_succ,
        reward_success=n_reward_succ,
        success_rate=n_succ / max(n_episodes, 1),
        reward_success_rate=n_reward_succ / max(n_episodes, 1),
        fixed_time_speed_violation_rate=(
            float(np.mean([e["fixed_time_speed_violation"] for e in episodes]))
            if episodes else 0.0
        ),
        fallback_rate=(
            float(np.mean([e["fallback_count"] > 0 for e in episodes])) if episodes else 0.0
        ),
        k_skip=k_skip,
        acc_rule=(WHOLE_CHUNK_ACC_RULE if exec_backend == "chunk_toppra" else None),
        speed_action_counts=action_counts,
        mean_dense_steps=float(np.mean([e["total_dense_steps"] for e in episodes])) if episodes else 0.0,
        mean_sim_time_s=float(np.mean([e["sim_time_s"] for e in episodes])) if episodes else 0.0,
        mean_end_to_end_wall_s=(float(np.mean([e["end_to_end_wall_s"] for e in episodes]))
                               if episodes else 0.0),
        mean_vla_wall_s=(float(np.mean([e["vla_wall_s"] for e in episodes])) if episodes else 0.0),
        mean_dqn_wall_s=(float(np.mean([e["dqn_wall_s"] for e in episodes])) if episodes else 0.0),
        mean_env_rpc_wall_s=(float(np.mean([e["env_rpc_wall_s"] for e in episodes]))
                             if episodes else 0.0),
        # 只在成功 episode 上算执行步数（加速对比应在「都成功」的前提下比，避免失败早停拉低步数）
        mean_dense_steps_success=(float(np.mean([e["total_dense_steps"] for e in succ_eps]))
                                  if succ_eps else float("nan")),
        aggr_in_contact_mean=c_mean, aggr_free_mean=f_mean,
        decel_near_contact=bool(c_mean < f_mean) if (all_contact_aggr and all_free_aggr) else None,
        n_contact_steps=len(all_contact_aggr), n_free_steps=len(all_free_aggr),
        episodes=episodes,
    )
    return result


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
            "k_skip": backend_k_skip(config, exec_backend),
            "stream_hold_steps": int(config.get("stream_hold_steps", 15)),
            "eval_video_save_freq": 25,
            "force_limit": parse_force_limit(config.get("force_limit", "")),
        },
        host=FLAGS.client_host, port=FLAGS.client_port,
    )
    env.reset()

    vla = _build_vla(config, config_task, FLAGS.seed, mesh,
                     (data_sharding, replicated_sharding), env, FLAGS.resume)
    dqnpack = _build_dqn(config, FLAGS.seed, exec_backend, FLAGS.dqn_ckpt,
                         vla["feat_dim"], vla["vfeat0"])

    result = run_backend_episodes(
        env, vla, dqnpack["backend"], dqnpack["learner"], exec_backend,
        n_episodes=FLAGS.n_episodes, max_decision_steps=FLAGS.max_decision_steps,
        seed=FLAGS.seed, out_dir=out_dir, record_video=FLAGS.record_video,
        k_skip=backend_k_skip(config, exec_backend),
    )

    # ---- 汇总写盘（保持原 eval_summary.json 字段，新增加速指标 mean_dense_steps）----
    summary = {
        "n_episodes": result["n_episodes"], "success": result["success"],
        "success_rate": result["success_rate"],
        "reward_success": result["reward_success"],
        "reward_success_rate": result["reward_success_rate"],
        "mean_dense_steps": result["mean_dense_steps"],
        "mean_dense_steps_success": result["mean_dense_steps_success"],
        "mean_sim_time_s": result["mean_sim_time_s"],
        "mean_end_to_end_wall_s": result["mean_end_to_end_wall_s"],
        "mean_vla_wall_s": result["mean_vla_wall_s"],
        "mean_dqn_wall_s": result["mean_dqn_wall_s"],
        "mean_env_rpc_wall_s": result["mean_env_rpc_wall_s"],
        "fixed_time_speed_violation_rate": result["fixed_time_speed_violation_rate"],
        "fallback_rate": result["fallback_rate"],
        "k_skip": result["k_skip"], "acc_rule": result["acc_rule"],
        "speed_action_counts": result["speed_action_counts"],
        "aggr_in_contact_mean": result["aggr_in_contact_mean"],
        "aggr_free_mean": result["aggr_free_mean"],
        "decel_near_contact": result["decel_near_contact"],
        "n_contact_steps": result["n_contact_steps"], "n_free_steps": result["n_free_steps"],
    }
    with open(os.path.join(out_dir, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logging.info("==== SpeedTune eval 汇总 (%s) ====", exec_backend)
    logging.info("success: %d/%d (%.0f%%)", result["success"], result["n_episodes"],
                 100 * result["success_rate"])
    logging.info("平均执行步数 dense_steps=%.0f (成功 episode=%.0f) | 平均仿真时间=%.2fs",
                 result["mean_dense_steps"], result["mean_dense_steps_success"], result["mean_sim_time_s"])
    logging.info("接触时段平均激进度 = %.3f | 非接触时段 = %.3f → 接触附近%s",
                 result["aggr_in_contact_mean"], result["aggr_free_mean"],
                 "减速 ✓" if result["decel_near_contact"] else
                 ("未减速" if result["decel_near_contact"] is False else "(数据不足)"))
    logging.info("输出目录: %s（videos/ + episode*_speed.{json,png} + eval_summary.json）", out_dir)


if __name__ == "__main__":
    app.run(main)
