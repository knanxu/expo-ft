#! /usr/bin/env python
"""SpeedTune 加速模块训练入口（branching Rainbow-DQN，冻结 VLA 之上，两阶段解耦）。

与 `train_pi_robo_async.py` 同构的 async 双线程：
  - **actor 主线程**：每个决策步（= 一整段 chunk）跑 ① 冻结 VLA 前向得 (action_chunk, suffix_feat)；
    ② DQN 读 detached suffix_feat 选速度档（epsilon-greedy）；③ decode 成 speed_params；
    ④ `env.step_chunk(real_chunk, speed_params, exec_backend)` 整段执行；⑤ 攒 episode transitions。
    episode 结束知道 success 后**回填 reward**（success-gated：成功 episode 的所有加速决策才获奖励，
    失败全 0 → 最严格的防 reward hacking），再写入共享 replay buffer。
  - **learner 后台线程**：buffer 攒够后采样 → DQN.update（double+dueling+C51+n-step+PER）→ 发布 q 参数。

**零侵入**：VLA 全程冻结（只前向、不进优化器）；不改 EXPO/BC 任何代码路径。env 经
`client_robotwin/` 的新增 `step_chunk` 协议（per-action `step` 不受影响）。

复用：`build_pi05`（加载冻结 drift VLA）、`make_drift_apply_fn`（一次前向得 mean+value_feat）、
`process_transformed_outputs`（delta 动作反归一化，state=归一化当前 state，避免 rollout 0% 坑）。
"""

import os
import logging
import threading
import time

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

import jax
import jax.numpy as jnp
import etils.epath as epath
import wandb

import openpi.training.sharding as openpi_sharding
from expo_ft.env.env_client import EnvClientWrapper
from expo_ft.utils.train_utils import init_logging, init_wandb

from expo_ft.agents.alg.speedtune_dqn import SpeedTuneLearner
from expo_ft.data.speedtune_buffer import SpeedTuneReplayBuffer
from expo_ft.speedtune.exec_backends import build_backend, parse_force_limit
from expo_ft.speedtune.runtime_config import (
    backend_k_skip,
    backend_support,
    episode_reward_success,
)
from expo_ft.speedtune.checkpoint_contract import (
    build_checkpoint_metadata,
    write_checkpoint_metadata,
)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "expo-ft-speedtune", "wandb project name.")
flags.DEFINE_string("run_name", None, "wandb run name.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm.")
flags.DEFINE_string("output_dir", "./logs", "Logs/checkpoints dir.")
flags.DEFINE_integer("fsdp_devices", 1, "FSDP devices for sharding.")
flags.DEFINE_string("client_host", "localhost", "Env server host.")
flags.DEFINE_integer("client_port", 8102, "Env server port.")
flags.DEFINE_boolean("checkpoint_model", False, "Save DQN checkpoints.")
flags.DEFINE_integer("checkpoint_interval", 0, "Save every N updates (0=end only).")
flags.DEFINE_boolean("resume", False, "Resume frozen-VLA load from checkpoint.")
flags.DEFINE_string("dataset_path", "", "Offline demo dataset (for example_action / env creation).")

config_flags.DEFINE_config_file(
    "config", "configs/model/speedtune_dqn_config.py", "SpeedTune DQN config.", lock_config=False)
config_flags.DEFINE_config_file(
    "config_task", "configs/task/robotwin_stack_blocks.py", "Task config.", lock_config=False)


def _setup_speedtune(config, config_task, seed, mesh, shardings, env):
    """加载冻结 VLA → 组装 drift 前向 / 反归一化 / DQN / exec_backend / buffer。

    Returns:
      dict(actor, drift_forward, unnormalize, preprocess, dqn, buffer, backend, feat_dim)。
    """
    from expo_ft.agents.vla.pi05 import build_pi05
    from expo_ft.agents.vla.drift_adapters import make_drift_apply_fn

    data_sharding, replicated_sharding = shardings
    n_real_dims = int(config.n_real_dims)
    resize_size = int(config.pi05_resize_size)

    # 冻结 pi0.5 drift VLA（use_drifting_loss=True）+ ckpt（config.pi05_*）。VLA 不训。
    actor, actor_train_state, _target, _agent_kwargs, _metadata = build_pi05(
        config, seed, mesh, data_sharding, replicated_sharding,
        FLAGS.resume, config_task.language_instruction,
    )
    H = int(actor.model_config.action_horizon)
    A = int(actor.model_config.action_dim)            # padded
    actor.action_dim = n_real_dims                    # process_raw_inputs/_unpad/_pad 用
    actor.state_dim = n_real_dims

    def preprocess(raw_obs):
        raw = dict(raw_obs)
        raw.setdefault("action", np.zeros((H, n_real_dims), dtype=np.float32))
        return actor.process_raw_inputs(raw, action_dim=n_real_dims, resize_size=resize_size)

    # 冻结 VLA 前向：一次得 (mean[1,H,A], cond_emb, value_feat[1,width])；frozen=None → 全参前向。
    drift_apply_fn = make_drift_apply_fn(actor_train_state.model_def)
    vla_params = actor_train_state.params

    @jax.jit
    def _forward(params, obs, z):
        return drift_apply_fn(params, obs, z, None)

    def drift_forward(obs_m, z):
        return _forward(vla_params, obs_m, z)  # (mean, cond_emb, value_feat)

    def unnormalize(mean, obs_m):
        """model-space mean[1,H,A] → 真实 qpos chunk[H,n_real_dims]（delta 反归一化）。"""
        unpad = actor._unpad_actions(mean)            # [1,H,n_real]
        real = actor.process_transformed_outputs(np.asarray(unpad), state=np.asarray(obs_m["state"]))
        return np.asarray(real[0])

    # 探针一次前向 → feat 维度 + exec_backend。
    example_obs = preprocess(env.get_observation())
    example_z = jax.random.normal(jax.random.PRNGKey(seed + 7), (1, H, A))
    _mean0, _cond0, vfeat0 = drift_forward(example_obs, example_z)
    feat_dim = int(np.asarray(vfeat0).shape[-1])
    logging.info("SpeedTune: VLA suffix feat_dim=%d, H=%d, A=%d", feat_dim, H, A)

    # exec_backend（含 reward α/β 统一覆盖钮）。
    backend_name = str(config.exec_backend)
    overrides = None
    ra, rb = config.get("reward_alpha", None), config.get("reward_beta", None)
    if backend_name in ("fixed_time", "chunk_toppra") and (ra is not None or rb is not None):
        base = build_backend(backend_name)
        overrides = {}
        for v in base.vars:
            o = {}
            if ra is not None:
                o["alpha"] = float(ra)
            if rb is not None:
                o["beta"] = float(rb)
            overrides[v.name] = o
    backend = build_backend(backend_name, overrides)
    logging.info("SpeedTune exec_backend=%s head_sizes=%s", backend.name, backend.head_sizes)
    support_min, support_max = backend_support(config, backend.name)

    dqn = SpeedTuneLearner.create(
        rng=jax.random.PRNGKey(seed), feat_dim=feat_dim, head_sizes=backend.head_sizes,
        n_atoms=int(config.n_atoms), v_min=support_min, v_max=support_max,
        hidden_dims=tuple(config.q_hidden_dims), dueling=bool(config.dueling),
        detach_input=bool(config.detach_q_input), lr=float(config.q_lr),
        max_grad_norm=float(config.max_grad_norm), tau=float(config.tau),
        epsilon=float(config.epsilon_start),
        example_feat=jnp.asarray(vfeat0),
    )
    buffer = SpeedTuneReplayBuffer(
        capacity=int(config.buffer_capacity), feat_dim=feat_dim, n_heads=backend.n_heads,
        n_step=int(config.n_step), gamma=float(config.gamma),
        per_alpha=float(config.per_alpha), per_beta=float(config.per_beta),
        per_eps=float(config.per_eps), seed=seed,
    )
    return dict(actor=actor, drift_forward=drift_forward, unnormalize=unnormalize,
                preprocess=preprocess, dqn=dqn, buffer=buffer, backend=backend,
                feat_dim=feat_dim, H=H, A=A)


def _epsilon_at(step, config):
    s, e, d = float(config.epsilon_start), float(config.epsilon_end), float(config.epsilon_decay_steps)
    if step >= d:
        return e
    return s + (e - s) * (step / max(d, 1.0))


def _beta_at(step, config):
    s, e, d = float(config.per_beta), float(config.per_beta_end), float(config.epsilon_decay_steps)
    if step >= d:
        return e
    return s + (e - s) * (step / max(d, 1.0))


def main(_):
    init_logging()
    config, config_task = FLAGS.config, FLAGS.config_task
    assert config.model_cls == "SpeedTuneLearner", f"expected SpeedTuneLearner, got {config.model_cls}"

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
    jax.config.update("jax_default_matmul_precision", "highest")

    mesh = openpi_sharding.make_mesh(FLAGS.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    log_dir = os.path.join(FLAGS.output_dir, FLAGS.run_name or "speedtune_run")
    os.makedirs(log_dir, exist_ok=True)
    init_wandb(epath.Path(log_dir), FLAGS.resume, FLAGS.project_name, FLAGS.run_name)
    wandb.config.update(FLAGS.flag_values_dict(), allow_val_change=FLAGS.resume)
    # wandb 横轴：所有 log 统一用决策步 step=it 做底层基准（同 step 多次 log 合并→不爆点）；
    # rollout/* 改用 episode 轴（define_metric），training/action/exec 用决策步轴。
    wandb.define_metric("episode")
    wandb.define_metric("rollout/*", step_metric="episode")

    # SpeedTune 是在线 RL，无需离线数据集；example_action 仅作 env_creation_request 占位
    # （RoboTwin create_env 不消费它，动作空间由 n_real_dims 决定）。
    example_action = np.asarray(
        config_task.get("example_action", np.zeros((1, int(config.n_real_dims)), dtype=np.float32)),
        dtype=np.float32)

    logging.info("Creating RoboTwin training env via env_client ...")
    env = EnvClientWrapper(
        env_creation_request={"example_action": example_action, "env_usage": "train",
                              "video_dir": os.path.join(log_dir, "train_videos"),
                              # SpeedTune：env 创建即设执行后端（RoboTwin take_chunk_action_backend 用）。
                              "exec_backend": str(config.exec_backend),
                              "k_skip": backend_k_skip(config, str(config.exec_backend)),
                              "stream_hold_steps": int(config.get("stream_hold_steps", 15)),
                              "force_limit": parse_force_limit(config.get("force_limit", ""))},
        host=FLAGS.client_host, port=FLAGS.client_port,
    )
    env.reset()

    setup = _setup_speedtune(config, config_task, FLAGS.seed, mesh,
                             (data_sharding, replicated_sharding), env)
    drift_forward = setup["drift_forward"]
    unnormalize = setup["unnormalize"]
    preprocess = setup["preprocess"]
    backend = setup["backend"]
    buffer = setup["buffer"]
    learner = setup["dqn"]
    H, A = setup["H"], setup["A"]
    exec_backend = str(config.exec_backend)
    checkpoint_metadata = build_checkpoint_metadata(config, backend)

    batch_size = int(config.batch_size)
    learning_starts = int(config.learning_starts)
    utd_ratio = int(config.utd_ratio)                    # P: 每次 update 调用的梯度步数
    update_per_episode = int(config.update_per_episode)  # 每 episode 的 update 调用次数
    max_iters = int(config.max_iters)
    log_every = 50   # wandb 训练/动作指标按决策步节流（统一 step=it，合并去重，防"一下子十万步"）

    # --------------------- async 双线程 --------------------- #
    _buf_lock = threading.Lock()
    _published = [None]
    _publish_lock = threading.Lock()
    _stop = threading.Event()
    _step = [0]          # 全局决策步计数（epsilon/beta schedule + 统一 wandb step）
    _n_updates = [0]
    _episode = [0]                       # 已完成 episode 数（rollout 横轴）
    _train_metrics = [None]              # update 线程暂存最新聚合 metrics，主线程统一 log(step=it)
    _can_update = threading.Event()      # buffer ready 才开 update（前期不空转）
    _new_episode = threading.Event()     # 每 episode 末放行一批 update（per-episode 控速）

    def _publish(lr):
        with _publish_lock:
            _published[0] = lr.q_net.params

    def _pickup(actor_lr):
        with _publish_lock:
            pub, _published[0] = _published[0], None
        if pub is None:
            return actor_lr
        return actor_lr.replace(q_net=actor_lr.q_net.replace(params=pub))

    def _update_worker():
        """P 方式 + per-episode 控速：每 episode 末做 update_per_episode 次 update 调用，每次
        utd_ratio 个梯度步（各自 PER 采样 + 回写优先级）→ 每 episode 共 update_per_episode×utd_ratio
        个梯度步。metrics 聚合后交主线程统一 log(step=it)，不在本线程 log（避免多线程 step 冲突）。"""
        learner_lr = learner
        _can_update.wait()                                  # buffer 未 ready 前阻塞，不空转
        while not _stop.is_set():
            if not _new_episode.wait(timeout=1.0):          # 等一个完整 episode 的数据（控速 + 防 stale）
                continue
            _new_episode.clear()
            beta = _beta_at(_step[0], config)
            agg = []
            for _ in range(update_per_episode):             # 每 episode update_per_episode 次 update 调用
                for _ in range(utd_ratio):                  # P: 每次调用 utd_ratio 个梯度步
                    with _buf_lock:
                        batch = buffer.sample(batch_size, beta=beta)
                    learner_lr, td, metrics = learner_lr.update(batch)
                    with _buf_lock:
                        buffer.update_priorities(batch["tree_indices"], np.asarray(td))
                    _n_updates[0] += 1
                    agg.append(metrics)
                _publish(learner_lr)                        # 每次调用后发布参数给 actor
            # 聚合本 episode 全部梯度步 metrics（均值），交主线程统一 log(step=it)。
            keys = [k for k, v in agg[0].items()
                    if np.isscalar(v) or getattr(v, "ndim", 1) == 0]
            tm = {f"training/{k}": float(np.mean([float(m[k]) for m in agg])) for k in keys}
            tm["training/per_beta"] = beta
            tm["training/n_updates"] = _n_updates[0]
            tm["training/replay_ratio"] = _n_updates[0] / max(_step[0], 1)
            _train_metrics[0] = tm
            if FLAGS.checkpoint_model and FLAGS.checkpoint_interval > 0 \
                    and _n_updates[0] % FLAGS.checkpoint_interval == 0:
                _save_checkpoint(log_dir, learner_lr, _n_updates[0], checkpoint_metadata)
        _save_checkpoint(log_dir, learner_lr, _n_updates[0], checkpoint_metadata)
        logging.info("Update thread exiting after %d updates.", _n_updates[0])

    _thread = threading.Thread(target=_update_worker, daemon=True)
    _thread.start()

    actor_lr = learner
    rng = np.random.default_rng(FLAGS.seed)
    obs = env.get_observation()
    pending = []          # 当前 episode 的决策步（episode 结束统一回填 reward 再入 buffer）
    ep_success = False
    ep_speed_violation = False
    ep_returns, ep_reward_returns, ep_lens, ep_exec_time, ep_dense = [], [], [], [], []
    _fallbacks = [0, 0]   # [topp_fallback 次数, 总 chunk 次数] → fallback 率

    def _flush_episode():
        """episode 结束：success-gated 回填 reward，按时序写入 buffer + 统计执行时间。"""
        nonlocal pending, ep_success, ep_speed_violation
        if not pending:
            return
        reward_success = episode_reward_success(
            backend.name, ep_success, ep_speed_violation
        )
        with _buf_lock:
            for t, p in enumerate(pending):
                next_feat = pending[t + 1]["feat"] if t + 1 < len(pending) else p["feat"]
                r = backend.total_reward(p["r_task"], reward_success, p["v_list"])
                buffer.insert(p["feat"], p["action_idxs"], r, next_feat, p["done"])
        ep_returns.append(bool(ep_success))
        ep_reward_returns.append(bool(reward_success))
        ep_lens.append(len(pending))
        ep_exec_time.append(float(sum(p["duration"] for p in pending)))   # episode 总 TOPPRA 执行时长(s)
        ep_dense.append(int(sum(p["n_exec"] for p in pending)))           # episode 总仿真帧数
        pending = []
        ep_success = False
        ep_speed_violation = False

    try:
        for it in tqdm.tqdm(range(max_iters), disable=not FLAGS.tqdm):
            _step[0] = it
            actor_lr = _pickup(actor_lr).replace(epsilon=_epsilon_at(it, config))

            obs_m = preprocess(obs)
            z = jax.random.normal(jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1))), (1, H, A))
            mean, _cond, value_feat = drift_forward(obs_m, z)
            feat = np.asarray(value_feat[0])                         # DQN state
            real_chunk = unnormalize(mean, obs_m)                    # [H, n_real]

            idxs, actor_lr = actor_lr.sample_action_idxs(feat[None])
            idxs = np.asarray(idxs[0], dtype=np.int32)
            speed_params, v_list = backend.decode(idxs)

            _executed, exec_info = env.step_chunk(real_chunk, speed_params, exec_backend)
            done, success, r_task, _mask = env.get_info_for_step()
            ep_success = ep_success or bool(success)
            ep_speed_violation = ep_speed_violation or bool(
                exec_info.get("fixed_time_speed_violation", False)
            )
            _fallbacks[1] += 1
            if exec_info.get("exec_status") == "topp_fallback":
                _fallbacks[0] += 1
            pending.append(dict(feat=feat, action_idxs=idxs, v_list=v_list,
                                r_task=float(r_task), done=bool(done),
                                duration=float(exec_info.get("duration", 0.0)),
                                n_exec=int(exec_info.get("n_exec_steps", 0))))

            # 每 log_every 决策步统一 log（step=it）：action/exec + 最新 training metrics 合并到同一 step。
            if it % log_every == 0:
                sp_log = {f"action/{k}": float(val) for k, val in speed_params.items()}
                sp_log["exec/dense_steps"] = int(exec_info.get("n_exec_steps", 0))
                sp_log["exec/duration_s"] = float(exec_info.get("duration", 0.0))
                sp_log["exec/planned_cruise_fraction"] = float(
                    exec_info.get("planned_cruise_fraction", 0.0)
                )
                sp_log["exec/max_planned_qvel"] = float(exec_info.get("max_planned_qvel", 0.0))
                sp_log["exec/fixed_time_speed_violation"] = float(
                    bool(exec_info.get("fixed_time_speed_violation", False))
                )
                if "acc_limit" in exec_info:
                    sp_log["action/derived_acc_limit"] = float(exec_info["acc_limit"])
                if _train_metrics[0] is not None:
                    sp_log.update(_train_metrics[0])   # 训练 metrics 合并到决策步轴
                wandb.log(sp_log, step=it)

            obs = env.reset() if done else env.get_observation()
            if done:
                _flush_episode()
                _episode[0] += 1
                # buffer ready 后：开 update 线程 + 放行本 episode 的 update_per_episode×utd_ratio 个梯度步。
                if buffer.ready(learning_starts):
                    if not _can_update.is_set():
                        _can_update.set()
                    _new_episode.set()
                if ep_returns:
                    fb_rate = _fallbacks[0] / max(_fallbacks[1], 1)
                    wandb.log({"rollout/success_rate": float(np.mean(ep_returns[-50:])),
                               "rollout/reward_success_rate": float(np.mean(ep_reward_returns[-50:])),
                               "rollout/ep_len_mean": float(np.mean(ep_lens[-50:])),
                               "rollout/exec_time_s": float(np.mean(ep_exec_time[-50:])),  # 对比核心：成功 vs 执行时间
                               "rollout/dense_steps": float(np.mean(ep_dense[-50:])),
                               "rollout/fallback_rate": float(fb_rate),
                               "rollout/epsilon": _epsilon_at(it, config),
                               "rollout/buffer_size": len(buffer),
                               "episode": _episode[0]},      # rollout 横轴 = episode
                              step=it)
    finally:
        _flush_episode()
        _stop.set()
        _can_update.set()     # 解除 update 线程 _can_update.wait() 阻塞（buffer 从未 ready 的边界）
        _new_episode.set()    # 解除 _new_episode.wait() 阻塞，让线程跑完 final checkpoint
        _thread.join(timeout=120)
        logging.info("SpeedTune async training finished after %d decision steps.", _step[0])


def _save_checkpoint(log_dir, learner, step, metadata):
    try:
        import orbax.checkpoint as ocp
        ckpt_dir = epath.Path(log_dir).resolve() / "checkpoints" / f"update_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with ocp.StandardCheckpointer() as ckptr:
            # force=True：final 与最后一次 interval 可能撞同一 update_N，覆盖而非报 "already exists"。
            ckptr.save(ckpt_dir / "q_net", learner.q_net.params, force=True)
            ckptr.save(ckpt_dir / "target", learner.target_params, force=True)
        write_checkpoint_metadata(ckpt_dir, metadata)
        logging.info("Saved SpeedTune DQN checkpoint at update %d", step)
    except Exception as e:
        logging.error("Checkpoint save failed: %s", e)


if __name__ == "__main__":
    app.run(main)
