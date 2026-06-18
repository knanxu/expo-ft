#! /usr/bin/env python
"""DBPO×BPO on-policy RL 训练入口（方案 A：最大保留 EXPO-FT async 双线程设计）。

与 `train_pi_robo_async.py` 同构：**actor 主线程**做 rollout（推理 + 驱动 env + 存 buffer），
**learner 后台线程**做参数更新（GAE + PPO/BPO + 发布），经 `_publish_lock`/`_published[0]` 原子发布。
**不改** EXPO/BC 路径——本文件是独立新增入口（CLAUDE.md 零侵入）。

与 EXPO-FT off-policy async 的差异（DBPO 是 on-policy，见 docs/ROBOTWIN_ADAPTATION_PLAN.md §4.5.2）：
  - `PiReplayBuffer`（环形复用）→ `RolloutBuffer`（每轮采满 rollout_size 决策步→GAE→K epoch→**用完即弃**）。
  - learner 线程：等**整轮** rollout → `run_ppo_iteration`（K=ppo_epochs）→ 发布 → 等下一轮。
  - actor：**一轮内冻结 π_old**（只在轮间拾取发布的新参数），每条 transition 存采集时 logp_old（z 复用 → ratio 精确）。
  - **1 轮滞后重叠**：in-flight rollout ≤ 1（actor 用第 N 轮参数采 N+1 轮，learner 训第 N 轮）→ 近 on-policy。

env 走分离① 的 `client_robotwin/`（RoboTwin，经 `env_client` per-action step）；learner 端 env_client 不变。
每个**决策步** = 1 次 chunk 推理 + per-action 执行前 H_e(=replan_steps) 步并**聚合 reward**（DBPO `venv.step(prefix)`
在 per-action 接口下的等价）。

两处与真实 pi0.5/checkpoint 对接的接缝（标 SEAM，待真实 DBP ckpt + RoboTwin 落地核对）：
  SEAM-1 `_build_dbpo_learner`：从 pi0.5 drift DBP checkpoint 装出 `DBPOLearner`。
  SEAM-2 `_preprocess_obs`：raw env obs → pi0.5 `Observation`（openpi RoboTwin input transform + tokenize）。
"""

import os
import logging
import threading
import time
from collections import deque

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

import jax
import etils.epath as epath
import wandb

import openpi.training.sharding as openpi_sharding
from expo_ft.env.env_client import EnvClientWrapper
from expo_ft.env.droid_utils import process_droid_dataset
from expo_ft.utils.train_utils import init_logging, init_wandb

from expo_ft.data.rollout_buffer import RolloutBuffer
from expo_ft.agents.alg.dbpo import run_ppo_iteration

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "expo-ft-dbpo", "wandb project name.")
flags.DEFINE_string("run_name", None, "wandb run name.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("max_iters", 1500, "PPO iterations (DBPO n_train_itr).")
flags.DEFINE_boolean("tqdm", True, "Use tqdm.")
flags.DEFINE_string("output_dir", "./logs", "Logs/checkpoints dir.")
flags.DEFINE_integer("fsdp_devices", 1, "FSDP devices for sharding.")
flags.DEFINE_string("client_host", "localhost", "Env server host.")
flags.DEFINE_integer("client_port", 8102, "Env server port.")
flags.DEFINE_boolean("checkpoint_model", False, "Save checkpoints.")
flags.DEFINE_integer("checkpoint_interval", 0, "Save every N iterations (0=end only).")
flags.DEFINE_boolean("overwrite", False, "Overwrite checkpoint dir.")
flags.DEFINE_boolean("resume", False, "Resume from checkpoint.")
flags.DEFINE_string("dataset_path", "", "Offline demo dataset (for example_action / env creation).")

config_flags.DEFINE_config_file(
    "config", "configs/model/dbpo_pi_config.py", "Model/algorithm config.", lock_config=False)
config_flags.DEFINE_config_file(
    "config_task", "configs/task/robotwin_stack_blocks.py", "Task config.", lock_config=False)


# --------------------------------------------------------------------------- #
# SEAM-1 + SEAM-2: build DBPOLearner from a real pi0.5 drift DBP checkpoint,
# and the raw-obs -> pi0.5 Observation preprocessor.  Both wired against pi05.py:
#   SEAM-1: TrainState 持 (model_def=nnx.graphdef, params=nnx.State) → 直接喂 build_dbpo_from_pi05。
#   SEAM-2: Pi05Agent.process_raw_inputs（input_transforms + Observation.from_dict）。
# --------------------------------------------------------------------------- #
def _setup_dbpo(config, config_task, seed, mesh, shardings, env):
    """`build_pi05` 一次 → 接 SEAM-1/2 → 组装 DBPOLearner。返回 (learner, preprocess)。"""
    from expo_ft.agents.vla.pi05 import build_pi05
    from expo_ft.agents.alg.dbpo_pi05 import build_dbpo_from_pi05

    data_sharding, replicated_sharding = shardings
    n_real_dims = int(config.n_real_dims)
    resize_size = int(config.pi05_resize_size)

    # 加载 use_drifting_loss=True 的 pi0.5 + DBP(Stage-1) checkpoint（config.pi05_*）。
    actor, actor_train_state, _target, _agent_kwargs, metadata = build_pi05(
        config, seed, mesh, data_sharding, replicated_sharding,
        FLAGS.resume, config_task.language_instruction,
    )
    # drift 在 model 空间（padded）出动作；env 侧 RoboTwinEnv 取前 n_real_dims 下发。
    H = int(actor.model_config.action_horizon)        # 50
    A = int(actor.model_config.action_dim)            # padded（如 32）
    actor.action_dim = n_real_dims                    # process_raw_inputs / _pad / _unpad 用
    actor.state_dim = n_real_dims

    # SEAM-2：raw env obs（observation.images.* / observation.state / prompt）→ batched Observation dict。
    # 注：robotwin RepackTransform 结构含 "actions":"action"（读扁平键 "action"），而 EXPO-FT 的
    # process_raw_inputs 只补 "actions" → 这里补 dummy "action"（shape=(H, n_real_dims)）供 repack 读取。
    def preprocess(raw_obs):
        raw = dict(raw_obs)
        raw.setdefault("action", np.zeros((H, n_real_dims), dtype=np.float32))
        return actor.process_raw_inputs(raw, action_dim=n_real_dims, resize_size=resize_size)

    example_obs = preprocess(env.get_observation())
    example_z = jax.random.normal(jax.random.PRNGKey(seed + 7), (1, H, A))

    # SEAM-1：TrainState.{model_def, params} = nnx.split 的 (graphdef, State)，build_dbpo_from_pi05 直接消费。
    learner = build_dbpo_from_pi05(
        rng=jax.random.PRNGKey(seed),
        model_def=actor_train_state.model_def, actor_params=actor_train_state.params,
        example_obs=example_obs, example_z=example_z,
        action_horizon=H, action_dim=A, replan_steps=int(config.replan_steps),
        trainable_filter=_make_trainable_filter(config.actor_trainable_regex),
        actor_lr=float(config.actor_lr), logstd_lr=float(config.logstd_lr), value_lr=float(config.value_lr),
        max_grad_norm=float(config.max_grad_norm) if config.get("max_grad_norm", None) else None,
        logstd_kwargs=dict(hidden_dims=tuple(config.logstd_hidden_dims),
                           log_std_init=float(config.logstd_init),
                           log_std_min=float(config.logstd_min),
                           log_std_max=float(config.logstd_max)),
        clip_eps=float(config.clip_eps), gamma=float(config.discount),
        gae_lambda=float(config.gae_lambda), c_value=float(config.c_value),
        c_entropy=float(config.c_entropy), lambda_anchor=float(config.lambda_anchor),
        rl_surrogate=str(config.rl_surrogate), bpo_lambda=float(config.bpo_lambda),
        n_real_dims=n_real_dims, normalize_adv=bool(config.normalize_adv),
        value_clip_eps=(None if config.get("value_clip_eps", None) is None
                        else float(config.value_clip_eps)),
        normalize_dims=bool(config.normalize_dims),
    )
    return learner, preprocess


def _make_trainable_filter(regex: str):
    """nnx Filter：path 命中 regex 的叶子可训，其余冻结（与 build_dbpo_from_pi05 的 trainable_filter 约定一致）。"""
    import re
    import flax.nnx as nnx
    pattern = re.compile(regex)

    def _pred(path, value):
        key = ".".join(str(p.key if hasattr(p, "key") else p) for p in path)
        return bool(pattern.search(key))

    return nnx.All(nnx.Param, _pred)


def main(_):
    init_logging()
    config, config_task = FLAGS.config, FLAGS.config_task
    assert config.model_cls == "DBPOLearner", f"expected DBPOLearner, got {config.model_cls}"

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
    # ratio 保真：fp32 matmul 用 highest（Phase-2 发现 TF32 致 ratio 漂 ~0.4%；归一化后已缓和，仍设以稳）。
    jax.config.update("jax_default_matmul_precision", "highest")

    mesh = openpi_sharding.make_mesh(FLAGS.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    log_dir = os.path.join(FLAGS.output_dir, FLAGS.run_name or "dbpo_run")
    os.makedirs(log_dir, exist_ok=True)
    init_wandb(epath.Path(log_dir), FLAGS.resume, FLAGS.project_name, FLAGS.run_name)
    wandb.config.update(FLAGS.flag_values_dict(), allow_val_change=FLAGS.resume)

    # --- dataset (for example_action + env creation), 同 train_pi_robo ---
    dataset = process_droid_dataset(FLAGS.dataset_path, config_task, num_data=0)
    example_action = dataset[0]["actions"][np.newaxis]  # (1, H, A)

    logging.info("Creating RoboTwin training env via env_client ...")
    env = EnvClientWrapper(
        env_creation_request={"example_action": example_action, "env_usage": "train",
                              "video_dir": os.path.join(log_dir, "train_videos")},
        host=FLAGS.client_host, port=FLAGS.client_port,
    )
    env.reset()
    logging.info("Env %s created.", env.env_id)

    # --- build DBPO learner + obs preprocessor (SEAM-1 + SEAM-2) ---
    learner, preprocess = _setup_dbpo(
        config, config_task, FLAGS.seed, mesh, (data_sharding, replicated_sharding), env)

    H_e = int(config.replan_steps)
    rollout_size = int(config.rollout_size)
    ppo_epochs = int(config.ppo_epochs)
    num_minibatches = int(config.num_minibatches)
    gamma, gae_lambda = float(config.discount), float(config.gae_lambda)
    target_kl = float(config.target_kl) if config.get("target_kl", None) is not None else None
    n_warmup = int(config.get("n_critic_warmup_iters", 0))   # 前 N 轮只更 critic

    # ===================== 方案 A：async 双线程 ===================== #
    # actor(主线程)采 rollout；learner(后台)更新+发布。1 轮滞后：in-flight rollout ≤ 1。
    _published = [None]                # learner→actor 发布的参数（{actor,logstd,value} params）
    _publish_lock = threading.Lock()
    _rollout_holder = [None]           # actor→learner 交付的已 finalize 的 RolloutBuffer + bootstrap
    _rollout_ready = threading.Event() # set: 有新 rollout 待消费
    _rollout_consumed = threading.Event()  # set: learner 已取走（背压：限 in-flight ≤ 1）
    _rollout_consumed.set()
    _stop = threading.Event()
    _iter = [0]

    def _publish(lr):
        with _publish_lock:
            _published[0] = {"actor": lr.actor.params,
                             "logstd": lr.logstd_head.params,
                             "value": lr.value_head.params}

    def _pickup(actor_lr):
        with _publish_lock:
            pub, _published[0] = _published[0], None
        if pub is None:
            return actor_lr
        return actor_lr.replace(
            actor=actor_lr.actor.replace(params=pub["actor"]),
            logstd_head=actor_lr.logstd_head.replace(params=pub["logstd"]),
            value_head=actor_lr.value_head.replace(params=pub["value"]),
        )

    def _update_worker():
        learner_lr = learner
        seed_ctr = FLAGS.seed + 1
        n_done = 0                                # learner 侧更新计数（critic warmup 用）
        _rollout_ready.wait()
        while not _stop.is_set():
            if not _rollout_ready.wait(timeout=1.0):
                continue
            _rollout_ready.clear()
            buf, _ = _rollout_holder[0]
            _rollout_holder[0] = None
            _rollout_consumed.set()              # 解除 actor 背压，可采下一轮（1 轮重叠）
            t0 = time.time()
            learner_lr, metrics = run_ppo_iteration(
                learner_lr, buf, ppo_epochs=ppo_epochs, num_minibatches=num_minibatches,
                seed=seed_ctr, target_kl=target_kl, critic_warmup=(n_done < n_warmup))
            seed_ctr += 1
            n_done += 1
            _publish(learner_lr)
            log_d = {f"training/{k}": float(v) for k, v in metrics.items()
                     if np.isscalar(v) or (hasattr(v, "ndim") and v.ndim == 0)}
            log_d["training/update_time_s"] = time.time() - t0
            wandb.log(log_d, step=_iter[0])
            if FLAGS.checkpoint_model and FLAGS.checkpoint_interval > 0 \
                    and _iter[0] % FLAGS.checkpoint_interval == 0:
                _save_checkpoint(log_dir, learner_lr, _iter[0])
        _save_checkpoint(log_dir, learner_lr, _iter[0])  # final
        logging.info("Update thread exiting at iter %d.", _iter[0])

    _thread = threading.Thread(target=_update_worker, daemon=True)
    _thread.start()

    actor_lr = learner
    obs = env.get_observation()
    try:
        for it in tqdm.tqdm(range(FLAGS.max_iters), disable=not FLAGS.tqdm):
            _iter[0] = it
            _rollout_consumed.wait()             # 背压：上一轮已被 learner 取走才采新轮（in-flight ≤ 1）
            _rollout_consumed.clear()
            actor_lr = _pickup(actor_lr)         # 轮间拾取新参数 → 本轮内 π_old 冻结

            buf = RolloutBuffer()
            ep_succ = []
            for _step in range(rollout_size):    # 采 rollout_size 个决策步（NOT episodes）
                obs_m = preprocess(obs)
                action_chunk, z, logp_old, value, actor_lr = actor_lr.sample_actions(obs_m)
                ac = np.asarray(action_chunk[0])         # (H, A)（B=1 去批）
                # ---- per-action 执行前 H_e 步并聚合 reward（DBPO venv.step(prefix) 的 per-action 等价）----
                r_agg, done = 0.0, False
                for h in range(H_e):
                    real_a, _ = env.step(ac[h].tolist())
                    done, success, r, mask = env.get_info_for_step()
                    r_agg += float(r)
                    if done:
                        ep_succ.append(bool(success)); break
                next_obs = env.get_observation()
                buf.add(obs_m, np.asarray(z[0]), ac[:H_e], float(logp_old[0]),
                        float(value[0]), r_agg, float(done))
                obs = env.reset() if done else next_obs

            # bootstrap V(s_T)：对最后一帧再过一次 value（用同一 actor_lr，frozen）。
            last_obs_m = preprocess(obs)
            _, _, _, last_value, actor_lr = actor_lr.sample_actions(last_obs_m)
            buf.finalize(float(last_value[0]), gamma=gamma, gae_lambda=gae_lambda)

            _rollout_holder[0] = (buf, None)     # 交付给 learner
            _rollout_ready.set()
            if ep_succ:
                wandb.log({"rollout/success_rate": float(np.mean(ep_succ)),
                           "rollout/episodes": len(ep_succ)}, step=it)
    finally:
        _stop.set()
        _rollout_ready.set()
        _thread.join(timeout=120)
        logging.info("DBPO async training finished.")


def _save_checkpoint(log_dir, learner, step):
    try:
        import orbax.checkpoint as ocp
        ckpt_dir = epath.Path(log_dir) / "checkpoints" / f"iter_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with ocp.StandardCheckpointer() as ckptr:
            ckptr.save(ckpt_dir / "actor", learner.actor.params)
            ckptr.save(ckpt_dir / "logstd", learner.logstd_head.params)
            ckptr.save(ckpt_dir / "value", learner.value_head.params)
        logging.info("Saved DBPO checkpoint at iter %d", step)
    except Exception as e:
        logging.error("Checkpoint save failed: %s", e)


if __name__ == "__main__":
    app.run(main)
