#!/usr/bin/env python
"""Synchronous SpeedTune training: rollout one episode, then update in-place.

The termination budget is the number of environment chunk decisions, matching
``train_pi_robo.py``'s environment-step loop. Gradient updates are triggered only
after a complete episode and do not control termination.
"""

import logging
import os

import etils.epath as epath
import jax
import numpy as np
import tqdm
import wandb
from absl import app

import openpi.training.sharding as openpi_sharding
from expo_ft.env.env_client import EnvClientWrapper
from expo_ft.speedtune.exec_backends import parse_force_limit
from expo_ft.speedtune.curriculum import (
    action_limits_for_backend,
    build_speed_curriculum,
    curriculum_metrics,
)
from expo_ft.speedtune.runtime_config import backend_k_skip
from expo_ft.speedtune.training_runtime import (
    flush_pending_episode,
    run_episode_updates,
    update_ready,
)
from expo_ft.utils.train_utils import init_logging, init_wandb
from train_speedtune_async import (
    FLAGS,
    _beta_at,
    _epsilon_at,
    _save_checkpoint,
    _setup_speedtune,
    build_checkpoint_metadata,
)


def main(_):
    init_logging()
    config, config_task = FLAGS.config, FLAGS.config_task
    assert config.model_cls == "SpeedTuneLearner", (
        f"expected SpeedTuneLearner, got {config.model_cls}"
    )

    jax.config.update(
        "jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser())
    )
    jax.config.update("jax_default_matmul_precision", "highest")

    mesh = openpi_sharding.make_mesh(FLAGS.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS)
    )
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )

    log_dir = os.path.join(FLAGS.output_dir, FLAGS.run_name or "speedtune_run")
    os.makedirs(log_dir, exist_ok=True)
    init_wandb(epath.Path(log_dir), FLAGS.resume, FLAGS.project_name, FLAGS.run_name)
    wandb.config.update(FLAGS.flag_values_dict(), allow_val_change=FLAGS.resume)
    wandb.config.update({"training_mode": "sync"}, allow_val_change=True)
    wandb.define_metric("episode")
    wandb.define_metric("rollout/*", step_metric="episode")

    example_action = np.asarray(
        config_task.get(
            "example_action",
            np.zeros((1, int(config.n_real_dims)), dtype=np.float32),
        ),
        dtype=np.float32,
    )
    logging.info("Creating RoboTwin synchronous SpeedTune env ...")
    env = EnvClientWrapper(
        env_creation_request={
            "example_action": example_action,
            "env_usage": "train",
            "video_dir": os.path.join(log_dir, "train_videos"),
            "exec_backend": str(config.exec_backend),
            "k_skip": backend_k_skip(config, str(config.exec_backend)),
            "stream_hold_steps": int(config.get("stream_hold_steps", 15)),
            "force_limit": parse_force_limit(config.get("force_limit", "")),
        },
        host=FLAGS.client_host,
        port=FLAGS.client_port,
    )
    env.reset()

    setup = _setup_speedtune(
        config,
        config_task,
        FLAGS.seed,
        mesh,
        (data_sharding, replicated_sharding),
        env,
    )
    drift_forward = setup["drift_forward"]
    unnormalize = setup["unnormalize"]
    preprocess = setup["preprocess"]
    backend = setup["backend"]
    buffer = setup["buffer"]
    learner = setup["dqn"]
    h, action_dim = setup["H"], setup["A"]
    exec_backend = str(config.exec_backend)
    checkpoint_metadata = build_checkpoint_metadata(config, backend)
    curriculum = build_speed_curriculum(config, backend)
    episode_action_limits = action_limits_for_backend(curriculum, backend)

    batch_size = int(config.batch_size)
    utd_ratio = int(config.utd_ratio)
    update_per_episode = int(config.update_per_episode)
    warmup_episodes = int(config.get("warmup_episodes", 10))
    max_iters = int(config.max_iters)
    log_every = 50

    rng = np.random.default_rng(FLAGS.seed)
    obs = env.get_observation()
    pending = []
    ep_success = False
    ep_speed_violation = False
    completed_episodes = 0
    n_updates = 0
    n_update_groups = 0
    decision_steps = 0
    ep_returns, ep_reward_returns, ep_speed_violations = [], [], []
    ep_lens, ep_exec_time, ep_dense = [], [], []
    fallbacks = [0, 0]
    latest_train_metrics = {}

    def checkpoint_metadata_snapshot():
        return {
            **checkpoint_metadata,
            "training_mode": "sync",
            "decision_steps": decision_steps,
            "completed_episodes": completed_episodes,
            "update_groups": n_update_groups,
            "gradient_updates": n_updates,
            "curriculum_state": (
                curriculum.state_dict() if curriculum is not None else None
            ),
        }

    def flush_episode(*, force_terminal=False):
        nonlocal pending, ep_success, ep_speed_violation
        summary = flush_pending_episode(
            pending,
            buffer,
            backend,
            task_success=ep_success,
            speed_violation=ep_speed_violation,
            force_terminal=force_terminal,
            reward_mode=str(config.get("reward_mode", "success_gated")),
            reward_alpha=(
                1.0 if config.get("reward_alpha", None) is None
                else float(config.get("reward_alpha", 1.0))
            ),
            reward_beta=(
                2.0 if config.get("reward_beta", None) is None
                else float(config.get("reward_beta", 2.0))
            ),
            gamma=float(config.gamma),
        )
        if summary is None:
            return None
        ep_returns.append(summary["task_success"])
        ep_reward_returns.append(summary["reward_success"])
        ep_speed_violations.append(summary["speed_violation"])
        ep_lens.append(summary["length"])
        ep_exec_time.append(summary["exec_time_s"])
        ep_dense.append(summary["dense_steps"])
        if not force_terminal:
            if curriculum is not None:
                curriculum.record_episode(
                    summary["reward_success"], episode=completed_episodes + 1,
                    decision=decision_steps,
                )
        pending = []
        ep_success = False
        ep_speed_violation = False
        return summary

    try:
        for it in tqdm.tqdm(range(max_iters), disable=not FLAGS.tqdm):
            decision_steps = it + 1
            learner = learner.replace(epsilon=_epsilon_at(it, config))

            obs_m = preprocess(obs)
            z = jax.random.normal(
                jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1))),
                (1, h, action_dim),
            )
            mean, _cond, value_feat = drift_forward(obs_m, z)
            feat = np.asarray(value_feat[0])
            real_chunk = unnormalize(mean, obs_m)

            idxs, learner = learner.sample_action_idxs(
                feat[None], max_action_idxs=episode_action_limits
            )
            idxs = np.asarray(idxs[0], dtype=np.int32)
            if curriculum is not None:
                curriculum.record_action(idxs)
            speed_params, reward_values = backend.decode(idxs)

            _executed, exec_info = env.step_chunk(
                real_chunk, speed_params, exec_backend
            )
            done, success, r_task, _mask = env.get_info_for_step()
            ep_success |= bool(success)
            ep_speed_violation |= bool(
                exec_info.get("fixed_time_speed_violation", False)
            )
            fallbacks[1] += 1
            if exec_info.get("exec_status") == "topp_fallback":
                fallbacks[0] += 1
            pending.append(
                {
                    "feat": feat,
                    "action_idxs": idxs,
                    "v_list": reward_values,
                    "r_task": float(r_task),
                    "done": bool(done),
                    "duration": float(exec_info.get("duration", 0.0)),
                    "n_exec": int(exec_info.get("n_exec_steps", 0)),
                    "execution_steps": int(exec_info.get("execution_steps", 1) or 1),
                }
            )

            if it % log_every == 0:
                step_log = {
                    f"action/{key}": float(value)
                    for key, value in speed_params.items()
                }
                step_log.update(
                    {
                        "exec/dense_steps": int(exec_info.get("n_exec_steps", 0)),
                        "exec/execution_steps": int(
                            exec_info.get("execution_steps", 0)
                        ),
                        "exec/duration_s": float(exec_info.get("duration", 0.0)),
                        "exec/planned_cruise_fraction": float(
                            exec_info.get("planned_cruise_fraction", 0.0)
                        ),
                        "exec/max_planned_qvel": float(
                            exec_info.get("max_planned_qvel", 0.0)
                        ),
                        "exec/fixed_time_speed_violation": float(
                            bool(exec_info.get("fixed_time_speed_violation", False))
                        ),
                        "exec/topp_torque_constrained": float(
                            bool(exec_info.get("topp_torque_constrained", False))
                        ),
                        "exec/topp_sd_start": float(
                            exec_info.get("topp_sd_start", 0.0)
                        ),
                        "training/n_updates": n_updates,
                        "training/n_update_groups": n_update_groups,
                        "training/decision_steps": decision_steps,
                    }
                )
                if "acc_limit" in exec_info:
                    step_log["action/derived_acc_limit"] = float(
                        exec_info["acc_limit"]
                    )
                step_log.update(latest_train_metrics)
                wandb.log(step_log, step=it)

            obs = env.reset() if done else env.get_observation()
            if not done:
                continue

            flush_episode()
            completed_episodes += 1
            update_action_limits = episode_action_limits
            if update_ready(
                completed_episodes=completed_episodes,
                replay_size=len(buffer),
                batch_size=batch_size,
                warmup_episodes=warmup_episodes,
            ):
                beta = _beta_at(it, config)
                learner, update_metrics, count = run_episode_updates(
                    learner,
                    buffer,
                    batch_size=batch_size,
                    beta=beta,
                    update_groups=update_per_episode,
                    utd_ratio=utd_ratio,
                    max_action_idxs=update_action_limits,
                )
                n_updates += count
                n_update_groups += update_per_episode
                latest_train_metrics = {
                    f"training/{key}": value
                    for key, value in update_metrics.items()
                }
                latest_train_metrics.update(
                    {
                        "training/per_beta": beta,
                        "training/n_updates": n_updates,
                        "training/n_update_groups": n_update_groups,
                        "training/replay_ratio": n_updates / max(decision_steps, 1),
                    }
                )
                if (
                    FLAGS.checkpoint_model
                    and FLAGS.checkpoint_interval > 0
                    and n_updates % FLAGS.checkpoint_interval == 0
                ):
                    _save_checkpoint(
                        log_dir, learner, n_updates, checkpoint_metadata_snapshot()
                    )

            fallback_rate = fallbacks[0] / max(fallbacks[1], 1)
            rollout_log = {
                    "rollout/success_rate": float(np.mean(ep_returns[-50:])),
                    "rollout/reward_success_rate": float(
                        np.mean(ep_reward_returns[-50:])
                    ),
                    "rollout/speed_violation_rate": float(
                        np.mean(ep_speed_violations[-50:])
                    ),
                    "rollout/ep_len_mean": float(np.mean(ep_lens[-50:])),
                    "rollout/exec_time_s": float(np.mean(ep_exec_time[-50:])),
                    "rollout/dense_steps": float(np.mean(ep_dense[-50:])),
                    "rollout/fallback_rate": float(fallback_rate),
                    "rollout/epsilon": _epsilon_at(it, config),
                    "rollout/buffer_size": len(buffer),
                    "rollout/n_updates": n_updates,
                    "episode": completed_episodes,
                }
            rollout_log.update(curriculum_metrics(curriculum))
            wandb.log(rollout_log, step=it)
            episode_action_limits = action_limits_for_backend(curriculum, backend)
    finally:
        # A 40,000-decision stop may occur mid-episode. Flush n-step state but do
        # not claim task success or trigger an episode update for this truncation.
        flush_episode(force_terminal=True)
        _save_checkpoint(log_dir, learner, n_updates, checkpoint_metadata_snapshot())
        logging.info(
            "SpeedTune sync training finished: decisions=%d episodes=%d "
            "update_groups=%d gradient_updates=%d",
            decision_steps,
            completed_episodes,
            n_update_groups,
            n_updates,
        )


if __name__ == "__main__":
    app.run(main)
