#! /usr/bin/env python
"""Whole-chunk TOPPRA fixed vel_limit × k_skip evaluation sweep.

This evaluates the frozen VLA policy with the ``chunk_toppra`` execution backend
only. It does not construct or restore a SpeedTune DQN; each configuration uses a
constant raw ``vel_limit`` and a fixed whole-chunk ``execution_steps``/k_skip.
"""

import logging
import os

import etils.epath as epath
import jax
import numpy as np
from absl import app, flags

import openpi.training.sharding as openpi_sharding
from expo_ft.env.env_client import EnvClientWrapper
from expo_ft.speedtune.chunk_kskip_sweep import (
    parse_k_skips,
    result_row,
    write_aggregate_artifacts,
    write_json,
)
from expo_ft.speedtune.exec_backends import build_backend, parse_force_limit
from expo_ft.speedtune.fixed_speed_sweep import ConstantSpeedLearner, parse_speeds
from expo_ft.utils.train_utils import init_logging

# Reuse VLA builder, deterministic rollout, and shared config flags.
import eval_speedtune as ev


FLAGS = flags.FLAGS

flags.DEFINE_string("vel_limits", "1,1.5,2,2.5,3,3.5,4", "Raw chunk_toppra vel_limit grid.")
flags.DEFINE_string("k_skips", "10,20,30,40,50", "Whole-chunk execution_steps/k_skip grid.")
flags.DEFINE_integer("chunk_toppra_port", 8102, "chunk_toppra RoboTwin server port.")


def _make_env(config, config_task, k_skip, output_dir):
    example_action = np.asarray(
        config_task.get(
            "example_action",
            np.zeros((1, int(config.n_real_dims)), dtype=np.float32),
        ),
        dtype=np.float32,
    )
    return EnvClientWrapper(
        env_creation_request={
            "example_action": example_action,
            "env_usage": "eval",
            "video_dir": os.path.join(output_dir, f"k{k_skip}", "videos"),
            "exec_backend": "chunk_toppra",
            "k_skip": int(k_skip),
            "stream_hold_steps": int(config.get("stream_hold_steps", 15)),
            "eval_video_save_freq": 25,
            "force_limit": parse_force_limit(config.get("force_limit", "")),
        },
        host=FLAGS.client_host,
        port=FLAGS.chunk_toppra_port,
    )


def _config_dir(output_dir, k_skip, vel_limit):
    vel_tag = f"{float(vel_limit):.1f}".replace(".", "p")
    return os.path.join(output_dir, f"k{k_skip}", f"vel_limit_{vel_tag}")


def main(_):
    init_logging()
    if FLAGS.n_episodes <= 0:
        raise ValueError("--n_episodes must be positive")
    config, config_task = FLAGS.config, FLAGS.config_task
    vel_limits = parse_speeds(FLAGS.vel_limits)
    k_skips = parse_k_skips(FLAGS.k_skips, max_k=int(config.action_horizon))
    output_dir = FLAGS.output_dir
    os.makedirs(output_dir, exist_ok=True)

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
    jax.config.update("jax_default_matmul_precision", "highest")
    mesh = openpi_sharding.make_mesh(FLAGS.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS)
    )
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    first_env = _make_env(config, config_task, k_skips[0], output_dir)
    first_env.reset()
    vla = ev._build_vla(
        config,
        config_task,
        FLAGS.seed,
        mesh,
        (data_sharding, replicated_sharding),
        first_env,
        FLAGS.resume,
    )

    backend = build_backend("chunk_toppra")
    rows = []
    envs = {k_skips[0]: first_env}
    for k_skip in k_skips:
        env = envs.get(k_skip)
        if env is None:
            env = _make_env(config, config_task, k_skip, output_dir)
            env.reset()
            envs[k_skip] = env
        for vel_limit in vel_limits:
            logging.info(
                "whole-chunk k_skip=%d vel_limit=%.1f episodes=%d",
                k_skip,
                vel_limit,
                FLAGS.n_episodes,
            )
            controller = ConstantSpeedLearner(backend, vel_limit)
            run_dir = _config_dir(output_dir, k_skip, vel_limit)
            os.makedirs(run_dir, exist_ok=True)
            result = ev.run_backend_episodes(
                env,
                vla,
                backend,
                controller,
                "chunk_toppra",
                n_episodes=FLAGS.n_episodes,
                max_decision_steps=FLAGS.max_decision_steps,
                seed=FLAGS.seed,
                out_dir=run_dir,
                record_video=False,
                k_skip=k_skip,
                save_episode_artifacts=False,
                max_action_idxs=None,
            )
            write_json(os.path.join(run_dir, "episodes.json"), result)
            rows.append(result_row(result, vel_limit=vel_limit, k_skip=k_skip))

    metadata = {
        "backend": "chunk_toppra",
        "n_episodes_per_config": int(FLAGS.n_episodes),
        "total_configs": int(len(k_skips) * len(vel_limits)),
        "total_episodes": int(len(k_skips) * len(vel_limits) * FLAGS.n_episodes),
        "seed": int(FLAGS.seed),
        "vel_limits": list(vel_limits),
        "k_skips": list(k_skips),
        "speed_parameter": "vel_limit",
        "physics_hz": 250,
        "vla_noise_key": "(seed, episode, decision_step)",
    }
    write_aggregate_artifacts(output_dir, rows, metadata)
    logging.info(
        "whole-chunk k-skip sweep complete: configs=%d total_episodes=%d output=%s",
        metadata["total_configs"],
        metadata["total_episodes"],
        output_dir,
    )


if __name__ == "__main__":
    app.run(main)
