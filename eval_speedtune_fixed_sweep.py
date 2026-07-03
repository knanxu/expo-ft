#! /usr/bin/env python
"""Evaluate fixed raw speeds for fixed-time and whole-chunk TOPPRA.

This loads the frozen VLA once and deliberately does not construct or restore a
SpeedTune DQN. Every backend/speed pair uses the same episode seeds and the same
per-decision VLA noise schedule as trained-policy evaluation.
"""

import logging
import os

import etils.epath as epath
import jax
import numpy as np
from absl import app, flags

import openpi.training.sharding as openpi_sharding
from expo_ft.env.env_client import EnvClientWrapper
from expo_ft.speedtune.exec_backends import build_backend, parse_force_limit
from expo_ft.speedtune.fixed_speed_sweep import (
    ConstantSpeedLearner,
    parse_speeds,
    result_row,
    write_aggregate_artifacts,
    write_json,
)
from expo_ft.utils.train_utils import init_logging

# Reuse the VLA builder, deterministic rollout, and shared config flags.
import eval_speedtune as ev


FLAGS = flags.FLAGS

flags.DEFINE_string("speeds", "1,1.5,2,2.5,3,3.5,4", "Raw v/vel_limit sweep grid.")
flags.DEFINE_integer("fixed_time_port", 8102, "fixed_time RoboTwin server port.")
flags.DEFINE_integer("chunk_toppra_port", 8103, "chunk_toppra RoboTwin server port.")


def _make_env(config, config_task, backend_name, port, output_dir):
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
            "video_dir": os.path.join(output_dir, backend_name, "videos"),
            "exec_backend": backend_name,
            "k_skip": ev.backend_k_skip(config, backend_name),
            "stream_hold_steps": int(config.get("stream_hold_steps", 15)),
            "eval_video_save_freq": 25,
            "force_limit": parse_force_limit(config.get("force_limit", "")),
        },
        host=FLAGS.client_host,
        port=port,
    )


def _config_dir(output_dir, backend_name, speed_param_name, speed):
    speed_tag = f"{float(speed):.1f}".replace(".", "p")
    return os.path.join(output_dir, backend_name, f"{speed_param_name}_{speed_tag}")


def main(_):
    init_logging()
    if FLAGS.n_episodes <= 0:
        raise ValueError("--n_episodes must be positive")
    speeds = parse_speeds(FLAGS.speeds)
    config, config_task = FLAGS.config, FLAGS.config_task
    output_dir = FLAGS.output_dir
    os.makedirs(output_dir, exist_ok=True)

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
    jax.config.update("jax_default_matmul_precision", "highest")
    mesh = openpi_sharding.make_mesh(FLAGS.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS)
    )
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    ports = {
        "fixed_time": FLAGS.fixed_time_port,
        "chunk_toppra": FLAGS.chunk_toppra_port,
    }
    envs = {
        name: _make_env(config, config_task, name, port, output_dir)
        for name, port in ports.items()
    }
    for env in envs.values():
        env.reset()

    # Observations have the same schema for both execution backends, so one VLA
    # instance built from fixed_time is reused for all 420 default episodes.
    vla = ev._build_vla(
        config,
        config_task,
        FLAGS.seed,
        mesh,
        (data_sharding, replicated_sharding),
        envs["fixed_time"],
        FLAGS.resume,
    )

    rows = []
    backends = ("fixed_time", "chunk_toppra")
    for backend_name in backends:
        backend = build_backend(backend_name)
        speed_param_name = backend.vars[0].name
        k_skip = ev.backend_k_skip(config, backend_name)
        for speed in speeds:
            logging.info(
                "fixed sweep backend=%s %s=%.1f episodes=%d k_skip=%d",
                backend_name,
                speed_param_name,
                speed,
                FLAGS.n_episodes,
                k_skip,
            )
            controller = ConstantSpeedLearner(backend, speed)
            run_dir = _config_dir(output_dir, backend_name, speed_param_name, speed)
            os.makedirs(run_dir, exist_ok=True)
            result = ev.run_backend_episodes(
                envs[backend_name],
                vla,
                backend,
                controller,
                backend_name,
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
            rows.append(result_row(result, speed))

    metadata = {
        "n_episodes_per_config": int(FLAGS.n_episodes),
        "total_episodes": int(len(backends) * len(speeds) * FLAGS.n_episodes),
        "seed": int(FLAGS.seed),
        "speeds": list(speeds),
        "backends": list(backends),
        "speed_parameter": {"fixed_time": "v", "chunk_toppra": "vel_limit"},
        "k_skip": {
            backend_name: ev.backend_k_skip(config, backend_name)
            for backend_name in backends
        },
        "physics_hz": 250,
        "vla_noise_key": "(seed, episode, decision_step)",
    }
    write_aggregate_artifacts(output_dir, rows, metadata)
    logging.info(
        "fixed-speed sweep complete: %d configs, %d total episodes, output=%s",
        len(rows),
        metadata["total_episodes"],
        output_dir,
    )


if __name__ == "__main__":
    app.run(main)
