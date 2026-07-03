import csv
import json
import tempfile
from pathlib import Path

import numpy as np

from expo_ft.speedtune.fixed_speed_sweep import (
    DEFAULT_SPEEDS,
    ConstantSpeedLearner,
    parse_speeds,
    result_row,
    write_aggregate_artifacts,
)
from expo_ft.speedtune.exec_backends import build_backend


def _result(backend, param):
    return {
        "exec_backend": backend,
        "n_episodes": 30,
        "success": 24,
        "success_rate": 0.8,
        "reward_success": 21,
        "reward_success_rate": 0.7,
        "mean_decision_steps": 20.0,
        "mean_decision_steps_success": 19.0,
        "mean_dense_steps": 1200.0,
        "mean_dense_steps_success": 1100.0,
        "mean_sim_time_s": 4.8,
        "fixed_time_speed_violation_rate": 0.1 if backend == "fixed_time" else 0.0,
        "fallback_rate": 0.05 if backend == "chunk_toppra" else 0.0,
        "k_skip": 10 if backend == "fixed_time" else 20,
        "speed_param_name": param,
        "max_planned_qvel": 4.3 if backend == "fixed_time" else 0.0,
        "mean_episode_max_planned_qvel": 3.7 if backend == "fixed_time" else 0.0,
    }


def test_default_grid_has_exact_half_step_values():
    assert DEFAULT_SPEEDS == (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
    assert parse_speeds("1,1.5,2,2.5,3,3.5,4") == DEFAULT_SPEEDS


def test_parse_speeds_rejects_duplicates_and_values_outside_supported_grid():
    for spec in ("1,1", "0.5,1", "1,4.5", "1,1.25"):
        try:
            parse_speeds(spec)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid speed spec accepted: {spec}")


def test_constant_controller_selects_requested_raw_backend_value():
    feat = np.zeros((3, 8), dtype=np.float32)
    for backend_name, param in (("fixed_time", "v"), ("chunk_toppra", "vel_limit")):
        backend = build_backend(backend_name)
        controller = ConstantSpeedLearner(backend, 2.5)
        idxs = controller.greedy_action_idxs(feat)
        assert idxs.shape == (3, 1)
        speed_params, _ = backend.decode(idxs[0])
        assert speed_params == {param: 2.5}


def test_constant_controller_rejects_multihead_and_unknown_values():
    for backend, speed in ((build_backend("per_action_toppra"), 2.0),
                           (build_backend("fixed_time"), 2.25)):
        try:
            ConstantSpeedLearner(backend, speed)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid fixed-speed controller configuration accepted")


def test_result_rows_and_artifacts_report_30_episode_physics_and_fixed_max_velocity():
    fixed = result_row(_result("fixed_time", "v"), 2.5)
    chunk = result_row(_result("chunk_toppra", "vel_limit"), 2.5)
    assert fixed["n_episodes"] == 30
    assert fixed["task_success"] == 24
    assert fixed["mean_dense_steps"] == 1200.0
    assert fixed["max_planned_qvel"] == 4.3
    assert fixed["speed_param_name"] == "v"
    assert chunk["speed_param_name"] == "vel_limit"
    assert chunk["k_skip"] == 20

    metadata = {
        "n_episodes_per_config": 30,
        "speeds": list(DEFAULT_SPEEDS),
        "k_skip": {"fixed_time": 10, "chunk_toppra": 20},
    }
    with tempfile.TemporaryDirectory() as tmp:
        write_aggregate_artifacts(tmp, [fixed, chunk], metadata, make_plots=False)
        summary = json.loads(Path(tmp, "fixed_speed_summary.json").read_text())
        assert summary["metadata"]["n_episodes_per_config"] == 30
        assert len(summary["results"]) == 2
        with Path(tmp, "fixed_speed_summary.csv").open(newline="") as file:
            rows = list(csv.DictReader(file))
        assert rows[0]["max_planned_qvel"] == "4.3"
        assert Path(tmp, "fixed_speed_report.md").exists()


def test_eval_entrypoint_reuses_vla_runner_without_dqn_checkpoint():
    entrypoint = Path(__file__).parents[1] / "eval_speedtune_fixed_sweep.py"
    text = entrypoint.read_text()
    assert "ConstantSpeedLearner" in text
    assert "ev._build_vla" in text
    assert "ev.run_backend_episodes" in text
    assert "ev._build_dqn" not in text
    assert "save_episode_artifacts=False" in text
    assert 'flags.DEFINE_string("speeds", "1,1.5,2,2.5,3,3.5,4"' in text


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} fixed-speed sweep tests passed")
