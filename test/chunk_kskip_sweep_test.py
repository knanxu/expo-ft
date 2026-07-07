import csv
import json
import math
import tempfile
from pathlib import Path

from expo_ft.speedtune.chunk_kskip_sweep import (
    DEFAULT_K_SKIPS,
    result_row,
    parse_k_skips,
    write_aggregate_artifacts,
)


def _result(k_skip=40, success=35, success_steps=1000.0):
    return {
        "exec_backend": "chunk_toppra",
        "n_episodes": 50,
        "success": success,
        "success_rate": success / 50,
        "reward_success": success,
        "reward_success_rate": success / 50,
        "fallback_rate": 0.1,
        "k_skip": k_skip,
        "speed_param_name": "vel_limit",
        "mean_decision_steps": 20.0,
        "mean_decision_steps_success": 18.0 if success else float("nan"),
        "mean_dense_steps": 1200.0,
        "mean_sim_time_s": 4.8,
        "mean_dense_steps_success": success_steps if success else float("nan"),
    }


def test_default_k_skip_grid_and_parser():
    assert DEFAULT_K_SKIPS == (10, 20, 30, 40, 50)
    assert parse_k_skips("10,20,30,40,50") == DEFAULT_K_SKIPS


def test_parse_k_skips_rejects_duplicates_unsorted_and_out_of_horizon():
    for spec in ("10,10", "40,20", "0,10", "10,51"):
        try:
            parse_k_skips(spec)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid k_skip spec accepted: {spec}")


def test_result_row_reports_success_rate_and_success_only_time():
    row = result_row(_result(k_skip=40), vel_limit=2.5, k_skip=40)
    assert row["backend"] == "chunk_toppra"
    assert row["vel_limit"] == 2.5
    assert row["k_skip"] == 40
    assert row["n_episodes"] == 50
    assert row["task_success"] == 35
    assert row["success_rate"] == 0.7
    assert row["mean_sim_time_s_success"] == 4.0


def test_result_row_uses_nan_when_no_successes():
    row = result_row(_result(success=0), vel_limit=4.0, k_skip=50)
    assert math.isnan(row["mean_dense_steps_success"])
    assert math.isnan(row["mean_sim_time_s_success"])


def test_write_aggregate_artifacts():
    rows = [
        result_row(_result(k_skip=40), vel_limit=2.0, k_skip=40),
        result_row(_result(k_skip=10), vel_limit=1.0, k_skip=10),
    ]
    metadata = {
        "n_episodes_per_config": 50,
        "vel_limits": [1.0, 2.0],
        "k_skips": [10, 40],
    }
    with tempfile.TemporaryDirectory() as tmp:
        write_aggregate_artifacts(tmp, rows, metadata, make_plots=False)
        summary = json.loads(Path(tmp, "wholechunk_kskip_vel_summary.json").read_text())
        assert summary["metadata"]["n_episodes_per_config"] == 50
        assert summary["results"][0]["k_skip"] == 10
        with Path(tmp, "wholechunk_kskip_vel_summary.csv").open(newline="") as file:
            csv_rows = list(csv.DictReader(file))
        assert csv_rows[0]["vel_limit"] == "1.0"
        assert Path(tmp, "wholechunk_kskip_vel_report.md").exists()


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} chunk k-skip sweep tests passed")
