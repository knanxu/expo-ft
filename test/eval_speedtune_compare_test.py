import json
import tempfile
from pathlib import Path

import eval_speedtune_compare as compare


def _result(backend, speed_name):
    episode = {
        "ep": 0,
        "task_success": True,
        "reward_success": True,
        "success": True,
        "n_decision_steps": 2,
        "total_dense_steps": 100,
        "end_to_end_wall_s": 1.0,
        "recs": [{"step": 0, "t_sim": 0.4, speed_name: 2.0}],
    }
    return {
        "exec_backend": backend,
        "n_episodes": 30,
        "success": 21,
        "success_rate": 0.7,
        "reward_success": 20,
        "reward_success_rate": 2 / 3,
        "mean_decision_steps": 2.0,
        "mean_decision_steps_success": 2.0,
        "mean_dense_steps": 100.0,
        "mean_dense_steps_success": 100.0,
        "mean_sim_time_s": 0.4,
        "mean_end_to_end_wall_s": 1.0,
        "mean_vla_wall_s": 0.1,
        "mean_dqn_wall_s": 0.01,
        "mean_env_rpc_wall_s": 0.8,
        "fixed_time_speed_violation_rate": 0.05,
        "fallback_rate": 0.0,
        "k_skip": 10 if backend == "fixed_time" else 20,
        "max_action_idxs": [6],
        "max_unlocked_speed": 4.0,
        "speed_action_counts": {speed_name: {"2.0": 2}},
        "speed_param_name": speed_name,
        "speed_in_contact_mean": 1.5,
        "speed_free_mean": 2.5,
        "slows_near_contact": True,
        "max_planned_qvel": 3.8 if backend == "fixed_time" else 0.0,
        "mean_episode_max_planned_qvel": 3.2 if backend == "fixed_time" else 0.0,
        "episodes": [episode],
    }


def test_param_keys_expose_only_backend_raw_speed():
    assert compare._param_keys(_result("fixed_time", "v")["episodes"][0]) == ["v"]
    whole = _result("chunk_toppra", "vel_limit")["episodes"][0]
    whole["recs"][0].update({"acc_limit": 16.0, "derived_acc_limit": 16.0})
    assert compare._param_keys(whole) == ["vel_limit"]


def test_compare_summary_contains_raw_speed_and_no_acceleration_or_aggressiveness():
    with tempfile.TemporaryDirectory() as tmp:
        compare._save_compare_summary(
            tmp, _result("fixed_time", "v"), _result("chunk_toppra", "vel_limit")
        )
        summary = json.loads(Path(tmp, "compare_summary.json").read_text())
    encoded = json.dumps(summary)
    assert summary["fixed_time"]["n_episodes"] == 30
    assert summary["chunk_toppra"]["n_episodes"] == 30
    assert "speed_param_name" in encoded
    assert "max_planned_qvel" in encoded
    assert "aggr" not in encoded
    assert "acc_limit" not in encoded
    assert "acc_rule" not in encoded


def test_standalone_compare_launcher_defaults_to_30_fixed_vs_chunk_episodes():
    script = Path(__file__).parents[1] / "scripts" / "run_eval_compare.sh"
    text = script.read_text()
    assert 'N_EPISODES="${N_EPISODES:-30}"' in text
    assert 'BACKEND_A="${BACKEND_A:-fixed_time}"' in text
    assert 'BACKEND_B="${BACKEND_B:-chunk_toppra}"' in text
    assert "setsid env CUDA_VISIBLE_DEVICES" in text
    assert 'kill -TERM -- "-$p"' in text


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} compare eval tests passed")
