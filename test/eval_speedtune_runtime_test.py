import tempfile

import numpy as np

import eval_speedtune as ev
from expo_ft.speedtune.exec_backends import build_backend


class FakeLearner:
    def __init__(self):
        self.action_limits = []

    def greedy_action_idxs(self, feat, *, max_action_idxs=None):
        self.action_limits.append(max_action_idxs)
        return np.asarray([[0]], dtype=np.int32)


class FakeMultiHeadLearner:
    def greedy_action_idxs(self, feat, *, max_action_idxs=None):
        del feat, max_action_idxs
        return np.asarray([[0, 0, 0]], dtype=np.int32)


class FakeEnv:
    def __init__(self):
        self.reseed_value = None
        self.started_videos = []
        self.stopped_videos = 0

    def reseed(self, seed):
        self.reseed_value = seed

    def reset(self):
        return {"state": np.zeros((1,))}

    def step_chunk(self, chunk, speed_params, backend):
        return chunk[-1], {
            "n_exec_steps": 15,
            "duration": 0.06,
            "exec_status": "success",
            "fixed_time_speed_violation": True,
            "max_planned_qvel": 4.2,
            "execution_steps": 10,
        }

    def get_info_for_step(self):
        return True, True, 1.0, 0.0

    def start_video(self, episode):
        self.started_videos.append(episode)

    def stop_video(self):
        self.stopped_videos += 1


def test_eval_records_physics_wall_and_safe_success(monkeypatch=None):
    original = ev._save_and_plot
    ev._save_and_plot = lambda *args, **kwargs: None
    try:
        vla = {
            "preprocess": lambda obs: obs,
            "drift_forward": lambda obs, z: (
                np.zeros((1, 2, 2)), None, np.ones((1, 4))
            ),
            "unnormalize": lambda mean, obs: np.zeros((2, 14)),
            "H": 2,
            "A": 2,
        }
        env = FakeEnv()
        with tempfile.TemporaryDirectory() as tmp:
            learner = FakeLearner()
            result = ev.run_backend_episodes(
                env, vla, build_backend("fixed_time"), learner, "fixed_time",
                n_episodes=1, max_decision_steps=3, seed=42, out_dir=tmp,
                record_video=False, k_skip=10, max_action_idxs=(0,),
            )
        episode = result["episodes"][0]
        assert env.reseed_value == 42
        assert episode["task_success"] is True
        assert episode["reward_success"] is False
        assert episode["total_dense_steps"] == 15
        assert episode["sim_time_s"] == 15 / 250
        assert episode["k_skip"] == 10
        assert episode["end_to_end_wall_s"] >= episode["vla_wall_s"]
        assert result["fixed_time_speed_violation_rate"] == 1.0
        assert result["speed_action_counts"] == {"v": {"1.0": 1}}
        assert episode["max_planned_qvel"] == 4.2
        assert result["max_planned_qvel"] == 4.2
        rec = episode["recs"][0]
        assert rec["v"] == 1.0
        for removed in ("aggr", "aggr_mean", "acc_limit", "derived_acc_limit"):
            assert removed not in rec
        assert learner.action_limits == [(0,)]
    finally:
        ev._save_and_plot = original


def test_whole_chunk_eval_records_only_raw_vel_limit():
    original = ev._save_and_plot
    ev._save_and_plot = lambda *args, **kwargs: None
    try:
        vla = {
            "preprocess": lambda obs: obs,
            "drift_forward": lambda obs, z: (
                np.zeros((1, 2, 2)), None, np.ones((1, 4))
            ),
            "unnormalize": lambda mean, obs: np.zeros((2, 14)),
            "H": 2,
            "A": 2,
        }
        env = FakeEnv()
        with tempfile.TemporaryDirectory() as tmp:
            result = ev.run_backend_episodes(
                env, vla, build_backend("chunk_toppra"), FakeLearner(), "chunk_toppra",
                n_episodes=1, max_decision_steps=3, seed=42, out_dir=tmp,
                record_video=False, k_skip=20, max_action_idxs=(0,),
            )
        rec = result["episodes"][0]["recs"][0]
        assert rec["vel_limit"] == 1.0
        assert result["speed_param_name"] == "vel_limit"
        assert result["speed_in_contact_mean"] is None
        assert result["speed_free_mean"] == 1.0
        for removed in ("aggr", "aggr_mean", "acc_limit", "derived_acc_limit"):
            assert removed not in rec
        for removed in ("aggr_in_contact_mean", "aggr_free_mean", "decel_near_contact"):
            assert removed not in result
    finally:
        ev._save_and_plot = original


def test_legacy_multihead_eval_uses_only_primary_raw_speed_without_crashing():
    original = ev._save_and_plot
    ev._save_and_plot = lambda *args, **kwargs: None
    try:
        vla = {
            "preprocess": lambda obs: obs,
            "drift_forward": lambda obs, z: (
                np.zeros((1, 2, 2)), None, np.ones((1, 4))
            ),
            "unnormalize": lambda mean, obs: np.zeros((2, 14)),
            "H": 2,
            "A": 2,
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = ev.run_backend_episodes(
                FakeEnv(), vla, build_backend("per_action_toppra"),
                FakeMultiHeadLearner(), "per_action_toppra",
                n_episodes=1, max_decision_steps=3, seed=42, out_dir=tmp,
                record_video=False, k_skip=10, max_action_idxs=(0, 0, 0),
            )
        assert result["speed_param_name"] == "v"
        assert result["speed_action_counts"] == {"v": {"1.0": 1}}
        assert set(result["episodes"][0]["recs"][0]) & {"vel_limit", "acc_limit"} == set()
    finally:
        ev._save_and_plot = original


def test_eval_records_only_selected_video_ids_without_overwriting_artifacts():
    original = ev._save_and_plot
    artifact_calls = []
    ev._save_and_plot = lambda *args, **kwargs: artifact_calls.append(args)
    try:
        vla = {
            "preprocess": lambda obs: obs,
            "drift_forward": lambda obs, z: (
                np.zeros((1, 2, 2)), None, np.ones((1, 4))
            ),
            "unnormalize": lambda mean, obs: np.zeros((2, 14)),
            "H": 2,
            "A": 2,
        }
        env = FakeEnv()
        with tempfile.TemporaryDirectory() as tmp:
            ev.run_backend_episodes(
                env, vla, build_backend("fixed_time"), FakeLearner(), "fixed_time",
                n_episodes=3, max_decision_steps=3, seed=42, out_dir=tmp,
                record_video=True, video_episode_ids={1}, save_episode_artifacts=False,
                k_skip=10, max_action_idxs=(0,),
            )
        assert env.started_videos == [1]
        assert env.stopped_videos == 1
        assert artifact_calls == []
    finally:
        ev._save_and_plot = original


if __name__ == "__main__":
    test_eval_records_physics_wall_and_safe_success()
    test_eval_records_only_selected_video_ids_without_overwriting_artifacts()
    test_whole_chunk_eval_records_only_raw_vel_limit()
    test_legacy_multihead_eval_uses_only_primary_raw_speed_without_crashing()
    print("4 eval runtime tests passed")
