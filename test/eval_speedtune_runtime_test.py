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
        assert learner.action_limits == [(0,)]
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
    print("2 eval runtime tests passed")
