from pathlib import Path

import numpy as np

from expo_ft.speedtune.training_runtime import (
    flush_pending_episode,
    run_episode_updates,
    update_ready,
)


class FakeLearner:
    def __init__(self):
        self.update_calls = 0
        self.action_limits = []

    def update(self, batch, *, max_action_idxs=None):
        self.update_calls += 1
        self.action_limits.append(max_action_idxs)
        td = np.ones(len(batch["tree_indices"]), dtype=np.float32)
        return self, td, {"loss": float(self.update_calls)}


class FakeBuffer:
    def __init__(self):
        self.sample_calls = 0
        self.priority_calls = 0
        self.inserted = []

    def sample(self, batch_size, beta):
        self.sample_calls += 1
        return {"tree_indices": np.arange(batch_size, dtype=np.int64)}

    def update_priorities(self, tree_indices, td):
        assert len(tree_indices) == len(td)
        self.priority_calls += 1

    def insert(self, feat, action_idxs, reward, next_feat, done, discount=None):
        self.inserted.append(
            (
                np.asarray(feat),
                np.asarray(action_idxs),
                reward,
                np.asarray(next_feat),
                done,
                discount,
            )
        )


class FakeBackend:
    name = "fixed_time"

    @staticmethod
    def total_reward(r_task, reward_success, values):
        del r_task
        return float(values[0] ** 2) if reward_success else 0.0


def test_update_ready_matches_train_pi_episode_warmup():
    assert not update_ready(completed_episodes=9, replay_size=1000, batch_size=64)
    assert not update_ready(completed_episodes=10, replay_size=63, batch_size=64)
    assert update_ready(completed_episodes=10, replay_size=64, batch_size=64)


def test_one_episode_runs_six_groups_of_twenty_gradient_updates():
    learner = FakeLearner()
    buffer = FakeBuffer()
    published = []

    learner, metrics, n_updates = run_episode_updates(
        learner,
        buffer,
        batch_size=8,
        beta=0.4,
        update_groups=6,
        utd_ratio=20,
        max_action_idxs=(2,),
        on_group_end=lambda current: published.append(current.update_calls),
    )

    assert n_updates == 120
    assert learner.update_calls == 120
    assert buffer.sample_calls == 120
    assert buffer.priority_calls == 120
    assert published == [20, 40, 60, 80, 100, 120]
    assert learner.action_limits == [(2,)] * 120
    assert metrics["loss"] == 60.5


def test_forced_terminal_flushes_partial_episode_without_success_reward():
    pending = [
        {
            "feat": np.asarray([0.0]),
            "action_idxs": np.asarray([0]),
            "v_list": [4.0],
            "r_task": 1.0,
            "done": False,
            "duration": 0.2,
            "n_exec": 10,
        },
        {
            "feat": np.asarray([1.0]),
            "action_idxs": np.asarray([1]),
            "v_list": [4.0],
            "r_task": 1.0,
            "done": False,
            "duration": 0.3,
            "n_exec": 20,
        },
    ]
    buffer = FakeBuffer()

    summary = flush_pending_episode(
        pending,
        buffer,
        FakeBackend(),
        task_success=True,
        speed_violation=False,
        force_terminal=True,
    )

    assert summary == {
        "task_success": False,
        "reward_success": False,
        "speed_violation": False,
        "length": 2,
        "exec_time_s": 0.5,
        "dense_steps": 30,
        "execution_steps": 2,
        "truncated": True,
    }
    assert [row[2] for row in buffer.inserted] == [0.0, 0.0]
    assert [row[4] for row in buffer.inserted] == [False, True]
    np.testing.assert_array_equal(buffer.inserted[0][3], np.asarray([1.0]))


def test_paper_speedtuning_reward_and_discount_are_chunk_level():
    pending = [
        {
            "feat": np.asarray([0.0]),
            "action_idxs": np.asarray([0]),
            "v_list": [4.0],
            "r_task": 1.0,
            "done": True,
            "duration": 0.2,
            "n_exec": 30,
            "execution_steps": 3,
        }
    ]
    buffer = FakeBuffer()

    summary = flush_pending_episode(
        pending,
        buffer,
        FakeBackend(),
        task_success=True,
        speed_violation=True,
        reward_mode="paper_speedtuning",
        reward_alpha=0.1,
        reward_beta=2.0,
        gamma=0.5,
    )

    expected_reward = 0.1 * 16.0 + 1.0
    assert abs(buffer.inserted[0][2] - expected_reward) < 1e-6
    assert abs(buffer.inserted[0][5] - 0.5) < 1e-9
    assert summary["reward_success"] is True
    assert summary["speed_violation"] is True
    assert summary["execution_steps"] == 3


def test_paper_speedtuning_failure_keeps_speed_reward_without_penalty():
    pending = [
        {
            "feat": np.asarray([0.0]),
            "action_idxs": np.asarray([0]),
            "v_list": [4.0],
            "r_task": 0.0,
            "done": True,
            "duration": 0.2,
            "n_exec": 30,
            "execution_steps": 2,
        }
    ]
    buffer = FakeBuffer()

    summary = flush_pending_episode(
        pending,
        buffer,
        FakeBackend(),
        task_success=False,
        speed_violation=False,
        reward_mode="paper_speedtuning",
        reward_alpha=0.1,
        reward_beta=2.0,
        gamma=0.5,
    )

    assert abs(buffer.inserted[0][2] - 1.6) < 1e-6
    assert abs(buffer.inserted[0][5] - 0.5) < 1e-9
    assert summary["reward_success"] is False


def test_success_gated_reward_uses_chunk_level_discount():
    pending = [
        {
            "feat": np.asarray([0.0]),
            "action_idxs": np.asarray([0]),
            "v_list": [4.0],
            "r_task": 1.0,
            "done": True,
            "duration": 0.2,
            "n_exec": 30,
            "execution_steps": 4,
        }
    ]
    buffer = FakeBuffer()

    summary = flush_pending_episode(
        pending,
        buffer,
        FakeBackend(),
        task_success=True,
        speed_violation=False,
        reward_mode="success_gated",
        gamma=0.5,
    )

    assert buffer.inserted[0][2] == 16.0
    assert abs(buffer.inserted[0][5] - 0.5) < 1e-9
    assert summary["reward_success"] is True


def test_sync_entry_uses_environment_decision_budget_and_partial_flush():
    source = (Path(__file__).parents[1] / "train_speedtune_sync.py").read_text()
    assert "for it in tqdm.tqdm(range(max_iters)" in source
    assert source.count("env.step_chunk(") == 1
    assert "run_episode_updates(" in source
    assert "force_terminal=True" in source
    assert "\"execution_steps\"" in source
    assert "ep_speed_violations" in source
    assert "\"rollout/speed_violation_rate\"" in source
    assert "reward_mode=str(config.get(\"reward_mode\", \"success_gated\"))" in source
    assert "gamma=float(config.gamma)" in source


def test_async_entry_uses_counted_episode_queue_instead_of_event_latch():
    source = (Path(__file__).parents[1] / "train_speedtune_async.py").read_text()
    assert "_episode_updates = queue.Queue()" in source
    assert "_episode_updates.put((_step[0], completed_action_limits))" in source
    assert "_episode_updates.put(_update_sentinel)" in source
    assert "_new_episode = threading.Event()" not in source
    assert "_worker_error = [None]" in source
    assert "SpeedTune async update worker failed" in source
    assert "\"execution_steps\"" in source
    assert "ep_speed_violations" in source
    assert "\"rollout/speed_violation_rate\"" in source
    assert "reward_mode=str(config.get(\"reward_mode\", \"success_gated\"))" in source
    assert "gamma=float(config.gamma)" in source


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} SpeedTune training runtime tests passed")


if __name__ == "__main__":
    main()
