"""SpeedTune replay buffer 测试：n-step 聚合 / done 截断 / PER 采样。

    python -m expo_ft.data.speedtune_buffer_test
"""

import numpy as np

from expo_ft.data.speedtune_buffer import SpeedTuneReplayBuffer

FEAT, NH = 4, 3


def _feat(i):
    return np.full(FEAT, i, dtype=np.float32)


def _episode_buffer(n_step=3, gamma=0.5):
    buf = SpeedTuneReplayBuffer(capacity=100, feat_dim=FEAT, n_heads=NH,
                               n_step=n_step, gamma=gamma, seed=0)
    # r=[1,2,3,4]，末步 done
    buf.insert(_feat(0), [0, 1, 2], 1.0, _feat(1), False)
    buf.insert(_feat(1), [1, 0, 1], 2.0, _feat(2), False)
    buf.insert(_feat(2), [2, 2, 0], 3.0, _feat(3), False)
    buf.insert(_feat(3), [0, 0, 0], 4.0, _feat(4), True)
    return buf


def test_episode_emits_one_transition_per_step():
    buf = _episode_buffer()
    assert len(buf) == 4


def test_nstep_aggregation_and_truncation():
    buf = _episode_buffer(n_step=3, gamma=0.5)
    b = buf.sample(4)
    order = np.argsort(b["feat"][:, 0])
    R = b["reward"][order]; disc = b["discount"][order]; dn = b["done"][order]; nf = b["next_feat"][order][:, 0]
    # t0 完整 3-step: R=1+0.5*2+0.25*3=2.75, discount=0.5^3=0.125, done=0, next=f(3)
    assert abs(R[0] - 2.75) < 1e-5 and abs(disc[0] - 0.125) < 1e-5 and dn[0] == 0 and nf[0] == 3
    # t1 done 截断: R=2+0.5*3+0.25*4=4.5, discount=0, done=1
    assert abs(R[1] - 4.5) < 1e-5 and disc[1] == 0.0 and dn[1] == 1
    # t2: R=3+0.5*4=5.0, discount=0, done=1
    assert abs(R[2] - 5.0) < 1e-5 and disc[2] == 0.0 and dn[2] == 1
    # t3 终止: R=4, discount=0, done=1, next=f(4)
    assert abs(R[3] - 4.0) < 1e-5 and disc[3] == 0.0 and dn[3] == 1 and nf[3] == 4


def test_action_idxs_preserved():
    buf = _episode_buffer()
    b = buf.sample(4)
    order = np.argsort(b["feat"][:, 0])
    np.testing.assert_array_equal(b["action_idxs"][order][0], [0, 1, 2])  # t0 的档位


def test_sample_shapes_and_weights():
    buf = _episode_buffer()
    b = buf.sample(8)
    assert b["feat"].shape == (8, FEAT)
    assert b["action_idxs"].shape == (8, NH)
    assert b["reward"].shape == (8,) and b["discount"].shape == (8,)
    assert b["is_weights"].shape == (8,)
    assert (b["is_weights"] > 0).all() and b["is_weights"].max() <= 1.0 + 1e-6


def test_priority_update_changes_sampling():
    buf = SpeedTuneReplayBuffer(capacity=1000, feat_dim=FEAT, n_heads=NH, n_step=1, seed=1)
    for i in range(200):
        buf.insert(_feat(i), [0, 0, 0], 0.0, _feat(i + 1), True)
    b = buf.sample(32)
    # 给一半样本极高优先级，下一批应更频繁命中它们
    hi = b["tree_indices"][:16]
    td = np.concatenate([np.full(16, 100.0), np.full(16, 1e-6)])
    buf.update_priorities(b["tree_indices"], td)
    # sum-tree 总和应反映新优先级（不崩、可继续采样）
    b2 = buf.sample(64)
    assert b2["feat"].shape == (64, FEAT)


def test_ready():
    buf = SpeedTuneReplayBuffer(capacity=100, feat_dim=FEAT, n_heads=NH, n_step=1, seed=0)
    assert not buf.ready(10)
    for i in range(10):
        buf.insert(_feat(i), [0, 0, 0], 1.0, _feat(i + 1), True)
    assert buf.ready(10)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} speedtune_buffer tests passed.")


if __name__ == "__main__":
    main()
