"""SpeedTuneLearner 测试：构造 / 采样 / 更新 / 端到端可学习性（branching C51 DQN）。

    python -m expo_ft.agents.alg.speedtune_dqn_test
"""

import jax
import numpy as np

from expo_ft.agents.alg.speedtune_dqn import SpeedTuneLearner
from expo_ft.data.speedtune_buffer import SpeedTuneReplayBuffer

FEAT = 8
HEADS = (5, 4, 4)


def _make(epsilon=0.0, v_min=-1.0, v_max=4.0, lr=3e-4, tau=0.01):
    return SpeedTuneLearner.create(
        rng=jax.random.PRNGKey(0), feat_dim=FEAT, head_sizes=HEADS,
        n_atoms=51, v_min=v_min, v_max=v_max, lr=lr, epsilon=epsilon, tau=tau)


def test_create_and_greedy_shape():
    lr = _make()
    feat = np.ones((3, FEAT), dtype=np.float32)
    idxs = np.asarray(lr.greedy_action_idxs(feat))
    assert idxs.shape == (3, len(HEADS))
    for i in range(len(HEADS)):
        assert (idxs[:, i] < HEADS[i]).all()


def test_epsilon_greedy_explores():
    lr = _make(epsilon=1.0)               # 纯探索
    feat = np.ones((256, FEAT), dtype=np.float32)
    idxs, lr2 = lr.sample_action_idxs(feat)
    idxs = np.asarray(idxs)
    # epsilon=1 → 每 head 应覆盖多个档位（greedy 在固定 feat 下会是单一档）
    for i in range(len(HEADS)):
        assert len(np.unique(idxs[:, i])) >= 2
    # rng 推进（返回新 learner）
    assert lr2 is not lr


def test_update_runs_and_moves_target():
    lr = _make(epsilon=0.1)
    buf = SpeedTuneReplayBuffer(capacity=2000, feat_dim=FEAT, n_heads=len(HEADS),
                               n_step=1, gamma=0.99, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(500):
        a = [rng.integers(0, h) for h in HEADS]
        buf.insert(np.ones(FEAT, np.float32), a, float(np.sum(a)), np.ones(FEAT, np.float32), True)
    tgt_before = jax.tree_util.tree_leaves(lr.target_params)[0]
    batch = buf.sample(64)
    lr2, td, metrics = lr.update(batch)
    assert "loss/dqn" in metrics
    assert np.asarray(td).shape == (64,)
    tgt_after = jax.tree_util.tree_leaves(lr2.target_params)[0]
    assert not np.allclose(np.asarray(tgt_before), np.asarray(tgt_after))  # 软更新生效


def test_learnability_bandit():
    """固定 state 的 bandit：每 head 有最优档；训练后 greedy 应收敛到最优档。"""
    best = np.array([2, 1, 3])
    lr = SpeedTuneLearner.create(
        rng=jax.random.PRNGKey(0), feat_dim=FEAT, head_sizes=HEADS,
        n_atoms=51, v_min=-1.0, v_max=4.0, lr=3e-4, epsilon=0.3, tau=0.01)
    buf = SpeedTuneReplayBuffer(capacity=5000, feat_dim=FEAT, n_heads=len(HEADS),
                               n_step=1, gamma=0.99, seed=0)
    state = np.ones(FEAT, np.float32)
    rng = np.random.default_rng(0)
    for _ in range(2000):
        a = np.array([rng.integers(0, h) for h in HEADS])
        r = float(np.sum(a == best))
        buf.insert(state, a, r, state, True)
    loss0 = None
    for step in range(800):
        batch = buf.sample(64, beta=0.5)
        lr, td, metrics = lr.update(batch)
        buf.update_priorities(batch["tree_indices"], np.asarray(td))
        if step == 0:
            loss0 = float(metrics["loss/dqn"])
    greedy = np.asarray(lr.greedy_action_idxs(state[None]))[0]
    assert (greedy == best).all(), f"未收敛: {greedy} != {best}"
    assert float(metrics["loss/dqn"]) < loss0     # loss 下降


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} speedtune_dqn tests passed.")


if __name__ == "__main__":
    main()
