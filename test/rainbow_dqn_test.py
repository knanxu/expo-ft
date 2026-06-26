"""Branching Rainbow-DQN 网络测试：shape / softmax 守恒 / C51 投影守恒。

    python -m expo_ft.networks.rainbow_dqn_test
"""

import jax
import jax.numpy as jnp
import numpy as np

from expo_ft.networks.rainbow_dqn import (
    BranchingRainbowQ,
    categorical_projection,
    expected_q,
    greedy_action_idxs,
    make_support,
)

B, FEAT, N_ATOMS = 4, 32, 51
HEADS = (5, 4, 4)
VMIN, VMAX = -1.0, 11.0


def _net_and_dists():
    net = BranchingRainbowQ(head_sizes=HEADS, n_atoms=N_ATOMS, v_min=VMIN, v_max=VMAX)
    key = jax.random.PRNGKey(0)
    feat = jax.random.normal(key, (B, FEAT))
    params = net.init(key, feat)["params"]
    return net, params, feat


def test_forward_shapes_and_simplex():
    net, params, feat = _net_and_dists()
    dists = net.apply({"params": params}, feat)
    assert len(dists) == len(HEADS)
    for i, d in enumerate(dists):
        assert d.shape == (B, HEADS[i], N_ATOMS)
        assert jnp.allclose(d.sum(-1), 1.0, atol=1e-4)   # softmax over atoms
        assert jnp.all(d > 0)                             # clip 后严格正（log 安全）


def test_greedy_idxs_shape():
    net, params, feat = _net_and_dists()
    dists = net.apply({"params": params}, feat)
    support = make_support(VMIN, VMAX, N_ATOMS)
    idxs = greedy_action_idxs(dists, support)
    assert idxs.shape == (B, len(HEADS))
    for i in range(len(HEADS)):
        assert jnp.all(idxs[:, i] < HEADS[i])


def test_projection_conserves_mass():
    net, params, feat = _net_and_dists()
    dists = net.apply({"params": params}, feat)
    support = make_support(VMIN, VMAX, N_ATOMS)
    sel = dists[0][:, 0, :]
    rewards = jnp.array([0.0, 1.0, 2.5, -1.0])
    discounts = jnp.array([0.99 ** 3, 0.0, 0.99 ** 3, 0.99 ** 2])
    m = categorical_projection(sel, rewards, discounts, support, VMIN, VMAX)
    assert m.shape == (B, N_ATOMS)
    assert jnp.allclose(m.sum(-1), 1.0, atol=1e-4)       # 概率守恒
    assert jnp.all(m >= 0)


def test_projection_terminal_mean_equals_reward():
    """discount=0（终止）：Tz=r 常数 → 投影后期望值 ≈ r。"""
    support = make_support(VMIN, VMAX, N_ATOMS)
    sel = jnp.ones((3, N_ATOMS)) / N_ATOMS               # 任意分布（终止时被 reward 覆盖）
    rewards = jnp.array([0.0, 3.0, -0.5])
    discounts = jnp.zeros(3)
    m = categorical_projection(sel, rewards, discounts, support, VMIN, VMAX)
    mean = expected_q(m, support)
    np.testing.assert_allclose(np.asarray(mean), np.asarray(rewards), atol=1e-4)


def test_no_dueling_variant():
    net = BranchingRainbowQ(head_sizes=(3,), n_atoms=11, v_min=0.0, v_max=1.0, dueling=False)
    key = jax.random.PRNGKey(1)
    feat = jax.random.normal(key, (2, FEAT))
    params = net.init(key, feat)["params"]
    dists = net.apply({"params": params}, feat)
    assert dists[0].shape == (2, 3, 11)
    assert jnp.allclose(dists[0].sum(-1), 1.0, atol=1e-4)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} rainbow_dqn tests passed.")


if __name__ == "__main__":
    main()
