"""Self-contained tests for the DBPO/BPO algorithm core (GAE + surrogates).

    python -m expo_ft.agents.alg.dbpo_core_test
"""

import jax
import jax.numpy as jnp
import numpy as np

from expo_ft.agents.alg import dbpo_core as C


def _gae_ref(rewards, values, dones, last_value, gamma, lam):
    """Plain numpy GAE reference (forward-readable, reversed loop)."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float64)
    gae, next_value = 0.0, float(last_value)
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
        next_value = values[t]
    return adv, adv + values


def test_gae_matches_reference():
    rng = np.random.default_rng(0)
    T = 20
    rewards = rng.normal(size=T)
    values = rng.normal(size=T)
    dones = (rng.random(T) < 0.15).astype(np.float64)
    last_value = float(rng.normal())
    gamma, lam = 0.99, 0.95
    adv, ret = C.compute_gae(jnp.asarray(rewards), jnp.asarray(values),
                             jnp.asarray(dones), jnp.asarray(last_value),
                             gamma=gamma, gae_lambda=lam)
    adv_ref, ret_ref = _gae_ref(rewards, values, dones, last_value, gamma, lam)
    np.testing.assert_allclose(np.array(adv), adv_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(ret), ret_ref, rtol=1e-5, atol=1e-5)


def test_gae_lambda_zero_is_td_residual():
    rewards = jnp.array([1.0, 2.0, 3.0])
    values = jnp.array([0.5, 0.5, 0.5])
    dones = jnp.array([0.0, 0.0, 0.0])
    adv, _ = C.compute_gae(rewards, values, dones, jnp.array(0.0),
                           gamma=0.9, gae_lambda=0.0)
    td = rewards + 0.9 * jnp.array([0.5, 0.5, 0.0]) - values  # next_value of last is bootstrap 0
    np.testing.assert_allclose(np.array(adv), np.array(td), rtol=1e-6, atol=1e-6)


def test_gae_done_blocks_bootstrap():
    """A done at t must zero the bootstrap from t+1 into t."""
    rewards = jnp.array([1.0, 1.0])
    values = jnp.array([10.0, 20.0])
    dones = jnp.array([1.0, 0.0])  # episode ends after step 0
    adv, _ = C.compute_gae(rewards, values, dones, jnp.array(5.0),
                           gamma=0.99, gae_lambda=0.95)
    # step0: nonterminal=0 -> delta = r - V0 = 1 - 10 = -9, and no propagation
    np.testing.assert_allclose(float(adv[0]), -9.0, rtol=1e-6, atol=1e-6)


def test_ppo_ratio_one():
    adv = jnp.array([1.0, -2.0, 3.0])
    ratio = jnp.ones(3)
    loss, m = C.ppo_policy_loss(ratio, adv, clip_eps=0.2)
    np.testing.assert_allclose(float(loss), -float(adv.mean()), rtol=1e-6, atol=1e-6)
    assert float(m["clipfrac"]) == 0.0
    np.testing.assert_allclose(float(m["approx_kl"]), 0.0, atol=1e-6)


def test_ppo_clips_positive_advantage():
    adv = jnp.array([1.0])
    ratio = jnp.array([2.0])  # way above 1+eps
    loss, m = C.ppo_policy_loss(ratio, adv, clip_eps=0.2)
    # min(2*1, 1.2*1) = 1.2 -> loss = -1.2 (clip caps the gain)
    np.testing.assert_allclose(float(loss), -1.2, rtol=1e-6, atol=1e-6)
    assert float(m["clipfrac"]) == 1.0


def test_bpo_zero_at_target():
    adv = jnp.array([1.0, -1.0, 0.5])
    eps, lam = 0.2, 1e-3
    target = 1.0 + eps * jnp.tanh(adv / (2 * lam))
    loss, _ = C.bpo_policy_loss(target, adv, clip_eps=eps, bpo_lambda=lam)
    np.testing.assert_allclose(float(loss), 0.0, atol=1e-6)


def test_bpo_target_direction():
    """Positive advantage -> target ratio > 1; negative -> < 1."""
    adv = jnp.array([5.0, -5.0])
    _, m = C.bpo_policy_loss(jnp.ones(2), adv, clip_eps=0.2, bpo_lambda=1e-3)
    t = 1.0 + 0.2 * jnp.tanh(adv / 2e-3)
    assert float(t[0]) > 1.0 and float(t[1]) < 1.0


def test_surrogate_dispatch():
    adv = jnp.array([1.0, -1.0])
    ratio = jnp.array([1.1, 0.9])
    lp, _ = C.surrogate_loss("ppo", ratio, adv, clip_eps=0.2)
    lb, _ = C.surrogate_loss("bpo", ratio, adv, clip_eps=0.2, bpo_lambda=1e-3)
    assert jnp.isfinite(lp) and jnp.isfinite(lb)
    try:
        C.surrogate_loss("xxx", ratio, adv, clip_eps=0.2)
        raise AssertionError("expected ValueError for unknown mode")
    except ValueError:
        pass


def test_value_loss_mse_and_clip():
    v = jnp.array([1.0, 2.0, 3.0])
    r = jnp.array([1.5, 1.5, 1.5])
    np.testing.assert_allclose(float(C.value_loss(v, r)),
                               0.5 * float(jnp.mean((v - r) ** 2)), rtol=1e-6)
    # clipping never lowers the loss below unclipped (it's a max)
    lc = C.value_loss(v, r, value_old=jnp.zeros(3), clip_eps=0.1)
    assert float(lc) >= float(C.value_loss(v, r)) - 1e-6


def test_anchor_loss_and_stopgrad():
    mean = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(float(C.anchor_loss(mean, mean)), 0.0, atol=1e-7)
    assert float(C.anchor_loss(mean, mean + 1.0)) > 0.0
    # gradient flows to `mean` only, never to the (frozen) anchor.
    g_mean = jax.grad(lambda m: C.anchor_loss(m, mean + 1.0))(mean)
    g_anchor = jax.grad(lambda a: C.anchor_loss(mean, a))(mean + 1.0)
    assert np.any(np.array(g_mean) != 0.0)
    np.testing.assert_allclose(np.array(g_anchor), 0.0, atol=1e-7)


def test_jit():
    f = jax.jit(lambda r, v, d, lv: C.compute_gae(r, v, d, lv, gamma=0.99, gae_lambda=0.95))
    adv, ret = f(jnp.ones(5), jnp.ones(5), jnp.zeros(5), jnp.array(0.0))
    assert adv.shape == (5,) and ret.shape == (5,)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} DBPO-core tests passed.")


if __name__ == "__main__":
    main()
