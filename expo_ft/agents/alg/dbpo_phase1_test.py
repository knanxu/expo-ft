"""Self-contained Phase-1 tests for the DBPO stochastic adapter.

The server venv has no pytest, so this runs as a plain script::

    python -m expo_ft.agents.alg.dbpo_phase1_test

It cross-checks :mod:`expo_ft.agents.alg.dbpo_utils` against tensorflow-probability
and validates the :mod:`expo_ft.networks.dbpo_heads` heads (shapes, init, clip).
"""

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp

from expo_ft.agents.alg import dbpo_utils as U
from expo_ft.networks.dbpo_heads import LogStdHead, ValueHead

tfd = tfp.distributions

# DROID pi0.5 shapes (config: expo_pi05_droid_lora_finetune_sft_cartesian_state).
B, H, A, HE = 4, 16, 32, 8  # batch, action_horizon, action_dim, replan_steps (H_e)


def _sample_inputs(seed=0):
    k = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(k, 3)
    mean = jax.random.normal(k1, (B, H, A))
    log_std = jax.random.uniform(k2, (B, H, A), minval=-2.0, maxval=0.5)
    actions = jax.random.normal(k3, (B, H, A))
    return mean, log_std, actions


def test_logprob_matches_tfp():
    mean, log_std, actions = _sample_inputs()
    got = U.gaussian_logprob(mean, log_std, actions, prefix_len=HE)
    elem = tfd.Normal(loc=mean, scale=jnp.exp(log_std)).log_prob(actions)  # (B,H,A)
    ref = elem[:, :HE, :].sum(axis=(-2, -1))
    np.testing.assert_allclose(np.array(got), np.array(ref), rtol=1e-5, atol=1e-5)
    assert got.shape == (B,)


def test_prefix_masking_ignores_suffix():
    """Perturbing mean/actions on the non-executed suffix must not change logp."""
    mean, log_std, actions = _sample_inputs()
    base = U.gaussian_logprob(mean, log_std, actions, prefix_len=HE)
    pert = jax.random.normal(jax.random.PRNGKey(99), (B, H, A)) * 5.0
    suffix = jnp.concatenate([jnp.zeros((HE, A)), jnp.ones((H - HE, A))], axis=0)
    mean2 = mean + pert * suffix  # only steps >= HE perturbed
    act2 = actions + pert * suffix
    after = U.gaussian_logprob(mean2, log_std, act2, prefix_len=HE)
    np.testing.assert_allclose(np.array(base), np.array(after), rtol=1e-6, atol=1e-6)


def test_prefix_start_offset():
    """prefix_start shifts the executed window (paper's T_o offset)."""
    mean, log_std, actions = _sample_inputs()
    elem = tfd.Normal(loc=mean, scale=jnp.exp(log_std)).log_prob(actions)
    got = U.gaussian_logprob(mean, log_std, actions, prefix_start=2, prefix_len=HE)
    ref = elem[:, 2:2 + HE, :].sum(axis=(-2, -1))
    np.testing.assert_allclose(np.array(got), np.array(ref), rtol=1e-5, atol=1e-5)


def test_dim_mask_restricts_dims():
    """dim_mask should restrict the per-dim sum to the real (non-padded) dims."""
    mean, log_std, actions = _sample_inputs()
    n_real = 7
    dim_mask = jnp.concatenate([jnp.ones(n_real), jnp.zeros(A - n_real)])
    got = U.gaussian_logprob(mean, log_std, actions, prefix_len=HE, dim_mask=dim_mask)
    elem = tfd.Normal(loc=mean, scale=jnp.exp(log_std)).log_prob(actions)
    ref = elem[:, :HE, :n_real].sum(axis=(-2, -1))
    np.testing.assert_allclose(np.array(got), np.array(ref), rtol=1e-5, atol=1e-5)


def test_entropy_matches_tfp():
    _, log_std, _ = _sample_inputs()
    got = U.gaussian_entropy(log_std, prefix_len=HE)
    elem = tfd.Normal(loc=jnp.zeros_like(log_std), scale=jnp.exp(log_std)).entropy()
    ref = elem[:, :HE, :].sum(axis=(-2, -1))
    np.testing.assert_allclose(np.array(got), np.array(ref), rtol=1e-5, atol=1e-5)


def test_sample_statistics():
    """Reparameterised sample mean/std should match (mu, sigma) over many draws."""
    mean = jnp.full((1, H, A), 0.7)
    log_std = jnp.full((1, H, A), jnp.log(0.3))
    rng = jax.random.PRNGKey(3)
    draws = jnp.stack([U.gaussian_sample(k, mean, log_std)
                       for k in jax.random.split(rng, 20000)])  # (N,1,H,A)
    np.testing.assert_allclose(float(draws.mean()), 0.7, atol=0.01)
    np.testing.assert_allclose(float(draws.std()), 0.3, atol=0.01)


def test_logstd_head_init_and_shape():
    head = LogStdHead(action_horizon=H, action_dim=A, log_std_init=-2.0,
                      log_std_min=-5.0, log_std_max=0.0)
    cond = jax.random.normal(jax.random.PRNGKey(1), (B, 2048))  # VLM prefix width
    params = head.init(jax.random.PRNGKey(2), cond)
    out = head.apply(params, cond)
    assert out.shape == (B, H, A), out.shape
    # zero final kernel + constant bias => log_std == log_std_init everywhere.
    np.testing.assert_allclose(np.array(out), -2.0, atol=1e-6)


def test_logstd_head_clip():
    head = LogStdHead(action_horizon=H, action_dim=A, log_std_init=2.0,
                      log_std_min=-5.0, log_std_max=0.0)  # init above max -> clip
    cond = jax.random.normal(jax.random.PRNGKey(1), (B, 1024))
    params = head.init(jax.random.PRNGKey(2), cond)
    out = head.apply(params, cond)
    np.testing.assert_allclose(np.array(out), 0.0, atol=1e-6)  # clipped to log_std_max
    assert float(out.max()) <= 0.0 + 1e-6 and float(out.min()) >= -5.0 - 1e-6


def test_value_head_shape():
    head = ValueHead()
    feat = jax.random.normal(jax.random.PRNGKey(1), (B, 2048))
    params = head.init(jax.random.PRNGKey(2), feat)
    out = head.apply(params, feat)
    assert out.shape == (B,), out.shape


def test_logprob_jit():
    """All utils must be jit-able (they run inside the learner's jitted update)."""
    mean, log_std, actions = _sample_inputs()
    f = jax.jit(lambda m, s, a: U.gaussian_logprob(m, s, a, prefix_len=HE))
    out = f(mean, log_std, actions)
    assert out.shape == (B,)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} Phase-1 DBPO tests passed.")


if __name__ == "__main__":
    main()
