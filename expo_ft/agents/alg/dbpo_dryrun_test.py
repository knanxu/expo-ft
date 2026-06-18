"""Phase-2 dry-run: end-to-end DBPOLearner on a tiny mock drift backbone (no pi0.5).

    python -m expo_ft.agents.alg.dbpo_dryrun_test

Validates the full on-policy loop -- rollout -> GAE -> PPO/BPO epochs -- and the
DBPO correctness centerpiece: because the rollout latent ``z`` is stored and
reused, the very first update's importance ratio is exactly 1.
"""

import jax

# DBPO's ratio relies on recomputing logp under the same params as rollout; on GPU
# the default fp32 matmul uses TF32 (~1e-3 rel. error) which makes the batched
# update-time forward differ from the B=1 rollout forward, nudging the "should be
# 1" first-update ratio to ~1.004.  Highest precision makes the z-reuse invariant
# exact and the test backend-independent.  (Dev doc: in the real learner this ratio
# noise is small and absorbed by PPO/BPO clipping; precision is a Phase 3/4 knob.)
jax.config.update("jax_default_matmul_precision", "highest")

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from expo_ft.agents.alg.dbpo import DBPOLearner, run_ppo_iteration
from expo_ft.data.rollout_buffer import RolloutBuffer
from expo_ft.networks.dbpo_heads import LogStdHead, ValueHead

# small shapes for a fast, CPU-friendly dry run
OBS_DIM, H, A, HE, FEAT = 10, 4, 6, 2, 16
T, NUM_MB, EPOCHS = 16, 4, 3


class MockDrift(nn.Module):
    """Stand-in for the pi0.5 drift backbone: (obs, z) -> (mean, cond_emb, value_feat)."""
    action_horizon: int
    action_dim: int
    feat_dim: int = FEAT

    @nn.compact
    def __call__(self, obs, z):
        b = obs.shape[0]
        h = nn.relu(nn.Dense(self.feat_dim)(obs))
        cond_emb = nn.Dense(self.feat_dim)(h)               # log-std head input
        x = jnp.concatenate([h, z.reshape(b, -1)], axis=-1)  # mean depends on obs AND z
        mean = nn.Dense(self.action_horizon * self.action_dim)(x)
        mean = mean.reshape(b, self.action_horizon, self.action_dim)
        value_feat = cond_emb                               # value head input (cond_emb, D §3.3)
        return mean, cond_emb, value_feat


def _build_learner(seed=0, surrogate="ppo"):
    rng = jax.random.PRNGKey(seed)
    rng, k_actor, k_ls, k_v = jax.random.split(rng, 4)

    drift = MockDrift(action_horizon=H, action_dim=A)
    obs0 = jnp.zeros((1, OBS_DIM))
    z0 = jnp.zeros((1, H, A))
    actor_vars = drift.init(k_actor, obs0, z0)
    _, cond0, vfeat0 = drift.apply(actor_vars, obs0, z0)

    def drift_apply_fn(params, obs, z, frozen=None):  # mock has no frozen backbone
        return drift.apply({"params": params}, obs, z)

    actor = TrainState.create(apply_fn=drift.apply, params=actor_vars["params"],
                              tx=optax.adam(3e-4))

    ls_head = LogStdHead(action_horizon=H, action_dim=A, log_std_init=-1.0)
    ls_params = ls_head.init(k_ls, cond0)["params"]
    logstd = TrainState.create(apply_fn=ls_head.apply, params=ls_params, tx=optax.adam(3e-4))

    v_head = ValueHead()
    v_params = v_head.init(k_v, vfeat0)["params"]
    value = TrainState.create(apply_fn=v_head.apply, params=v_params, tx=optax.adam(1e-3))

    return DBPOLearner.create(
        rng=rng, drift_apply_fn=drift_apply_fn, actor=actor,
        logstd_head=logstd, value_head=value,
        action_horizon=H, action_dim=A, replan_steps=HE,
        rl_surrogate=surrogate, c_entropy=0.0, lambda_anchor=1.0, n_real_dims=A,
    )


def _collect_rollout(learner, seed=1):
    buf = RolloutBuffer()
    rng = np.random.default_rng(seed)
    for t in range(T):
        obs = jnp.asarray(rng.normal(size=(1, OBS_DIM)), dtype=jnp.float32)
        action, z, logp, value, learner = learner.sample_actions(obs)
        reward = float(rng.normal())
        done = 1.0 if (t + 1) % 8 == 0 else 0.0  # an episode boundary mid-rollout
        buf.add(obs[0], z[0], action[0], float(logp[0]), float(value[0]), reward, done)
    buf.finalize(last_value=0.0, gamma=0.99, gae_lambda=0.95)
    return buf, learner


def test_rollout_buffer_shapes_and_gae():
    learner = _build_learner()
    buf, _ = _collect_rollout(learner)
    assert buf.z.shape == (T, H, A)
    assert buf.actions.shape == (T, H, A)
    assert buf.advantages.shape == (T,) and buf.returns.shape == (T,)
    # minibatches cover every index exactly once
    seen = np.concatenate([np.asarray(mb["logp_old"]) for mb in buf.iterate_minibatches(0, NUM_MB)])
    assert len(seen) == T


def test_first_update_ratio_is_one():
    """z reuse => logp_new == logp_old on the first update => ratio == 1 (Eq. 50)."""
    learner = _build_learner()
    buf, learner = _collect_rollout(learner)
    mb = next(buf.iterate_minibatches(0, NUM_MB))
    _, metrics = learner.update(mb)
    np.testing.assert_allclose(float(metrics["ppo/ratio_mean"]), 1.0, atol=1e-3)
    np.testing.assert_allclose(float(metrics["ppo/approx_kl"]), 0.0, atol=1e-4)


def test_full_ppo_run_is_finite_and_updates_params():
    learner = _build_learner()
    buf, learner = _collect_rollout(learner)
    p0 = jax.tree_util.tree_leaves(learner.actor.params)[0].copy()
    n_steps = 0
    for ep in range(EPOCHS):
        for mb in buf.iterate_minibatches(ep, NUM_MB):
            learner, metrics = learner.update(mb)
            for k, v in metrics.items():
                assert np.isfinite(float(v)), f"non-finite {k}={v}"
            n_steps += 1
    assert n_steps == EPOCHS * NUM_MB
    p1 = jax.tree_util.tree_leaves(learner.actor.params)[0]
    assert np.any(np.array(p0) != np.array(p1)), "actor params did not change"


def test_critic_fits_returns_on_fixed_batch():
    """Overfitting one minibatch should drive the value loss down."""
    learner = _build_learner()
    buf, learner = _collect_rollout(learner)
    mb = next(buf.iterate_minibatches(0, 1))  # single full-rollout batch
    _, m0 = learner.update(mb)
    for _ in range(60):
        learner, m = learner.update(mb)
    assert float(m["loss/value"]) < float(m0["loss/value"]), \
        f"value loss did not decrease: {float(m0['loss/value'])} -> {float(m['loss/value'])}"


def test_bpo_surrogate_runs():
    learner = _build_learner(surrogate="bpo")
    buf, learner = _collect_rollout(learner)
    for mb in buf.iterate_minibatches(0, NUM_MB):
        learner, metrics = learner.update(mb)
        assert np.isfinite(float(metrics["loss/total"]))
    assert "ppo/target_ratio_mean" in metrics  # bpo metrics surfaced


def test_critic_warmup_freezes_actor():
    """critic_warmup=True: only the value head steps; actor/logstd are held (DBPO warmup)."""
    learner = _build_learner()
    buf, learner = _collect_rollout(learner)
    a0 = jax.tree_util.tree_leaves(learner.actor.params)[0].copy()
    v0 = jax.tree_util.tree_leaves(learner.value_head.params)[0].copy()
    for mb in buf.iterate_minibatches(0, NUM_MB):
        learner, _ = learner.update(mb, critic_warmup=True)
    a1 = jax.tree_util.tree_leaves(learner.actor.params)[0]
    v1 = jax.tree_util.tree_leaves(learner.value_head.params)[0]
    np.testing.assert_allclose(np.array(a0), np.array(a1), atol=1e-7)  # actor frozen during warmup
    assert np.any(np.array(v0) != np.array(v1)), "value head should still update during warmup"


def test_run_ppo_iteration_stabilizers():
    """run_ppo_iteration honors critic_warmup (actor frozen) + target_kl early-stop, finite."""
    learner = _build_learner()
    buf, learner = _collect_rollout(learner)
    a0 = jax.tree_util.tree_leaves(learner.actor.params)[0].copy()
    learner, _ = run_ppo_iteration(learner, buf, ppo_epochs=EPOCHS, num_minibatches=NUM_MB,
                                   seed=0, critic_warmup=True)  # whole iteration in warmup
    a1 = jax.tree_util.tree_leaves(learner.actor.params)[0]
    np.testing.assert_allclose(np.array(a0), np.array(a1), atol=1e-7)
    # post-warmup with a tiny target_kl should early-stop and still return finite metrics.
    learner, metrics = run_ppo_iteration(learner, buf, ppo_epochs=EPOCHS, num_minibatches=NUM_MB,
                                         seed=1, target_kl=1e-6)
    assert np.isfinite(float(metrics["loss/total"]))


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} DBPO dry-run tests passed.")


if __name__ == "__main__":
    main()
