"""Phase-3 integration test: DBPOLearner on the REAL pi0.5 JAX drift backbone.

Uses the tiny ``dummy`` gemma variant so the full path (nnx merge -> preprocess ->
drift sampler -> heads -> PPO update with frozen anchor) runs on CPU in seconds.
Verifies gradients actually flow into the pi0.5 actor params.

    python -m expo_ft.agents.alg.dbpo_pi05_test
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # heavy integration test: keep off the GPU

import jax
# DBPO ratio fidelity: recompute must match rollout (see dbpo_dryrun_test).
jax.config.update("jax_default_matmul_precision", "highest")

import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx

import openpi.models.pi0_config as pi0_config
import openpi.shared.nnx_utils as nnx_utils

from expo_ft.agents.alg.dbpo_pi05 import build_dbpo_from_pi05, make_drift_apply_fn

# Train only the action projections; freeze the VLM/SigLIP backbone (real-model setup).
_TRAINABLE_FILTER = nnx_utils.PathRegex(".*action_.*proj.*")

H, A, HE, B = 4, 8, 2, 3


def _dummy_pi05(action_horizon=H, action_dim=A):
    cfg = pi0_config.Pi0Config(
        paligemma_variant="dummy", action_expert_variant="dummy",
        pi05=True, use_drifting_loss=True,
        action_horizon=action_horizon, action_dim=action_dim,
    )
    model = cfg.create(jax.random.PRNGKey(0))
    graphdef, state = nnx.split(model)
    return cfg, graphdef, state


def test_drift_apply_contract_shapes():
    cfg, gd, state = _dummy_pi05()
    f = make_drift_apply_fn(gd)
    obs = cfg.fake_obs(B)
    z = jnp.zeros((B, cfg.action_horizon, cfg.action_dim))
    mean, cond_emb, value_feat = f(state, obs, z, None)  # full-finetune: no frozen split
    assert mean.shape == (B, cfg.action_horizon, cfg.action_dim)
    assert cond_emb.ndim == 2 and cond_emb.shape[0] == B          # (B, prefix_width)
    assert value_feat.shape == cond_emb.shape


def _build_learner_and_minibatch(surrogate="ppo"):
    cfg, gd, state = _dummy_pi05()
    obs = cfg.fake_obs(B)
    z_probe = jax.random.normal(jax.random.PRNGKey(2), (B, H, A))
    learner = build_dbpo_from_pi05(
        rng=jax.random.PRNGKey(3), model_def=gd, actor_params=state,
        example_obs=obs, example_z=z_probe,
        action_horizon=H, action_dim=A, replan_steps=HE,
        rl_surrogate=surrogate, lambda_anchor=1.0, c_entropy=0.0, n_real_dims=A,
    )
    # one rollout-style step to get a self-consistent (z, action, logp, value)
    action, z, logp, value, learner = learner.sample_actions(obs)
    rng = np.random.default_rng(0)
    mb = {
        "obs": obs.to_dict(),
        "z": z,
        "actions": action,
        "logp_old": logp,
        "value_old": value,
        "advantages": jnp.asarray(rng.normal(size=B), dtype=jnp.float32),
        "returns": jnp.asarray(rng.normal(size=B), dtype=jnp.float32),
    }
    return learner, mb


def test_update_flows_grad_into_pi05_actor():
    learner, mb = _build_learner_and_minibatch("ppo")
    p_before = jax.tree_util.tree_leaves(learner.actor.params)
    new_learner, metrics = learner.update(mb)
    for k, v in metrics.items():
        assert np.isfinite(float(v)), f"non-finite metric {k}={v}"
    assert float(metrics["grad_norm/actor"]) > 0.0, "no gradient reached the pi0.5 actor"
    # z reuse => first-update ratio ~ 1 (real backbone, highest precision)
    np.testing.assert_allclose(float(metrics["ppo/ratio_mean"]), 1.0, atol=1e-2)
    p_after = jax.tree_util.tree_leaves(new_learner.actor.params)
    changed = any(np.any(np.array(a) != np.array(b)) for a, b in zip(p_before, p_after))
    assert changed, "pi0.5 actor params did not update"


def test_anchor_keeps_pretrained_frozen():
    """anchor_params must remain the initial snapshot after an update (frozen theta-bar)."""
    learner, mb = _build_learner_and_minibatch("ppo")
    anchor_before = jax.tree_util.tree_leaves(learner.anchor_params)
    new_learner, _ = learner.update(mb)
    anchor_after = jax.tree_util.tree_leaves(new_learner.anchor_params)
    for a, b in zip(anchor_before, anchor_after):
        np.testing.assert_array_equal(np.array(a), np.array(b))


def test_bpo_runs_on_real_backbone():
    learner, mb = _build_learner_and_minibatch("bpo")
    _, metrics = learner.update(mb)
    assert np.isfinite(float(metrics["loss/total"]))
    assert "ppo/target_ratio_mean" in metrics


def test_trainable_filter_freezes_backbone():
    """With a trainable_filter, only the filtered params are optimized; the frozen
    backbone (carried in actor_frozen) is shared by actor+anchor and never changes.
    This is what makes the real 3B model fit (adam over a small subset)."""
    cfg, gd, state = _dummy_pi05()
    obs = cfg.fake_obs(B)
    z_probe = jax.random.normal(jax.random.PRNGKey(2), (B, H, A))
    learner = build_dbpo_from_pi05(
        rng=jax.random.PRNGKey(3), model_def=gd, actor_params=state,
        example_obs=obs, example_z=z_probe, action_horizon=H, action_dim=A, replan_steps=HE,
        trainable_filter=_TRAINABLE_FILTER, rl_surrogate="ppo", n_real_dims=A,
    )
    # optimizer covers only the small trainable subset
    n_trainable = sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(learner.actor.params))
    n_frozen = sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(learner.actor_frozen))
    assert n_trainable < n_frozen and n_frozen > 1_000_000, (n_trainable, n_frozen)

    action, z, logp, value, learner = learner.sample_actions(obs)
    rng = np.random.default_rng(0)
    mb = {"obs": obs.to_dict(), "z": z, "actions": action, "logp_old": logp, "value_old": value,
          "advantages": jnp.asarray(rng.normal(size=B), jnp.float32),
          "returns": jnp.asarray(rng.normal(size=B), jnp.float32)}
    frozen_before = jax.tree_util.tree_leaves(learner.actor_frozen)
    trainable_before = jax.tree_util.tree_leaves(learner.actor.params)
    new_learner, metrics = learner.update(mb)
    assert float(metrics["grad_norm/actor"]) > 0.0
    # frozen backbone untouched
    for a, b in zip(frozen_before, jax.tree_util.tree_leaves(new_learner.actor_frozen)):
        np.testing.assert_array_equal(np.array(a), np.array(b))
    # trainable subset moved
    moved = any(np.any(np.array(a) != np.array(b))
                for a, b in zip(trainable_before, jax.tree_util.tree_leaves(new_learner.actor.params)))
    assert moved, "trainable action projections did not update"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} DBPO×pi0.5 integration tests passed.")


if __name__ == "__main__":
    main()
