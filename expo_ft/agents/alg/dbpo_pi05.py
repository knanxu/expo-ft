"""Wire the real pi0.5 JAX drift backbone into :class:`DBPOLearner` (Phase 3).

Builds the abstract ``drift_apply_fn(params, obs, z) -> (mean, cond_emb, value_feat)``
that the framework-agnostic learner consumes, by reconstructing the nnx pi0.5 model
from ``(graphdef, params)`` and calling its single-step drift sampler with hidden
features (``_sample_actions_drifting(return_hidden=True)``).

Key facts that make this drop-in (verified on a dummy pi0.5):
  * ``nnx.split(model)`` yields a graphdef (static) + an ``nnx.State`` whose leaves
    are plain arrays, so the State is a clean pytree -- ``optax`` operates on it
    directly (as ``pi05.py`` already relies on) and plain ``jax.grad`` flows through
    ``nnx.merge`` into the backbone.  Hence the actor is just a Flax ``TrainState``
    whose ``params`` is the ``nnx.State``, and the Phase-2 update path is reused.
  * The log-std head consumes ``cond_emb`` (pooled VLM prefix feature); the value
    head consumes the mean-pooled action-expert ``suffix_feat`` (RLinf critic design,
    docs/RLINF_ROBOTWIN_REFERENCE.md §4) with a stop-gradient input.

Freezing (LoRA) note: this differentiates the full ``nnx.State``.  Respecting
``config.trainable_filter`` (freeze the VLM, train only LoRA/expert) is a Phase-4
optimization; see dev doc.
"""

import jax
import jax.numpy as jnp
import optax
import flax.nnx as nnx
from flax.training.train_state import TrainState

import openpi.models.model as _model

from expo_ft.agents.alg.dbpo import DBPOLearner
from expo_ft.networks.dbpo_heads import LogStdHead, ValueHead

# rng is unused by the drift sampler when an explicit noise (z) is provided.
_UNUSED_KEY = jax.random.PRNGKey(0)


def _actor_apply_unused(*args, **kwargs):
    raise RuntimeError("actor.apply_fn is unused; the drift forward goes through drift_apply_fn")


def make_drift_apply_fn(model_def):
    """Return ``drift_apply_fn(params, obs, z, frozen) -> (mean, cond_emb, value_feat)``.

    ``params`` is the (trainable) pi0.5 ``nnx.State`` being differentiated; ``frozen``
    is the complementary non-trainable backbone state (VLM/SigLIP) or ``None`` for a
    full-finetune split.  ``obs`` is an :class:`Observation` or its dict; ``z`` is the
    drift latent ``(B, H, A)``.  Differentiable wrt ``params`` only.
    """

    def drift_apply_fn(params, obs, z, frozen):
        # ``frozen is None`` is a static (Python-level) branch at trace time.
        model = nnx.merge(model_def, params) if frozen is None else nnx.merge(model_def, params, frozen)
        observation = obs if isinstance(obs, _model.Observation) else _model.Observation.from_dict(obs)
        observation = _model.preprocess_observation(None, observation, train=False)
        mean, suffix_feat, cond_emb = model._sample_actions_drifting(
            _UNUSED_KEY, observation, z, return_hidden=True
        )
        # value_feat := mean-pooled action-expert suffix features (RLinf critic
        # design: V reads suffix_out, not cond_emb -- RL-validated on VLA, see
        # docs/RLINF_ROBOTWIN_REFERENCE.md §4).  The value head stop-gradients this
        # input (ValueHead.detach_input) so the critic loss never perturbs the
        # shared action expert -- matching RLinf's ``detach_critic_input``.
        value_feat = jnp.mean(suffix_feat, axis=-2)  # [B, H, width] -> [B, width]
        return mean, cond_emb, value_feat

    return drift_apply_fn


def build_dbpo_from_pi05(
    *,
    rng,
    model_def,
    actor_params,
    example_obs,
    example_z,
    action_horizon: int,
    action_dim: int,
    replan_steps: int,
    trainable_filter=None,
    actor_lr: float = 1e-5,
    logstd_lr: float = 3e-4,
    value_lr: float = 3e-4,
    max_grad_norm: float | None = None,
    logstd_kwargs: dict | None = None,
    **dbpo_kwargs,
) -> DBPOLearner:
    """Assemble a :class:`DBPOLearner` around an initialized pi0.5 nnx model.

    Args:
      model_def:    nnx graphdef from ``nnx.split(model)`` (or pi0.5 ``TrainState.model_def``).
      actor_params: full nnx ``State`` (or pi0.5 ``TrainState.params`` / ``ema_params``).
      example_obs / example_z: one batched observation + latent to probe head input dims.
      trainable_filter: optional nnx Filter selecting which params get an optimizer +
                        gradient.  The complement is carried frozen and shared by
                        actor & anchor.  **Essential for the real 3B model** -- adam
                        over the full state needs ~24GB; freezing the VLM/SigLIP and
                        training only the action expert/projections makes it feasible.
                        ``None`` = full finetune (only viable for tiny/dummy models).
      ``dbpo_kwargs``: forwarded to :meth:`DBPOLearner.create` (clip_eps, gamma,
                       gae_lambda, c_value, c_entropy, lambda_anchor, rl_surrogate,
                       bpo_lambda, n_real_dims, normalize_adv, value_clip_eps).
    """
    rng, k_ls, k_v = jax.random.split(rng, 3)
    drift_apply_fn = make_drift_apply_fn(model_def)

    # grad-norm clip（DBPO max_grad_norm）：每个 TrainState 各自裁剪（actor/logstd/value 独立组），
    # 与 DBPO 分别对 actor_ft / critic 调 clip_grad_norm_ 同精神。None → 不裁剪（旧行为，测试不受影响）。
    def _tx(lr):
        adam = optax.adam(lr)
        if max_grad_norm is not None and max_grad_norm > 0:
            return optax.chain(optax.clip_by_global_norm(max_grad_norm), adam)
        return adam

    if trainable_filter is None:
        trainable_params, frozen_params = actor_params, None
    else:
        trainable_params = actor_params.filter(trainable_filter)
        frozen_params = actor_params.filter(nnx.Not(trainable_filter))

    actor = TrainState.create(apply_fn=_actor_apply_unused, params=trainable_params, tx=_tx(actor_lr))

    _mean0, cond0, vfeat0 = drift_apply_fn(trainable_params, example_obs, example_z, frozen_params)

    ls = LogStdHead(action_horizon=action_horizon, action_dim=action_dim, **(logstd_kwargs or {}))
    logstd_head = TrainState.create(apply_fn=ls.apply, params=ls.init(k_ls, cond0)["params"],
                                    tx=_tx(logstd_lr))
    vh = ValueHead()
    value_head = TrainState.create(apply_fn=vh.apply, params=vh.init(k_v, vfeat0)["params"],
                                   tx=_tx(value_lr))

    return DBPOLearner.create(
        rng=rng, drift_apply_fn=drift_apply_fn, actor=actor, actor_frozen=frozen_params,
        logstd_head=logstd_head, value_head=value_head,
        action_horizon=action_horizon, action_dim=action_dim, replan_steps=replan_steps,
        **dbpo_kwargs,
    )
