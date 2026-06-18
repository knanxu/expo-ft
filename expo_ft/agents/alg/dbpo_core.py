"""Framework-agnostic DBPO/BPO algorithm core: GAE + surrogate/value losses.

These are pure JAX functions with no dependency on the drift backbone or the
learner, so they are exhaustively unit-testable on synthetic data and reused
verbatim once the real pi0.5 drift backbone is wired in (Phase 3).

Conventions:
  * A rollout is a length-``T`` sequence of *decision steps* (each decision
    executes an action chunk prefix of ``replan_steps``).  ``rewards``/``values``/
    ``dones`` are ``(T,)``; ``gamma`` is the per-decision discount chosen by the
    learner (e.g. ``discount ** replan_steps`` to match EXPO-FT's chunked TD).
  * ``done_t = 1`` marks an episode boundary after step ``t`` (zeros both the
    bootstrap and the GAE propagation across it).
"""

from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    last_value: jnp.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generalized Advantage Estimation (Schulman 2016), reverse ``lax.scan``.

    Args:
      rewards, values, dones: ``(T,)`` per decision step.
      last_value: scalar bootstrap ``V(s_T)`` for the step after the last one.
      gamma, gae_lambda: discount and GAE lambda.
    Returns:
      ``(advantages, returns)`` each ``(T,)``; ``returns = advantages + values``.
    """

    def scan_fn(carry, x):
        gae, next_value = carry
        reward, value, done = x
        nonterminal = 1.0 - done
        delta = reward + gamma * next_value * nonterminal - value
        gae = delta + gamma * gae_lambda * nonterminal * gae
        return (gae, value), gae

    init = (jnp.zeros((), dtype=values.dtype), last_value)
    _, advantages = jax.lax.scan(scan_fn, init, (rewards, values, dones), reverse=True)
    returns = advantages + values
    return advantages, returns


def normalize(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Whiten advantages (standard PPO trick); guarded against zero variance."""
    return (x - x.mean()) / (x.std() + eps)


def ppo_policy_loss(
    ratio: jnp.ndarray,
    advantages: jnp.ndarray,
    *,
    clip_eps: float,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """PPO clipped surrogate (DBPO Eq. 51). Returns ``(loss, metrics)``.

    Loss is the negated clipped objective (so gradient descent maximizes return).
    """
    unclipped = ratio * advantages
    clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss = -jnp.mean(jnp.minimum(unclipped, clipped))
    metrics = {
        "ratio_mean": jnp.mean(ratio),
        "ratio_max": jnp.max(ratio),
        "clipfrac": jnp.mean((jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32)),
        "approx_kl": jnp.mean(ratio - 1.0 - jnp.log(ratio + 1e-8)),
    }
    return loss, metrics


def bpo_policy_loss(
    ratio: jnp.ndarray,
    advantages: jnp.ndarray,
    *,
    clip_eps: float,
    bpo_lambda: float,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """BPO advantage-weighted TV surrogate (BPO Eq. 8/16); replaces PPO's clip.

    ``target_ratio = 1 + eps * tanh(A / (2*lambda))``; loss = ``|A| * |ratio - target|``.
    Uses the mean advantage in place of the soft-median (BPO ablation Fig. 9 shows
    parity), so no extra median network is needed initially.
    """
    target_ratio = 1.0 + clip_eps * jnp.tanh(advantages / (2.0 * bpo_lambda))
    loss = jnp.mean(jnp.abs(advantages) * jnp.abs(ratio - target_ratio))
    metrics = {
        "ratio_mean": jnp.mean(ratio),
        "ratio_max": jnp.max(ratio),
        "target_ratio_mean": jnp.mean(target_ratio),
        "approx_kl": jnp.mean(ratio - 1.0 - jnp.log(ratio + 1e-8)),
    }
    return loss, metrics


def surrogate_loss(
    mode: str,
    ratio: jnp.ndarray,
    advantages: jnp.ndarray,
    *,
    clip_eps: float,
    bpo_lambda: float = 1e-3,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Pluggable policy surrogate dispatcher (config ``rl_surrogate``)."""
    if mode == "ppo":
        return ppo_policy_loss(ratio, advantages, clip_eps=clip_eps)
    if mode == "bpo":
        return bpo_policy_loss(ratio, advantages, clip_eps=clip_eps, bpo_lambda=bpo_lambda)
    raise ValueError(f"unknown surrogate mode: {mode!r} (expected 'ppo' or 'bpo')")


def value_loss(
    value_pred: jnp.ndarray,
    returns: jnp.ndarray,
    *,
    value_old: Optional[jnp.ndarray] = None,
    clip_eps: Optional[float] = None,
) -> jnp.ndarray:
    """Critic loss (DBPO Eq. 52): ``0.5 * mean((V - R)^2)``.

    If ``value_old`` and ``clip_eps`` are given, applies PPO-style value clipping
    (max of clipped/unclipped squared error) for stability.
    """
    err = value_pred - returns
    if value_old is not None and clip_eps is not None:
        v_clipped = value_old + jnp.clip(value_pred - value_old, -clip_eps, clip_eps)
        err_clipped = v_clipped - returns
        return 0.5 * jnp.mean(jnp.maximum(err ** 2, err_clipped ** 2))
    return 0.5 * jnp.mean(err ** 2)


def anchor_loss(mean: jnp.ndarray, anchor_mean: jnp.ndarray) -> jnp.ndarray:
    """Stay-close-to-pretrained-drift regularizer (DBPO Eq. 54).

    Aligned with the DBPO reference (`ppo_adapter.py`: ``F.mse_loss(new, old)``):
    **per-element MSE** over the full chunk (H x d_a), with the same latent ``z``;
    ``anchor_mean`` is computed under the frozen Stage-1 params (stop-gradient).
    The mse (mean, not sum) is dimension-normalized, so ``lambda_anchor=1.0`` (DBPO
    ``anchor_loss_coeff``) transfers directly.
    """
    return jnp.mean((mean - jax.lax.stop_gradient(anchor_mean)) ** 2)
