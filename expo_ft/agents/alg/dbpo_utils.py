"""Gaussian stochastic-adapter utilities for DBPO (Drift-Based Policy Optimization).

DBPO turns the deterministic single-step drift output ``mu_theta(o, z)`` into a
stochastic actor ``pi(x | o, z) = N(x; mu, diag(sigma^2))`` so that PPO/BPO can
estimate likelihood ratios.  Per the DBPO paper (Eq. 48 / 57) the executed-prefix
log-likelihood is the sum of per-step, per-dim diagonal-Gaussian log-densities
over ONLY the executed steps of the action chunk::

    log pi(x_exec | o, z) = sum_{h in exec} sum_{m} log N(a_{h,m}; mu_{h,m}, sigma_{h,m}^2)

In EXPO-FT the executed prefix is ``action_chunk[:replan_steps]`` (start index 0,
no observation-history offset), so ``prefix_len = replan_steps`` and
``prefix_start = 0``.

All functions are pure (jit-friendly).  The chunk arrays use the convention
``(..., action_horizon, action_dim)``; ``log_std`` is expected at that same shape
(the :class:`~expo_ft.networks.dbpo_heads.LogStdHead` emits it so), though
``gaussian_logprob`` also accepts any ``log_std`` broadcastable to it.
"""

import math
from typing import Optional

import jax
import jax.numpy as jnp

_LOG2PI = math.log(2.0 * math.pi)


def step_mask(horizon: int, prefix_len: Optional[int], prefix_start: int = 0,
              dtype=jnp.float32) -> jnp.ndarray:
    """Mask over the step axis selecting ``[prefix_start, prefix_start + prefix_len)``.

    ``prefix_len=None`` selects every step from ``prefix_start`` onward.  Returns a
    ``(horizon,)`` array of 0/1 in ``dtype``.
    """
    idx = jnp.arange(horizon)
    if prefix_len is None:
        mask = idx >= prefix_start
    else:
        mask = (idx >= prefix_start) & (idx < prefix_start + prefix_len)
    return mask.astype(dtype)


def gaussian_logprob(
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
    actions: jnp.ndarray,
    *,
    prefix_len: Optional[int] = None,
    prefix_start: int = 0,
    dim_mask: Optional[jnp.ndarray] = None,
    normalize_dims: bool = False,
) -> jnp.ndarray:
    """Executed-prefix diagonal-Gaussian log-likelihood (DBPO Eq. 48 / 57).

    Args:
      mean:    ``(..., H, A)`` Gaussian mean (the drift output ``mu_theta(o, z)``).
      log_std: broadcastable to ``(..., H, A)`` (state-conditioned ``log sigma``).
      actions: ``(..., H, A)`` the (executed) action chunk.
      prefix_len:   number of executed steps ``H_e``; ``None`` = all steps.
      prefix_start: first executed step index (EXPO-FT uses ``0``).
      dim_mask:     optional ``(A,)`` 0/1 mask restricting the sum to real
                    (non-padded) action dims; padded dims cancel in the PPO ratio
                    but masking keeps entropy / logp numerically clean.
      normalize_dims: if True, divide the summed log-prob by the number of credited
                    coordinates ``H_e * (#real dims)`` -> per-coordinate *mean* log-prob
                    (DBPO ``normalize_act_space_dimension=True``, its default).  Makes
                    the PPO/BPO ratio ``exp((1/N) sum delta_j)`` dimension-invariant
                    (Var[log r] ~ 1/N), so a high-dim chunk (pi0.5 H*d_a=700) behaves
                    like the low-dim regime the hyperparameters were tuned for.  Note:
                    this is the geometric-mean ratio, not the exact joint ratio of
                    paper Eq. 50 -- DBPO trades that exactness for high-dim stability.

    Returns:
      ``(...,)`` per batch element: summed (or per-coordinate-mean) executed-prefix log-prob.
    """
    inv_std = jnp.exp(-log_std)
    z = (actions - mean) * inv_std
    elem = -0.5 * (z ** 2) - log_std - 0.5 * _LOG2PI  # (..., H, A), broadcast on log_std
    return _masked_sum(elem, mean.shape[-2], prefix_len, prefix_start, dim_mask,
                       normalize=normalize_dims)


def gaussian_entropy(
    log_std: jnp.ndarray,
    *,
    prefix_len: Optional[int] = None,
    prefix_start: int = 0,
    dim_mask: Optional[jnp.ndarray] = None,
    normalize_dims: bool = False,
) -> jnp.ndarray:
    """Summed diagonal-Gaussian differential entropy over the executed prefix (Eq. 53).

    ``log_std`` must be ``(..., H, A)`` (full per-element shape).  Returns ``(...,)``.
    Per-element entropy of ``N(mu, sigma^2)`` is ``0.5*log(2*pi*e) + log sigma``.
    ``normalize_dims`` mirrors :func:`gaussian_logprob` (per-coordinate mean) so the
    entropy bonus stays on the same scale as the ratio when DBPO's dimension
    normalization is on -- keeps ``c_entropy`` transferable across action dims.
    """
    elem = 0.5 * (_LOG2PI + 1.0) + log_std  # (..., H, A)
    return _masked_sum(elem, log_std.shape[-2], prefix_len, prefix_start, dim_mask,
                       normalize=normalize_dims)


def gaussian_sample(rng, mean: jnp.ndarray, log_std: jnp.ndarray) -> jnp.ndarray:
    """Reparameterised sample ``x = mu + sigma * eps``, ``eps ~ N(0, I)``.

    ``log_std`` broadcastable to ``mean``.  Note this exploration noise is separate
    from the drift latent ``z`` (which produces ``mean``); for PPO/BPO only the
    sampled ``x`` and its old log-prob are stored, while ``z`` must also be stored
    so the mean can be recomputed under updated params (DBPO Eq. 47/50).
    """
    eps = jax.random.normal(rng, mean.shape, dtype=mean.dtype)
    return mean + jnp.exp(log_std) * eps


def _masked_sum(elem, horizon, prefix_len, prefix_start, dim_mask, normalize=False):
    """Apply optional dim mask + executed-step mask, then sum over (step, dim) axes.

    If ``normalize``, divide by the number of credited coordinates
    ``(#executed steps) * (#real dims)`` -> per-coordinate mean (DBPO
    ``normalize_act_space_dimension``).
    """
    n_dims = elem.shape[-1]
    if dim_mask is not None:
        elem = elem * dim_mask  # broadcasts over the last (A) axis
        n_dims = jnp.sum(dim_mask)
    smask = step_mask(horizon, prefix_len, prefix_start, dtype=elem.dtype)  # (H,)
    elem = elem * smask[:, None]  # broadcasts over (..., H, A)
    total = elem.sum(axis=(-2, -1))
    if normalize:
        total = total / (jnp.sum(smask) * n_dims + 1e-8)
    return total
