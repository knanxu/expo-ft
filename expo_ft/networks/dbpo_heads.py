"""DBPO stochastic-adapter heads (Flax linen), kept in the same idiom as the other
``expo_ft.networks`` modules (``MLP`` + ``default_init`` + ``@nn.compact``).

Two small heads bolt the DBPO stochastic actor / critic onto the frozen drift
backbone, consuming the hidden features the drift sampler exposes
(``_sample_actions_drifting(return_hidden=True) -> (mean, suffix_feat, cond_emb)``):

  * :class:`LogStdHead` -- state-conditioned diagonal ``log sigma`` ``g_psi(c_theta(o))``
    (DBPO Eq. 44-45), fed by ``cond_emb`` (mean-pooled VLM prefix features).
  * :class:`ValueHead`  -- critic ``V_phi`` (DBPO Eq. 52), fed by the mean-pooled
    action-expert ``suffix_feat`` (RLinf's RL-validated critic-input design; see
    ``docs/RLINF_ROBOTWIN_REFERENCE.md`` §4).  Its input is ``stop_gradient``-ed
    (``detach_input``) so the critic loss never flows into the shared action expert
    -- mirroring RLinf's ``detach_critic_input``.

Both infer their input dim from the passed feature, so they work regardless of the
VLM / action-expert widths.
"""

from typing import Sequence

import jax
import flax.linen as nn
import jax.numpy as jnp

from expo_ft.networks import default_init
from expo_ft.networks.mlp import MLP


class LogStdHead(nn.Module):
    """State-conditioned diagonal log-std head ``g_psi`` (DBPO Eq. 44-45).

    Maps an observation feature (``cond_emb``) to ``log sigma`` of shape
    ``(..., action_horizon, action_dim)``, clipped to ``[log_std_min, log_std_max]``.

    The final layer is initialised with a zero kernel and a constant bias equal to
    ``log_std_init``, so at init ``log sigma == log_std_init`` everywhere (a
    controlled, state-independent starting scale) and state dependence is learned
    during training.
    """

    action_horizon: int
    action_dim: int
    hidden_dims: Sequence[int] = (256, 256)
    log_std_min: float = -5.0
    log_std_max: float = 0.0
    log_std_init: float = -2.0

    @nn.compact
    def __call__(self, cond_emb: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = MLP(self.hidden_dims, activate_final=True)(cond_emb, training=training)
        out_dim = self.action_horizon * self.action_dim
        log_std = nn.Dense(
            out_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.constant(self.log_std_init),
            name="OutLogStd",
        )(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return log_std.reshape(log_std.shape[:-1] + (self.action_horizon, self.action_dim))


class ValueHead(nn.Module):
    """Critic ``V_phi`` (DBPO Eq. 52): observation feature -> scalar value.

    Fed the mean-pooled action-expert ``suffix_feat`` (RLinf critic design).  When
    ``detach_input`` (default), the input is ``stop_gradient``-ed so the value loss
    only updates this head and never perturbs the shared action expert -- the JAX
    equivalent of RLinf's ``detach_critic_input=True``.  Returns the squeezed scalar
    of shape ``(...,)``.
    """

    hidden_dims: Sequence[int] = (256, 256)
    detach_input: bool = True

    @nn.compact
    def __call__(self, obs_feat: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        if self.detach_input:
            obs_feat = jax.lax.stop_gradient(obs_feat)
        x = MLP(self.hidden_dims, activate_final=True)(obs_feat, training=training)
        value = nn.Dense(1, kernel_init=default_init())(x)
        return jnp.squeeze(value, -1)
