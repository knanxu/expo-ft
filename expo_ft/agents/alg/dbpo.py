"""DBPOLearner: on-policy (PPO/BPO) RL on a single-step drift actor.

Wiring of the Phase-1 stochastic adapter (``dbpo_utils``) and the Phase-2 algorithm
core (``dbpo_core``) into a Flax ``struct.PyTreeNode`` learner.  The drift backbone
is held behind an abstract ``drift_apply_fn(params, obs, z) -> (mean, cond_emb,
value_feat)`` so the same learner runs against a tiny mock backbone now and the
real pi0.5 drift backbone after Phase 3 (only :meth:`create` differs).

Differentiated parameter groups:
  * ``actor``       -- drift backbone params ``theta`` (Gaussian mean ``mu_theta(o,z)``)
  * ``logstd_head`` -- ``psi`` (state-conditioned ``log sigma``)
  * ``value_head``  -- ``phi`` (critic ``V_phi(o)``)
``anchor_params`` is a frozen snapshot of ``theta`` at init (DBPO anchor, Eq. 54).

NOTE: this is on-policy, so its ``update`` consumes a rollout minibatch (not the
off-policy ``(agent, batch, utd_ratio)`` signature of ``AgentLearner``); the
train-loop adapter is Phase 3.  Kept as a plain ``struct.PyTreeNode`` for now.
"""

from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState

from expo_ft.agents.alg import dbpo_core as C
from expo_ft.agents.alg import dbpo_utils as U


def _leading_batch(obs: Any) -> int:
    return jax.tree_util.tree_leaves(obs)[0].shape[0]


@partial(
    jax.jit,
    static_argnames=(
        "drift_apply_fn", "logstd_apply_fn", "value_apply_fn",
        "replan_steps", "clip_eps", "c_value", "c_entropy", "lambda_anchor",
        "rl_surrogate", "bpo_lambda", "n_real_dims", "normalize_adv", "value_clip_eps",
        "normalize_dims", "critic_warmup",
    ),
)
def _dbpo_update_step(
    actor_ts, logstd_ts, value_ts, anchor_params, frozen_params, mb,
    *, drift_apply_fn, logstd_apply_fn, value_apply_fn,
    replan_steps, clip_eps, c_value, c_entropy, lambda_anchor,
    rl_surrogate, bpo_lambda, n_real_dims, normalize_adv, value_clip_eps,
    normalize_dims, critic_warmup=False,
):
    obs, z, actions = mb["obs"], mb["z"], mb["actions"]
    logp_old, value_old = mb["logp_old"], mb["value_old"]
    adv, ret = mb["advantages"], mb["returns"]
    if normalize_adv:
        adv = C.normalize(adv)

    dim_mask = None
    if n_real_dims is not None:
        dim_mask = (jnp.arange(actions.shape[-1]) < n_real_dims).astype(actions.dtype)

    # Anchor mean: frozen Stage-1 params, same latent z, no gradient (Eq. 54).
    # ``frozen_params`` carries the non-trainable backbone (VLM/SigLIP) that is
    # shared by actor & anchor and never differentiated (None for the full-finetune
    # / mock path).
    anchor_mean = jax.lax.stop_gradient(drift_apply_fn(anchor_params, obs, z, frozen_params)[0])

    def loss_fn(params):
        mean, cond_emb, value_feat = drift_apply_fn(params["actor"], obs, z, frozen_params)
        log_std = logstd_apply_fn({"params": params["logstd"]}, cond_emb)
        value_pred = value_apply_fn({"params": params["value"]}, value_feat)

        logp_new = U.gaussian_logprob(mean, log_std, actions,
                                      prefix_len=replan_steps, dim_mask=dim_mask,
                                      normalize_dims=normalize_dims)
        ratio = jnp.exp(logp_new - logp_old)  # latent z reused -> valid ratio (Eq. 50)
        p_loss, p_metrics = C.surrogate_loss(rl_surrogate, ratio, adv,
                                             clip_eps=clip_eps, bpo_lambda=bpo_lambda)
        v_loss = C.value_loss(value_pred, ret, value_old=value_old, clip_eps=value_clip_eps)
        ent = jnp.mean(U.gaussian_entropy(log_std, prefix_len=replan_steps, dim_mask=dim_mask,
                                          normalize_dims=normalize_dims))
        a_loss = C.anchor_loss(mean, anchor_mean)  # full chunk (DBPO mse over H x d_a)

        total = p_loss + c_value * v_loss - c_entropy * ent + lambda_anchor * a_loss
        metrics = {
            "loss/total": total, "loss/policy": p_loss, "loss/value": v_loss,
            "loss/entropy": ent, "loss/anchor": a_loss,
            "logstd/mean": jnp.mean(log_std),
            **{f"ppo/{k}": v for k, v in p_metrics.items()},
        }
        return total, metrics

    params = {"actor": actor_ts.params, "logstd": logstd_ts.params, "value": value_ts.params}
    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    # critic warmup（DBPO n_critic_warmup_itr）：前若干 iter 只更 value，actor/logstd 不动。
    # critic_warmup 为静态值，分支在 trace 期解析（warmup 结束时仅重编译一次）。
    if not critic_warmup:
        actor_ts = actor_ts.apply_gradients(grads=grads["actor"])
        logstd_ts = logstd_ts.apply_gradients(grads=grads["logstd"])
    value_ts = value_ts.apply_gradients(grads=grads["value"])
    metrics["grad_norm/actor"] = optax.global_norm(grads["actor"])
    metrics["grad_norm/logstd"] = optax.global_norm(grads["logstd"])
    metrics["grad_norm/value"] = optax.global_norm(grads["value"])
    return actor_ts, logstd_ts, value_ts, metrics


class DBPOLearner(struct.PyTreeNode):
    rng: jax.Array
    drift_apply_fn: Callable = struct.field(pytree_node=False)
    actor: TrainState
    anchor_params: Any
    actor_frozen: Any  # non-trainable backbone (VLM/SigLIP) shared by actor+anchor; None = full finetune
    logstd_head: TrainState
    value_head: TrainState
    # static hyperparameters
    action_horizon: int = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False)
    replan_steps: int = struct.field(pytree_node=False)
    clip_eps: float = struct.field(pytree_node=False)
    gamma: float = struct.field(pytree_node=False)
    gae_lambda: float = struct.field(pytree_node=False)
    c_value: float = struct.field(pytree_node=False)
    c_entropy: float = struct.field(pytree_node=False)
    lambda_anchor: float = struct.field(pytree_node=False)
    rl_surrogate: str = struct.field(pytree_node=False)
    bpo_lambda: float = struct.field(pytree_node=False)
    n_real_dims: Optional[int] = struct.field(pytree_node=False)
    normalize_adv: bool = struct.field(pytree_node=False)
    value_clip_eps: Optional[float] = struct.field(pytree_node=False)
    normalize_dims: bool = struct.field(pytree_node=False)  # DBPO normalize_act_space_dimension

    @classmethod
    def create(
        cls, *, rng, drift_apply_fn, actor, logstd_head, value_head, actor_frozen=None,
        action_horizon, action_dim, replan_steps,
        clip_eps=0.2, gamma=0.99, gae_lambda=0.95, c_value=0.5, c_entropy=0.0,
        lambda_anchor=1.0, rl_surrogate="ppo", bpo_lambda=1e-3,
        n_real_dims=None, normalize_adv=True, value_clip_eps=None,
        normalize_dims=False,
    ) -> "DBPOLearner":
        return cls(
            rng=rng, drift_apply_fn=drift_apply_fn, actor=actor,
            anchor_params=actor.params, actor_frozen=actor_frozen,
            logstd_head=logstd_head, value_head=value_head,
            action_horizon=action_horizon, action_dim=action_dim, replan_steps=replan_steps,
            clip_eps=clip_eps, gamma=gamma, gae_lambda=gae_lambda, c_value=c_value,
            c_entropy=c_entropy, lambda_anchor=lambda_anchor, rl_surrogate=rl_surrogate,
            bpo_lambda=bpo_lambda, n_real_dims=n_real_dims, normalize_adv=normalize_adv,
            value_clip_eps=value_clip_eps, normalize_dims=normalize_dims,
        )

    def _dim_mask(self):
        if self.n_real_dims is None:
            return None
        return (jnp.arange(self.action_dim) < self.n_real_dims).astype(jnp.float32)

    def sample_actions(self, obs: Any) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, "DBPOLearner"]:
        """Rollout-time action sampling for a batch of observations.

        Returns ``(action, z, logp, value, new_self)`` where ``action``/``z`` are
        ``(B, H, A)``, ``logp``/``value`` are ``(B,)``.  ``z`` MUST be stored in the
        rollout buffer so the ratio can be recomputed under updated params.
        """
        rng, zk, ak = jax.random.split(self.rng, 3)
        b = _leading_batch(obs)
        z = jax.random.normal(zk, (b, self.action_horizon, self.action_dim))
        mean, cond_emb, value_feat = self.drift_apply_fn(self.actor.params, obs, z, self.actor_frozen)
        log_std = self.logstd_head.apply_fn({"params": self.logstd_head.params}, cond_emb)
        value = self.value_head.apply_fn({"params": self.value_head.params}, value_feat)
        action = U.gaussian_sample(ak, mean, log_std)
        logp = U.gaussian_logprob(mean, log_std, action,
                                  prefix_len=self.replan_steps, dim_mask=self._dim_mask(),
                                  normalize_dims=self.normalize_dims)
        return action, z, logp, value, self.replace(rng=rng)

    def update(self, minibatch: Dict[str, Any], *, critic_warmup: bool = False
              ) -> Tuple["DBPOLearner", Dict[str, jnp.ndarray]]:
        """One gradient step on a rollout minibatch (recomputes ratio with stored z).

        ``critic_warmup`` (DBPO ``n_critic_warmup_itr``): when True, only the value head
        steps; actor/logstd are held (lets the critic fit before the policy moves).
        """
        actor, logstd, value, metrics = _dbpo_update_step(
            self.actor, self.logstd_head, self.value_head, self.anchor_params,
            self.actor_frozen, minibatch,
            drift_apply_fn=self.drift_apply_fn,
            logstd_apply_fn=self.logstd_head.apply_fn,
            value_apply_fn=self.value_head.apply_fn,
            replan_steps=self.replan_steps, clip_eps=self.clip_eps,
            c_value=self.c_value, c_entropy=self.c_entropy, lambda_anchor=self.lambda_anchor,
            rl_surrogate=self.rl_surrogate, bpo_lambda=self.bpo_lambda,
            n_real_dims=self.n_real_dims, normalize_adv=self.normalize_adv,
            value_clip_eps=self.value_clip_eps, normalize_dims=self.normalize_dims,
            critic_warmup=critic_warmup,
        )
        return self.replace(actor=actor, logstd_head=logstd, value_head=value), metrics


def run_ppo_iteration(learner: "DBPOLearner", buffer, *, ppo_epochs: int,
                      num_minibatches: int, seed: int,
                      target_kl: Optional[float] = None,
                      critic_warmup: bool = False) -> Tuple["DBPOLearner", Dict[str, Any]]:
    """Drive one PPO/BPO iteration over a finalized rollout (the on-policy loop core).

    Consumes ``buffer`` (already ``finalize()``-d with GAE) over ``ppo_epochs`` x
    ``num_minibatches`` gradient steps and returns ``(updated_learner, last_metrics)``.
    This is the entry point train_pi_robo* calls each iteration: collect a rollout ->
    ``buffer.finalize(...)`` -> ``run_ppo_iteration(...)`` -> publish new params.

    In-loop stabilizers (DBPO-faithful):
      * ``target_kl``: stop the iteration once minibatch ``approx_kl`` exceeds it
        (DBPO ``target_kl=0.02``); ``None`` disables.
      * ``critic_warmup``: forwarded to :meth:`update` (value-only steps this iteration).
    (grad-norm clipping lives in the optimizers; see ``build_dbpo_from_pi05(max_grad_norm=...)``.)
    """
    metrics: Dict[str, Any] = {}
    stopped = False
    for epoch in range(ppo_epochs):
        for mb in buffer.iterate_minibatches(seed + epoch, num_minibatches):
            learner, metrics = learner.update(mb, critic_warmup=critic_warmup)
            if target_kl is not None and float(metrics.get("ppo/approx_kl", 0.0)) > target_kl:
                metrics["ppo/kl_early_stop"] = 1.0
                stopped = True
                break
        if stopped:
            break
    return learner, metrics
