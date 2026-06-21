"""SpeedTuneLearner: branching Rainbow-DQN on a frozen VLA's suffix features.

两阶段解耦的加速模块（与 EXPO/BC/DBPO 平级，零侵入）。VLA 冻结，``suffix_feat``
（mean-pooled action-expert 特征）在 rollout 时算好并缓存进 ``SpeedTuneReplayBuffer``，
所以本 learner 是**纯 DQN**——不在 ``update`` 里重跑 3B VLA 前向，只对 Q 网络求梯度。

Rainbow 子集（计划默认）：
  * double DQN  : next-action 用 online 网选，分布用 target 网取（防过估计）。
  * dueling     : 见 ``BranchingRainbowQ``。
  * distributional C51 : 每 head 独立的 value 分布 + categorical projection。
  * n-step      : 已在 buffer 聚合（``reward``=Σγᵏr，``discount``=γⁿ·(1-done)）。
  * PER         : ``is_weights`` 加权 loss，返回 per-sample priority 回写 sum-tree。
  * branching (BDQ) : 每速度变量一个 head，跨 head TD loss 求和。
NoisyNet 未实现（后置）；探索用 epsilon-greedy（``sample_action_idxs``）。
"""

from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState

from expo_ft.networks.rainbow_dqn import (
    BranchingRainbowQ,
    categorical_projection,
    expected_q,
    greedy_action_idxs,
    make_support,
)


@partial(jax.jit, static_argnames=("q_apply_fn", "v_min", "v_max", "n_heads"))
def _speedtune_update_step(
    q_ts: TrainState,
    target_params: Any,
    batch: Dict[str, jnp.ndarray],
    support: jnp.ndarray,
    *,
    q_apply_fn,
    v_min: float,
    v_max: float,
    n_heads: int,
):
    feat, next_feat = batch["feat"], batch["next_feat"]
    actions = batch["action_idxs"]          # [B, n_heads]
    reward, discount = batch["reward"], batch["discount"]  # [B], [B]
    is_w = batch["is_weights"]              # [B]
    B = feat.shape[0]
    bidx = jnp.arange(B)

    # ---- target 分布（无梯度，double DQN）-------------------------------
    next_online = q_apply_fn({"params": q_ts.params}, next_feat)   # list[B,Nᵢ,atoms]
    next_target = q_apply_fn({"params": target_params}, next_feat)
    m_list = []
    for i in range(n_heads):
        next_a = jnp.argmax(expected_q(next_online[i], support), axis=-1)  # [B] online 选动作
        next_dist = next_target[i][bidx, next_a]                          # [B, atoms] target 取分布
        m_i = categorical_projection(next_dist, reward, discount, support, v_min, v_max)
        m_list.append(jax.lax.stop_gradient(m_i))

    # ---- online loss（跨 head cross-entropy 求和）------------------------
    def loss_fn(params):
        online = q_apply_fn({"params": params}, feat)  # list[B,Nᵢ,atoms]
        per_sample = jnp.zeros((B,))
        for i in range(n_heads):
            p_taken = online[i][bidx, actions[:, i]]                 # [B, atoms]
            ce = -jnp.sum(m_list[i] * jnp.log(p_taken), axis=-1)     # [B]  C51 cross-entropy
            per_sample = per_sample + ce
        loss = jnp.mean(is_w * per_sample)
        return loss, per_sample

    (loss, per_sample), grads = jax.value_and_grad(loss_fn, has_aux=True)(q_ts.params)
    q_ts = q_ts.apply_gradients(grads=grads)
    metrics = {
        "loss/dqn": loss,
        "q/grad_norm": optax.global_norm(grads),
        "q/td_priority_mean": jnp.mean(per_sample),
    }
    return q_ts, per_sample, metrics


class SpeedTuneLearner(struct.PyTreeNode):
    rng: jax.Array
    q_net: TrainState
    target_params: Any
    support: jnp.ndarray
    # static hyperparameters
    head_sizes: Tuple[int, ...] = struct.field(pytree_node=False)
    n_atoms: int = struct.field(pytree_node=False)
    v_min: float = struct.field(pytree_node=False)
    v_max: float = struct.field(pytree_node=False)
    n_heads: int = struct.field(pytree_node=False)
    tau: float = struct.field(pytree_node=False)
    epsilon: float = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        *,
        rng,
        feat_dim: int,
        head_sizes,
        n_atoms: int = 51,
        v_min: float = -1.0,
        v_max: float = 11.0,
        hidden_dims=(256, 256),
        dueling: bool = True,
        detach_input: bool = True,
        lr: float = 1e-4,
        max_grad_norm: float = 10.0,
        tau: float = 0.005,
        epsilon: float = 0.05,
        example_feat=None,
    ) -> "SpeedTuneLearner":
        head_sizes = tuple(int(x) for x in head_sizes)
        net = BranchingRainbowQ(
            head_sizes=head_sizes, n_atoms=n_atoms, v_min=v_min, v_max=v_max,
            hidden_dims=tuple(hidden_dims), dueling=dueling, detach_input=detach_input,
        )
        rng, init_rng = jax.random.split(rng)
        if example_feat is None:
            example_feat = jnp.zeros((1, feat_dim), dtype=jnp.float32)
        params = net.init(init_rng, example_feat)["params"]
        if max_grad_norm and max_grad_norm > 0:
            tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr))
        else:
            tx = optax.adam(lr)
        q_net = TrainState.create(apply_fn=net.apply, params=params, tx=tx)
        return cls(
            rng=rng, q_net=q_net, target_params=params,
            support=make_support(v_min, v_max, n_atoms),
            head_sizes=head_sizes, n_atoms=n_atoms, v_min=v_min, v_max=v_max,
            n_heads=len(head_sizes), tau=tau, epsilon=epsilon,
        )

    def greedy_action_idxs(self, feat: jnp.ndarray) -> jnp.ndarray:
        """纯贪心选档 ``[B, n_heads]``（eval / 确定性推理用）。"""
        dists = self.q_net.apply_fn({"params": self.q_net.params}, feat)
        return greedy_action_idxs(dists, self.support)

    def sample_action_idxs(self, feat: jnp.ndarray, *, greedy: bool = False
                           ) -> Tuple[jnp.ndarray, "SpeedTuneLearner"]:
        """rollout 选档：每 head 独立 epsilon-greedy。返回 ``([B, n_heads], new_self)``。"""
        greedy_idxs = self.greedy_action_idxs(feat)  # [B, n_heads]
        if greedy or self.epsilon <= 0.0:
            return greedy_idxs, self
        B = feat.shape[0]
        rng = self.rng
        out = []
        for i in range(self.n_heads):
            rng, rk1, rk2 = jax.random.split(rng, 3)
            rand_a = jax.random.randint(rk1, (B,), 0, self.head_sizes[i])
            explore = jax.random.uniform(rk2, (B,)) < self.epsilon
            out.append(jnp.where(explore, rand_a, greedy_idxs[:, i]))
        idxs = jnp.stack(out, axis=-1)
        return idxs, self.replace(rng=rng)

    def update(self, batch: Dict[str, Any]
               ) -> Tuple["SpeedTuneLearner", jnp.ndarray, Dict[str, jnp.ndarray]]:
        """一次 DQN 梯度步 + target 软更新。

        ``batch`` 含 ``feat/next_feat/action_idxs/reward/discount/is_weights``（numpy 或 jnp）。
        Returns ``(new_self, td_priorities[B], metrics)``；``td_priorities`` 回写 PER sum-tree。
        """
        jbatch = {k: jnp.asarray(v) for k, v in batch.items()
                  if k in ("feat", "next_feat", "action_idxs", "reward", "discount", "is_weights")}
        q_net, per_sample, metrics = _speedtune_update_step(
            self.q_net, self.target_params, jbatch, self.support,
            q_apply_fn=self.q_net.apply_fn, v_min=self.v_min, v_max=self.v_max,
            n_heads=self.n_heads,
        )
        new_target = optax.incremental_update(q_net.params, self.target_params, self.tau)
        new_self = self.replace(q_net=q_net, target_params=new_target)
        return new_self, per_sample, metrics
