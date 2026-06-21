"""Branching (factored) Rainbow Q-network for SpeedTune.

读 pi0.5 action expert 的 detached suffix 特征（mean-pooled 成 ``[B, feat_dim]``），输出
**多个互不耦合的 head**（BDQ 风格，Tavakoli et al. 2018）：每个速度变量一个 head，各自
独立的 C51 分布（Bellemare et al. 2017）+ dueling（Wang et al. 2016）。总输出 bin 数
= Σ Nᵢ（相加），避免笛卡尔积爆炸。

  - distributional (C51): 每 head 输出 ``[B, Nᵢ, n_atoms]`` 的概率分布（over value atoms）。
  - dueling: value 流 ``V(s)`` ``[B, n_atoms]`` 各 head 共享；每 head 自己的 advantage 流
    ``A_i(s,a)`` ``[B, Nᵢ, n_atoms]``，组合 ``logits = V + A - mean_a A``。
  - NoisyNet 暂未实现（计划：可选、后置）；探索用 epsilon-greedy（在 learner 端）。

输入特征默认 ``stop_gradient``（``detach_input``，与 ``dbpo_heads.ValueHead`` 同惯例）：
SpeedTune 两阶段解耦，VLA 冻结，DQN 梯度绝不回灌 action expert。
"""

from typing import List, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from expo_ft.networks.mlp import MLP, default_init


def make_support(v_min: float, v_max: float, n_atoms: int) -> jnp.ndarray:
    """C51 的固定 value support ``z`` ``[n_atoms]``（等距）。"""
    return jnp.linspace(v_min, v_max, n_atoms)


def expected_q(probs: jnp.ndarray, support: jnp.ndarray) -> jnp.ndarray:
    """分布 → 期望 Q：``E[Z] = Σ p·z``。

    Args:
      probs:   ``[..., N, n_atoms]`` 每动作的概率分布。
      support: ``[n_atoms]``。
    Returns:
      ``[..., N]`` 每动作的期望 Q。
    """
    return jnp.sum(probs * support, axis=-1)


class BranchingRainbowQ(nn.Module):
    """共享 backbone + 每变量一个独立 C51(+dueling) head。

    Attributes:
      head_sizes:   ``(N₁, N₂, ...)`` 各 head 的离散档位数（来自 ExecBackend.head_sizes）。
      n_atoms:      C51 原子数。
      v_min/v_max:  value support 范围（按 reward 量级设；见 config）。
      hidden_dims:  共享 backbone MLP 宽度。
      dueling:      是否用 dueling 分解。
      detach_input: 是否 stop_gradient 输入特征（默认 True）。
    """

    head_sizes: Sequence[int]
    n_atoms: int = 51
    v_min: float = -1.0
    v_max: float = 11.0
    hidden_dims: Sequence[int] = (256, 256)
    dueling: bool = True
    detach_input: bool = True

    @nn.compact
    def __call__(self, feat: jnp.ndarray, training: bool = False) -> List[jnp.ndarray]:
        """Args: feat ``[B, feat_dim]`` (mean-pooled suffix). Returns: list of
        ``[B, Nᵢ, n_atoms]`` 概率分布（softmax over atoms），每 head 一个。"""
        if self.detach_input:
            feat = jax.lax.stop_gradient(feat)
        x = MLP(self.hidden_dims, activate_final=True)(feat, training=training)

        value = None
        if self.dueling:
            value = nn.Dense(self.n_atoms, kernel_init=default_init(), name="value")(x)  # [B, atoms]

        dists: List[jnp.ndarray] = []
        for i, n_act in enumerate(self.head_sizes):
            adv = nn.Dense(n_act * self.n_atoms, kernel_init=default_init(), name=f"adv_{i}")(x)
            adv = adv.reshape(adv.shape[:-1] + (n_act, self.n_atoms))  # [B, n_act, atoms]
            if self.dueling:
                adv = adv - jnp.mean(adv, axis=-2, keepdims=True)
                logits = value[..., None, :] + adv  # [B, n_act, atoms]
            else:
                logits = adv
            probs = nn.softmax(logits, axis=-1)
            probs = jnp.clip(probs, 1e-8, 1.0)  # 数值稳定（后面取 log）
            dists.append(probs)
        return dists


def greedy_action_idxs(dists: Sequence[jnp.ndarray], support: jnp.ndarray) -> jnp.ndarray:
    """每 head 各自按期望 Q argmax → 档位索引 ``[B, n_heads]``。"""
    idxs = [jnp.argmax(expected_q(d, support), axis=-1) for d in dists]  # each [B]
    return jnp.stack(idxs, axis=-1)


def categorical_projection(
    next_probs: jnp.ndarray,
    rewards: jnp.ndarray,
    discounts: jnp.ndarray,
    support: jnp.ndarray,
    v_min: float,
    v_max: float,
) -> jnp.ndarray:
    """C51 distributional Bellman 投影（单 head，batched）。

    把 ``Tz = r + γⁿ·z`` 的分布 ``next_probs`` 投影回固定 support。

    Args:
      next_probs: ``[B, n_atoms]`` 目标网络在 next-state、所选 next-action 上的分布。
      rewards:    ``[B]`` n-step 聚合奖励。
      discounts:  ``[B]`` 有效折扣（``γⁿ·(1-done)``；episode 在 n 步内终止则为 0）。
      support:    ``[n_atoms]``。
    Returns:
      ``[B, n_atoms]`` 投影后的目标分布（每行和为 1）。
    """
    n_atoms = support.shape[0]
    delta_z = (v_max - v_min) / (n_atoms - 1)

    tz = rewards[:, None] + discounts[:, None] * support[None, :]  # [B, atoms]
    tz = jnp.clip(tz, v_min, v_max)
    b = (tz - v_min) / delta_z  # [B, atoms] ∈ [0, n_atoms-1]
    lo = jnp.floor(b).astype(jnp.int32)
    hi = jnp.ceil(b).astype(jnp.int32)

    # 当 b 落在整数点（lo==hi）时，把权重整体记到该格（下面两项之一会被 (hi==b) 触发）。
    lo_w = next_probs * (hi.astype(jnp.float32) - b)
    hi_w = next_probs * (b - lo.astype(jnp.float32))
    # lo==hi 时上面两权重都为 0，需补回整概率到该格。
    eq = (lo == hi)
    lo_w = jnp.where(eq, next_probs, lo_w)

    batch = next_probs.shape[0]
    m = jnp.zeros((batch, n_atoms), dtype=jnp.float32)
    bidx = jnp.arange(batch)[:, None]  # [B,1]
    m = m.at[bidx, lo].add(lo_w)
    m = m.at[bidx, hi].add(hi_w)
    return m
