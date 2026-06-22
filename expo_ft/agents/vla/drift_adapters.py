"""pi0.5 drift 前向适配器：把 nnx pi0.5 模型包成 ``drift_apply_fn``。

构造抽象的 ``drift_apply_fn(params, obs, z, frozen) -> (mean, cond_emb, value_feat)``，
通过从 ``(graphdef, params)`` 重建 nnx pi0.5 模型并调用其单步 drift 采样器
（``_sample_actions_drifting(return_hidden=True)``）拿到隐藏特征。**算法无关**：任何想在
冻结/可训 pi0.5 drift 通路上读 (动作均值, VLM prefix 特征, action-expert suffix 特征)
的下游模块都可复用（当前 SpeedTune DQN 用它取 ``suffix_feat`` 作 Q 输入）。

设计要点（在 dummy pi0.5 上验证过）：
  * ``nnx.split(model)`` 产出 graphdef（静态）+ 叶子为纯数组的 ``nnx.State``，后者是干净
    pytree —— ``optax`` 直接作用其上（``pi05.py`` 已依赖此性质），``jax.grad`` 经
    ``nnx.merge`` 流回 backbone。
  * value_feat 取 mean-pooled action-expert ``suffix_feat``（RLinf critic 设计：V 读
    suffix_out 而非 cond_emb，docs/RLINF_ROBOTWIN_REFERENCE.md §4）。该输入应由下游
    value/critic head 做 stop-gradient（如 ``rainbow_dqn`` 的 ``detach_input``），
    使 critic 损失不扰动共享 action expert。
"""

import jax
import jax.numpy as jnp
import flax.nnx as nnx

import openpi.models.model as _model

# rng is unused by the drift sampler when an explicit noise (z) is provided.
_UNUSED_KEY = jax.random.PRNGKey(0)


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
        # design: V reads suffix_out, not cond_emb).  Downstream value/critic heads
        # should stop-gradient this input so the critic loss never perturbs the
        # shared action expert.
        value_feat = jnp.mean(suffix_feat, axis=-2)  # [B, H, width] -> [B, width]
        return mean, cond_emb, value_feat

    return drift_apply_fn
