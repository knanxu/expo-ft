"""On-policy rollout buffer for DBPO (PPO/BPO).

Unlike EXPO-FT's off-policy :class:`PiReplayBuffer`, DBPO is on-policy: each PPO
iteration collects a fresh rollout, computes GAE, and is consumed over a few
epochs of minibatches before being discarded.

The single most important field is ``latent_z`` -- the drift latent used to
generate the action mean at rollout time.  DBPO's importance ratio is only valid
if the SAME ``z`` is reused when recomputing the log-prob under updated params
(paper Eq. 47/50); storing it here is what makes that possible.

Per decision step (each executes an action-chunk prefix of ``replan_steps``):
  observation (pytree) · latent_z · action chunk · logp_old · value · reward · done
"""

from typing import Any, Dict, Iterator

import jax
import jax.numpy as jnp
import numpy as np

from expo_ft.agents.alg.dbpo_core import compute_gae


class RolloutBuffer:
    """Accumulates one on-policy rollout, then yields GAE-annotated minibatches."""

    def __init__(self):
        self._obs: list = []
        self._z: list = []
        self._actions: list = []
        self._logp: list = []
        self._value: list = []
        self._reward: list = []
        self._done: list = []
        self.finalized = False

    def add(self, obs: Any, latent_z, action, logp_old, value, reward, done) -> None:
        """Append one decision step. ``obs`` may be any pytree (dict/array)."""
        assert not self.finalized, "cannot add to a finalized RolloutBuffer"
        self._obs.append(jax.tree.map(np.asarray, obs))
        self._z.append(np.asarray(latent_z))
        self._actions.append(np.asarray(action))
        self._logp.append(float(logp_old))
        self._value.append(float(value))
        self._reward.append(float(reward))
        self._done.append(float(done))

    def __len__(self) -> int:
        return len(self._reward)

    def finalize(self, last_value: float, *, gamma: float, gae_lambda: float) -> None:
        """Stack the rollout into arrays and fill ``advantages`` / ``returns`` via GAE."""
        assert not self.finalized and len(self) > 0
        self.obs = jax.tree.map(lambda *xs: np.stack(xs), *self._obs)
        self.z = np.stack(self._z)
        self.actions = np.stack(self._actions)
        self.logp_old = np.asarray(self._logp, dtype=np.float32)
        self.value = np.asarray(self._value, dtype=np.float32)
        self.reward = np.asarray(self._reward, dtype=np.float32)
        self.done = np.asarray(self._done, dtype=np.float32)
        adv, ret = compute_gae(
            jnp.asarray(self.reward), jnp.asarray(self.value), jnp.asarray(self.done),
            jnp.asarray(np.float32(last_value)), gamma=gamma, gae_lambda=gae_lambda,
        )
        self.advantages = np.asarray(adv, dtype=np.float32)
        self.returns = np.asarray(ret, dtype=np.float32)
        self.finalized = True

    def iterate_minibatches(self, seed: int, num_minibatches: int) -> Iterator[Dict[str, Any]]:
        """Yield ``num_minibatches`` disjoint shuffled minibatches covering the rollout.

        Each minibatch is a dict of stacked arrays (jax) ready for the learner's
        jitted update; ``obs`` keeps its original pytree structure.
        """
        assert self.finalized, "call finalize() before iterating"
        T = len(self)
        perm = np.random.default_rng(seed).permutation(T)
        for mb_idx in np.array_split(perm, num_minibatches):
            mb_idx = np.asarray(mb_idx)
            yield {
                "obs": jax.tree.map(lambda x: jnp.asarray(x[mb_idx]), self.obs),
                "z": jnp.asarray(self.z[mb_idx]),
                "actions": jnp.asarray(self.actions[mb_idx]),
                "logp_old": jnp.asarray(self.logp_old[mb_idx]),
                "value_old": jnp.asarray(self.value[mb_idx]),
                "advantages": jnp.asarray(self.advantages[mb_idx]),
                "returns": jnp.asarray(self.returns[mb_idx]),
            }
