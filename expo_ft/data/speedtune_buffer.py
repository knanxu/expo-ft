"""SpeedTune DQN 经验回放（n-step 聚合 + 比例 PER / sum-tree）。

存的是 transition 而非图像：``(feat, action_idxs, reward, next_feat, done)``，其中
``feat`` 是 mean-pooled 的 detached suffix 特征 ``[feat_dim]``（DQN 的 state），
``action_idxs`` 是各 head 的离散档位 ``[n_heads]``。决策粒度 = 一整段 chunk。

  - n-step 聚合：维护长度 n 的滑窗，把 ``(s_t, a_t, Σγᵏr, s_{t+n}, γⁿ·(1-done))`` 写入树；
    episode 结束时 flush 余下的短窗（截断到 done）。
  - 比例 PER（Schaul et al. 2016）：sum-tree 按 ``pᵅ`` 采样，重要度权重 ``(N·P)⁻ᵝ``，
    新样本给当前最大优先级（保证至少被采一次）。

纯 numpy（rollout/learner CPU 端）；``sample`` 返回 numpy dict，learner 端转 jax。
"""

import collections
from typing import Any, Dict, List

import numpy as np


class _SumTree:
    """比例优先级用的 sum-tree（叶子存优先级，内部节点存子树和）。"""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)
        self.write = 0
        self.size = 0

    def total(self) -> float:
        return float(self.tree[0])

    def add(self, p: float) -> int:
        """写入一个新叶子（环形），返回其 tree 索引。"""
        tree_idx = self.write + self.capacity - 1
        self.update(tree_idx, p)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return tree_idx

    def update(self, tree_idx: int, p: float) -> None:
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        idx = tree_idx
        while idx != 0:  # 迭代向上传播（避免递归深度）
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, s: float):
        """按前缀和 s 定位叶子，返回 (tree_idx, priority, data_idx)。"""
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):  # 到叶子
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - (self.capacity - 1)
        return idx, float(self.tree[idx]), data_idx


_Trans = collections.namedtuple("_Trans", "feat action_idxs reward next_feat done")


class SpeedTuneReplayBuffer:
    """n-step + PER replay buffer for the SpeedTune DQN."""

    def __init__(
        self,
        capacity: int,
        feat_dim: int,
        n_heads: int,
        *,
        n_step: int = 3,
        gamma: float = 0.99,
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        per_eps: float = 1e-6,
        seed: int = 0,
    ):
        self.capacity = int(capacity)
        self.feat_dim = int(feat_dim)
        self.n_heads = int(n_heads)
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.per_alpha = float(per_alpha)
        self.per_beta = float(per_beta)
        self.per_eps = float(per_eps)
        self._rng = np.random.default_rng(seed)

        self._feat = np.zeros((self.capacity, self.feat_dim), dtype=np.float32)
        self._next_feat = np.zeros((self.capacity, self.feat_dim), dtype=np.float32)
        self._action_idxs = np.zeros((self.capacity, self.n_heads), dtype=np.int32)
        self._reward = np.zeros((self.capacity,), dtype=np.float32)   # n-step 聚合 R
        self._discount = np.zeros((self.capacity,), dtype=np.float32)  # γⁿ·(1-done)
        self._done = np.zeros((self.capacity,), dtype=np.float32)

        self._tree = _SumTree(self.capacity)
        self._max_priority = 1.0
        self._nstep_q: collections.deque = collections.deque(maxlen=self.n_step)

    def __len__(self) -> int:
        return self._tree.size

    def ready(self, min_size: int) -> bool:
        return len(self) >= int(min_size)

    # ---- 写入 -------------------------------------------------------------
    def _emit_nstep(self) -> None:
        """把 n-step 窗的最早一条聚合成 transition 写入树。"""
        q = self._nstep_q
        R = 0.0
        discount = 1.0
        last_next_feat = q[-1].next_feat
        done_flag = 0.0
        for t in q:
            R += discount * float(t.reward)
            discount *= self.gamma
            if t.done:
                last_next_feat = t.next_feat
                done_flag = 1.0
                break
        eff_discount = 0.0 if done_flag > 0.5 else discount  # 终止则不 bootstrap
        first = q[0]
        write = self._tree.write
        self._feat[write] = first.feat
        self._action_idxs[write] = first.action_idxs
        self._reward[write] = R
        self._next_feat[write] = last_next_feat
        self._discount[write] = eff_discount
        self._done[write] = done_flag
        self._tree.add(self._max_priority ** self.per_alpha)

    def insert(self, feat, action_idxs, reward: float, next_feat, done: bool) -> None:
        """加入一步 transition；n-step 窗满或 episode done 时落盘。"""
        feat = np.asarray(feat, dtype=np.float32).reshape(self.feat_dim)
        next_feat = np.asarray(next_feat, dtype=np.float32).reshape(self.feat_dim)
        action_idxs = np.asarray(action_idxs, dtype=np.int32).reshape(self.n_heads)
        self._nstep_q.append(_Trans(feat, action_idxs, float(reward), next_feat, bool(done)))
        if done:
            # episode 结束：flush 整个窗，每条都作为起点聚合一次（到 done 截断），清空。
            while self._nstep_q:
                self._emit_nstep()
                self._nstep_q.popleft()
        elif len(self._nstep_q) >= self.n_step:
            # 窗满：聚合最早一条的完整 n-step，弹出最早一条。
            self._emit_nstep()
            self._nstep_q.popleft()

    # ---- 采样 -------------------------------------------------------------
    def sample(self, batch_size: int, beta: float = None) -> Dict[str, Any]:
        """比例 PER 采样。返回 numpy batch + ``tree_indices``/``is_weights``。"""
        assert len(self) > 0, "buffer 为空"
        beta = self.per_beta if beta is None else float(beta)
        total = self._tree.total()
        segment = total / batch_size
        data_idxs: List[int] = []
        tree_idxs: List[int] = []
        priorities: List[float] = []
        for i in range(batch_size):
            s = self._rng.uniform(segment * i, segment * (i + 1))
            tree_idx, p, data_idx = self._tree.get(s)
            # 极少数边界（p=0 的未写满槽）兜底：重采到有效叶子
            if p <= 0 or data_idx >= len(self):
                s = self._rng.uniform(0, total)
                tree_idx, p, data_idx = self._tree.get(s)
            data_idxs.append(data_idx)
            tree_idxs.append(tree_idx)
            priorities.append(p)

        probs = np.asarray(priorities, dtype=np.float64) / max(total, 1e-12)
        n = len(self)
        is_weights = (n * probs) ** (-beta)
        is_weights = (is_weights / is_weights.max()).astype(np.float32)
        di = np.asarray(data_idxs, dtype=np.int64)
        return {
            "feat": self._feat[di],
            "action_idxs": self._action_idxs[di],
            "reward": self._reward[di],
            "discount": self._discount[di],
            "next_feat": self._next_feat[di],
            "done": self._done[di],
            "is_weights": is_weights,
            "tree_indices": np.asarray(tree_idxs, dtype=np.int64),
        }

    def update_priorities(self, tree_indices, td_errors) -> None:
        """用新的 |TD-error| 更新优先级（learner 每步调）。"""
        td = np.abs(np.asarray(td_errors, dtype=np.float64)) + self.per_eps
        for tree_idx, e in zip(np.asarray(tree_indices).tolist(), td.tolist()):
            self._max_priority = max(self._max_priority, e)
            self._tree.update(int(tree_idx), e ** self.per_alpha)
