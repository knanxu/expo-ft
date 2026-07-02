"""三种执行方式的动作空间（branching DQN 离散档位）+ success-gated reward。

每种执行方式（exec_backend）声明若干**互不耦合**的速度变量，每个变量对应一个独立的
RainbowDQN head（BDQ 风格）。一个变量 = 一组离散档位（grid），DQN 在该 head 上独立
argmax 选一档；总 bin 数 = Σ Nᵢ（相加）而非 Π Nᵢ（相乘）。

  - 方式1 ``fixed_time``        : 1 变量 v —— RoboTwin streaming（论文式固定时长，reconstruct(v)）。
  - 方式2 ``per_action_toppra`` : 3 变量 —— v + vel_limit + acc_limit（RoboTwin per_action，逐 action TOPP）。
  - 方式3 ``chunk_toppra``      : 1 变量 vel_limit（acc_limit=4*vel_limit² 由执行层派生）。

动作空间值绝对值制（v=chunk 压缩比∈[1,4]；vel_limit/acc_limit=绝对关节速度/加速度上限
rad/s、rad/s²）。whole-chunk 不再独立输出 acc_limit，而是在执行层直接取 ``4*vel_limit**2``，
且不施加额外加速度物理上限。

reward（防 reward hacking，success-gated）：

    r = 1[success] · α · speed^β

fixed-time 与 whole-chunk 使用原始 ``speed∈[1,4]``、默认 α=1、β=2，任务失败奖励为 0，
不额外叠加任务奖励。per-action 作为兼容路径仍使用归一化激进度及稀疏任务项。

这是纯 numpy（rollout/CPU 端用），learner 端的 DQN 用 jax；二者通过 ``head_sizes`` /
``decode`` / ``total_reward`` 解耦。
"""

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclasses.dataclass(frozen=True)
class SpeedVar:
    """一个速度控制变量 = 一个 DQN head。

    Attributes:
      name:      变量名（也是 ``speed_params`` 的 key），如 "v"/"vel_limit"/"acc_limit"。
      grid:      离散档位值（下发给 env 的真实参数值）。len(grid) = 该 head 的输出维度 Nᵢ。
      faster_is: "smaller" 表示档位值越小越快（如 compress 压缩率），"larger" 表示越大越快
                 （如 vel/acc scale）。决定归一化激进度 ``v`` 的方向。
      alpha:     reward 速度项系数 αᵢ（≥0）。
      beta:      reward 速度项指数 βᵢ（>0；>1 凸=奖励极端加速，<1 凹=边际递减偏好适度）。
    """

    name: str
    grid: Tuple[float, ...]
    faster_is: str = "larger"
    alpha: float = 1.0
    beta: float = 1.0
    reward_input: str = "normalized"

    def __post_init__(self):
        assert len(self.grid) >= 1, f"SpeedVar {self.name} grid 不能为空"
        assert self.faster_is in ("smaller", "larger"), self.faster_is
        assert self.reward_input in ("normalized", "raw"), self.reward_input
        assert self.alpha >= 0.0 and self.beta > 0.0

    @property
    def n_bins(self) -> int:
        return len(self.grid)

    def value(self, idx: int) -> float:
        """档位索引 → 下发给 env 的真实参数值。"""
        return float(self.grid[idx])

    def aggressiveness(self, idx: int) -> float:
        """档位索引 → 归一化激进度 v ∈ [0,1]（越快越大），用于 reward 速度项。"""
        g = np.asarray(self.grid, dtype=np.float64)
        lo, hi = float(g.min()), float(g.max())
        if hi <= lo:  # 单档位（无可调空间）→ 无加速激励
            return 0.0
        norm = (float(self.grid[idx]) - lo) / (hi - lo)  # ∈[0,1]
        return (1.0 - norm) if self.faster_is == "smaller" else norm

    def reward_value(self, idx: int) -> float:
        """用于 reward 的值：论文式 raw speed 或兼容 per-action 的归一化激进度。"""
        return self.value(idx) if self.reward_input == "raw" else self.aggressiveness(idx)


@dataclasses.dataclass(frozen=True)
class ExecBackend:
    """一种执行方式：一组 SpeedVar（=一组独立 DQN head）+ success-gated reward。"""

    name: str
    vars: Tuple[SpeedVar, ...]
    include_task_reward: bool = True

    def __post_init__(self):
        assert len(self.vars) >= 1, f"exec_backend {self.name} 至少 1 个变量"
        names = [v.name for v in self.vars]
        assert len(set(names)) == len(names), f"变量名重复: {names}"

    @property
    def n_heads(self) -> int:
        return len(self.vars)

    @property
    def head_sizes(self) -> Tuple[int, ...]:
        """各 head 的输出维度 (N₁, N₂, ...)，喂给 RainbowDQN 构造。"""
        return tuple(v.n_bins for v in self.vars)

    def decode(self, action_idxs: Sequence[int]) -> Tuple[Dict[str, float], List[float]]:
        """DQN 每 head 的档位索引 → (speed_params, reward_values)。

        Args:
          action_idxs: 长度 = n_heads 的离散档位索引（每 head 一个）。

        Returns:
          speed_params: {var_name: 真实参数值}，传给 ``env.step_chunk(..., speed_params=...)``。
          v_list:       fixed/chunk 为原始 1–4，per-action 为归一化激进度。
        """
        assert len(action_idxs) == self.n_heads, (len(action_idxs), self.n_heads)
        speed_params: Dict[str, float] = {}
        v_list: List[float] = []
        for var, idx in zip(self.vars, action_idxs):
            idx = int(idx)
            assert 0 <= idx < var.n_bins, f"{var.name} idx {idx} 越界 [0,{var.n_bins})"
            speed_params[var.name] = var.value(idx)
            v_list.append(var.reward_value(idx))
        return speed_params, v_list

    def speed_reward(self, success: bool, v_list: Sequence[float]) -> float:
        """纯速度项 Σ αᵢ vᵢ^βᵢ，success-gated（失败→0）。"""
        if not success:
            return 0.0
        assert len(v_list) == self.n_heads, (len(v_list), self.n_heads)
        total = 0.0
        for var, v in zip(self.vars, v_list):
            v = max(0.0, float(v))  # 数值兜底，避免负底数^beta
            total += var.alpha * (v ** var.beta)
        return float(total)

    def total_reward(self, r_task: float, success: bool, v_list: Sequence[float]) -> float:
        """完整 reward；fixed/chunk 仅用 success-gated raw-speed，per-action 保留任务项。"""
        task = float(r_task) if self.include_task_reward else 0.0
        return task + self.speed_reward(success, v_list)


# ---------------------------------------------------------------------------
# 默认动作空间（可被 config 覆盖；见 configs/model/speedtune_dqn_config.py）
# ---------------------------------------------------------------------------

# RoboTwin 真实速度参数（绝对值制：vel_limit/acc_limit 直接是关节上限，对接 take_chunk_action_per_action）。
# v = chunk 压缩比 ∈[1,4]（reconstruct_chunk: 1=原速，越大帧数越少=越快）。faster_is="larger"。
_DEFAULT_SPEED = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
_DEFAULT_V = (1.0, 1.5, 2.0, 3.0, 4.0)
# vel_limit = 绝对关节速度上限 (rad/s, ≤真机 5~5.5)；acc_limit = 绝对加速度上限 (rad/s²)。越大越快。
# 占位值：Task 6 (calibrate_acc_grid) 标定后回填 acc_limit；vel_limit 按真机速度定。
_DEFAULT_VEL_LIMIT = (1.0, 2.0, 3.0)
_DEFAULT_ACC_LIMIT = (1.0, 3.0, 5.0, 7.0, 9.0)


def _default_specs() -> Dict[str, dict]:
    """每个 exec_backend 的默认变量规格（grid/faster_is/alpha/beta），供 config 浅覆盖。

    grid 用 RoboTwin 真实参数值（decode 后直接传执行后端）。fixed/chunk reward 使用原始
    1–4 数值；per-action 保留归一化激进度。
    """
    return {
        # 方式1 fixed_time（RoboTwin streaming）：只调标量 v（论文式固定时长）。
        "fixed_time": {
            "v": dict(grid=_DEFAULT_SPEED, faster_is="larger", alpha=1.0, beta=2.0,
                      reward_input="raw"),
        },
        # 方式2 per_action_toppra（RoboTwin per_action，逐 action TOPP）：v + vel_limit + acc_limit。
        "per_action_toppra": {
            "v": dict(grid=_DEFAULT_V, faster_is="larger", alpha=0.5, beta=1.0),
            "vel_limit": dict(grid=_DEFAULT_VEL_LIMIT, faster_is="larger", alpha=0.5, beta=1.0),
            "acc_limit": dict(grid=_DEFAULT_ACC_LIMIT, faster_is="larger", alpha=0.5, beta=1.0),
        },
        # 方式3 chunk_toppra：网络只选 vel_limit；acc_limit=4*vel_limit² 由执行层派生。
        "chunk_toppra": {
            "vel_limit": dict(grid=_DEFAULT_SPEED, faster_is="larger", alpha=1.0, beta=2.0,
                              reward_input="raw"),
        },
    }


def list_backends() -> Tuple[str, ...]:
    return tuple(_default_specs().keys())


def build_backend(name: str, overrides: Dict[str, dict] = None) -> ExecBackend:
    """构造一个 exec_backend。

    Args:
      name:      "fixed_time" / "per_action_toppra" / "chunk_toppra"。
      overrides: 可选，``{var_name: {grid/faster_is/alpha/beta: ...}}``，浅覆盖默认规格
                 （只覆盖提供的键，未提供的沿用默认）。config 用它调 grid / α / β。

    Returns:
      ExecBackend（变量顺序固定，决定 DQN head 顺序与 decode 顺序）。
    """
    specs = _default_specs()
    if name not in specs:
        raise ValueError(f"未知 exec_backend {name!r}；可选 {list_backends()}")
    spec = specs[name]
    overrides = overrides or {}
    out_vars: List[SpeedVar] = []
    for var_name, base in spec.items():
        merged = dict(base)
        merged.update(overrides.get(var_name, {}))
        merged["grid"] = tuple(float(x) for x in merged["grid"])
        out_vars.append(SpeedVar(name=var_name, **merged))
    return ExecBackend(
        name=name,
        vars=tuple(out_vars),
        include_task_reward=name == "per_action_toppra",
    )


def parse_force_limit(spec) -> Optional[List[float]]:
    """force_limit 配置 → per-joint 单臂力矩上限 list（空/None → None = 不施加 = ∞）。

    接受逗号分隔字符串 "30,40,30,15,10,10" / list / tuple；空串、纯空白、None → None。
    长度应 = 单臂 arm dof（6, ARX5）；不强制校验——由 RoboTwinEnv._apply_force_limit 的 zip
    自然处理（多余截断、不足只设前 N 个关节）。SpeedTune 执行层力矩底座用，EXPO/BC 不传 → None。
    """
    if spec is None:
        return None
    if isinstance(spec, (list, tuple)):
        vals = [float(x) for x in spec]
        return vals or None
    s = str(spec).strip()
    if not s:
        return None
    return [float(x) for x in s.split(",") if x.strip()]
