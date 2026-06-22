# SpeedTune per_action 执行核重构（方法 B）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 RoboTwin `per_action` 后端从「逐 action 两点零速 TOPP（stop-and-go 伪快）」改为「逐 action 两点 + 非零边界速度 TOPPRA」，消除段间走停、实现真加速，且保留逐 action 粒度以兼容 partial chunk / async inference。

**Architecture:** 绕过 mplib 写死的 `compute_trajectory()`（零边界），改用 `toppra_chunk_executor.retime_chunk` 逐 action 调用并传非零 `sd_start/sd_end`。`sd_start` = 实测关节速度在本段切向的投影（闭环）；`sd_end` = 固定 `v_cruise`（末段 0）。弧长参数化下 `sd` 直接 = 关节空间速度幅值 (rad/s)。

**Tech Stack:** Python, numpy, toppra（`SplineInterpolator` + `TOPPRA`），SAPIEN（关节 PD 驱动 `set_arm_joints`），RoboTwin fork（`knanxu/RoboTwin`）。

## Global Constraints

- **执行/验证模型（CLAUDE.md，最高优先级）**：本地**只写代码 + `python3 -m py_compile` 静态语法检查**；**禁止**本地 `import sapien/toppra/torch`、跑 `pytest`/仿真/训练（显存不足）。所有 `pytest`（云端）+ 集成 rollout 验证由**维护人在云端**跑。本计划每个「运行测试」步骤标 **【云端】**=维护人执行，**【本地】**=Claude/实现者执行。
- **零侵入**：**不改** mplib（`.../mplib/planner.py`，第三方）、原版/fork `take_action`（`_base_task.py:1562`，policy-eval 仍用，保留不动）、`take_chunk_action`（whole_chunk，`:1781`）、`take_chunk_action_streaming`（`:2056`）。`retime_chunk` 新增参数默认 `(0,0)` → whole_chunk 行为逐位不变。
- **repo 分离**：RoboTwin 改动 commit 在 `/home/xukainan/RoboTwin`（`git -C /home/xukainan/RoboTwin`）；expo-ft 改动 commit 在 `/home/xukainan/expo-ft`。**两 repo 分别提交。**
- **弧长参数化语义**：`retime_chunk` 用累积弧长作 path_s（`toppra_chunk_executor.py:107`），故 `|dq/ds|=1`、`sd` = 关节空间速度幅值 (rad/s)。`v_cruise` 是 **12 dof 联合速度向量的范数**（非单关节），与单关节 `PHYS_VEL_CEIL=3.0` 量纲不同。
- **vel/acc 约束**：base limits `× vel_scale/acc_scale` 后钳 `PHYS_VEL_CEIL=PHYS_ACC_CEIL=3.0`（沿用 `toppra_chunk_executor.py:117-120`）。

---

### Task 1: toppra_chunk_executor 层扩展（非零边界 + 常数 + sd 计算纯函数）

**Files:**
- Modify: `/home/xukainan/RoboTwin/envs/robot/toppra_chunk_executor.py`（`retime_chunk:29` 签名 + `compute_trajectory(0, 0):139`；文件顶部加常数；文件末尾加纯函数）
- Test: `/home/xukainan/RoboTwin/envs/robot/toppra_chunk_executor_test.py`（新建）

**Interfaces:**
- Produces:
  - `retime_chunk(..., sd_start: float = 0.0, sd_end: float = 0.0)` — 新增两可选参数，内部 `compute_trajectory(sd_start, sd_end)`；默认 `(0,0)` 行为不变。
  - `DEFAULT_CRUISE_SD: float`（模块常数，12dof 联合速度幅值初值，云端调）。
  - `compute_segment_sd_bounds(q_current, qd_actual, target, joint_vel_limits, vel_scale, is_last, v_cruise, phys_vel_ceil=PHYS_VEL_CEIL, safety=0.9) -> tuple[float, float, np.ndarray|None, float]` 返回 `(sd_start, sd_end, tangent, seg_len)`；`seg_len<1e-6` 表退化（`tangent=None`，调用方应跳过）。

- [ ] **Step 1: 【本地】写失败测试** `toppra_chunk_executor_test.py`

```python
import numpy as np
import pytest
from envs.robot.toppra_chunk_executor import (
    retime_chunk, compute_segment_sd_bounds, DEFAULT_CRUISE_SD, PHYS_VEL_CEIL,
)

# --- 12 dof 假数据: 一条非退化两点路径 ---
def _dummy_retime(sd_start=0.0, sd_end=0.0):
    cur = np.zeros(12)
    target = np.full(12, 0.3)                     # 各关节移动 0.3 rad
    return retime_chunk(
        current_state_arm=cur, chunk_arm=target[None, :],
        current_gripper=np.zeros(2), chunk_gripper=np.zeros((1, 2)),
        joint_vel_limits=np.full(12, 2.0), joint_acc_limits=np.full(12, 2.0),
        vel_scale=1.0, acc_scale=1.0, exec_hz=250, sd_start=sd_start, sd_end=sd_end,
    )

def test_default_boundary_backward_compatible():
    # 默认 (0,0): 末端速度 ≈ 0 (whole_chunk 行为不变)
    r = _dummy_retime()
    assert r["status"] == "success"
    assert np.linalg.norm(r["dense_arm_vel"][-1]) < 1e-2

def test_nonzero_sd_end_gives_nonzero_terminal_speed():
    # sd_end=V: 弧长参数化下末端速度幅值 ≈ V
    V = 0.8
    r = _dummy_retime(sd_end=V)
    assert r["status"] == "success"
    assert abs(np.linalg.norm(r["dense_arm_vel"][-1]) - V) < 0.1

def test_sd_bounds_last_segment_is_zero():
    sd_start, sd_end, tangent, seg_len = compute_segment_sd_bounds(
        q_current=np.zeros(12), qd_actual=np.zeros(12), target=np.full(12, 0.2),
        joint_vel_limits=np.full(12, 2.0), vel_scale=1.0, is_last=True,
        v_cruise=DEFAULT_CRUISE_SD,
    )
    assert sd_end == 0.0
    assert seg_len > 1e-6 and tangent is not None

def test_sd_start_is_velocity_projection_clipped():
    # qd 沿 +tangent → sd_start = |qd 投影|; 反向 → 0
    target = np.full(12, 1.0)
    tang = target / np.linalg.norm(target)
    qd_forward = tang * 0.5
    s1, _, _, _ = compute_segment_sd_bounds(np.zeros(12), qd_forward, target,
                                            np.full(12, 3.0), 1.0, False, DEFAULT_CRUISE_SD)
    assert abs(s1 - 0.5) < 1e-6
    s2, _, _, _ = compute_segment_sd_bounds(np.zeros(12), -qd_forward, target,
                                            np.full(12, 3.0), 1.0, False, DEFAULT_CRUISE_SD)
    assert s2 == 0.0

def test_sd_end_clamped_by_vel_limit():
    # v_cruise 极大 → sd_end 被 0.9*sd_max 钳住 (sd_max 由 per-joint vel 约束推)
    target = np.full(12, 1.0)
    _, sd_end, _, _ = compute_segment_sd_bounds(np.zeros(12), np.zeros(12), target,
                                                joint_vel_limits=np.full(12, 2.0), vel_scale=1.0,
                                                is_last=False, v_cruise=1e6)
    # tangent 各分量 = 1/sqrt(12); scaled_vel=min(2.0,3.0)=2.0; sd_max=2.0/(1/sqrt(12))=2*sqrt(12)
    assert abs(sd_end - 0.9 * 2.0 * np.sqrt(12)) < 1e-3

def test_degenerate_segment_flagged():
    sd_start, sd_end, tangent, seg_len = compute_segment_sd_bounds(
        q_current=np.full(12, 0.5), qd_actual=np.zeros(12), target=np.full(12, 0.5),
        joint_vel_limits=np.full(12, 2.0), vel_scale=1.0, is_last=False, v_cruise=1.0,
    )
    assert seg_len < 1e-6 and tangent is None
```

- [ ] **Step 2: 【本地】py_compile 测试文件**

Run: `python3 -m py_compile /home/xukainan/RoboTwin/envs/robot/toppra_chunk_executor_test.py`
Expected: 无输出（语法 OK）。

- [ ] **Step 3: 【本地】改 `retime_chunk` 签名 + compute_trajectory**

在 `toppra_chunk_executor.py` 文件顶部常数区（`PHYS_ACC_CEIL` 之后）加：

```python
# 方法 B (per_action 逐 action 非零边界) 的默认巡航路径速度 (= 12dof 联合关节速度幅值, rad/s).
# 保守初值, 云端按 "时间↓ / success 不降 / jerk 可接受" 实测调.
DEFAULT_CRUISE_SD: float = 1.0
```

`retime_chunk` 签名增加两参数（在 `phys_acc_ceil` 之后）：

```python
def retime_chunk(
    current_state_arm: np.ndarray,
    chunk_arm: np.ndarray,
    current_gripper: np.ndarray,
    chunk_gripper: np.ndarray,
    joint_vel_limits: np.ndarray,
    joint_acc_limits: np.ndarray,
    vel_scale: float = 1.0,
    acc_scale: float = 1.0,
    exec_hz: int = 250,
    phys_vel_ceil: float = PHYS_VEL_CEIL,
    phys_acc_ceil: float = PHYS_ACC_CEIL,
    sd_start: float = 0.0,
    sd_end: float = 0.0,
):
```

把 `:139` 的 `retimed = instance.compute_trajectory(0, 0)` 改为：

```python
        retimed = instance.compute_trajectory(sd_start, sd_end)
```

- [ ] **Step 4: 【本地】在文件末尾加 `compute_segment_sd_bounds` 纯函数**

```python
def compute_segment_sd_bounds(
    q_current: np.ndarray,
    qd_actual: np.ndarray,
    target: np.ndarray,
    joint_vel_limits: np.ndarray,
    vel_scale: float,
    is_last: bool,
    v_cruise: float,
    phys_vel_ceil: float = PHYS_VEL_CEIL,
    safety: float = 0.9,
):
    """方法 B: 算逐 action 两点段在弧长参数化下的边界路径速度 (sd_start, sd_end).

    弧长参数化下 |dq/ds|=1, 故 sd = 关节空间速度幅值 (rad/s).
      - sd_start = 实测关节速度在本段单位切向上的投影 (闭环), clip >= 0.
      - sd_end   = 末段 0; 否则 min(v_cruise, safety * sd_max),
                   sd_max = min_j(scaled_vel_j / |tangent_j|) 由 per-joint vel 约束推.
    Returns: (sd_start, sd_end, tangent, seg_len). seg_len<1e-6 表退化 (tangent=None).
    """
    q_current = np.asarray(q_current, dtype=np.float64)
    qd_actual = np.asarray(qd_actual, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    seg = target - q_current
    seg_len = float(np.linalg.norm(seg))
    if seg_len < 1e-6:
        return 0.0, 0.0, None, seg_len
    tangent = seg / seg_len
    sd_start = max(0.0, float(qd_actual @ tangent))
    if is_last:
        sd_end = 0.0
    else:
        scaled_vel = np.minimum(
            np.asarray(joint_vel_limits, dtype=np.float64) * float(vel_scale),
            float(phys_vel_ceil),
        )
        sd_max = float(np.min(scaled_vel / np.maximum(np.abs(tangent), 1e-9)))
        sd_end = min(float(v_cruise), float(safety) * sd_max)
    return sd_start, sd_end, tangent, seg_len
```

- [ ] **Step 5: 【本地】py_compile 实现文件**

Run: `python3 -m py_compile /home/xukainan/RoboTwin/envs/robot/toppra_chunk_executor.py`
Expected: 无输出。

- [ ] **Step 6: 【云端】维护人跑单元测试**

Run（云端，RoboTwin 根目录）: `pytest envs/robot/toppra_chunk_executor_test.py -v`
Expected: 6 个测试全 PASS。重点 `test_default_boundary_backward_compatible`（向后兼容）+ `test_nonzero_sd_end_gives_nonzero_terminal_speed`（非零边界生效）。

- [ ] **Step 7: 提交（RoboTwin repo）**

```bash
git -C /home/xukainan/RoboTwin add envs/robot/toppra_chunk_executor.py envs/robot/toppra_chunk_executor_test.py
git -C /home/xukainan/RoboTwin commit -m "feat(per_action): retime_chunk 支持非零 sd 边界 + compute_segment_sd_bounds (方法B基础)"
```

---

### Task 2: robot.py 新增 arm 实测关节速度 getter

**Files:**
- Modify: `/home/xukainan/RoboTwin/envs/robot/robot.py`（在 `get_right_arm_real_jointState`，约 `:524` 之后新增）

**Interfaces:**
- Consumes: `self.left_entity.get_qvel()` / `get_active_joints()`、`self.left_arm_joints`（已有，`robot.py:179-180`）。
- Produces: `get_left_arm_real_jointVelocity() -> list[float]`、`get_right_arm_real_jointVelocity() -> list[float]`（**仅 arm joints 实测速度，不含 gripper**）。

- [ ] **Step 1: 【本地】写实现**（类比 `get_left_arm_real_jointState:508`，把 `get_qpos` 换 `get_qvel`、不追加 gripper）

```python
    def get_left_arm_real_jointVelocity(self) -> list:
        velocity_list = []
        left_joints_qvel = self.left_entity.get_qvel()
        left_active_joints = self.left_entity.get_active_joints()
        for joint in self.left_arm_joints:
            velocity_list.append(float(left_joints_qvel[left_active_joints.index(joint)]))
        return velocity_list

    def get_right_arm_real_jointVelocity(self) -> list:
        velocity_list = []
        right_joints_qvel = self.right_entity.get_qvel()
        right_active_joints = self.right_entity.get_active_joints()
        for joint in self.right_arm_joints:
            velocity_list.append(float(right_joints_qvel[right_active_joints.index(joint)]))
        return velocity_list
```

- [ ] **Step 2: 【本地】py_compile**

Run: `python3 -m py_compile /home/xukainan/RoboTwin/envs/robot/robot.py`
Expected: 无输出。

- [ ] **Step 3: 【云端】smoke（随 Task 3 集成验证一并确认）**

无独立单元测试（依赖 SAPIEN articulation，难离线构造）。云端集成时确认：env reset 后调 `robot.get_left_arm_real_jointVelocity()` 返回 `len == left_arm_dim` 的有限浮点 list（静止时各分量 ≈ 0）。

- [ ] **Step 4: 提交（RoboTwin repo）**

```bash
git -C /home/xukainan/RoboTwin add envs/robot/robot.py
git -C /home/xukainan/RoboTwin commit -m "feat(per_action): robot 新增 arm 实测关节速度 getter (方法B sd_start 闭环)"
```

---

### Task 3: 重写 `take_chunk_action_per_action` 执行核（方法 B）

**Files:**
- Modify: `/home/xukainan/RoboTwin/envs/_base_task.py`（`take_chunk_action_per_action:1983-2054` 整体重写函数体）

**Interfaces:**
- Consumes: `retime_chunk(..., sd_start, sd_end)`、`compute_segment_sd_bounds(...)`、`DEFAULT_CRUISE_SD` [Task 1]；`get_left_arm_real_jointVelocity/get_right_arm_real_jointVelocity` [Task 2]；`get_left_arm_real_jointState`、`set_arm_joints`、`set_gripper`、`_tick_eval_video`、`check_success`、`take_action`（fallback）、模块函数 `_apply_k_skip`、`reconstruct_chunk`（均已有）。
- Produces: `take_chunk_action_per_action(...)` 返回 `info` dict，字段同现状（`status/fallback_reason/duration/dense_steps/take_action_cnt_delta/topp_return_code/success_obs_time`），但 `duration` 改为**累加真实 `result["duration"]`**（口径修正）。新增可选参数 `v_cruise: float = None`（None→用 `DEFAULT_CRUISE_SD`）。

- [ ] **Step 1: 【本地】用新执行核替换整个函数体**（`def take_chunk_action_per_action` 起至 `return info` 止）

```python
    def take_chunk_action_per_action(self, action_chunk, vel_scale: float = 1.0,
                                     acc_scale: float = 1.0, v: float = 1.0,
                                     video_save_freq: int = -1, max_actions: int = None,
                                     v_cruise: float = None):
        """后端 B: 逐 action 两点 TOPPRA + 段间非零边界速度 (方法 B), 消除 stop-and-go.

        与原 (复用 take_action 的两点零速) 的差异:
          - 每个 action 绕过 mplib, 自调 retime_chunk(单 target) 并传非零 sd_start/sd_end,
            段间速度幅值不归零 (真加速); 速度方向在段间仍突变 (两点直线几何必然), 靠 PD 兜底.
          - current 用实测 qpos/qvel (闭环纠累积误差); sd_start = 实测速度切向投影,
            sd_end = v_cruise (末段 0), 钳到 per-joint vel 约束可行上限.
          - duration 累加真实 TOPP 时长 (非 dense_steps/250, 消除伪快统计).
          - 保留逐 action 粒度: 每 action 后查 success / 可中断 (partial chunk / async).
        retime_chunk 默认 (0,0) 不变, whole_chunk 不受影响.
        """
        from .robot.toppra_chunk_executor import (
            retime_chunk, compute_segment_sd_bounds, DEFAULT_CRUISE_SD,
        )
        if v_cruise is None:
            v_cruise = DEFAULT_CRUISE_SD

        info = {
            "status": "success",
            "fallback_reason": None,
            "duration": 0.0,
            "dense_steps": 0,
            "take_action_cnt_delta": 0,
            "topp_return_code": None,
            "success_obs_time": 0.0,
        }
        if self.take_action_cnt >= self.step_lim or self.eval_success:
            info["status"] = "truncated"
            return info

        action_chunk = np.asarray(action_chunk)
        if action_chunk.ndim != 2 or action_chunk.shape[0] == 0:
            info["status"] = "topp_fallback"
            info["fallback_reason"] = f"invalid chunk shape: {action_chunk.shape}"
            return info

        if v != 1.0:
            from .utils.chunk_accel import reconstruct_chunk
            action_chunk = reconstruct_chunk(action_chunk, float(v))
            if action_chunk.shape[0] == 0:
                info["status"] = "topp_fallback"
                info["fallback_reason"] = f"reconstruct_chunk produced empty chunk at v={v}"
                return info

        remaining_budget = int(self.step_lim - self.take_action_cnt)
        if action_chunk.shape[0] > remaining_budget:
            action_chunk = action_chunk[:remaining_budget]
            info["chunk_truncated"] = True
            if action_chunk.shape[0] == 0:
                info["status"] = "truncated"
                return info

        action_chunk = _apply_k_skip(action_chunk, max_actions)
        M = int(action_chunk.shape[0])

        # ---- 拆臂维度 (用命令值拿 dim, 与既有一致) + 拼 12dof chunk / gripper ----
        left_jointstate = self.robot.get_left_arm_jointState()
        right_jointstate = self.robot.get_right_arm_jointState()
        left_arm_dim = len(left_jointstate) - 1
        right_arm_dim = len(right_jointstate) - 1
        left_arm_actions = action_chunk[:, :left_arm_dim]
        left_gripper_actions = action_chunk[:, left_arm_dim]
        right_arm_actions = action_chunk[:, left_arm_dim + 1: left_arm_dim + 1 + right_arm_dim]
        right_gripper_actions = action_chunk[:, left_arm_dim + 1 + right_arm_dim]
        chunk_arm = np.concatenate([left_arm_actions, right_arm_actions], axis=1)        # (M, 12)
        chunk_gripper = np.stack([left_gripper_actions, right_gripper_actions], axis=1)  # (M, 2)

        # ---- 12 dof base vel/acc limits ----
        left_p = self.robot.left_mplib_planner.planner
        right_p = self.robot.right_mplib_planner.planner
        joint_vel_limits = np.concatenate([
            np.asarray(left_p.joint_vel_limits, dtype=np.float64),
            np.asarray(right_p.joint_vel_limits, dtype=np.float64),
        ])
        joint_acc_limits = np.concatenate([
            np.asarray(left_p.joint_acc_limits, dtype=np.float64),
            np.asarray(right_p.joint_acc_limits, dtype=np.float64),
        ])

        dense_steps = 0
        executed = 0
        total_duration = 0.0
        _t_obs0 = float(getattr(self, "_last_success_obs_time", 0.0))

        for i in range(M):
            if self.take_action_cnt >= self.step_lim or self.eval_success:
                break

            # --- 实测 current (闭环): 位置 + 速度 (去掉 gripper, 仅 arm dof) ---
            cur_left = self.robot.get_left_arm_real_jointState()[:left_arm_dim]
            cur_right = self.robot.get_right_arm_real_jointState()[:right_arm_dim]
            q_current = np.asarray(list(cur_left) + list(cur_right), dtype=np.float64)   # (12,)
            qd_left = self.robot.get_left_arm_real_jointVelocity()
            qd_right = self.robot.get_right_arm_real_jointVelocity()
            qd_actual = np.asarray(list(qd_left) + list(qd_right), dtype=np.float64)     # (12,)

            target = np.asarray(chunk_arm[i], dtype=np.float64)                          # (12,)
            sd_start, sd_end, tangent, seg_len = compute_segment_sd_bounds(
                q_current, qd_actual, target, joint_vel_limits, vel_scale,
                is_last=(i == M - 1), v_cruise=v_cruise,
            )

            # 退化段 (current≈target): 不 TOPP, 仅推进 cnt
            if tangent is None:
                self.take_action_cnt += 1
                executed += 1
                continue

            current_gripper = np.array([
                self.robot.get_left_gripper_val(), self.robot.get_right_gripper_val(),
            ]).reshape(-1)

            result = retime_chunk(
                current_state_arm=q_current,
                chunk_arm=target[None, :],                  # (1, 12) → 内部 vstack 成两点
                current_gripper=current_gripper,
                chunk_gripper=chunk_gripper[i][None, :],     # (1, 2)
                joint_vel_limits=joint_vel_limits,
                joint_acc_limits=joint_acc_limits,
                vel_scale=vel_scale, acc_scale=acc_scale, exec_hz=250,
                sd_start=sd_start, sd_end=sd_end,
            )

            # TOPP 失败 → 该段回退原 mplib take_action (stop-and-go), 保证不崩
            if result["status"] != "success":
                n = self.take_action(action_chunk[i], action_type="qpos",
                                     vel_scale=vel_scale, acc_scale=acc_scale,
                                     video_save_freq=video_save_freq)
                dense_steps += int(n or 0)   # take_action 内部已 cnt += 1
                executed += 1
                if self.eval_success:
                    break
                continue

            # --- 下发 dense 轨迹 (复刻 take_chunk_action 下发循环) ---
            dense_arm = result["dense_arm_pos"]        # (T, 12)
            dense_arm_vel = result["dense_arm_vel"]    # (T, 12)
            dense_gripper = result["dense_gripper"]    # (T, 2)
            total_duration += float(result["duration"])
            info["topp_return_code"] = result["return_code"]
            T = dense_arm.shape[0]
            T_cap = min(T, 100000)
            seg_success = False
            for t in range(T_cap):
                self._update_render()
                if self.render_freq:
                    self.viewer.render()
                self.robot.set_arm_joints(
                    dense_arm[t, :left_arm_dim], dense_arm_vel[t, :left_arm_dim], "left")
                self.robot.set_gripper(float(dense_gripper[t, 0]), "left")
                self.robot.set_arm_joints(
                    dense_arm[t, left_arm_dim:left_arm_dim + right_arm_dim],
                    dense_arm_vel[t, left_arm_dim:left_arm_dim + right_arm_dim], "right")
                self.robot.set_gripper(float(dense_gripper[t, 1]), "right")
                self.scene.step()
                self._update_render()
                dense_steps += 1
                self._tick_eval_video(video_save_freq)
                if self.check_success():
                    self.eval_success = True
                    _t = time.perf_counter()
                    self.get_obs()
                    if self.eval_video_path is not None and hasattr(self, "eval_video_ffmpeg"):
                        try:
                            self.eval_video_ffmpeg.stdin.write(
                                self.now_obs["observation"]["head_camera"]["rgb"].tobytes())
                        except Exception:
                            pass
                    self._last_success_obs_time = time.perf_counter() - _t
                    seg_success = True
                    break

            self.take_action_cnt += 1
            executed += 1
            if seg_success or self.eval_success:
                break

        info["dense_steps"] = dense_steps
        info["duration"] = total_duration   # 真实 TOPP 时长累加 (口径修正, 不再 dense_steps/250)
        info["take_action_cnt_delta"] = executed
        info["success_obs_time"] = max(
            0.0, float(getattr(self, "_last_success_obs_time", 0.0)) - _t_obs0
        )
        return info
```

- [ ] **Step 2: 【本地】py_compile**

Run: `python3 -m py_compile /home/xukainan/RoboTwin/envs/_base_task.py`
Expected: 无输出。

- [ ] **Step 3: 【本地】逻辑 review checklist**（人工核对，无法本地运行）

确认：① cnt 每 action 恰好 +1（正常分支末尾 +1；fallback 分支 take_action 内部 +1 后 continue；退化分支 +1 后 continue）——无双加/漏加；② 不写 chunk 开头 video header 帧（靠 `_tick_eval_video`）；③ `take_action`/`take_chunk_action`/streaming 函数体未被改动。

- [ ] **Step 4: 【云端】维护人集成验证（per_action 后端 rollout/eval）**

Run（云端，参考 `scripts/run_speedtune_*.sh` / `eval_speedtune.py` 的 per_action backend；具体 ckpt/路径/任务名由维护人按云端环境填）：跑 `per_action` 后端的 eval rollout。
Expected / 检查项：
1. **真加速**：执行时间 ≈/≤ `fixed_time` 后端（且因 duration 已改真实口径，`info["duration"]` 不再被低估）；
2. **运动连续**：可视化/视频里机械臂不再逐 action 走停（段间速度幅值不归零）；
3. **正确性**：task success 率不低于改前（P0）；
4. **逐 action 兼容**：执行到第 k 个 action 中断（partial chunk）/ 中途换 chunk（async）路径正常；
5. **fallback 率**：`info["fallback_reason"]` 频率低（高则说明 `v_cruise` 偏大致 toppra 频繁 infeasible，调小）；
6. **回归**：`whole_chunk` 后端行为与改前一致（`retime_chunk` 默认 `(0,0)`）。

- [ ] **Step 5: 提交（RoboTwin repo）**

```bash
git -C /home/xukainan/RoboTwin add envs/_base_task.py
git -C /home/xukainan/RoboTwin commit -m "feat(per_action): 执行核改方法B(逐action两点非零边界TOPPRA), 消除stop-and-go + duration真实口径"
```

---

## 后续可选（本计划不展开完整代码，留作独立小迭代）

这两项是 design §4/§7 的**验证/调参增强**，非核心功能；核心交付（Task 1-3）已可独立云端验证。实现前需先读对应文件确定接入点。

- **eval 平滑度指标**（`eval_speedtune.py`，expo-ft）：在 episode 汇总处新增 jerk / 速度过零次数统计（数据来源需让执行核在 `info` 暴露每段速度序列或过零计数 → 属对 Task 3 `info` 的增量扩展）。目的：量化"连续性"，避免再被伪快误导。**实现前读 `eval_speedtune.py` 的 `t_acc += info["duration"]`（约 `:240`）附近汇总逻辑。**
- **`v_cruise` 透传**（`client_robotwin/envs/robotwin_env.py::step_chunk` + `take_chunk_action_backend`，expo-ft/RoboTwin）：按既有透传 `v/vel_scale/acc_scale` 的方式把 `v_cruise` 从 expo-ft 端传到 `take_chunk_action_per_action`，便于云端不改 RoboTwin 即可调参。**起步用 `DEFAULT_CRUISE_SD` 默认常数即可，无需此项。**

---

## Self-Review

**1. Spec coverage**（对照 design 各节）：
- §3.2 执行循环 → Task 3 Step 1 ✓
- §3.3 sd_end=v_cruise 固定+末段0+钳 → Task 1 `compute_segment_sd_bounds` + Task 3 调用 ✓
- §3.4 sd_start 闭环读实测投影 → Task 2 qvel getter + Task 1 `compute_segment_sd_bounds` ✓
- §3.5 双臂12dof/gripper不参与TOPP/duration真实口径/fallback → Task 3 ✓（gripper 经 `retime_chunk` 插值）
- §4 扩 retime_chunk 默认(0,0)兼容 + 不改 mplib/take_action/whole_chunk/streaming → Task 1 + Global Constraints ✓
- §6 fallback（退化/解不出）→ Task 3 退化分支 + take_action fallback ✓
- §7 验证 → Task 3 Step 4 检查项 1-6 ✓（jerk/过零指标移「后续可选」）
- §8 v_cruise 云端调 / 升级路径 → `DEFAULT_CRUISE_SD` 常数 + 「后续可选」 ✓

**2. Placeholder scan**：无 TBD/TODO；每个改码步骤含完整代码；云端命令的环境参数（ckpt/路径）依赖云端，属 CLAUDE.md 既定约束非 placeholder。

**3. Type consistency**：`compute_segment_sd_bounds` 返回 `(sd_start, sd_end, tangent, seg_len)` 在 Task 1 定义、Task 3 解包一致；`retime_chunk` 新参 `sd_start/sd_end` 默认 `0.0` 在 Task 1 加、Task 3 传一致；`get_*_arm_real_jointVelocity` 返回 list 在 Task 2 定义、Task 3 `list(...)` 拼接一致；`info` 字段与现状/whole_chunk 对齐。
