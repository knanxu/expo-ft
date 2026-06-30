# per_action 提速 阶段 1 实现计划（绝对值制 + 力矩底座）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 SpeedTune `per_action_toppra`（连带 `chunk_toppra`）执行速度由 DQN 输出的绝对 `vel_limit`/`acc_limit` 唯一决定，并在执行层挂真机力矩 `force_limit` 作物理底座。

**Architecture:** TOPPRA executor 改绝对值制（去 `×base`、去 `v_cruise`、去 chunk 边界归零）；`_base_task` 两个 chunk 后端透传绝对值；`exec_backends` grid 改绝对 rad/s；`RoboTwinEnv` 透传新 key + 挂 `force_limit`；新增无负载 acc 标定脚本。抓取负载由 `force_limit` + RL 自动吸收，标定任务无关。

**Tech Stack:** Python, numpy, toppra, sapien（RoboTwin venv，云端）; jax（learner，与本阶段无关）。

## Global Constraints

- **零侵入 EXPO/BC**：不改 `robot.py`；不改 `take_action`(:1568) 主体；`force_limit` 默认 `None`(=∞=原行为)，仅 SpeedTune env 开启；`planner.py` 的 `PHYS_*_CEIL` 不动。
- **本地能跑验证**（探测确认：RTX 4060 8GB + `sapien`/`toppra`/`curobo`/`torch-cuda` 全可用，`envs._base_task` 可 import）：执行后端单测、RoboTwin 仿真、`play_once` 专家数据验证（时间/成功率）、acc 标定脚本均**本地跑**，用 RoboTwin conda env：`RT=/home/xukainan/miniforge3/envs/RoboTwin/bin/python`（pytest 用 `$RT -m pytest`）。expo-ft 纯 numpy 测试可用 stspeed/RoboTwin env。仅 VLA(3B)/完整 SpeedTune DQN 训练显存不够 → 云端。原计划标 "Run(云端)" 的执行后端/标定步骤改为**本地 `$RT` 跑**。
- **commit 需维护人批准**（项目约定）：commit 步骤照写，执行时由维护人确认。
- **两个 repo**：`RoboTwin`（`envs/robot/toppra_chunk_executor.py`、`envs/_base_task.py`）与 `expo-ft`（`expo_ft/speedtune/exec_backends.py`、`client_robotwin/...`、`scripts/...`、`test/...`）分别提交。
- **绝对值制**：`vel_limit`(rad/s)、`acc_limit`(rad/s²) 直接是关节上限；标量广播到 12 dof；钳 `PHYS_CEIL` 安全网。
- **grid 档数若变 → `ExecBackend.head_sizes` 变 → DQN 须重训**（本就是新 run）。
- grid 具体数值是**占位**，由 Task 6 标定脚本（云端）产出后回填；占位不影响接口与逻辑正确性。

---

## File Structure

**RoboTwin repo**（云端 `/home/chenlu/RoboTwin`，本地 `/home/xukainan/RoboTwin` 对照可改）：
- `envs/robot/toppra_chunk_executor.py` — `retime_chunk` / `compute_segment_sd_bounds` 绝对值化、去隐藏约束、`PHYS_VEL_CEIL→5.0`（Task 1）
- `envs/robot/toppra_chunk_executor_test.py` — 上述单元测试更新（Task 1）
- `envs/_base_task.py` — `take_chunk_action_per_action` / `take_chunk_action` 透传绝对值、去取 base、回退适配（Task 2）

**expo-ft repo**（本地 `/home/xukainan/expo-ft`）：
- `expo_ft/speedtune/exec_backends.py` — grid 改绝对值 + 变量改名（Task 3）
- `test/exec_backends_test.py` — 上述单元测试更新（Task 3）
- `client_robotwin/envs/robotwin_env.py` — `step_chunk` 透传新 key（Task 4）+ `force_limit` 机制（Task 5）
- `scripts/calibrate_acc_grid.py` — 无负载 q̈_max 标定脚本（Task 6，新建）

---

## Task 1: toppra_chunk_executor 绝对值化 + 去隐藏约束

**Files:**
- Modify: `RoboTwin/envs/robot/toppra_chunk_executor.py`（`retime_chunk`、`compute_segment_sd_bounds`、`PHYS_VEL_CEIL`、删 `DEFAULT_CRUISE_SD`）
- Test: `RoboTwin/envs/robot/toppra_chunk_executor_test.py`

**Interfaces:**
- Produces:
  - `retime_chunk(current_state_arm, chunk_arm, current_gripper, chunk_gripper, vel_limit: float, acc_limit: float, exec_hz=250, phys_vel_ceil=PHYS_VEL_CEIL, phys_acc_ceil=PHYS_ACC_CEIL, sd_start=0.0, sd_end=0.0) -> dict`（去掉 `joint_vel_limits/joint_acc_limits/vel_scale/acc_scale`，改收绝对标量 `vel_limit/acc_limit`）
  - `compute_segment_sd_bounds(q_current, qd_actual, target, vel_limit: float, phys_vel_ceil=PHYS_VEL_CEIL, safety=0.99) -> (sd_start, sd_end, tangent, seg_len)`（去掉 `joint_vel_limits/vel_scale/is_last/v_cruise`；不再有 `is_last` 归零）
  - `PHYS_VEL_CEIL=5.0`；`PHYS_ACC_CEIL`（占位 `10.0`，Task 6 标定后回填）；`DEFAULT_CRUISE_SD` 删除

- [ ] **Step 1: 更新单元测试（写期望行为）**

替换 `toppra_chunk_executor_test.py` 中 `retime_chunk`/`compute_segment_sd_bounds` 相关用例为绝对值制：

```python
import numpy as np
from envs.robot.toppra_chunk_executor import retime_chunk, compute_segment_sd_bounds, PHYS_VEL_CEIL

def test_retime_chunk_absolute_limits_no_base():
    # 绝对值制：vel_limit/acc_limit 直接作上限，不依赖 base
    res = retime_chunk(
        current_state_arm=np.zeros(12),
        chunk_arm=np.tile(np.linspace(0, 1.0, 12), (3, 1)),
        current_gripper=np.zeros(2),
        chunk_gripper=np.zeros((3, 2)),
        vel_limit=2.0, acc_limit=2.0, exec_hz=250,
    )
    assert res["status"] == "success"
    # 关节速度不超过 vel_limit（含数值容差）
    assert np.max(np.abs(res["dense_arm_vel"])) <= 2.0 + 1e-3

def test_retime_chunk_clamped_by_phys_ceil():
    # vel_limit 超 PHYS_VEL_CEIL 时被钳
    res = retime_chunk(
        current_state_arm=np.zeros(12),
        chunk_arm=np.tile(np.linspace(0, 1.0, 12), (3, 1)),
        current_gripper=np.zeros(2), chunk_gripper=np.zeros((3, 2)),
        vel_limit=999.0, acc_limit=999.0, exec_hz=250,
    )
    assert res["status"] == "success"
    assert np.max(np.abs(res["dense_arm_vel"])) <= PHYS_VEL_CEIL + 1e-3

def test_sd_bounds_no_vcruise_no_islast_zero():
    # 去 v_cruise + 去 is_last 归零：段末速度 = safety * sd_max（非 0）
    q = np.zeros(12); qd = np.zeros(12)
    target = np.zeros(12); target[0] = 1.0  # 单关节运动，tangent[0]=1
    sd_start, sd_end, tangent, seg_len = compute_segment_sd_bounds(
        q, qd, target, vel_limit=2.0, safety=0.99,
    )
    assert tangent is not None and seg_len > 0
    # sd_max = vel_limit / max|tangent| = 2.0 / 1.0 = 2.0；sd_end = 0.99*2.0
    assert abs(sd_end - 0.99 * 2.0) < 1e-6
    assert sd_end > 0.0  # 不再归零

def test_sd_bounds_degenerate_returns_none():
    q = np.zeros(12); qd = np.zeros(12)
    sd_start, sd_end, tangent, seg_len = compute_segment_sd_bounds(
        q, qd, target=np.zeros(12), vel_limit=2.0,
    )
    assert tangent is None  # current≈target
```

- [ ] **Step 2: Run(云端) 测试确认失败**

Run(云端): `cd /home/chenlu/RoboTwin && python -m pytest envs/robot/toppra_chunk_executor_test.py -v`
Expected: FAIL（旧签名 `retime_chunk` 不接受 `vel_limit`，`compute_segment_sd_bounds` 仍要 `is_last/v_cruise`）

- [ ] **Step 3: 改 PHYS_CEIL 常量 + 删 DEFAULT_CRUISE_SD**

`toppra_chunk_executor.py:25-30` 改为：

```python
PHYS_VEL_CEIL: float = 5.0   # rad/s（真机 ARX5 5~5.5，安全网）
PHYS_ACC_CEIL: float = 10.0  # rad/s²（占位安全网；Task 6 标定 q̈_max 峰值后回填，须 ≥ acc grid 上界）
# DEFAULT_CRUISE_SD 删除（绝对值制下段间速度由 vel_limit 决定，不再有独立巡航钳）
```

- [ ] **Step 4: 改 retime_chunk 签名与 vel/acc 计算**

签名（`:33-47`）去掉 `joint_vel_limits, joint_acc_limits, vel_scale, acc_scale`，改 `vel_limit, acc_limit`：

```python
def retime_chunk(
    current_state_arm: np.ndarray,
    chunk_arm: np.ndarray,
    current_gripper: np.ndarray,
    chunk_gripper: np.ndarray,
    vel_limit: float,
    acc_limit: float,
    exec_hz: int = 250,
    phys_vel_ceil: float = PHYS_VEL_CEIL,
    phys_acc_ceil: float = PHYS_ACC_CEIL,
    sd_start: float = 0.0,
    sd_end: float = 0.0,
):
```

`:122-126` 的 vel/acc 计算改为绝对值标量广播（dof 用 `kept_arm.shape[1]`）：

```python
    # 5. TOPPRA 约束（绝对值制：vel_limit/acc_limit 直接作上限，钳物理天花板安全网）
    dof = kept_arm.shape[1]
    vel_lim = np.full(dof, min(float(vel_limit), float(phys_vel_ceil)), dtype=np.float64)
    acc_lim = np.full(dof, min(float(acc_limit), float(phys_acc_ceil)), dtype=np.float64)
```

（删除原 `vel_lim = joint_vel_limits * vel_scale ...` 四行及其后的 `limits dof mismatch` 检查——绝对值标量广播 dof 必然一致。）

- [ ] **Step 5: 改 compute_segment_sd_bounds 签名与逻辑**

整函数（`:241-278`）替换为：

```python
def compute_segment_sd_bounds(
    q_current: np.ndarray,
    qd_actual: np.ndarray,
    target: np.ndarray,
    vel_limit: float,
    phys_vel_ceil: float = PHYS_VEL_CEIL,
    safety: float = 0.99,
):
    """逐 action 两点段在弧长参数化下的边界路径速度 (sd_start, sd_end)。

    绝对值制 + 去隐藏约束：
      - sd_start = 实测关节速度在本段单位切向上的投影 (闭环), clip >= 0.
      - sd_end   = safety * sd_max（**不再有 is_last 归零、不再有 v_cruise 钳**）。
        sd_max = vel_limit / max_j|tangent_j|（标量 vel_limit，钳 phys_vel_ceil）。
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
    vel_lim = min(float(vel_limit), float(phys_vel_ceil))
    sd_max = vel_lim / float(np.max(np.abs(tangent)))
    sd_end = float(safety) * sd_max
    return sd_start, sd_end, tangent, seg_len
```

- [ ] **Step 6: 本地语法检查**

Run: `python -m ast /home/xukainan/RoboTwin/envs/robot/toppra_chunk_executor.py && echo OK`
Expected: `OK`（仅语法；逻辑由云端 pytest 验证）

- [ ] **Step 7: Run(云端) 测试确认通过**

Run(云端): `cd /home/chenlu/RoboTwin && python -m pytest envs/robot/toppra_chunk_executor_test.py -v`
Expected: PASS

- [ ] **Step 8: Commit（RoboTwin repo，维护人确认后）**

```bash
cd /home/chenlu/RoboTwin && git add envs/robot/toppra_chunk_executor.py envs/robot/toppra_chunk_executor_test.py
git commit -m "feat(speedtune): toppra executor 改绝对值制 + 去 v_cruise/chunk边界归零, PHYS_VEL_CEIL->5.0"
```

---

## Task 2: _base_task 两个 chunk 后端透传绝对值

**Files:**
- Modify: `RoboTwin/envs/_base_task.py`（`take_chunk_action_per_action`:1989、`take_chunk_action`:1787）

**Interfaces:**
- Consumes: Task 1 的 `retime_chunk(...,vel_limit,acc_limit,...)`、`compute_segment_sd_bounds(q_current,qd_actual,target,vel_limit,safety=0.99)`
- Produces:
  - `take_chunk_action_per_action(action_chunk, vel_limit=5.0, acc_limit=10.0, v=1.0, video_save_freq=-1, max_actions=None) -> info`
  - `take_chunk_action(action_chunk, vel_limit=5.0, acc_limit=10.0, v=1.0, video_save_freq=-1) -> info`

- [ ] **Step 1: 改 take_chunk_action_per_action 签名 + 删 v_cruise/base 取用**

签名（`:1989-1992`）改：

```python
    def take_chunk_action_per_action(self, action_chunk, vel_limit: float = 5.0,
                                     acc_limit: float = 10.0, v: float = 1.0,
                                     video_save_freq: int = -1, max_actions: int = None):
```

`:2004-2008` 的 import 改为（删 `compute_segment_sd_bounds, DEFAULT_CRUISE_SD` 之外的 v_cruise 逻辑）：

```python
        from .robot.toppra_chunk_executor import retime_chunk, compute_segment_sd_bounds
```

删除 `:2007-2008`（`if v_cruise is None: v_cruise = DEFAULT_CRUISE_SD`）。

删除 `:2060-2070`（取 mplib base 的 `joint_vel_limits/joint_acc_limits` 整段——绝对值制不再需要 base）。

- [ ] **Step 2: 改 per_action 内循环的 sd_bounds / retime_chunk 调用**

`:2090-2093` 改为（去 `joint_vel_limits/vel_scale/is_last/v_cruise`）：

```python
            sd_start, sd_end, tangent, seg_len = compute_segment_sd_bounds(
                q_current, qd_actual, target, vel_limit=vel_limit,
            )
```

`:2105-2114` 的 `retime_chunk` 调用改为：

```python
            result = retime_chunk(
                current_state_arm=q_current,
                chunk_arm=target[None, :],
                current_gripper=current_gripper,
                chunk_gripper=chunk_gripper[i][None, :],
                vel_limit=vel_limit, acc_limit=acc_limit, exec_hz=250,
                sd_start=sd_start, sd_end=sd_end,
            )
```

`:2118-2120` 回退 `take_action`：绝对值制下无 scale，回退用 mplib 原生约束（`scaled_limits(1,1)` no-op）：

```python
                n = self.take_action(action_chunk[i], action_type="qpos",
                                     video_save_freq=video_save_freq)
```

- [ ] **Step 3: 改 take_chunk_action（whole_chunk）签名 + base 取用 + retime 调用**

签名（`:1787`）改 `vel_scale/acc_scale` → `vel_limit/acc_limit`（默认 5.0/10.0）。

删除 `:1886-1896`（取 mplib base 的 `left_p/right_p` + `joint_vel_limits/joint_acc_limits` 整段）。

`:1899-1909` 的 `retime_chunk` 调用改为：

```python
        retimed = retime_chunk(
            current_state_arm=current_state_arm,
            chunk_arm=chunk_arm,
            current_gripper=current_gripper,
            chunk_gripper=chunk_gripper,
            vel_limit=vel_limit, acc_limit=acc_limit, exec_hz=250,
        )
```

- [ ] **Step 4: 本地语法检查**

Run: `python -m ast /home/xukainan/RoboTwin/envs/_base_task.py && echo OK`
Expected: `OK`

- [ ] **Step 5: Run(云端) 冒烟（import + 构造，无 sapien 场景）**

Run(云端): `cd /home/chenlu/RoboTwin && python -c "import envs._base_task; print('import OK')"`
Expected: `import OK`（完整执行验证留待 Task 4 后的 eval 集成）

- [ ] **Step 6: Commit（RoboTwin repo，维护人确认后）**

```bash
cd /home/chenlu/RoboTwin && git add envs/_base_task.py
git commit -m "feat(speedtune): per_action/whole_chunk 透传绝对 vel_limit/acc_limit, 去 base 取用"
```

---

## Task 3: exec_backends grid 改绝对值 + 变量改名

**Files:**
- Modify: `expo-ft/expo_ft/speedtune/exec_backends.py`（`_DEFAULT_*`、`_default_specs`）
- Test: `expo-ft/test/exec_backends_test.py`

**Interfaces:**
- Produces: `per_action_toppra` / `chunk_toppra` 的 vars = `(v, vel_limit, acc_limit)`；`decode` 返回 `speed_params = {"v":…, "vel_limit":…, "acc_limit":…}`。`fixed_time` 仍只 `(v,)`，不变。

- [ ] **Step 1: 更新单元测试**

`test/exec_backends_test.py` 增改：

```python
from expo_ft.speedtune.exec_backends import build_backend

def test_per_action_vars_renamed_absolute():
    be = build_backend("per_action_toppra")
    names = [v.name for v in be.vars]
    assert names == ["v", "vel_limit", "acc_limit"]

def test_per_action_decode_keys_absolute():
    be = build_backend("per_action_toppra")
    speed_params, v_list = be.decode([0, 0, 0])
    assert set(speed_params.keys()) == {"v", "vel_limit", "acc_limit"}
    assert len(v_list) == 3

def test_fixed_time_unchanged():
    be = build_backend("fixed_time")
    assert [v.name for v in be.vars] == ["v"]
```

- [ ] **Step 2: Run(云端) 测试确认失败**

Run(云端): `cd /home/chenlu/expo-ft && python -m pytest test/exec_backends_test.py -v`
Expected: FAIL（变量名还是 `vel_scale/acc_scale`）

- [ ] **Step 3: 改默认 grid 常量（绝对值占位，Task 6 回填）**

`exec_backends.py:138-141` 改为：

```python
# 绝对值制：grid 值直接是关节上限（rad/s, rad/s²），不再 ×base。占位值，Task 6 标定脚本产出后回填。
_DEFAULT_V = (1.0, 1.5, 2.0, 3.0, 4.0)           # chunk 压缩比（MDP horizon），不变
_DEFAULT_VEL_LIMIT = (1.0, 2.0, 3.0, 4.0, 5.0)   # rad/s（真机 ≤5.5）；占位
_DEFAULT_ACC_LIMIT = (2.0, 4.0, 6.0, 8.0)        # rad/s²；占位，Task 6 标定后回填
```

- [ ] **Step 4: 改 _default_specs（变量改名 + 绝对值）**

`exec_backends.py:150-167` 的 `per_action_toppra` / `chunk_toppra` 改为：

```python
        "per_action_toppra": {
            "v": dict(grid=_DEFAULT_V, faster_is="larger", alpha=0.5, beta=1.0),
            "vel_limit": dict(grid=_DEFAULT_VEL_LIMIT, faster_is="larger", alpha=0.5, beta=1.0),
            "acc_limit": dict(grid=_DEFAULT_ACC_LIMIT, faster_is="larger", alpha=0.5, beta=1.0),
        },
        "chunk_toppra": {
            "v": dict(grid=_DEFAULT_V, faster_is="larger", alpha=0.5, beta=1.0),
            "vel_limit": dict(grid=_DEFAULT_VEL_LIMIT, faster_is="larger", alpha=0.5, beta=1.0),
            "acc_limit": dict(grid=_DEFAULT_ACC_LIMIT, faster_is="larger", alpha=0.5, beta=1.0),
        },
```

（`fixed_time` 的 `"v"` 项不变。文件顶部模块 docstring 里 "v + TOPPRA vel_scale + acc_scale" 改 "v + vel_limit + acc_limit"。）

- [ ] **Step 5: 本地语法检查**

Run: `python -m ast /home/xukainan/expo-ft/expo_ft/speedtune/exec_backends.py && echo OK`
Expected: `OK`

- [ ] **Step 6: Run(云端) 测试确认通过**

Run(云端): `cd /home/chenlu/expo-ft && python -m pytest test/exec_backends_test.py -v`
Expected: PASS

- [ ] **Step 7: Commit（expo-ft repo，维护人确认后）**

```bash
cd /home/chenlu/expo-ft && git add expo_ft/speedtune/exec_backends.py test/exec_backends_test.py
git commit -m "feat(speedtune): exec_backends grid 改绝对值 vel_limit/acc_limit"
```

---

## Task 4: robotwin_env.step_chunk 透传新 key

**Files:**
- Modify: `expo-ft/client_robotwin/envs/robotwin_env.py`（`step_chunk`:337-387）

**Interfaces:**
- Consumes: Task 2 的 `take_chunk_action_per_action(...,vel_limit,acc_limit,v,...)` / `take_chunk_action(...,vel_limit,acc_limit,v,...)`；Task 3 的 `speed_params` key `{v, vel_limit, acc_limit}`
- Produces: `step_chunk(chunk, speed_params, exec_backend)` 不变的返回 dict

- [ ] **Step 1: 改 step_chunk 读 key + 后端调用**

`:359-361` 改为：

```python
        v = float(speed_params.get("v", 1.0))
        vel_limit = float(speed_params.get("vel_limit", 5.0))
        acc_limit = float(speed_params.get("acc_limit", 10.0))
```

`:364-375` 的三分支改为（streaming 不变；per_action/whole_chunk 传 vel_limit/acc_limit）：

```python
        if rt == "streaming":
            info = self._call_backend(self.env.take_chunk_action_streaming, chunk,
                                      v=v, hold_steps=self._stream_hold_steps,
                                      max_actions=self._k_skip, video_save_freq=vsf)
        elif rt == "per_action":
            info = self._call_backend(self.env.take_chunk_action_per_action, chunk,
                                      vel_limit=vel_limit, acc_limit=acc_limit, v=v,
                                      max_actions=self._k_skip, video_save_freq=vsf)
        else:  # whole_chunk
            info = self._call_backend(self.env.take_chunk_action, chunk,
                                      vel_limit=vel_limit, acc_limit=acc_limit, v=v,
                                      video_save_freq=vsf)
```

- [ ] **Step 2: 本地语法检查**

Run: `python -m ast /home/xukainan/expo-ft/client_robotwin/envs/robotwin_env.py && echo OK`
Expected: `OK`

- [ ] **Step 3: Commit（expo-ft repo，维护人确认后）**

```bash
cd /home/chenlu/expo-ft && git add client_robotwin/envs/robotwin_env.py
git commit -m "feat(speedtune): step_chunk 透传绝对 vel_limit/acc_limit"
```

---

## Task 5: robotwin_env force_limit 机制（执行层力矩底座）

**Files:**
- Modify: `expo-ft/client_robotwin/envs/robotwin_env.py`（`__init__`、`reset` 后施加）

**Interfaces:**
- Produces: `RoboTwinEnv(..., force_limit=None)` —— `force_limit` 为 per-joint 单臂力矩上限 list（长度=单臂 arm dof，左右臂同值）或标量或 `None`(=∞，不施加)。每次 `reset()` 布场后对左右臂 arm 关节施加。

- [ ] **Step 1: __init__ 接收 force_limit**

`:42-78` 的 `__init__` kwargs 解析区加：

```python
        # SpeedTune 执行层力矩底座（真机 τ_max）。None=∞=原行为（EXPO/BC 零侵入）。
        # 可传标量（所有 arm 关节同）或 per-joint list（长度=单臂 arm dof）。
        _fl = kwargs.get("force_limit", None)
        self._force_limit = _fl if _fl not in (None, "", 0) else None
```

- [ ] **Step 2: 新增 _apply_force_limit 方法**

在 `step_chunk` 之前插入：

```python
    def _apply_force_limit(self):
        """对左右臂 arm 关节施加 PD force_limit（真机 τ_max 物理底座）。

        None → 不施加（默认 ∞，EXPO/BC 零侵入）。不改 robot.py：在 env 层重设
        set_drive_property，仅覆盖 force_limit，stiffness/damping 沿用 robot 已设值。
        """
        if self._force_limit is None:
            return
        robot = getattr(self.env, "robot", None)
        if robot is None:
            return
        try:
            arms = [(robot.left_arm_joints, robot.left_joint_stiffness, robot.left_joint_damping),
                    (robot.right_arm_joints, robot.right_joint_stiffness, robot.right_joint_damping)]
            for joints, stiff, damp in arms:
                fl = self._force_limit
                fl_list = fl if isinstance(fl, (list, tuple)) else [float(fl)] * len(joints)
                for j, jf in zip(joints, fl_list):
                    j.set_drive_property(stiffness=stiff, damping=damp, force_limit=float(jf))
        except Exception:
            logging.exception("[RoboTwin] apply force_limit failed (继续不限力矩)")
```

- [ ] **Step 3: reset 后调用 _apply_force_limit**

`reset()` 中 `setup_demo` 成功后（`:234` 设 instruction 附近）加一行 `self._apply_force_limit()`。

- [ ] **Step 4: 本地语法检查**

Run: `python -m ast /home/xukainan/expo-ft/client_robotwin/envs/robotwin_env.py && echo OK`
Expected: `OK`

- [ ] **Step 5: Run(云端) 集成冒烟**

Run(云端): 起 per_action server（`force_limit` 经 config-as-kwargs 传入），跑 1 episode，确认 `set_drive_property(force_limit=...)` 无异常、episode 正常结束。

- [ ] **Step 6: Commit（expo-ft repo，维护人确认后）**

```bash
cd /home/chenlu/expo-ft && git add client_robotwin/envs/robotwin_env.py
git commit -m "feat(speedtune): RoboTwinEnv 可选 force_limit 力矩底座(默认 None, EXPO/BC 零侵入)"
```

---

## Task 6: 无负载 acc 标定脚本

**Files:**
- Create: `expo-ft/scripts/calibrate_acc_grid.py`

**Interfaces:**
- Produces: 命令行脚本，输入 `--tau_max`（per-joint 单臂，逗号分隔）、`--task_config`、`--robotwin_root`；输出每关节 `q̈_max=(τ_max−g)/M_robot` 的 min/max/分位，并打印建议 acc grid（回填 Task 3 的 `_DEFAULT_ACC_LIMIT` 与 `PHYS_ACC_CEIL`）。

- [ ] **Step 1: 写标定脚本**

```python
"""无负载 acc 上界标定（任务无关）：q̈_j,max = (τ_j,max − g_j(q)) / M_robot_jj(q)。

抓取负载不在此标定——由执行层 force_limit + RL 吸收（见 spec §3#3）。
扫机器人工作空间随机位形，统计 q̈_max 范围，给 acc grid 建议值。
云端跑（需 sapien + RoboTwin env）。
"""
import argparse
import numpy as np

from client_robotwin.envs.robotwin_env import RoboTwinEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau_max", required=True,
                    help="per-joint 单臂力矩上限 N·m，逗号分隔，如 '40,40,20,20,10,10'")
    ap.add_argument("--task_config", default="configs/task/robotwin_stack_blocks.py")
    ap.add_argument("--robotwin_root", default="/home/chenlu/RoboTwin")
    ap.add_argument("--n_samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import os
    os.chdir(args.robotwin_root)
    tau = np.array([float(x) for x in args.tau_max.split(",")], dtype=np.float64)

    # 构造 env（复用 RoboTwinEnv 的场景解析；不跑 policy，只取 robot articulation）
    import importlib.util
    spec = importlib.util.spec_from_file_location("task_cfg", args.task_config)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    cfg = mod.get_config() if hasattr(mod, "get_config") else mod.config
    env = RoboTwinEnv(task_name=cfg["task_name"],
                      robotwin_task_config=cfg.get("robotwin_task_config", {}))
    env.reset()
    robot = env.env.robot
    art = robot.left_entity  # 单臂 articulation
    arm_joints = robot.left_arm_joints
    dof = len(arm_joints)
    rng = np.random.default_rng(args.seed)

    qmin, qmax = _joint_limits(art, arm_joints)
    pin = art.create_pinocchio_model()   # sapien pinocchio：质量矩阵/动力学（API 已本地验证）
    qddmax_samples = []
    for _ in range(args.n_samples):
        q = rng.uniform(qmin, qmax)
        _set_full_qpos(art, arm_joints, q)
        qpos_full = art.get_qpos()
        M = np.asarray(pin.compute_generalized_mass_matrix(qpos_full))  # 不是 art.compute_...！
        g = np.asarray(art.compute_passive_force(gravity=True,
                                                 coriolis_and_centrifugal=False))
        idx = _arm_joint_indices(art, arm_joints)
        M_jj = np.array([M[i, i] for i in idx])
        g_j = np.array([g[i] for i in idx])
        qddmax = np.maximum((tau - np.abs(g_j)) / np.maximum(M_jj, 1e-9), 0.0)
        qddmax_samples.append(qddmax)

    arr = np.stack(qddmax_samples)  # (n, dof)
    print("per-joint q̈_max (rad/s^2):")
    print("  min   :", np.round(arr.min(0), 3))
    print("  p10   :", np.round(np.percentile(arr, 10, 0), 3))
    print("  median:", np.round(np.median(arr, 0), 3))
    print("  max   :", np.round(arr.max(0), 3))
    # 标量 grid 建议：上界取全关节 p10 的较高分位（弱关节由 per-joint force_limit 兜底）
    upper = float(np.percentile(arr.min(1), 90))  # 每样本最弱关节的 90 分位
    lower = float(np.percentile(arr.min(1), 10))
    grid = np.round(np.linspace(lower, upper, 4), 2)
    print(f"\n建议 _DEFAULT_ACC_LIMIT = {tuple(grid)}")
    print(f"建议 PHYS_ACC_CEIL >= {round(upper * 1.2, 2)}")


def _arm_joint_indices(art, arm_joints):
    active = art.get_active_joints()
    return [active.index(j) for j in arm_joints]


def _joint_limits(art, arm_joints):
    lim = np.asarray(art.get_qlimits())  # (n_active, 2)
    idx = _arm_joint_indices(art, arm_joints)
    return lim[idx, 0], lim[idx, 1]


def _set_full_qpos(art, arm_joints, q_arm):
    full = np.asarray(art.get_qpos()).copy()
    for i, qi in zip(_arm_joint_indices(art, arm_joints), q_arm):
        full[i] = qi
    art.set_qpos(full)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 本地语法检查**

Run: `python -m ast /home/xukainan/expo-ft/scripts/calibrate_acc_grid.py && echo OK`
Expected: `OK`

- [ ] **Step 3: Run(云端) 标定，回填 grid**

Run(云端): `cd /home/chenlu/expo-ft && python -m scripts.calibrate_acc_grid --tau_max 40,40,20,20,10,10`
Expected: 打印 per-joint q̈_max 范围 + 建议 `_DEFAULT_ACC_LIMIT` / `PHYS_ACC_CEIL`。维护人据此回填 Task 3 的 `_DEFAULT_ACC_LIMIT` 与 Task 1 的 `PHYS_ACC_CEIL`，并 commit。

- [ ] **Step 4: Commit（expo-ft repo，维护人确认后）**

```bash
cd /home/chenlu/expo-ft && git add scripts/calibrate_acc_grid.py
git commit -m "feat(speedtune): 无负载 acc 上界标定脚本(任务无关, 含重力)"
```

---

## 集成验证（全部任务后，云端）

- [ ] 重跑 `eval_speedtune_compare`（确保 `9713d8a` 端口预检生效），确认：
  1. per_action `dense_steps` 大幅下降（对比基线 12003）；
  2. `success_rate` 不降；
  3. `vel_limit`/`acc_limit` 档位与实测速度/加速度单调对应；
  4. per_action 挂 force_limit 后正常**不饱和**（轨迹力矩可行）。
- [ ] `print(robot.left_entity.compute_generalized_mass_matrix(...))` 抽查标定脚本数值合理。

---

## Self-Review

**Spec coverage:**
- §4#1 grid 绝对值改名 → Task 3 ✓
- §4#2 retime_chunk 绝对值 → Task 1 ✓
- §4#3 sd_bounds 去 v_cruise/去归零/safety 0.99 → Task 1 ✓
- §4#4 PHYS_VEL_CEIL→5.0、PHYS_ACC_CEIL → Task 1（值待 Task 6 回填）✓
- §4#5 调用方传绝对值/去 base/回退 → Task 2 ✓
- §4#6 step_chunk 透传 → Task 4 ✓
- §4#7 force_limit → Task 5 ✓
- §4#8 TOPP 回退适配 → Task 2 Step 2 ✓
- §5 acc 力矩标定（无负载+重力） → Task 6 ✓
- §7 零侵入（force_limit 默认 None、不改 robot.py、planner PHYS_CEIL 不动）→ Task 5 + Global Constraints ✓
- 阶段 2（fixed_time 验证）→ 不在本计划（独立计划）✓

**Placeholder scan:** grid 数值为标注的"占位，Task 6 回填"，非漏洞（标定流程在 Task 6 完整给出）；其余步骤均有实际代码/命令。

**Type consistency:** `vel_limit/acc_limit`（绝对标量 float）贯穿 Task 1→2→3→4；`speed_params` key `{v,vel_limit,acc_limit}` 在 Task 3 产、Task 4 消，一致；`retime_chunk`/`compute_segment_sd_bounds` 新签名 Task 1 定、Task 2 用，一致。
