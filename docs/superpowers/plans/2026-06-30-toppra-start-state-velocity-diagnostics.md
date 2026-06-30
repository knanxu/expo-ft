# TOPPRA Start-State and Velocity Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make whole-chunk TOPPRA start from measured arm qpos and compare per-action/whole-chunk real joint dynamics over one expert episode at progressively increasing constraints.

**Architecture:** Apply one surgical RoboTwin execution-kernel change and lock it with a fake-robot regression test. Add a standalone expo-ft diagnostic runner that wraps `scene.step`, records measured and commanded joint state at 250 Hz, runs identical expert actions through both backends without `k_skip`, and writes reproducible numeric artifacts and plots.

**Tech Stack:** Python, NumPy, pytest, SAPIEN/RoboTwin, TOPPRA, h5py, matplotlib.

---

### Task 1: Lock whole-chunk start-state semantics with TDD

**Files:**
- Create: `/home/xukainan/RoboTwin/envs/whole_chunk_start_state_test.py`
- Modify: `/home/xukainan/RoboTwin/envs/_base_task.py:1861-1877`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

from envs._base_task import Base_Task
import envs.robot.toppra_chunk_executor as tce


class FakeRobot:
    def get_left_arm_jointState(self):
        return [10.0] * 6 + [0.0]

    def get_right_arm_jointState(self):
        return [20.0] * 6 + [0.0]

    def get_left_arm_real_jointState(self):
        return [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0]

    def get_right_arm_real_jointState(self):
        return [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, 0.0]

    def get_left_gripper_val(self):
        return 0.0

    def get_right_gripper_val(self):
        return 0.0


def test_whole_chunk_retime_starts_from_measured_qpos(monkeypatch):
    captured = {}

    def fake_retime_chunk(**kwargs):
        captured.update(kwargs)
        return {
            "status": "fallback",
            "fallback_reason": "stop after input capture",
            "duration": 0.0,
            "dense_arm_pos": None,
            "dense_arm_vel": None,
            "dense_gripper": None,
            "return_code": None,
        }

    monkeypatch.setattr(tce, "retime_chunk", fake_retime_chunk)
    env = Base_Task()
    env.robot = FakeRobot()
    env.take_action_cnt = 0
    env.step_lim = 100
    env.eval_success = False
    env.take_chunk_action(np.zeros((2, 14)), vel_limit=1.0, acc_limit=1.0, v=1.0)

    expected = np.array([1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6], dtype=float)
    np.testing.assert_allclose(captured["current_state_arm"], expected)
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
cd /home/xukainan/RoboTwin
/home/xukainan/miniforge3/envs/RoboTwin/bin/python -m pytest envs/whole_chunk_start_state_test.py -q
```

Expected: one assertion failure showing `[10, 20]` drive-target groups rather than measured values.

- [ ] **Step 3: Implement the minimal production change**

Replace construction of `current_state_arm` in `take_chunk_action` with:

```python
left_real_qpos = np.asarray(
    self.robot.get_left_arm_real_jointState()[:left_arm_dim], dtype=np.float64)
right_real_qpos = np.asarray(
    self.robot.get_right_arm_real_jointState()[:right_arm_dim], dtype=np.float64)
current_state_arm = np.concatenate([left_real_qpos, right_real_qpos])
```

Keep drive-target reads only for arm dimension discovery. Do not alter `get_obs`, per-action execution, boundary speed, or `k_skip`.

- [ ] **Step 4: Verify GREEN and run TOPPRA unit regression**

```bash
cd /home/xukainan/RoboTwin
/home/xukainan/miniforge3/envs/RoboTwin/bin/python -m pytest envs/whole_chunk_start_state_test.py envs/robot/toppra_chunk_executor_test.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the isolated RoboTwin fix**

```bash
git -C /home/xukainan/RoboTwin add envs/_base_task.py envs/whole_chunk_start_state_test.py
git -C /home/xukainan/RoboTwin commit -m "fix(toppra): start whole chunk from measured qpos"
```

### Task 2: Add a reusable 250 Hz backend diagnostic runner

**Files:**
- Create: `/home/xukainan/expo-ft/scripts/diag_toppra_episode_dynamics.py`
- Read: `/home/xukainan/expo-ft/scripts/bench_exec_backends.py`

- [ ] **Step 1: Implement a scene proxy that records one row after every physics step**

The proxy exposes the wrapped scene through `__getattr__` and appends `qpos`, `qvel`,
`drive_qpos`, and `plan_qvel` after delegated `scene.step()`. Measured state comes from
articulation `get_qpos/get_qvel`; commands come from arm-joint drive targets.

```python
class SceneProbe:
    def __init__(self, scene, robot):
        self._scene = scene
        self.robot = robot
        self.qpos, self.qvel = [], []
        self.drive_qpos, self.plan_qvel = [], []

    def __getattr__(self, name):
        return getattr(self._scene, name)

    def _measured(self, side, velocity):
        entity = getattr(self.robot, f"{side}_entity")
        joints = getattr(self.robot, f"{side}_arm_joints")
        active = entity.get_active_joints()
        raw = entity.get_qvel() if velocity else entity.get_qpos()
        return [float(raw[active.index(joint)]) for joint in joints]

    def step(self):
        result = self._scene.step()
        self.qpos.append(self._measured("left", False) + self._measured("right", False))
        self.qvel.append(self._measured("left", True) + self._measured("right", True))
        joints = list(self.robot.left_arm_joints) + list(self.robot.right_arm_joints)
        self.drive_qpos.append([float(j.get_drive_target()[0]) for j in joints])
        self.plan_qvel.append([float(j.get_drive_velocity_target()[0]) for j in joints])
        return result
```

- [ ] **Step 2: Implement one-run collection without action truncation**

Use `bench_exec_backends.build_env`, `apply_real_params`, episode 0 HDF5 actions, seed 0,
and 50-action chunks. Dispatch one of these exact calls, always with `v=1.0`:

```python
env.take_chunk_action_per_action(
    chunk,
    vel_limit=vel_limit,
    acc_limit=acc_limit,
    v=1.0,
    max_actions=None,
    video_save_freq=-1,
)
env.take_chunk_action(
    chunk,
    vel_limit=vel_limit,
    acc_limit=acc_limit,
    v=1.0,
    video_save_freq=-1,
)
```

Compute:

```python
qacc = np.vstack([np.full((1, 12), np.nan), np.diff(qvel, axis=0) * 250.0])
tracking_error = drive_qpos - qpos
```

Write backend/config-specific NPZ and CSV. CSV contains `step,time_s` and 12 columns for each
of `qpos,qvel,qacc,plan_qvel,drive_qpos,tracking_error`. Assert all arrays contain exactly
`sum(info["dense_steps"])` rows.

- [ ] **Step 3: Implement the ordered safe sweep and incremental output**

```python
LIMIT_GRID = [(1.0, 1.0), (2.0, 2.0), (3.0, 4.0), (5.0, 8.0)]
BACKENDS = ("per_action", "whole_chunk")
```

At each grid point run per-action then whole-chunk. Write NPZ/CSV/JSON immediately after each
run and close the scene. Stop increasing limits only if a run has non-finite state. Preserve and
report TOPPRA fallback counts, but continue the ordered sweep because per-action deliberately falls
back to its original point-to-point executor and those transitions are part of the backend behavior.
Do not treat measured acceleration above the configured TOPPRA limit as a reason to discard the run:
that difference is a diagnostic result of PD/contact dynamics and must be reported.

- [ ] **Step 4: Implement statistics and plots**

Create `per_joint_summary.csv` with backend/limits/joint and absolute P50/P95/P99/max for
qvel/qacc/plan_qvel/tracking_error plus configured-limit violation fractions. Create:

- `velocity_timeseries_<limits>.png`: per-action and whole-chunk panels, six left-arm traces and velocity-limit lines;
- `acceleration_timeseries_<limits>.png`: equivalent acceleration panels and acceleration-limit lines;
- `peak_by_joint_<limits>.png`: grouped qvel/qacc maxima for all 12 joints.

- [ ] **Step 5: Syntax-check and commit the diagnostic runner**

```bash
cd /home/xukainan/expo-ft
/home/xukainan/miniforge3/envs/RoboTwin/bin/python -m py_compile scripts/diag_toppra_episode_dynamics.py
git add scripts/diag_toppra_episode_dynamics.py
git commit -m "feat: record TOPPRA episode joint dynamics"
```

### Task 3: Execute the sweep and verify artifacts

**Files:**
- Produce: `/home/xukainan/expo-ft/scripts/bench_out/toppra_episode_dynamics/`

- [ ] **Step 1: Run the diagnostic from the RoboTwin root**

```bash
cd /home/xukainan/RoboTwin
PYTHONPATH=/home/xukainan/RoboTwin:/home/xukainan/expo-ft \
  /home/xukainan/miniforge3/envs/RoboTwin/bin/python \
  /home/xukainan/expo-ft/scripts/diag_toppra_episode_dynamics.py \
  --task stack_blocks_two --rollout demo_clean --episode 0 --seed 0 \
  --output_dir /home/xukainan/expo-ft/scripts/bench_out/toppra_episode_dynamics
```

Expected: both backends finish `(1,1)` before higher grids; each completed run prints dense-step,
success, fallback, qvel peak, and qacc peak summaries.

- [ ] **Step 2: Verify numeric consistency from every NPZ**

For every run reload NPZ/JSON and assert:

```python
assert len(data["qpos"]) == summary["dense_steps"]
assert data["qpos"].shape == data["qvel"].shape == data["qacc"].shape
assert data["qpos"].shape[1] == 12
assert np.allclose(data["tracking_error"], data["drive_qpos"] - data["qpos"])
assert np.allclose(data["qacc"][1:], np.diff(data["qvel"], axis=0) * 250.0)
```

- [ ] **Step 3: Run fresh regression verification**

```bash
cd /home/xukainan/RoboTwin
/home/xukainan/miniforge3/envs/RoboTwin/bin/python -m pytest envs/whole_chunk_start_state_test.py envs/robot/toppra_chunk_executor_test.py -q
cd /home/xukainan/expo-ft
git diff --check
```

Expected: pytest exits zero and `git diff --check` emits no errors.
