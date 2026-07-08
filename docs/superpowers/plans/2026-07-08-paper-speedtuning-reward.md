# Paper SpeedTuning Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a config-selectable paper SpeedTuning reward mode with chunk-level reward/discount while preserving the current success-gated default.

**Architecture:** Keep reward selection inside `flush_pending_episode`, because that is the existing episode relabel boundary. The SpeedTuning paper baseline treats one chunk as the action/reward/discount unit, so paper mode stores one chunk-level reward and one `gamma` per DQN transition. `SpeedTuneReplayBuffer` still accepts per-transition discounts for compatibility/future use. Use dynamic C51 support for paper mode so alpha sweeps keep useful atom resolution.

**Tech Stack:** Python, numpy, ml_collections ConfigDict, existing SpeedTune DQN and local script-style tests.

---

### Task 1: ReplayBuffer Variable Discount

**Files:**
- Modify: `expo_ft/data/speedtune_buffer.py`
- Test: `test/speedtune_buffer_test.py`

- [x] **Step 1: Write the failing test**

Add this test to `test/speedtune_buffer_test.py`:

```python
def test_variable_transition_discount_enters_nstep_return():
    buf = SpeedTuneReplayBuffer(
        capacity=10, feat_dim=FEAT, n_heads=NH, n_step=2, gamma=0.99, seed=0
    )
    buf.insert(_feat(0), [0, 0, 0], 1.0, _feat(1), False, discount=0.5)
    buf.insert(_feat(1), [0, 0, 0], 2.0, _feat(2), False, discount=0.25)
    buf.insert(_feat(2), [0, 0, 0], 3.0, _feat(3), True, discount=0.1)

    batch = buf.sample(3)
    order = np.argsort(batch["feat"][:, 0])
    rewards = batch["reward"][order]
    discounts = batch["discount"][order]
    done = batch["done"][order]

    np.testing.assert_allclose(rewards, [2.0, 2.75, 3.0], rtol=1e-6)
    np.testing.assert_allclose(discounts, [0.125, 0.0, 0.0], rtol=1e-6)
    np.testing.assert_array_equal(done, [0.0, 1.0, 1.0])
```

- [x] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest test/speedtune_buffer_test.py::test_variable_transition_discount_enters_nstep_return -q
```

Expected: fail with `TypeError: insert() got an unexpected keyword argument 'discount'`.

- [x] **Step 3: Implement variable discount**

Change `_Trans` to include `discount`, update `insert(..., discount=None)`, and update `_emit_nstep` to multiply by `t.discount` after non-terminal rewards. Keep default `discount=self.gamma` for old callers.

- [x] **Step 4: Run buffer tests**

Run:

```bash
python -m pytest test/speedtune_buffer_test.py -q
```

Expected: all buffer tests pass.

### Task 2: Paper Reward Runtime

**Files:**
- Modify: `expo_ft/speedtune/training_runtime.py`
- Test: `test/speedtune_training_runtime_test.py`

- [x] **Step 1: Write failing tests**

Update `FakeBuffer.insert` in `test/speedtune_training_runtime_test.py` to accept `discount=None` and store it. Add:

```python
def test_paper_speedtuning_reward_and_discount_are_chunk_level():
    pending = [{
        "feat": np.asarray([0.0]),
        "action_idxs": np.asarray([0]),
        "v_list": [4.0],
        "r_task": 1.0,
        "done": True,
        "duration": 0.2,
        "n_exec": 30,
        "execution_steps": 3,
    }]
    buffer = FakeBuffer()

    summary = flush_pending_episode(
        pending,
        buffer,
        FakeBackend(),
        task_success=True,
        speed_violation=False,
        reward_mode="paper_speedtuning",
        reward_alpha=0.1,
        reward_beta=2.0,
        gamma=0.5,
    )

    expected_reward = 0.1 * 16.0 + 1.0
    assert abs(buffer.inserted[0][2] - expected_reward) < 1e-6
    assert abs(buffer.inserted[0][5] - 0.5) < 1e-9
    assert summary["reward_success"] is True
    assert summary["execution_steps"] == 3


def test_paper_speedtuning_failure_keeps_speed_reward_without_penalty():
    pending = [{
        "feat": np.asarray([0.0]),
        "action_idxs": np.asarray([0]),
        "v_list": [4.0],
        "r_task": 0.0,
        "done": True,
        "duration": 0.2,
        "n_exec": 30,
        "execution_steps": 2,
    }]
    buffer = FakeBuffer()

    summary = flush_pending_episode(
        pending,
        buffer,
        FakeBackend(),
        task_success=False,
        speed_violation=False,
        reward_mode="paper_speedtuning",
        reward_alpha=0.1,
        reward_beta=2.0,
        gamma=0.5,
    )

    assert abs(buffer.inserted[0][2] - 1.6) < 1e-6
    assert summary["reward_success"] is False
```

- [x] **Step 2: Run the runtime tests to verify failure**

Run:

```bash
python -m pytest test/speedtune_training_runtime_test.py::test_paper_speedtuning_reward_and_discount_are_chunk_level test/speedtune_training_runtime_test.py::test_paper_speedtuning_failure_keeps_speed_reward_without_penalty -q
```

Expected: fail because `flush_pending_episode` does not accept `reward_mode`.

- [x] **Step 3: Implement paper reward mode**

Add helper functions in `training_runtime.py`:

```python
def paper_speedtuning_reward(v_list, r_task, *, alpha, beta):
    speed = sum(max(0.0, float(v)) ** float(beta) for v in v_list)
    return float(float(alpha) * speed + float(r_task))
```

Extend `flush_pending_episode` with keyword args:

```python
reward_mode: str = "success_gated",
reward_alpha: float = 1.0,
reward_beta: float = 2.0,
gamma: float = 0.99,
```

When `reward_mode == "paper_speedtuning"`, pass chunk-level `discount=gamma` to the buffer and use `r_task=0` for forced terminal partial flushes. The default success-gated mode also writes chunk-level `gamma`.

- [x] **Step 4: Run runtime tests**

Run:

```bash
python -m pytest test/speedtune_training_runtime_test.py -q
```

Expected: all runtime tests pass.

### Task 3: Config and Support

**Files:**
- Modify: `configs/model/speedtune_dqn_config.py`
- Modify: `expo_ft/speedtune/runtime_config.py`
- Test: `test/speedtune_config_test.py`

- [x] **Step 1: Write failing config tests**

Add assertions that default reward mode remains success-gated:

```python
assert config.reward_mode == "success_gated"
assert config.paper_speedtuning_episode_steps == 800
```

Add a paper support test:

```python
def test_paper_speedtuning_support_scales_with_alpha():
    config = get_config()
    config.reward_mode = "paper_speedtuning"
    config.reward_alpha = 1e-4
    config.reward_beta = 2.0
    support = backend_support(config, "fixed_time")
    expected_max = 1.0 + finite_horizon_q_max(
        1e-4 * 4.0 ** 2, config.gamma, 800 // config.fixed_time_k_skip
    )
    assert support[0] == 0.0
    assert abs(support[1] - expected_max) < 1e-9
```

- [x] **Step 2: Run config tests to verify failure**

Run:

```bash
python -m pytest test/speedtune_config_test.py -q
```

Expected: fail because the config fields and support branch do not exist.

- [x] **Step 3: Implement config and dynamic support**

In `speedtune_dqn_config.py` add:

```python
config.reward_mode = "success_gated"
config.paper_speedtuning_episode_steps = 800
config.paper_speedtuning_task_reward_max = 1.0
```

Change `epsilon_decay_steps` default from `20000` to `4000` decisions.

In `runtime_config.backend_support`, if `reward_mode == "paper_speedtuning"`, compute:

```python
max_speed_reward = reward_alpha * (4.0 ** reward_beta)
max_chunks = paper_speedtuning_episode_steps // fixed_time_k_skip
support_max = task_reward_max + finite_horizon_q_max(max_speed_reward, gamma, max_chunks)
return 0.0, float(support_max)
```

- [x] **Step 4: Run config tests**

Run:

```bash
python -m pytest test/speedtune_config_test.py -q
```

Expected: all config tests pass.

### Task 4: Trainer Wiring

**Files:**
- Modify: `train_speedtune_async.py`
- Modify: `train_speedtune_sync.py`
- Test: `test/speedtune_training_runtime_test.py`

- [x] **Step 1: Write source-level wiring assertions**

Update existing trainer source tests to assert:

```python
assert "\"execution_steps\"" in source
assert "reward_mode=str(config.get(\"reward_mode\", \"success_gated\"))" in source
assert "gamma=float(config.gamma)" in source
```

- [x] **Step 2: Run source tests to verify failure**

Run:

```bash
python -m pytest test/speedtune_training_runtime_test.py::test_sync_entry_uses_environment_decision_budget_and_partial_flush test/speedtune_training_runtime_test.py::test_async_entry_uses_counted_episode_queue_instead_of_event_latch -q
```

Expected: fail on the new string assertions.

- [x] **Step 3: Wire both trainers**

Add `execution_steps=int(exec_info.get("execution_steps", 1) or 1)` to pending transition dicts in both trainers.

Pass reward config into `flush_pending_episode`:

```python
reward_mode=str(config.get("reward_mode", "success_gated")),
reward_alpha=float(config.get("reward_alpha", 1.0)),
reward_beta=float(config.get("reward_beta", 2.0)),
gamma=float(config.gamma),
```

Add `exec/execution_steps` to step logs.

- [x] **Step 4: Run runtime tests**

Run:

```bash
python -m pytest test/speedtune_training_runtime_test.py -q
```

Expected: all runtime tests pass.

### Task 5: Focused Verification

**Files:**
- No new files.

- [x] **Step 1: Run focused tests**

Run:

```bash
python -m pytest test/speedtune_buffer_test.py test/speedtune_training_runtime_test.py test/speedtune_config_test.py -q
```

Expected: all focused tests pass.

- [x] **Step 2: Inspect diff**

Run:

```bash
git diff -- docs/superpowers/specs/2026-07-08-paper-speedtuning-reward-design.md docs/superpowers/plans/2026-07-08-paper-speedtuning-reward.md configs/model/speedtune_dqn_config.py expo_ft/speedtune/runtime_config.py expo_ft/speedtune/training_runtime.py expo_ft/data/speedtune_buffer.py train_speedtune_async.py train_speedtune_sync.py test/speedtune_buffer_test.py test/speedtune_training_runtime_test.py test/speedtune_config_test.py
```

Expected: only scoped changes for paper reward, variable discount, config, tests, and docs.

## Plan Self-Review

- Spec coverage: reward mode, alpha, no failure speed penalty, final-task reward, and chunk-level `gamma` are covered by Tasks 1-4.
- Placeholder scan: no TBD/TODO placeholders remain.
- Type consistency: `execution_steps` is an int, `discount` is optional float, and old buffer callers still work through the default discount.
