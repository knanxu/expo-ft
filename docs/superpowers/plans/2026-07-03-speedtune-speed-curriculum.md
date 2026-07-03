# SpeedTune Speed Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a persistent 20-episode/70%-success speed-bin curriculum to fixed-time and whole-chunk SpeedTune training while retaining the current epsilon-greedy Rainbow-DQN.

**Architecture:** A pure Python curriculum state machine owns the unlocked-bin frontier, rolling success window, counts and JSON state. The DQN receives a per-head maximum-action mask for greedy, epsilon-random and double-DQN bootstrap selection. Trainers update the curriculum only after complete episodes and save it in checkpoint metadata; eval restores the final mask.

**Tech Stack:** Python, NumPy, JAX/Flax, C51 Rainbow-DQN, W&B, Orbax, standalone Python tests.

---

### Task 1: Implement the curriculum state machine

**Files:**
- Create: `expo_ft/speedtune/curriculum.py`
- Create: `test/speedtune_curriculum_test.py`

- [ ] **Step 1: Write failing state-machine tests**

Test the public API:

```python
state = SpeedCurriculum(
    speed_grid=(1, 1.5, 2, 2.5, 3, 3.5, 4),
    window_size=20,
    success_threshold=0.7,
)
assert state.max_unlocked_idx == 0
for success in [True] * 14 + [False] * 6:
    state.record_episode(success, episode=20, decision=100)
assert state.max_unlocked_idx == 1
assert state.recent_results == []
```

Also test fewer than 20 results, a sliding failed window, one-bin-at-a-time unlock, maximum-boundary behavior, action counting and `state_dict` round trip.

- [ ] **Step 2: Run RED**

Run: `uv run python test/speedtune_curriculum_test.py`

Expected: `ModuleNotFoundError` for `expo_ft.speedtune.curriculum`.

- [ ] **Step 3: Implement `SpeedCurriculum`**

Provide:

```python
class SpeedCurriculum:
    @property
    def action_limits(self) -> tuple[int, ...]: ...
    def record_action(self, action_idxs) -> None: ...
    def record_episode(self, reward_success, *, episode, decision) -> bool: ...
    def state_dict(self) -> dict: ...
    @classmethod
    def from_state_dict(cls, state) -> "SpeedCurriculum": ...
```

`record_episode` returns `True` only when exactly one new bin is unlocked.

- [ ] **Step 4: Run GREEN**

Run: `uv run python test/speedtune_curriculum_test.py`

Expected: all curriculum tests pass.

### Task 2: Mask Rainbow-DQN action selection and bootstrap

**Files:**
- Modify: `expo_ft/networks/rainbow_dqn.py`
- Modify: `expo_ft/agents/alg/speedtune_dqn.py`
- Create: `test/speedtune_action_mask_test.py`

- [ ] **Step 1: Write failing action-mask tests**

Create a one-head learner with seven bins. Under `max_action_idxs=(0,)`, verify 512 epsilon-random samples are all zero. Under `(2,)`, verify every sample is at most two. Verify deterministic greedy obeys a mask and `learner.update(batch, max_action_idxs=(1,))` runs.

- [ ] **Step 2: Run RED**

Run: `uv run python test/speedtune_action_mask_test.py`

Expected: `sample_action_idxs` rejects `max_action_idxs`.

- [ ] **Step 3: Implement masked selection**

Add `masked_expected_q_argmax(dists, support, max_action_idxs)` and use it in rollout greedy selection and `_speedtune_update_step`. Random exploration samples from `[0, max_idx+1)` for each head. Default `None` maps to each head's full action range, preserving existing callers.

- [ ] **Step 4: Run GREEN and legacy DQN tests**

Run:

```bash
uv run python test/speedtune_action_mask_test.py
timeout 120 uv run python test/speedtune_dqn_test.py
```

Expected: new and existing DQN tests pass.

### Task 3: Pass the mask through episode update blocks

**Files:**
- Modify: `expo_ft/speedtune/training_runtime.py`
- Modify: `test/speedtune_training_runtime_test.py`

- [ ] **Step 1: Write a failing propagation test**

Extend the fake learner to record `max_action_idxs`; call:

```python
run_episode_updates(..., max_action_idxs=(2,))
```

Assert all 120 gradient updates receive `(2,)`.

- [ ] **Step 2: Run RED**

Run: `uv run python test/speedtune_training_runtime_test.py`

Expected: unexpected keyword argument `max_action_idxs`.

- [ ] **Step 3: Implement propagation and run GREEN**

Forward the optional mask into every `learner.update` call. Run the same test and expect all training-runtime tests to pass.

### Task 4: Integrate curriculum into sync and async trainers

**Files:**
- Modify: `train_speedtune_sync.py`
- Modify: `train_speedtune_async.py`
- Modify: `configs/model/speedtune_dqn_config.py`
- Create: `test/speedtune_trainer_curriculum_test.py`

- [ ] **Step 1: Write failing trainer-contract tests**

Inspect both trainer sources and require curriculum construction, masked `sample_action_idxs`, complete-episode `record_episode`, checkpoint `curriculum_state`, and no curriculum update in `force_terminal=True`. Require async queue tokens to include the episode action limit.

- [ ] **Step 2: Run RED**

Run: `uv run python test/speedtune_trainer_curriculum_test.py`

Expected: source-contract assertions fail.

- [ ] **Step 3: Add config and trainer integration**

Add:

```python
config.curriculum_enabled = True
config.curriculum_window_size = 20
config.curriculum_success_threshold = 0.7
```

Construct the curriculum only for `fixed_time` and `chunk_toppra`. At episode start retain its action limit. Record every action. After a complete flush, update the window for the next episode. Sync update blocks and async queue tokens use the retained limit. Add W&B curriculum fields and `curriculum_state` to checkpoint metadata snapshots.

- [ ] **Step 4: Run GREEN and compile**

Run:

```bash
uv run python test/speedtune_trainer_curriculum_test.py
uv run python test/speedtune_training_runtime_test.py
uv run python -m py_compile train_speedtune_sync.py train_speedtune_async.py
```

Expected: all pass.

### Task 5: Persist contract and apply the final mask in eval

**Files:**
- Modify: `expo_ft/speedtune/checkpoint_contract.py`
- Modify: `eval_speedtune.py`
- Modify: `eval_speedtune_compare.py`
- Create: `test/speedtune_curriculum_checkpoint_eval_test.py`
- Modify: `test/eval_speedtune_runtime_test.py`

- [ ] **Step 1: Write failing checkpoint/eval tests**

Require checkpoint metadata version 2 and static curriculum config. Test extraction of `(max_unlocked_idx,)` from saved `curriculum_state`. Update the fake eval learner to record that greedy action selection receives the restored mask.

- [ ] **Step 2: Run RED**

Run:

```bash
uv run python test/speedtune_curriculum_checkpoint_eval_test.py
uv run python test/eval_speedtune_runtime_test.py
```

Expected: version/mask extraction and eval call assertions fail.

- [ ] **Step 3: Implement persistence and eval mask**

Add `curriculum_config` to the static contract and `curriculum_action_limits(metadata, head_sizes)`. `_build_dqn` returns the restored limit. Pass it through primary and diagnostic single/paired eval, and record final unlocked index/speed in summaries.

- [ ] **Step 4: Run GREEN**

Run both tests plus `uv run python test/speedtune_checkpoint_contract_test.py`; expect all to pass.

### Task 6: Full verification, commit, and push

**Files:**
- All files above plus this plan.

- [ ] **Step 1: Run focused verification**

```bash
uv run python test/speedtune_curriculum_test.py
uv run python test/speedtune_action_mask_test.py
uv run python test/speedtune_training_runtime_test.py
uv run python test/speedtune_trainer_curriculum_test.py
uv run python test/speedtune_curriculum_checkpoint_eval_test.py
uv run python test/eval_speedtune_runtime_test.py
timeout 120 uv run python test/speedtune_dqn_test.py
uv run python test/speedtune_checkpoint_contract_test.py
bash test/run_speedtune_train_eval_test.sh
uv run python -m py_compile train_speedtune_sync.py train_speedtune_async.py \
  eval_speedtune.py eval_speedtune_compare.py
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 2: Commit only curriculum-related paths**

Use explicit `git add`; preserve `CLAUDE.md`, benchmark deletions, historical outputs and unrelated pre-existing tests. Commit as `feat: add SpeedTune speed curriculum`.

- [ ] **Step 3: Push the existing feature branch**

Run: `git push origin dbpo-robotwin`

Expected: remote `dbpo-robotwin` matches local HEAD.
