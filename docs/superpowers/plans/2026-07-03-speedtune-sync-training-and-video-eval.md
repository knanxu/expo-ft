# SpeedTune Sync Training and Video Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add selectable synchronous SpeedTune training with a 40,000 environment-decision budget, reliable asynchronous episode accounting, five deterministic diagnostic eval videos, and publish the required RoboTwin executor implementation.

**Architecture:** Keep the existing async entry point and add a sync entry point that reuses its VLA/DQN construction helpers. Put testable update scheduling and episode-buffer logic in a small shared module. Keep strict timing eval video-free, then deterministically replay selected episode ids only for video capture. Fail fast when the connected RoboTwin lacks the new executor contract.

**Tech Stack:** Python, JAX/Flax, Rainbow-DQN, NumPy, W&B, Bash, RoboTwin/SAPIEN, pytest and standalone Python tests.

---

### Task 1: Publish the RoboTwin executor contract

**Files:**
- Modify: `/home/xukainan/RoboTwin/envs/_base_task.py`
- Modify: `/home/xukainan/RoboTwin/envs/robot/toppra_chunk_executor.py`
- Test: `/home/xukainan/RoboTwin/envs/whole_chunk_kskip_test.py`
- Test: `/home/xukainan/RoboTwin/envs/whole_chunk_start_state_test.py`
- Test: `/home/xukainan/RoboTwin/envs/whole_chunk_vel_control_test.py`
- Test: `/home/xukainan/RoboTwin/envs/streaming_speed_violation_test.py`

- [ ] **Step 1: Run the focused executor tests**

Run:

```bash
conda run --no-capture-output -n RoboTwin python -m pytest -q \
  envs/whole_chunk_kskip_test.py envs/whole_chunk_start_state_test.py \
  envs/whole_chunk_vel_control_test.py envs/streaming_speed_violation_test.py
```

Expected: 7 tests pass.

- [ ] **Step 2: Inspect the exact staged scope**

Run:

```bash
git diff --check
git diff -- envs/_base_task.py envs/robot/toppra_chunk_executor.py \
  envs/whole_chunk_kskip_test.py envs/whole_chunk_start_state_test.py \
  envs/whole_chunk_vel_control_test.py envs/streaming_speed_violation_test.py
```

Expected: only `execution_steps`, `4*vel_limit**2`, cruise telemetry, and fixed-time speed-gate changes.

- [ ] **Step 3: Commit only the six executor files**

```bash
git add envs/_base_task.py envs/robot/toppra_chunk_executor.py \
  envs/whole_chunk_kskip_test.py envs/whole_chunk_start_state_test.py \
  envs/whole_chunk_vel_control_test.py envs/streaming_speed_violation_test.py
git commit -m "feat: enforce SpeedTune executor contract"
```

### Task 2: Add shared, testable episode-update scheduling

**Files:**
- Create: `expo_ft/speedtune/training_runtime.py`
- Create: `test/speedtune_training_runtime_test.py`
- Modify: `train_speedtune_async.py`

- [ ] **Step 1: Write failing scheduling tests**

Tests cover:

```python
assert update_ready(completed_episodes=9, replay_size=1000, batch_size=64) is False
assert update_ready(completed_episodes=10, replay_size=63, batch_size=64) is False
assert update_ready(completed_episodes=10, replay_size=64, batch_size=64) is True
```

and a fake learner/buffer proving one episode executes exactly `6 * 20` calls to
`learner.update`, with the same number of priority updates.

- [ ] **Step 2: Verify RED**

Run: `uv run python test/speedtune_training_runtime_test.py`

Expected: import failure because `training_runtime.py` does not exist.

- [ ] **Step 3: Implement the minimal shared runtime**

Implement:

```python
def update_ready(completed_episodes, replay_size, batch_size, warmup_episodes=10): ...
def run_episode_updates(learner, buffer, *, batch_size, beta,
                        update_groups, utd_ratio): ...
def flush_pending_episode(pending, buffer, backend, *, task_success,
                          speed_violation, force_terminal=False): ...
```

The forced-terminal path marks only the last transition terminal so a 40,000-decision
mid-episode stop flushes n-step state without awarding success.

- [ ] **Step 4: Verify GREEN**

Run: `uv run python test/speedtune_training_runtime_test.py`

Expected: all runtime tests pass.

- [ ] **Step 5: Replace async Event with a counted FIFO**

Use `queue.Queue`; enqueue one decision-step token for every update-eligible completed
episode and enqueue a final sentinel after rollout. FIFO shutdown drains all earlier episode
tokens. Use `update_ready(..., warmup_episodes=10)` and the shared update function.

- [ ] **Step 6: Verify async source and runtime tests**

Run:

```bash
uv run python test/speedtune_training_runtime_test.py
python -m py_compile train_speedtune_async.py expo_ft/speedtune/training_runtime.py
```

Expected: pass with no `threading.Event` episode latch.

### Task 3: Add synchronous SpeedTune training

**Files:**
- Create: `train_speedtune_sync.py`
- Modify: `configs/model/speedtune_dqn_config.py`
- Modify: `test/speedtune_training_runtime_test.py`

- [ ] **Step 1: Add a failing source-contract test**

Verify the sync entry exists, iterates `range(max_iters)`, calls `env.step_chunk` once per
iteration, invokes shared episode updates only after `done`, and force-flushes a final partial
episode.

- [ ] **Step 2: Verify RED**

Run: `uv run python test/speedtune_training_runtime_test.py`

Expected: failure because `train_speedtune_sync.py` does not exist.

- [ ] **Step 3: Implement sync rollout/update loop**

Reuse `_setup_speedtune`, epsilon/beta schedules and checkpoint writer from the async module.
Run exactly one `step_chunk` per decision, relabel the complete episode before replay insertion,
then synchronously run six update groups with UTD 20 after warmup. Log cumulative decisions,
episodes, update groups and gradient updates. Always save the final learner checkpoint.

- [ ] **Step 4: Update config comments without changing the 40,000 budget**

Document `max_iters` as environment chunk decisions and add `warmup_episodes=10`; retain
`update_per_episode=6` and `utd_ratio=20`.

- [ ] **Step 5: Verify sync entry**

Run:

```bash
uv run python test/speedtune_training_runtime_test.py
python -m py_compile train_speedtune_sync.py
```

Expected: all tests pass.

### Task 4: Enforce the cloud RoboTwin contract and add decision telemetry

**Files:**
- Modify: `client_robotwin/envs/robotwin_env.py`
- Modify: `test/robotwin_step_chunk_test.py`

- [ ] **Step 1: Write failing stale-RoboTwin tests**

Add fake old whole-chunk and streaming methods. Expect `RuntimeError` when
`execution_steps` is absent or when whole/fixed telemetry fields are missing. Verify returned
info includes input chunk length, requested action count, actual execution steps and dense steps.

- [ ] **Step 2: Verify RED**

Run: `uv run python test/robotwin_step_chunk_test.py`

Expected: stale methods are silently accepted by the current signature filter.

- [ ] **Step 3: Implement fail-fast validation**

Require `execution_steps` in the whole-chunk method signature and validate whole-chunk
`vel_limit/acc_limit/execution_steps/planned_cruise_fraction`; validate fixed-time
`fixed_time_speed_violation/max_planned_qvel`. Check returned acceleration equals `4V²`.

- [ ] **Step 4: Verify GREEN**

Run: `uv run python test/robotwin_step_chunk_test.py`

Expected: all adapter tests pass.

### Task 5: Add paired decision metrics and deterministic video selection

**Files:**
- Modify: `expo_ft/speedtune/paired_eval.py`
- Modify: `test/paired_speedtune_eval_test.py`
- Modify: `eval_speedtune.py`
- Modify: `eval_speedtune_compare.py`
- Modify: `test/eval_speedtune_runtime_test.py`

- [ ] **Step 1: Write failing pure-helper tests**

Test that decision ratios use only jointly successful ids and that five video ids prioritize
any backend failure, then safe-reward failure, then joint success, with stable episode ordering.

- [ ] **Step 2: Verify RED**

Run: `uv run python test/paired_speedtune_eval_test.py`

Expected: missing decision metrics and selector.

- [ ] **Step 3: Implement metrics and selector**

Extend `paired_metrics` with `decision_step_ratio` and implement
`select_diagnostic_episode_ids(..., limit=5)`.

- [ ] **Step 4: Verify GREEN**

Run: `uv run python test/paired_speedtune_eval_test.py`

Expected: all paired helper tests pass.

- [ ] **Step 5: Add selective video recording to the rollout helper**

Add optional `video_episode_ids` and `save_episode_artifacts`. Record input/requested/actual
chunk counts in every decision record. Primary eval always runs video-free.

- [ ] **Step 6: Add diagnostic replay**

For single and paired eval, select five ids after the primary run, reseed each environment,
replay sequentially through the maximum selected id, save video only for selected ids, discard
diagnostic metrics, and write `video_manifest.json`. Paired eval uses identical selected ids for
both backends.

- [ ] **Step 7: Verify eval tests and syntax**

Run:

```bash
uv run python test/paired_speedtune_eval_test.py
uv run python test/eval_speedtune_runtime_test.py
python -m py_compile eval_speedtune.py eval_speedtune_compare.py
```

Expected: all tests pass.

### Task 6: Add training-mode selection and five-video cloud defaults

**Files:**
- Modify: `scripts/run_speedtune_train_eval.sh`
- Modify: `test/run_speedtune_train_eval_test.sh`

- [ ] **Step 1: Write failing shell assertions**

Require default `TRAIN_MODE=sync`, async override, invalid-mode rejection, selected trainer in
dry-run output, `VIDEO_EPISODES=5`, eval `--record_video`, and both repository commit ids.

- [ ] **Step 2: Verify RED**

Run: `bash test/run_speedtune_train_eval_test.sh`

Expected: missing training-mode/video assertions fail.

- [ ] **Step 3: Implement script selection**

Map `sync -> train_speedtune_sync.py`, `async -> train_speedtune_async.py`; prompt only in an
interactive terminal, default sync otherwise. Pass `--record_video --video_episodes 5` to eval
and print expo-ft/RoboTwin commit ids before launch.

- [ ] **Step 4: Verify GREEN and shell syntax**

Run:

```bash
bash -n scripts/run_speedtune_train_eval.sh
bash test/run_speedtune_train_eval_test.sh
```

Expected: dry-run, override, invalid input and cleanup tests pass.

### Task 7: Full verification, intentional commits, and push

**Files:**
- All files listed above.

- [ ] **Step 1: Run the complete focused expo-ft suite**

```bash
uv run python test/speedtune_training_runtime_test.py
uv run python test/paired_speedtune_eval_test.py
uv run python test/eval_speedtune_runtime_test.py
uv run python test/robotwin_step_chunk_test.py
uv run python test/speedtune_buffer_test.py
bash test/run_speedtune_train_eval_test.sh
python -m py_compile train_speedtune_async.py train_speedtune_sync.py \
  eval_speedtune.py eval_speedtune_compare.py
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 2: Review expo-ft scope and commit only relevant files**

Use explicit `git add` paths; do not stage `CLAUDE.md`, removed benchmark files, historical
results, or unrelated tests. Commit with `feat: add synchronous SpeedTune training and video eval`.

- [ ] **Step 3: Push both current feature branches**

```bash
git -C /home/xukainan/RoboTwin push -u origin feat/slow-expert-data
git -C /home/xukainan/expo-ft push -u origin dbpo-robotwin
```

Expected: both remotes accept the new commits.

- [ ] **Step 4: Report cloud command contract**

Report branch/commit ids, validation results, default sync behavior, async override, video output
location, and the requirement that cloud pulls both repositories before running the script.
