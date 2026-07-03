# SpeedTune Raw-Speed Eval and Fixed Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct SpeedTune eval to report raw backend speed values over 30 episodes and provide a robust cloud fixed-speed sweep for fixed-time and whole-chunk TOPPRA.

**Architecture:** Keep VLA rollout and physics accounting in `eval_speedtune.run_backend_episodes`, but replace normalized aggressiveness with one backend-specific raw speed field and aggregate maximum planned velocity. Build the fixed-speed sweep as a thin constant-action controller over the same runner, so seed/noise, `k_skip`, success gating, and dense-step accounting stay identical to trained-DQN eval.

**Tech Stack:** Python, JAX, NumPy, absl flags, RoboTwin WebSocket environment client, pytest, Bash.

---

### Task 1: Specify and test raw-speed eval records

**Files:**
- Modify: `test/eval_speedtune_runtime_test.py`
- Modify: `eval_speedtune.py`

- [ ] **Step 1: Write failing runtime assertions**

Add assertions that fixed-time records `v` and `max_planned_qvel`, whole-chunk records only `vel_limit`, and neither path emits `aggr`, `aggr_mean`, `acc_limit`, or `derived_acc_limit`. Assert aggregate success uses the requested episode count and aggregate maximum planned velocity is present.

- [ ] **Step 2: Verify RED**

Run: `pytest -q test/eval_speedtune_runtime_test.py`

Expected: failure on legacy aggressiveness/acceleration fields or missing maximum-velocity summary.

- [ ] **Step 3: Implement the raw-speed contract**

Derive the raw field from the single backend variable (`v` or `vel_limit`), store only that field in each decision record, compute raw contact/free means, and plot it against `t_sim`. Add per-episode and aggregate fixed-time `max_planned_qvel`; remove eval-only acceleration/aggressiveness fields and logs.

- [ ] **Step 4: Verify GREEN**

Run: `pytest -q test/eval_speedtune_runtime_test.py`

Expected: all tests pass.

### Task 2: Correct paired eval summaries and plots

**Files:**
- Create: `test/eval_speedtune_compare_test.py`
- Modify: `eval_speedtune_compare.py`
- Modify: `scripts/run_eval_compare.sh`

- [ ] **Step 1: Write failing paired-output tests**

Test that parameter discovery returns only `v` or `vel_limit`, compare summaries contain raw-speed and maximum-velocity fields but no aggressiveness/acceleration fields, and launcher defaults are 30 episodes with `fixed_time chunk_toppra`.

- [ ] **Step 2: Verify RED**

Run: `pytest -q test/eval_speedtune_compare_test.py`

Expected: legacy `acc_limit`/aggressiveness behavior fails the assertions.

- [ ] **Step 3: Update paired eval**

Replace normalized-aggressiveness plots with raw `v`/`vel_limit` curves, remove acceleration keys from parameter discovery and summary output, update documentation/defaults to fixed-time versus whole-chunk TOPPRA, and make the standalone launcher default to 30 episodes.

- [ ] **Step 4: Verify GREEN**

Run: `pytest -q test/eval_speedtune_compare_test.py`

Expected: all tests pass.

### Task 3: Add fixed-speed sweep logic

**Files:**
- Create: `expo_ft/speedtune/fixed_speed_sweep.py`
- Create: `eval_speedtune_fixed_sweep.py`
- Create: `test/fixed_speed_sweep_test.py`

- [ ] **Step 1: Write failing pure-function tests**

Test the exact seven-value grid, constant controller index selection, backend parameter mapping, row aggregation, fixed-time maximum planned velocity, and rejection of unsupported backends/values.

- [ ] **Step 2: Verify RED**

Run: `pytest -q test/fixed_speed_sweep_test.py`

Expected: import failure because the sweep module does not exist.

- [ ] **Step 3: Implement reusable sweep helpers**

Add strict grid parsing, a constant greedy controller compatible with `run_backend_episodes`, aggregation to serializable rows, and JSON/CSV/Markdown/plot writers.

- [ ] **Step 4: Implement the executable**

Create both backend environments with their configured `k_skip`, load one frozen VLA, run every backend/speed pair with the same base seed and deterministic per-decision noise, and write per-config episode results plus aggregate artifacts.

- [ ] **Step 5: Verify GREEN**

Run: `pytest -q test/fixed_speed_sweep_test.py`

Expected: all tests pass.

### Task 4: Add robust cloud launcher

**Files:**
- Create: `scripts/run_speedtune_fixed_sweep.sh`
- Create: `test/run_speedtune_fixed_sweep_test.sh`

- [ ] **Step 1: Write the failing dry-run test**

Assert defaults of 30 episodes, the seven speeds, fixed-time `k_skip=10`, chunk TOPPRA `k_skip=20`, two backend servers, commit logging, and process-group cleanup declarations.

- [ ] **Step 2: Verify RED**

Run: `bash test/run_speedtune_fixed_sweep_test.sh`

Expected: failure because the launcher does not exist.

- [ ] **Step 3: Implement the launcher**

Add path/tool/port/GPU validation, two process-group RoboTwin servers, readiness checks, one fixed-sweep eval process, structured logs, dry-run output, and TERM/KILL cleanup with failure-stage reporting.

- [ ] **Step 4: Verify GREEN**

Run: `bash test/run_speedtune_fixed_sweep_test.sh`

Expected: `run_speedtune_fixed_sweep dry-run tests passed`.

### Task 5: Full verification and publication

**Files:**
- Verify all files above

- [ ] **Step 1: Run targeted test suite**

Run: `pytest -q test/eval_speedtune_runtime_test.py test/eval_speedtune_compare_test.py test/fixed_speed_sweep_test.py test/paired_speedtune_eval_test.py && bash test/run_speedtune_fixed_sweep_test.sh && bash test/run_speedtune_train_eval_test.sh`

Expected: all Python and shell tests pass.

- [ ] **Step 2: Run syntax/static validation**

Run: `python -m py_compile eval_speedtune.py eval_speedtune_compare.py eval_speedtune_fixed_sweep.py expo_ft/speedtune/fixed_speed_sweep.py && bash -n scripts/run_eval_compare.sh scripts/run_speedtune_fixed_sweep.sh`

Expected: exit code 0.

- [ ] **Step 3: Review scope**

Run: `git status -sb && git diff --check && git diff -- <explicit task files>`

Expected: no whitespace errors; unrelated dirty files remain unstaged.

- [ ] **Step 4: Commit and push**

Explicitly stage only the task files, commit with `fix: correct SpeedTune eval and add fixed-speed sweep`, then push the current `dbpo-robotwin` branch to `origin`.
