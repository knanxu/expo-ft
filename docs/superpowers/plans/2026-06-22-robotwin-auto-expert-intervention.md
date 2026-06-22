# RoboTwin 自动专家介入 Implementation Plan（含实现状态 + 云端验证清单）

> 设计见 `docs/superpowers/specs/2026-06-22-robotwin-auto-expert-intervention-design.md`。
> **运行环境**：本地只写代码 + `py_compile` 静态检查（显存不足，跑不了 pytest/sapien）；
> 所有 pytest / 训练 / rollout 由维护人在**云端**跑。

**Goal:** RoboTwin 自动专家介入（在线 on-policy 专家纠正）等价替代 DROID 人类在环，打破 EXPO rollout 0% 死锁。

**Architecture:** server 端 rollout 到 X% step 预算仍未 success → 从当前状态调 RoboTwin `play_once` 录「专家录播带」（monkey-patch `_take_picture` 内存收集 `get_obs()`），逐 step 回放为 `action_type="human"` transition；learner 复用既有 `is_hil` 通路，零改动。

## Global Constraints
- 零侵入：只改 `client_robotwin/` + `configs/task/robotwin_stack_blocks.py` + 新增 `scripts/run_expo_robotwin.sh`；不改 `client/`、`train_pi_robo_async.py`、`expo_ft/` 算法、RoboTwin 本体（仅运行时 monkey-patch）。
- 14-D aloha 布局 `[左臂6,左夹爪1,右臂6,右夹爪1]`；obs 扁平键 `observation.images.{cam_high,cam_left_wrist,cam_right_wrist}` + `observation.state` + `prompt`；`action[t]=vector[t+1]`。
- commit message 结尾加 `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`。

## 实现状态（已完成，本地 py_compile 通过）

| Task | 文件 | 内容 | 状态 |
|---|---|---|---|
| 1 | `client_robotwin/envs/expert_takeover.py::should_takeover` | 触发判定（纯逻辑） | ✅ |
| 2 | `…/expert_takeover.py::TapeReplayer` | 逐帧回放状态机（current_obs/current_info/next_action） | ✅ |
| 3 | `…/expert_takeover.py::record_expert_tape` | 录播（monkey-patch `_take_picture` 内存收集，规划失败→None） | ✅ |
| — | `client_robotwin/envs/expert_takeover_test.py` | Task 1-3 单测（6 测试，云端 pytest 跑） | ✅ |
| 4 | `client_robotwin/envs/robotwin_env.py` | `__init__` 加 takeover 参数 + 抽 `_flatten_obs` + `get_observation` 接管分发 | ✅ |
| 5 | `…/robotwin_env.py` | `_should_takeover`/`_start_takeover` + `step`(接管返回 human)/`get_info_for_step`(走 tape)/`reset` 清状态 | ✅ |
| 6 | `client_robotwin/run_robotwin_client.py` | step op 透传 `env.step` 的 `action_type` | ✅ |
| 6 | `configs/task/robotwin_stack_blocks.py` | `takeover_enable=True/step_frac=0.5/save_freq=15` | ✅ |
| — | `scripts/run_expo_robotwin.sh` | 云端运行脚本（server + EXPO train，含 takeover） | ✅ |

## 接口契约（跨文件一致）
- `should_takeover(*, steps_since_reset, step_budget, success_once, step_frac, enabled) -> bool`
- `TapeReplayer(frames: list[(obs_flat dict, qpos ndarray)])`：`current_obs()`、`current_info()->(done,success,reward,mask)`、`next_action()->ndarray`（推进 cursor）、`exhausted`。
- `record_expert_tape(env, flatten_obs_fn, *, save_freq=15, n_real_dims=14) -> list[(obs,qpos)] | None`
- `RoboTwinEnv._flatten_obs(raw)->dict`；`step(action)->{"executed_action","action_type"}`。

## 云端验证清单（维护人上传后执行）

- [ ] **单元测试**：`python -m pytest client_robotwin/envs/expert_takeover_test.py -v` → 6 passed。
- [ ] **回归**：`python -m pytest expo_ft/speedtune/exec_backends_test.py -v` 通过；`python -c "import expo_ft.agents.alg.expo_ft, expo_ft.agents.alg.bc; print('ok')"`。
- [ ] **起训练**：`export VLA_CKPT/VLA_ASSETS_DIR/VLA_ASSET_ID/DATASET_PATH` 后 `bash scripts/run_expo_robotwin.sh`。
- [ ] **接管触发**：`grep "expert takeover started" logs/<run>/server.log`（应在 step ≈ 0.5·step_lim 出现 `N frames`）。
- [ ] **打破死锁**：wandb `rollout 成功率脱离 0%` 且上升；`training/target_q_max` 上升（critic 不再恒 0）；`is_hil` 比例 > 0；success-only actor batch 非空。
- [ ] **自适应退出**：随成功率上升，接管触发率下降。
- [ ] **不回归**：DROID/SpeedTune run 冒烟正常；`--config_task.takeover_enable=False` 复现旧行为（rollout 仍 0%）作对照。
- [ ] **回填**：把 rollout/Q/接管率曲线结论写回 design doc「验证结果」节并 commit。

## 已知风险（design §8/§11）
- `play_once` 从 agent 奇异/碰撞位形规划可能失败 → `record_expert_tape` 返回 None → 放弃接管、episode 走到超时（agent 失败 episode 仍进 buffer，合理）。
- 录播期 `get_observation`/`step`/`get_info_for_step` 三者**必须走 tape**（已实现）：play_once 已把物理推到末态，真实 `check_success` 会过早 True。
- 触发那一步 `play_once` 阻塞数秒~数十秒（< `ep_timeout_secs=120`，不误触发 update 暂停）。
