# RoboTwin 自动专家介入设计（替代人类在环，打破 EXPO rollout 0% 死锁）

- 日期：2026-06-22
- 维护人：polar823
- 状态：设计已对齐，待写实现 plan
- 关联问题：EXPO-FT 在 RoboTwin 在线 RL 训练时 rollout 成功率恒 0%

---

## 1. 背景与问题

EXPO-FT（`EXPOLearner`）= 冻结/微调的 pi0.5 drift base policy + residual edit actor + critic(Q ensemble)。
它在 DROID 真机上靠**人类在环**（spacemouse 逐 step 接管，`action_type="human"`/`is_hil=True`）提供
on-policy 纠正样本。RoboTwin 仿真没有这条路径——`client_robotwin/run_robotwin_client.py` 的 step op
固定返回 `action_type="policy"`（注释明确"RoboTwin 仿真无 human override"）。

现象（云端观察，由维护人确认）：
- base policy 单独（RoboTwin 自带 `eval_policy_client` 或 `--actor_only_base_actions=True`）成功率 30–60%；
- 提供了较多 offline 专家 demo（`process_robotwin_dataset`，末帧 reward=1）；
- 但 EXPO 完整训练（带 residual+critic）时 **rollout 成功率恒 0%**。

> 注：本仓库本地只有代码，所有训练/rollout/eval 在云端完成（见 CLAUDE.md「运行环境」一节）。

## 2. 根因：在线学习的死锁

EXPO 唯一的正反馈来源是 reward（稀疏 0/1 终局 success）：
- `update_critic`（`expo_ft/agents/alg/expo_ft.py:765`）：`target_q = rewards + discount^replan · masks · next_q`；
- `update_residual_actor`（`:684`）：最大化 critic 的 Q；
- `update_actor`（`:724`）：base pi0.5 做 `actor_success_only` 的 BC。

死锁链：
1. critic 只在 **off-policy** 的 offline demo 状态上暖启，泛化不到 agent rollout 实际进入的状态；
2. 在那些状态上 Q 是噪声 → `sample_actions`（`:510`）用 argmax Q 从 base/residual candidates 里**乱选**；
3. 动作变差 → rollout 0% success → 永远拿不到 **on-policy** 成功样本；
4. critic 永远学不好 → 回到第 2 步。

offline demo（off-policy，只覆盖专家自身状态轨迹）打不破这个死锁，因为它覆盖不到 agent 自己会走到的
"坏状态"。**只有 on-policy 纠正（人类在环的等价物）能打破**：在 agent 真实失败状态下提供"如何成功"的样本，
让 critic 学到"失败状态也能救回来"。

## 3. 目标 / 非目标

**目标**：
- 在 RoboTwin 用**自动专家介入**等价替代 DROID 的人类在环，产生覆盖 agent 失败状态的 on-policy 纠正样本，
  使 EXPO 的 critic/residual 获得正信号，rollout 成功率脱离 0%。
- 严格零侵入：不改 EXPO/BC 算法、`train_pi_robo_async.py` 主循环、`client/` 内 DROID 文件、RoboTwin 本体。

**非目标**：
- 不做多 env 并行（沿用单 env，CLAUDE.md 约定）。
- 不针对单一算法定制 env（介入机制对 EXPO/BC 通用）。
- 首版不做课程式介入率衰减（YAGNI；step 预算阈值已自带自适应退出）。
- 首版只在 `stack_blocks_two` 打通；设计为通用，但不为其它任务预写进度检测。

## 4. 设计决策汇总（已与维护人对齐）

| # | 决策 | 选择 |
|---|---|---|
| D1 | 触发策略 / 数据来源 | **失败/卡住时从当前状态接管到底**（真 on-policy DAgger） |
| D2 | 触发判据 | **step 预算阈值**：用到 `X%` 预算仍未 success → 接管 |
| D3 | 接管语义 | **从当前状态重做整任务**（直接调 `play_once`，不检测子任务进度；规划失败丢弃 episode） |
| D4 | 数据注入 | **逐 step 复用 DROID `action_type="human"` 协议**（learner 零改动） |
| D5 | 逐步化实现 | **录播带（在线专家采集 + 回放）**，非实时线程驱动 |
| D6 | base actor BC 范围 | **(a) 沿用现有 `actor_success_only`**（BC 整个 success episode，含 agent 前缀+expert 段），与 DROID EXPOLearner 一致；(b) 只 BC `is_hil` 段为备选 |

## 5. 架构

### 5.1 数据流（单个 episode）

```
learner 逐 step rollout (agent base+residual)
   │  每步: get_observation → get_info_for_step → step(action)
   │  transition is_hil=False 进 buffer (真实 env 推进)
   ▼
[server 触发器] steps_since_reset ≥ X%·step_budget 且 未 success?
   │ 否 → 继续 agent rollout
   │ 是 → 进入「专家接管模式」(episode 内不可逆)
   ▼
[专家录播] 从 agent 当前状态调 env.play_once()，按 save_freq 录:
   tape = [(obs_i, qpos_i, reward_i, done_i), ...]   (复用 RoboTwin 专家采集机制)
   末帧成功 → reward=1/done=1；规划失败/未成功 → episode 作废
   ▼
[逐 step 回放] 之后 learner 每次 get_observation/step/get_info_for_step:
   server 忽略 learner action，从 tape 逐帧回放:
   step() → (qpos_i, action_type="human")，cursor++
   ▼
learner 收到 "human" → 清空 action_plan、标 is_hil=True、进 buffer
   (train_pi_robo_async.py:374/396-409 已支持，零改动)
   ▼
tape 尽 → episode done(success) → on_episode_done(success=True)
   → mark_episode_success → critic 学到 on-policy 状态上的 reward=1
                          + base actor BC expert 动作
```

### 5.2 组件分解（各自单一职责、可独立测试）

| 组件 | 位置 | 职责 / 接口 | 依赖 |
|---|---|---|---|
| **TakeoverTrigger** | `client_robotwin/envs/robotwin_env.py` | 纯判定 `should_takeover(steps_since_reset, step_budget, success_once) → bool`。无副作用 | 无 |
| **ExpertTapeRecorder** | 新增 `client_robotwin/envs/expert_takeover.py` | 从 agent 当前状态调 `env.play_once()`，按 `save_freq` 录 `tape=[(obs, qpos, reward, done)]` | RoboTwin `play_once`/`get_obs` |
| **TapeReplayer** | 新增 `client_robotwin/envs/expert_takeover.py` | 持 `tape+cursor`，提供 `obs()/next_action()/info()` 逐帧回放。纯状态机，无 RoboTwin 依赖 | 无 |
| **RoboTwinEnv 集成** | `client_robotwin/envs/robotwin_env.py` | `step/get_observation/get_info_for_step` 按 `_takeover_active` 分发到正常路径 or TapeReplayer | 上三者 |
| **server 协议** | `client_robotwin/run_robotwin_client.py` | step op 接管模式返回 `action_type="human"`（其余分支不动） | RoboTwinEnv |

### 5.3 episode 状态机

```
AGENT_ROLLOUT (真实推进, is_hil=False)
   │ trigger: steps ≥ X%·budget 且 未 success
   ▼
RECORDING (跑 play_once 录 tape; learner 那次 step 阻塞一次, 等同 DROID 人类操作)
   │ play_once 成功
   ▼
REPLAY (逐帧回放 action_type="human", is_hil=True)
   │ cursor 到末帧
   ▼
DONE(success) → reset 清接管状态
```
- trigger 在 episode 内**不可逆**（接管到底，对应 D1）。
- `play_once` 失败（`plan_success=False` 或 `check_success=False`）→ episode 作废（见 §8）。

## 6. 零侵入边界（严格遵守 CLAUDE.md）

| 改 | 不改 |
|---|---|
| `client_robotwin/run_robotwin_client.py`（step op 加接管分支） | `client/` 所有 DROID 文件（`run_client.py`/`droid_env.py`/…） |
| `client_robotwin/envs/robotwin_env.py`（触发器 + 分发） | `train_pi_robo_async.py` 主循环 / `expo_ft/env/env_client.py` |
| 新增 `client_robotwin/envs/expert_takeover.py`（录播 + 回放） | EXPO/BC 算法、replay buffer、RoboTwin 本体（`/home/xukainan/RoboTwin`） |

learner 端**完全复用** `action_type=="human"` → `is_hil` 通路；DROID 路径与既有 SpeedTune `step_chunk`
路径均不受影响。

## 7. reward / is_hil / 学习语义

- 接管前 agent 段：`is_hil=False, reward=0`（非终局）；接管后 expert 段：`is_hil=True`；
  expert 完成 → 末帧 `reward=1/done=1` → `BatchProcessor.on_episode_done(success=True)` →
  `mark_episode_success`（`expo_ft/data/batch_processor.py:80`）。
- **critic**：学整个 episode——从 agent 真实失败前缀（reward=0）经 expert 段到 reward=1，学到
  "从 agent 失败状态出发存在通往 success 的路径"，Q 不再把失败状态判死 → 打破死锁（核心收益）。
- **base actor BC**（`actor_success_only`，决策 D6=a）：沿用现有，BC 整个 success episode（含 agent 前缀
  + expert 段），与 DROID EXPOLearner 完全一致。若云端 wandb 显示 agent 前缀稀释纠正信号，再升级为
  (b) 只 BC `is_hil` 段（需在 `batch_processor` success actor 采样加 `hil_only` filter，轻微触碰 learner 侧）。

## 8. 错误处理

- **curobo/`play_once` 规划失败**（`plan_success=False`）或 expert 执行后 `check_success=False`：
  该 episode **作废**（不注入任何 expert transition），server 端 reset 重来——避免给 critic 灌
  "expert 也没救成"的低信息/误导样本。
- **阻塞时长**：`play_once` 几秒~几十秒，远小于 learner `ep_timeout_secs=120`（`train_pi_robo_async.py:37`），
  不会误触发 update 暂停。
- **自适应退出**：随 agent 改进，越来越多 episode 在 `X%` 预算前就 success → 不触发接管 → expert
  依赖自然衰减（step 预算阈值的内在好处）。

## 9. 配置参数（在 `configs/task/robotwin_stack_blocks.py` / RoboTwinEnv kwargs 暴露）

- `takeover_step_frac`（默认 `0.5`）：用到该比例 step 预算仍未 success 即接管。
- `takeover_enable`（默认 `True`）：可关用于对照实验（关掉退化为当前纯 offline-demo 行为）。
- `takeover_prob`（默认 `1.0`，预留）：触发后实际接管的概率；首版不实现课程衰减。

## 10. 测试与不回归

- **单元**：TakeoverTrigger 阈值边界、TapeReplayer 回放/边界、协议返回 `action_type="human"`
  （mock tape，不依赖 sapien）。新增 `client_robotwin/**/*_test.py`。
- **集成（云端）**：rollout 成功率脱离 0；wandb 观测 `is_hil` 比例、critic `target_q_max`↑、
  success-only actor batch 非空。
- **不回归**：`client/` 未改（DROID 路径不变）；`expo_ft.agents.{vla,alg}` 与 EXPO/BC 导入/构造不变；
  既有 `expo_ft/**/*_test.py` 通过。

## 11. 实现待验证点（plan 阶段细化）

1. **录播带取 obs 的接入点（已核实）**：`get_obs()`（`_base_task.py:446`）任何时刻返回完整 obs
   （三相机 rgb + `joint_action.vector` 14-D qpos），格式正是 `RoboTwinEnv.get_observation`/
   `process_robotwin_dataset` 所用。`play_once` 执行中已在 `save_freq` 边界调 `_take_picture`
   （`together_move_to_pose:959` / `take_dense_action:1553`），但 `_take_picture`（`:517`）**存盘且
   受 `save_data` gate**，不直接用。**结论**：录播时在 `client_robotwin` 侧运行时把
   `self.env._take_picture` **monkey-patch 成"内存收集 `get_obs()`"**（跑完恢复），
   `play_once(save_freq=目标采样率)` → 得内存 tape；不改 RoboTwin 本体、与 offline demo 同粒度。
   `action[t] = vector[t+1]`（下一帧绝对 qpos）与 `process_robotwin_dataset` 对齐。
2. **触发时序对齐 + 回放期一律走 tape**：触发那一步 learner 已 `get_observation` 拿到真实当前 obs
   （≈ tape[0].obs），`step` 内触发+录 tape 后返回 tape[0].qpos；之后 `get_observation` 走 tape 回放。
   需保证 `(obs_t, action_t, reward_t, next_obs)` 衔接正确。
   **⚠️ 正确性要点**：回放期 `get_observation`/`step`/`get_info_for_step` **三者都必须读 tape（按 cursor）**，
   **不能**读真实 env——因为 `play_once` 已把物理 env 一次性推进到末态，真实 `check_success()` 会从回放第一帧
   就返回 True → episode 过早 done、tape 没放完。`get_info_for_step` 的 `(done, success)` 须按 cursor 是否到
   tape 末帧给出，而非 `env.check_success()`。
3. **qpos 粒度**：expert tape 的 action 必须是与 agent rollout / offline demo 同布局的 14-D 绝对 qpos
   目标（`[左臂6,左夹爪1,右臂6,右夹爪1]`），经同一套 openpi transforms。
4. **`play_once` 从中间状态的鲁棒性**：`grasp_actor`/`place_actor` 读实时位姿 + curobo 从当前 qpos 规划
   理论可行（见 `stack_blocks_two.py:81` `pick_and_place_block`），但 agent 留下的奇异/碰撞位形可能使
   规划失败——按 §8 作废处理。
