# RoboTwin 适配实施计划（Phase 4）

> **本文档自包含**：为「/clear 后只留本计划继续执行」而写。读本计划 + 下列 4 份即可接手，无需历史对话：
> `CLAUDE.md`（零侵入约定）、`docs/DBPO_DEV.md`（决策/进展）、`docs/RLINF_ROBOTWIN_REFERENCE.md`（RLinf 参考实现）、
> `docs/BIMANUAL_RL_HORIZON_SURVEY.md`（H/H_e 惯例）。维护人：polar823。建立：2026-06-18。

---

## 0. 目标与现状

- **目标**：在 **RoboTwin 仿真（stack two blocks，双臂 aloha-agilex）** 上跑通在线 RL 微调，作为**真机 RL 流程**的验证，
  后续再迁真机。**两条算法路径同等重要**：① 新增的 **DBPO×drift-pi0.5**；② **EXPO-FT 原算法（EXPOLearner/BCLearner）适配到新任务 env**。
  RoboTwin env 对两者通用。
- **已完成（Phase 0–3 + value 头改造，全套 31 测试通过）**：drift 合并进 openpi；`DBPOLearner`（`agents/alg/dbpo*.py`）+
  `RolloutBuffer`（`data/rollout_buffer.py`）+ GAE + PPO/BPO surrogate + 随机适配头（`networks/dbpo_heads.py`）+
  真实 pi0.5 接线（`dbpo_pi05.py`，`build_dbpo_from_pi05`/`run_ppo_iteration`）+ `configs/model/dbpo_pi_config.py`。
  value 头已改为读 **pooled `suffix_feat` + stop_gradient**（RLinf 设计）。
- **本计划范围 = Phase 4**：补齐"环境"与"训练入口"，让上述 learner 真正在 RoboTwin 里转起来。

## 1. 锁定的决策（除非主动推翻，不再讨论）

| # | 决策 | 出处 |
|---|---|---|
| 1 | **零侵入**：RoboTwin 适配写成**独立包 `client_robotwin/`**（与 `client/` 平级，绝不进 `client/`，不改任何 DROID 文件）| CLAUDE.md |
| 2 | **执行方式 2**（逐 action TOPPRA、变帧数）为默认；方式 1/3 后续适配 | CLAUDE.md |
| 3 | **稀疏 0/1 终局奖励，无 reward shaping**；macro-action 奖励 = 段内 success | CLAUDE.md |
| 4 | **DBPO rollout 执行整段 chunk**：`H_e = H = 50`（对齐 RLinf，**不做部分执行/前缀截断**）。**EXPO-FT 原算法保留部分执行**（per-step + `replan_steps`）| 用户 2026-06-18 |
| 5 | `action_dim=14`、`n_real_dims=14`（2×7 真实维，余 padding 靠 dim_mask）| CLAUDE.md |
| 6 | value 头 = pooled `suffix_feat` + stop_gradient（已实现）| DEV §3.3 |
| 7 | actor 初值 = RoboTwin stack-two-blocks DBP(Stage-1) checkpoint；anchor θ̄ = 同快照 | CLAUDE.md |
| 8 | learner 端 `expo_ft/env/env_client.py` **不动**（通用 websocket）| CLAUDE.md |
| 9 | **单 env，不并行**（不引入 RLinf 多 env 向量化）：仿真只为跑通**真机 RL 流程**，真机本就无法并行 env。样本效率不足为已知代价 | 用户 2026-06-18 |
| 10 | **env 算法无关（同等重要）**：`client_robotwin/` 的 env 同时服务 **DBPO 与 EXPO-FT 原算法（EXPOLearner/BCLearner）**；接口对所有 `model_cls` 通用，差异只在训练入口循环 | 用户 2026-06-18 |

## 2. 目标架构与接口契约

```
 learner (expo-ft/.venv, JAX)                         client_robotwin/ (自有 sim 依赖)
 ┌─────────────────────────────┐                      ┌───────────────────────────────────┐
 │ train_pi_robo_dbpo.py        │   websocket          │ run_robotwin_client.py (server)    │
 │  DBPOLearner.sample_actions  │  msgpack_numpy  ⇄    │  操作: create_env/reset/step/      │
 │  决策步循环: 执行 H_e 步      │  (复用现协议)         │       get_observation/get_info     │
 │  → RolloutBuffer → GAE       │                      │   ↓ 调 RoboTwinEnv (mirror droid)  │
 │  → run_ppo_iteration → 发布   │                      │ RoboTwinEnv ← robotwin VectorEnv   │
 └─────────────────────────────┘                      └───────────────────────────────────┘
        expo_ft/env/env_client.py (EnvClientWrapper, 不动)
```

**复用不变的 websocket 协议**（见 `expo_ft/env/env_client.py` 与 `client/run_client.py`）：

| 操作 | 请求字段 | 响应 | RoboTwinEnv 对应方法 |
|---|---|---|---|
| `create_env` | `{example_action, env_usage, video_dir, ...}` | `{env_id, task_description}` | `__init__` |
| `reset` | `{env_id}` | `{observation, done}` | `reset()→obs` |
| `step` | `{env_id, action}`（**单个 action**，list）| `{action: executed_action, action_type}` | `step(action)→{executed_action}` |
| `get_observation` | `{env_id}` | `{observation}` | `get_observation()` |
| `get_info_for_step` | `{env_id}` | `{done, success, reward, mask}` | `get_info_for_step()→(done,success,reward,mask)` |

**要 mirror 的接口**（`client/envs/droid_env.py`）：`reset`/`get_observation`/`step(action)`/`get_info_for_step()`。
droid_env 已是**稀疏 0/1 终局**（`reward=1.0 if success else 0.0`，`mask=0.0 if done else 1.0`，:115-118）——RoboTwin 照此。

**obs 格式**：必须产出 pi0.5 的 **RoboTwin 双臂 config（`pi05_aloha_robotwin`）** 期望的 keys（约定：
`base_image`/`left_wrist_image`/`right_wrist_image` + 14 维关节 `state` + `prompt/instruction`）。
**与 DBP 训练数据管线严格对齐**（图像分辨率、相机映射、state 维序、归一化）——见 T2。

## 3. 包结构（要新建/修改的文件）

```
client_robotwin/                      # ★ 新建独立包
  __init__.py
  run_robotwin_client.py              # server，复用 client/run_client.py 的协议骨架（不 import DROID 专有依赖）
  envs/
    __init__.py
    robotwin_env.py                   # ★ 核心适配器，mirror droid_env 接口；参考 RLinf robotwin_env.py
    utils.py                          # 图像/状态处理（按需）
  (pyproject.toml / 独立 venv)         # 隔离 sapien/robotwin 重依赖（讨论点 D5）

# learner 侧（不污染 DROID 路径）
configs/model/dbpo_pi_config.py       # 改：双臂（pi05_aloha_robotwin / action_dim=14 / n_real_dims=14 / H=50 / ckpt 路径）
configs/task/robotwin_stack_blocks.py # ★ 新增 task config（env_type/control_hz/replan_steps 等）
train_pi_robo_dbpo.py                 # ★ 新增训练入口（决策步 on-policy 循环；不改 train_pi_robo.py）
```

## 4. 分步任务（每步：做什么 / 参考 / 交付 / 验证）

### 4a — RoboTwin 独立 client 包

- **T1 `RoboTwinEnv` 骨架**：建 `client_robotwin/envs/robotwin_env.py`，定义类 + `__init__`（实例化已改造的
  RoboTwin，单 env 或 VectorEnv n_envs=1）。**参考** `RLinf/rlinf/envs/robotwin/robotwin_env.py`（`_init_env` 用
  `robotwin.envs.vector_env.VectorEnv`）。交付：可构造、可 `close`。验证：`python -c` 构造一个 env 不报错。
- **T2 obs 格式对齐**：实现 `get_observation()` → 产出 pi0.5 RoboTwin config 期望的 dict（images+state+prompt）。
  **参考** RLinf `_extract_obs_image`(:161)（full_image/left_wrist_image/right_wrist_image/state/instruction）。
  **核对**与 DBP 训练数据完全一致（图像 resize=224、相机 id、14 维 state 顺序）。交付：obs dict。
  验证：把一帧 obs 喂给 `build_dbpo_from_pi05` 的 `preprocess_observation` + drift forward 不报形状错。
- **T3 `step(action)` 单 action 执行（方式 2）**：接收**一个** 14 维 action → RoboTwin 方式 2（逐 action TOPPRA、
  变帧数）执行 → 返回 `{executed_action}`。交付：单步执行。验证：随机 action step 若干次仿真不崩、executed_action 形状对。
- **T4 reward/done/success/mask**：实现 `get_info_for_step()→(done,success,reward,mask)`，**稀疏 0/1 终局**
  （成功=1 否则 0；done 时 mask=0）。**参考** RLinf `_calc_step_reward`(:206) + droid_env :115-118。交付：四元组。
- **T5 `reset()` + 种子**：`reset()→obs`；可选 success_seeds（reset 到可解初始态）。**参考** RLinf `_init_reset_state_ids`(:505)。
- **T6 `run_robotwin_client.py` server**：复用 `client/run_client.py` 的 websocket+msgpack 协议骨架，
  把 5 个操作转发到 `RoboTwinEnv`。**不 import** DROID/zed 专有依赖。交付：可启动的 server。
- **T7 冒烟测试**：起 server → learner 侧用 `EnvClientWrapper` 连 → `reset`/随机 `step`/`get_info_for_step` 跑通一个 episode。
  交付：端到端连通。验证：日志显示 obs/executed_action/reward 正常流转。
- **T8 ⭐ greedy DBP baseline 闸门**：用 RoboTwin DBP checkpoint 在该 env 里**纯 mean（关噪声）**评 stack-two-blocks，
  确认成功率 **>0**。**这是 RL 前的硬门槛**（RL 靠 BC 流形 bootstrap，做不动则 PPO/BPO 无正信号）。

### 4b — 训练入口接线（决策步 on-policy 循环）

- **T9 config 改双臂**：`dbpo_pi_config.py` 改 `pi05_config_name=pi05_aloha_robotwin`、`action_dim=14`、
  `n_real_dims=14`、`action_horizon=50`、**`replan_steps=50`（=H，DBPO 整段执行，D1 已定）**、`pi05_weight_loader_path=<DBP ckpt>`；
  新增 `configs/task/robotwin_stack_blocks.py`（`env_type='sim'`、`control_hz` 等）。
- **T10 `train_pi_robo_dbpo.py`**：新入口（**不改** `train_pi_robo.py` 的 EXPO/BC 路径）：
  `build_dbpo_from_pi05` 加载 DBP ckpt → **决策步（macro-action）循环**：
  ① `obs→sample_actions→(chunk[50], z[50], logp_old, value)`；② 逐个 `env.step` **执行整段 50 步**（H_e=H=50），
  聚合 reward（稀疏终局；参考 RLinf chunk_level sum），取执行后 obs 为 next_obs、done 取段内终止；
  ③ 存 `(o,z,x_exec[:50],logp_old,value,reward,done)` 进 `RolloutBuffer`（决策步粒度，非 env-step；logp 在整 50 步上求和）；
  ④ 采满 `rollout_size`→`buffer.finalize`(GAE)→`run_ppo_iteration`→发布参数→清空。
- **T11 小规模跑通 + 监控（DBPO）**：reward 曲线、**ratio 分布/approx-KL**（整段 700 维易爆 → 超界则加 ratio clamp，D1 残留项）、
  log-std 演化、anchor loss、value 拟合。定 D8（rollout/epoch）+ §6.4 超参。

### 4c — EXPO-FT 原算法适配同一新 env（同等优先）

- **T12 EXPO-FT 跑新任务 env**：复用现有 `train_pi_robo.py`（**不改其逻辑**）——它已支持 `env_type='sim'`（:113）。
  只需 ① 提供 `configs/task/robotwin_*.py`（`env_type='sim'` + RoboTwin 数据集路径）；② EXPO/BC 的 model config
  指向双臂 pi0.5；③ 连同一个 `client_robotwin/` server。EXPO-FT 保留其 **per-step + `replan_steps` 部分执行**。
  交付：`EXPOLearner`/`BCLearner` 能在 RoboTwin stack-two-blocks 上采样+更新。验证：短跑不崩、reward/episode 正常。
- **注**：env 侧无需为 EXPO-FT 做任何特化——per-action `step` + 稀疏奖励接口对两算法通用；差异只在 train 入口的 rollout 粒度。

---

## 4.5 PPO 训练伪代码（DBPO 权威 + EXPO-FT 方案 A 映射）

> 权威来源：`/home/xukainan/DBPO/dbpo/workspace/dbpo_finetune_workspace.py`（`run()` + `_ppo_update()`）。
> **关键事实**：PPO 按**固定决策步数 `n_steps` 更新，不是按 episode 数**；episode **跨迭代边界**（rollout 间不 reset，
> `buffer.firsts[0]=上一轮 done`，GAE 末步 bootstrap）。每决策步 = 1 次 chunk 推理 + 执行 `act_steps(=H_e)` 子步（env 内聚合 1 reward）。

### 4.5.1 DBPO 原版（权威，n_envs 并行）
```
init: actor_ft = deepcopy(actor_old);  actor_old 冻结(=anchor θ̄);  critic
prev_obs = venv.reset();  done = 0
for itr in 1..n_train_itr(1500):
    buffer.reset();  buffer.firsts[0] = done           # 不 reset env；episode 跨 itr
    # ---- Rollout：恰好 n_steps(500) 个决策步（NOT episodes）----
    for step in 1..n_steps:
        value          = critic(prev_obs)
        chunk,(z,act),logp_old = actor.get_actions(prev_obs)   # 推理；存 z
        prefix         = chunk[:, :act_steps]                  # 执行前缀 = H_e
        obs,reward,term,trunc = venv.step(prefix)              # env 执行 H_e 子步→聚合 1 reward/env
        buffer.add(step, prev_obs, (z,act), reward, term, trunc, value, logp_old)
        prev_obs = obs                                          # venv 内部 auto-reset
    # ---- Update：K(5) epoch 在 n_steps×n_envs 批上 ----
    buffer.update(prev_obs, critic)        # reward 归一 + GAE（bootstrap V(prev_obs)，γ=0.99/决策步）
    for epoch in 1..update_epochs(5):
        for mb in shuffle(n_steps×n_envs, size=batch_size):
            logp_new = actor.logprob(mb.obs, mb.z, mb.act)     # 同 z 复用 → ratio 精确
            ratio    = exp(logp_new - logp_old)                # normalize_dims=True → 每坐标均值比
            pg       = PPO_clip(ratio, adv, eps=0.02)
            v        = 0.5(critic(obs)-returns)²  [+value clip]
            ent, anchor = entropy(mb), mse(actor_ft.mean(z), actor_old.mean(z))
            loss = pg + 0.01·ent_loss + 0.5·v + 1.0·anchor
            backward; clip_grad_norm(0.5); critic.step(); if itr>=critic_warmup(5): actor.step()
            if approx_kl > target_kl(0.02): break              # KL 早停
```
超参：`n_steps=500, update_epochs=5, batch=50000, γ=0.99, λ=0.95, eps=0.02, ent=0.01, vf=0.5, anchor=1.0,
target_kl=0.02, critic_warmup=5, actor_lr=1e-5, critic_lr=1e-4`。

### 4.5.2 EXPO-FT 方案 A 映射（async 双线程，单 env，per-action env_client，1 轮滞后）
```
# train_pi_robo_dbpo_async.py —— 保留 EXPO-FT async 骨架（_publish_lock/_buffer_lock/_published[0]）
# 改：PiReplayBuffer→RolloutBuffer；_update_worker 改 on-policy；actor 一轮内冻结 π_old

ACTOR 线程（rollout 角色，device[0]）:
  π_old = 当前 _infer_cache               # 整轮冻结，轮内不拾取新参数
  for step in 1..rollout_size(256):       # 决策步（NOT episodes）
      obs = env.get_observation()
      chunk, z, logp_old, value = agent.sample_actions(obs)        # 推理(π_old)，含归一化 logp
      r_agg = 0
      for h in 1..H_e(25):                  # per-action env_client 执行前缀
          real_a, _ = env.step(chunk[h])    # → client_robotwin take_action(方式2)
          done,succ,r,mask = env.get_info_for_step();  r_agg += r
          if done: break
      next_obs = env.get_observation()
      with _buffer_lock: rollout_buffer.add(obs, z, chunk[:H_e], logp_old, value, r_agg, done)
      if done: env.reset()
  signal: rollout 满 → 取 _published[0] 更新 π_old（1 轮滞后重叠）

LEARNER 线程（update 角色，device[1:]）:
  wait 整轮 rollout
  buffer.finalize(GAE, bootstrap V(next_obs))
  for epoch in 1..ppo_epochs(5):
      for mb in buffer.iterate_minibatches(num_minibatches(8)):
          agent, m = agent.update(mb)        # ratio(z复用), PPO clip 0.02, value, ent, anchor
          if m.approx_kl > target_kl(0.02): break        # KL 早停（待接入）
  with _publish_lock: _published[0] = agent._infer_cache  # 发布
  buffer.clear()                              # on-policy：用完即弃
```
要点：① 每条存采集时 `logp_old`（z 复用 → ratio 精确）；② 整轮同一 π_old；③ 用完即弃（非 replay 复用）；
④ 1 轮滞后 = async 重叠（actor 用第 N 轮参数采 N+1 轮，learner 训第 N 轮）。per-action env_client 下 actor 每决策步调 H_e 次 `env.step` 并聚合 reward（DBPO 的 `venv.step(prefix)` 在 per-action 接口下的等价）。

## 5. 待讨论的开放问题（执行前先和用户敲定）

- **D1 — H_e ✅ 已定：DBPO 执行整段 chunk（`H_e=H=50`，对齐 RLinf，不部分执行）**；EXPO-FT 保留部分执行。
  **残留待定**：整段 50×14=700 维 → chunk_level ratio 易爆/消失，是否引入 **ratio clamp**（RLinf `bpo_ratio_clamp=4.0`）；
  起步监控 ratio 分布/approx-KL，超界再加 clamp。
- **D2 — env 并行 ✅ 已定：单 env，不并行**（不引入 RLinf 向量化）。理由：仿真只为跑通真机 RL 流程，真机无法并行 env。
  代价：样本效率低（D8 调 rollout/epoch 缓解）。
- **D3 — reward 聚合**：macro-action reward = 段内 sum / 取终局 success（稀疏下二者近似）。
- **D4 — obs/action 维度精确对齐**：14 维 state 的关节顺序、夹爪定义、图像相机映射，必须与 DBP 训练数据逐一核对。
- **D5 — 依赖隔离**：`client_robotwin/` 用独立 venv（sapien/robotwin 重）还是共用 client/.venv？
- **D6 — clip_eps**：DBPO repo 实测 `clip_ploss_coef=0.02`（你 config 现 0.2，差 10×），是否对齐。
- **D7 — task/config 命名**：RoboTwin 配置名、task config 文件名、checkpoint 路径占位符填写。

## 6. 关键文件索引

- **learner（要改/新增）**：`configs/model/dbpo_pi_config.py`、`configs/task/robotwin_stack_blocks.py`(新)、`train_pi_robo_dbpo.py`(新)
- **learner（不动，复用）**：`expo_ft/env/env_client.py`、`agents/alg/dbpo*.py`、`data/rollout_buffer.py`、`networks/dbpo_heads.py`、`agents/alg/dbpo_pi05.py`
- **要 mirror 的接口**：`client/envs/droid_env.py`、`client/run_client.py`（协议骨架）
- **RLinf 参考**：`RLinf/rlinf/envs/robotwin/robotwin_env.py`（适配核心）、`docs/RLINF_ROBOTWIN_REFERENCE.md`（已剖析）
- **决策/惯例**：`CLAUDE.md`、`docs/DBPO_DEV.md`、`docs/BIMANUAL_RL_HORIZON_SURVEY.md`
