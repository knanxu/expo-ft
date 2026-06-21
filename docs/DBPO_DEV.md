# DBPO × EXPO-FT 集成开发文档

> 目标：在 EXPO-FT 真机在线 RL 框架中，用「改进版 pi0.5（drift head 单步生成策略）」替换原有 flow-matching 策略，并实现 **DBPO**（Drift-Based Policy Optimization）在线 RL 算法，其策略优化器用 **BPO**（Bounded Ratio Policy Optimization）替代标准 PPO。

- 维护人：polar823
- 起始日期：2026-06-17
- 状态：**调研完成，进入实现阶段（Phase 0）**

本文件是**活文档（living doc）**：第 1–6 节记录调研结论与设计决策（相对稳定），第 7 节是滚动更新的「进展日志 / Changelog」，每完成一个里程碑追加一条。

---

## 0. TL;DR — 一句话总览

把 EXPO-FT 现有的 **JAX/Flax 在线 RL 基础设施**（真机 env client、async actor-learner、数据管线、配置体系）保留并复用，**新增一个 `DBPOLearner`**（与现有 `EXPOLearner`/`BCLearner` 并列），它：

1. 用 `/home/xukainan/openpi` 改进版 pi0.5 的 **drift head** 作为 actor backbone（单步生成，`_sample_actions_drifting(return_hidden=True)` 返回 `(mean, suffix_feat, cond_emb)`）；
2. 在 backbone 上挂一个 **state-conditioned 对数标准差头 `g_ψ`** 和一个 **value 头 `V_φ`**，把确定性 drift 输出变成有解析 log-prob 的高斯 actor；
3. 用 **on-policy rollout buffer**（关键：存储并复用 latent `z`）+ GAE + 多 epoch 更新，按 **DBPO** 目标（PPO clip + anchor 正则）训练；
4. 策略 surrogate loss 做成**可插拔**，默认用 **BPO 的 ATV loss** 替代 PPO clip。

---

## 1. 项目背景与现状

### 1.1 三个输入

| 来源 | 路径 / 链接 | 角色 |
|---|---|---|
| EXPO-FT 框架 | `/home/xukainan/expo-ft` | 真机在线 RL 基础设施（JAX/Flax）|
| 改进版 openpi | `/home/xukainan/openpi` | 含 drift head / heatmap head 的 pi0.5（JAX + PyTorch 双实现）|
| DBPO 论文 | Gao et al. 2026, arXiv:2604.03540；repo `github.com/YuxuanGao0822/DBPO` | drift 策略的 PPO 在线微调 |
| BPO 论文 | Ao et al. 2026, arXiv:2604.18578；repo `github.com/bounded-ratio-rl/bounded_ratio_rl` | PPO 的最小改动替代，收敛更快 |

### 1.2 EXPO-FT 框架关键事实（调研结论）

- **框架**：服务端 learner 与 VLA 封装 **全部是 JAX/Flax/nnx**（`expo_ft/agents/vla/pi05.py` 用 `flax.nnx` + `jax`）。客户端（真机 SDK）是独立 numpy/mujoco 环境。
- **现有算法**：`EXPOLearner`（`expo_ft/agents/alg/expo_ft.py`）是 **off-policy actor-critic（SAC 风格）**：base policy（pi0.5）+ residual actor + Ensemble-Q（REDQ）+ Q-selection。`BCLearner`（`bc.py`）是纯模仿。
- **算法分发**：配置里 `config.model_cls = "EXPOLearner" | "BCLearner"` 字符串，在 `train_pi_robo.py:139-147` 分支构造。**新增算法 = 新增一个 `model_cls` 分支即可**，对框架是低侵入扩展点。
- **数据管线**：`expo_ft/data/replay_buffer.py`（`PiReplayBuffer`，环形 off-policy buffer）+ `batch_processor.py`。字段含 images / state / actions(chunk) / tokenized_prompt / rewards / masks / dones。
- **Actor-Learner**：`train_pi_robo_async.py` 主线程采样（device[0]），后台线程更新（mesh on device[1:]），`_publish_lock`/`_buffer_lock`/`_episode_done` 三把锁同步，参数通过 `cache_infer_params()` + `_published[0]` 原子发布。
- **真机环境**：`expo_ft/env/env_client.py`，receding-horizon 执行（`replan_steps`），动作以 chunk 形式规划。
- **当前挂载的 openpi**：`expo_ft/agents/vla/openpi` 是 **pd-perry fork**（flow-matching，**无 drift/heatmap**）。需替换为改进版。

### 1.3 改进版 openpi 关键事实（调研结论）

- **drift head 双实现**：JAX `src/openpi/models/pi0.py`（`_compute_loss_drifting`、`_sample_actions_drifting`）+ PyTorch `models_pytorch/pi0_pytorch.py`。
- **RL 契约已预埋**：`_sample_actions_drifting(return_hidden=True)` 在**两个框架里都返回相同三元组** `(mean, suffix_feat, cond_emb)`，源码注释明确写 *"so the DBPO hooks have an identical (mean, suffix_out, cond_emb) contract across frameworks"*：
  - `mean`：`[B, action_horizon, action_dim]` —— 确定性 drift 动作，**作为高斯 actor 的均值**。
  - `suffix_feat`：`[B, action_horizon, expert_width]` —— post-LLM 动作 token 特征，**value 头输入**。
  - `cond_emb`：`[B, expert_width]` —— 对 prefix 有效 token 均值池化的观测条件特征，**state-conditioned log-std 头输入**。
- **单步推理**：drift 推理是 1-NFE（一次前向），`time=1.0` 纯噪声直接出动作，无迭代去噪。比 flow-matching 的 `num_steps=10` 快约 10×。
- **配置开关**：`pi0_config.py` 中 `use_drifting_loss=True`（与 `use_heatmap_loss` 互斥，默认都 False → flow matching）。
- **尚不存在的部分**：**没有** log-std 头 / value 头 / anchor / rollout buffer / DBPO learner —— 只有模型层 `return_hidden` 脚手架。这些是本项目要开发的。

### 1.4 DBPO 算法关键事实（论文精读结论）

DBPO 把确定性 one-step（drift）生成器变成可做 PPO 的随机 actor。核心机制（论文 Eq. 编号）：

1. **随机适配（Eq. 43–46）**：actor 高斯 `π_{θ,ψ}(x|o,z) = N(x; μ_θ(o,z), diag(σ̃_ψ(o)²))`。均值 `μ_θ` = drift 输出（依赖 latent `z`，提供多模态）；**对角 log-std `log σ_ψ(o) = g_ψ(c_θ(o))` 是 state-conditioned 网络头**（接在观测特征 `cond_emb` 上），并做 clip（Eq. 45）`log σ̃ = clip(log σ, log σ_min, log σ_max)`。
2. **latent 复用（Eq. 47, 49–50）—— 正确性关键**：rollout 时采 `z~p_0=N(0,I)`，再采 `x~π(·|o,z)`，**把 `(z,x)` 都存进 buffer**；PPO 更新时**复用同一个 `z`**。因为 `p_0(z)` 与参数无关，joint ratio 精确等于 conditional ratio：`r_t = π_{θ,ψ}(x_exec|o,z) / π_{θ_k}(x_exec|o,z)`。**不复用 z，ratio 就错。**
3. **executed-prefix log-prob（Eq. 13/48/57）**：只对**实际执行的 prefix**（`H_e` 步，从 index `T_o` 起）算 log-prob，逐步逐维对角高斯求和，未执行的 suffix 不进入 credit assignment。
   `log π(x_exec|o,z) = Σ_{h∈[T_o, T_o+H_e)} Σ_{m} log N(a_{h,m}; μ_{h,m}, σ̃_{h,m}²)`。
4. **目标函数（Eq. 51–55）**：标准 PPO clip + value loss + entropy + **anchor 正则**（DBPO 特有）：
   `L_RL = -J_clip + c_v·L_value - c_e·H + λ_anchor·L_anchor`，
   其中 `L_anchor = E[‖μ_θ(o,z) - μ_{θ̄}(o,z)‖²]`，`θ̄` 是**冻结的 Stage-1（drift 预训练）参数**，相同 `z`、相同 `o` 下约束 mean 不偏离预训练流形。**消融显示 anchor 对稳定性至关重要**（去掉后 RoboMimic 0.90→0.75）。
5. **value 只输入观测 `o`**（Eq. 52），`V_φ(o_t)`，不依赖 z / action。
6. **论文未给 RL 超参数值**（ε, λ_anchor, GAE-λ, c_v, c_e, log σ 范围）——需从 repo 取或自调（见 §6.4）。

### 1.5 BPO 算法关键事实（论文精读结论）

BPO 用对 likelihood ratio 的**有界约束**的解析最优解去替代 PPO 的启发式 clip。**对 PPO 是最小改动——只改 policy loss 那几行**，value loss / GAE / entropy / 训练循环全不变。

- 目标 ratio（带正则解析解）：`ρ* = 1 + ε·tanh(Ã / 2λ)`（`λ`→0 退化为硬 sign 形式 `1 + ε·sign(Ã)`）。
- **BPO surrogate（ATV loss，Eq. 8/16）**：`l_BPO = |A| · |ρ - ρ*|`，advantage 加权的 total-variation（L1）损失。
- 与 PPO 两点差异：① PPO 越界后梯度为 0（单边停），BPO 是**对称斜率**（越界仍有梯度往回拉）；② PPO 用 mean advantage，BPO 原版用 **median advantage**（需多一个 median 头 `μ_ψ`，但消融证明**用 mean 性能相当**，工程上可省）。
- 新超参：`λ=0.001`（推荐），TV 权重 `α₁=0`，value/median loss 系数 `w₁=w₂=0.5`。调参建议：复用 PPO 超参，clip ε 调大 0.1，epoch 翻倍，entropy 系数取 PPO 的 0.1×。

---

## 2. 核心架构决策（Decision Log）

> 每条决策含「决策 / 理由 / 备选与权衡 / 状态」。这是文档最重要的部分。

### D1 — 框架：DBPO learner 用 **JAX/Flax**（不是 PyTorch）
- **决策**：DBPO learner、log-std 头、value 头、rollout buffer 全部用 JAX/Flax 实现，与 `EXPOLearner` 同构。
- **理由**：(1) EXPO-FT 的 learner + VLA 封装（`pi05.py`）+ async 训练全是 JAX/nnx；(2) 改进版 openpi 的 **JAX 路径已内置 drift + `return_hidden` 契约**，注释显式为「DBPO hooks」准备；(3) 避免 JAX↔PyTorch 跨框架桥接（梯度、设备、sharding 都要打通，成本高、易错）；(4) 复用 EXPO-FT 的 sharding/FSDP、`cache_infer_params` 发布机制。
- **备选**：PyTorch learner（参考 repo 是 PyTorch，drift 也有 PyTorch 实现）。**否决**，因为要把整个 EXPO-FT 真机 RL infra 迁到 PyTorch，工作量与风险远大于在 JAX 里实现 DBPO 数学。
- **状态**：✅ 已定。（若后续发现 JAX 内 PPO/GAE 实现成本过高，再评估 PyTorch 旁路。）

### D2 — Action head：用 **drift head（单步）**
- **决策**：actor 用 `use_drifting_loss=True` 的 pi0.5，推理走 `_sample_actions_drifting(return_hidden=True)`。heatmap head 暂不接 RL（保留为 BC/离散化分支）。
- **理由**：用户主用 drift head；单步推理满足真机控制频率；`(mean, suffix_feat, cond_emb)` 契约天然适配 DBPO 的高斯均值 / value 输入 / log-std 输入。
- **状态**：✅ 已定。

### D3 — 用 **git merge** 把改进版 drift 并入 EXPO-FT 自带的 openpi checkout（不是整体替换）
- **决策（Phase 0 后落定）**：在自带 fork 的 editable checkout（`expo_ft/agents/vla/openpi`，branch `expo_ft`）里 **`git merge` 改进版 fork 的 `feat/jax-return-hidden`**，手工解决 4 个冲突文件后保留双方增量。**否决**了「重指 uv source / 整体替换」方案。
- **否决理由（Phase 0 实测）**：两 fork 是**双向分叉**，整体替换会丢掉 EXPO-FT 必需的定制：
  - `pi05.py:218` 调用 `_policy.Policy.infer(is_batch=True, for_training=True)` —— 这两个参数是 pd-perry 给 `policy.py` 加的，改进版没有，**整体替换即崩**。
  - EXPO-FT 加载的具名 config `expo_pi05_droid_lora_finetune_sft_cartesian_state` **只在自带 fork** 的 `config.py`，改进版没有。
- **可行性（Phase 0 实测，低风险）**：两 fork 共享近期 merge-base `54cbaee`（各 28 / 15 个提交）；EXPO-FT 依赖的**基础设施模块全部字节相同**（`training/utils.py` TrainState、`sharding`、`optimizer`、`weight_loaders`、`transforms`、`nnx_utils`、`array_typing` 均 0 行差异）。冲突仅 4 个文件，详见 §Phase 0 结论。
- **状态**：✅ 已定。⬜ 合并执行 = Phase 0 的收尾动作（下一步）。

### D4 — RL 范式：**on-policy DBPO**，不复用 EXPO-FT 的 off-policy critic
- **决策**：新建 on-policy 训练路径（rollout buffer + GAE + 多 epoch），不复用 `EXPOLearner` 的 replay buffer / Ensemble-Q / residual / Q-selection。
- **理由**：DBPO/BPO 本质是 PPO 系 on-policy 算法，与 EXPO-FT 的 SAC 系 off-policy 架构不兼容（ratio 需要 `π_old`、需要 rollout-time 的 `z` 与 `logp_old`）。强行复用 off-policy buffer 会破坏 ratio 正确性。
- **复用 vs 新建**：
  - **复用**：真机 `env_client`、async actor-learner 线程模型与三锁、配置体系、数据预处理 transforms、checkpoint/日志工具、websocket 推理服务。
  - **新建**：`RolloutBuffer`（存 `o, z, x_exec, logp_old, value, reward, done, adv, ret`）、GAE、PPO/BPO 更新、log-std 头、value 头、anchor。
- **状态**：✅ 已定。

### D5 — 随机适配实现细节
- **决策**：
  - 高斯均值 = `mean`（drift 输出，依赖 `z`）。
  - log-std 头 `g_ψ`：小 MLP，输入 `cond_emb`（`[B, expert_width]`），输出 `[B, exec_action_dim]`，clip 到 `[log σ_min, log σ_max]`。**state-conditioned，不是全局标量**。
  - value 头 `V_φ`：MLP，输入对观测的特征（见 §3.3 决策）→ 标量。
  - log-prob：只在 **executed prefix（`H_e`=`replan_steps` 步）** 上逐步逐维高斯求和。
  - **rollout buffer 存 `z`，更新时复用**（D4 正确性核心）。
  - anchor：保存一份 **frozen 的初始 drift 参数 `θ̄`**，每次更新算 `‖μ_θ(o,z) - μ_{θ̄}(o,z)‖²`。
- **状态**：✅ 已定（细节字段在 §3、§5 展开）。

### D6 — Surrogate loss 可插拔，默认 **BPO**
- **决策**：实现 `surrogate_loss(ratio, adv, *, mode, eps, lam, ...)`，`mode ∈ {"ppo", "bpo"}`，配置项 `config.rl_surrogate`。先实现并验证 `ppo`（基线、好调试），再切 `bpo`（默认目标）。
  - PPO：`-mean(min(ρ·A, clip(ρ,1±ε)·A))`。
  - BPO：`ρ* = 1 + ε·tanh(Ã/2λ)`；`mean(|ρ - ρ*|·|A|)`。初版用 **mean advantage**（省 median 头），`Ã ≈ A`；median 头作为可选 `config.bpo_use_median` 后续加。
- **理由**：BPO 是用户目标，但 PPO 作基线便于隔离「drift+stochastic 适配是否正确」与「surrogate 选择」两类 bug。
- **状态**：✅ 已定。

### D7 — 新算法以独立 `model_cls = "DBPOLearner"` 注册，零侵入现有算法
- **决策**：`expo_ft/agents/alg/dbpo.py` 新增 `DBPOLearner`，在 `alg/__init__.py` 导出，在 `train_pi_robo*.py` 加分支；新增 `configs/model/dbpo_pi_config.py`。`EXPOLearner`/`BCLearner` 完全不动。
- **状态**：✅ 已定。

### D8（开放）— rollout 长度 / 真机 on-policy 的数据量
- **问题**：真机 on-policy PPO 每轮要重采，样本效率低于 off-policy。真机采样昂贵。需确定 rollout horizon、每轮 episode 数、PPO epoch 数，平衡墙钟时间与稳定性。
- **倾向**：小 rollout（如 1–几个 episode）+ 较多 epoch（BPO 建议 epoch 翻倍）+ 较大 batch 复用。可能需要引入有限的「near-on-policy」重用窗口。
- **状态**：⬜ 开放，Phase 4 真机前定，先在 §6.4 给初始默认值。

---

## 3. 系统架构设计

### 3.1 模块总览

```
真机 (client/.venv, numpy/droid)                服务端 learner (.venv, JAX)
┌────────────────────────────┐                  ┌─────────────────────────────────┐
│ env_client (真机 IO)        │  obs/action ⇄    │ Actor 线程 (device[0])           │
│ receding-horizon 执行        │  websocket/共享   │  DBPOLearner.sample_actions      │
└────────────────────────────┘                  │   = drift mean + N(·,σ) 采样      │
                                                 │   产出 (z, x_exec, logp, value)   │
                                                 │         ↓ 写入 RolloutBuffer      │
                                                 │ Learner 线程 (mesh device[1:])    │
                                                 │  每凑满 rollout → GAE → 多 epoch  │
                                                 │  DBPO/BPO 更新 (θ,ψ,φ), 冻结 θ̄   │
                                                 │         ↓ cache_infer_params 发布 │
                                                 └─────────────────────────────────┘
```

### 3.2 `DBPOLearner` 状态（Flax PyTreeNode）

```
DBPOLearner:
  actor: Pi05DriftAgent           # drift backbone (use_drifting_loss=True), 不入 pytree
  actor_train_state               # openpi TrainState (params, opt_state, ...)
  anchor_params: at.Params        # ❄️ frozen Stage-1 drift 参数 θ̄ (anchor 用)
  logstd_head: TrainState         # g_ψ: cond_emb -> log σ
  value_head:  TrainState         # V_φ: obs feature -> scalar
  # 超参
  replan_steps (=H_e), action_dim, action_horizon (=H)
  clip_eps, gae_lambda, discount, c_value, c_entropy, lambda_anchor
  logstd_min, logstd_max, logstd_init
  rl_surrogate ("ppo"|"bpo"), bpo_lambda, bpo_use_median
  ppo_epochs, minibatch_size, rollout_size
```

### 3.3 关键设计点：value 头的输入特征（✅ 已定：suffix_feat + stop_gradient）
论文说 `V_φ(o_t)` 只输入观测。候选：
- (a) `cond_emb`（prefix 池化特征，`[expert_width]`）—— 纯观测条件，贴论文 `V_φ(o)` 语义。
- (b) `suffix_feat` 池化（`[B,H,width]→mean→[B,width]`）—— 含 action-query 上下文。
- **决策（2026-06-18，改用 (b)）**：value 头读 **mean-pooled `suffix_feat`**，并在头输入处 **`jax.lax.stop_gradient`**。
  理由：① **RLinf 的 PPO-on-VLA 实测用的就是 suffix（`_compute_value_from_suffix`），已在 RoboTwin RL 微调上验证**（见
  `docs/RLINF_ROBOTWIN_REFERENCE.md` §4）；② stop_gradient = RLinf 的 `detach_critic_input=True`，使 value loss 只更新
  value head、**不回传共享 action expert**（避免大 value 梯度污染 actor，value_lr≫actor_lr）。log-std 头仍用 `cond_emb`。
  代码：`networks/dbpo_heads.py ValueHead(detach_input=True)` + `agents/alg/dbpo_pi05.py value_feat=mean(suffix_feat)`。

### 3.4 RolloutBuffer 字段（on-policy，新建）

| 字段 | 形状 | 说明 |
|---|---|---|
| `observations` | images/state/tokenized_prompt | 复用现有 transforms 预处理 |
| `latent_z` | `[H, action_dim]` | **rollout 时的噪声，必须存，更新复用** |
| `actions_exec` | `[H_e, action_dim]` | 实际执行的 prefix 动作（采样得到）|
| `logp_old` | scalar | rollout 时 prefix log-prob（旧策略）|
| `value` | scalar | rollout 时 `V_φ(o)` |
| `reward` | scalar | 真机奖励 |
| `done` / `mask` | scalar | episode 终止 |
| `advantage` / `return` | scalar | GAE 后回填 |

> 注：动作时间尺度——EXPO-FT 以 chunk + `replan_steps` 执行。`H_e` 对齐 `replan_steps`；log-prob 的求和窗口 = 执行的 `H_e` 步。需与 `env_client` 的执行语义严格对齐（Phase 3 重点核对）。

---

## 4. 实现计划（分阶段）

> 每阶段有明确交付物与验收。完成后在 §7 追加 Changelog。

### Phase 0 — 环境打通与 API 兼容性核对（**先做，去风险**）
- [ ] 决定 D3 方案 (a)/(b)，让 EXPO-FT 的 `.venv` 装上含 drift 的 openpi。
- [ ] 核对改进版 openpi 与 `pi05.py` 依赖的接口 diff（`model.py`/`policy.py`/`config.py`/transforms/checkpoint）。
- [ ] 跑通：用 `use_drifting_loss=True` 的 pi0.5 在 `pi05.py` 里加载 + 单步 `sample_actions`（仅推理，验证 drift 路径在 EXPO-FT 进程内可用）。
- [ ] 验证 `_sample_actions_drifting(return_hidden=True)` 在 JAX 路径返回三元组、形状正确。
- **交付**：一份 API 兼容性 note（追加到本文档 §7 或单独 `docs/PHASE0_API_NOTES.md`）+ 可加载的 drift pi0.5。

### Phase 1 — 随机适配头（log-std 头 + value 头 + 高斯 log-prob）✅ 完成
- [x] 新建 **`expo_ft/networks/dbpo_heads.py`**（放 `networks/` 而非 `vla/`，与其它 `nn.Module` 同处更一致）：Flax linen `LogStdHead`（`cond_emb→logσ`，reshape `(...,H,A)`，clip；final 层 zero-kernel + bias=`log_std_init` → 初始 logσ 恒为 `log_std_init`、状态无关，训练中学状态依赖）、`ValueHead`（obs feat→标量，仅依赖观测）。输入维运行时推断（兼容 VLM 2048 / expert 1024）。
- [x] 新建 `expo_ft/agents/alg/dbpo_utils.py`：`gaussian_logprob(mean, log_std, actions, prefix_len, prefix_start, dim_mask)`（executed-prefix 逐步逐维对角高斯求和，Eq.48/57）、`gaussian_entropy(log_std, prefix_len, ...)`、`gaussian_sample(rng, mean, log_std)`、`step_mask(...)`。纯函数、jit 友好。
  - **EXPO-FT 适配点**：执行窗口 = `action_chunk[:replan_steps]` → `prefix_start=0, prefix_len=replan_steps(=8)`；`dim_mask` 可选只算真实动作维（padded 维在 ratio 中本就抵消，masking 让 logp/entropy 数值更干净）。
- [x] 单元测试 `expo_ft/agents/alg/dbpo_phase1_test.py`（自包含，无需 pytest）：**10/10 通过** —— log-prob 对拍 tfp、prefix masking 忽略 suffix、prefix_start 偏移、dim_mask、entropy 对拍 tfp、sample 统计(μ/σ)、LogStdHead 初始化恒等 + clip、ValueHead 形状、jit 可编译。
- **交付**：✅ 通过单测的随机适配工具集（utils + heads）。
- **注**：新文件均在 expo-ft 主仓工作区，**未提交**（待用户 review；主仓在 `main` 分支）。

### Phase 2 — `DBPOLearner` 离线 dry-run（无真机）✅ 完成
- [x] `expo_ft/agents/alg/dbpo_core.py`（**框架无关算法核**）：`compute_gae`(reverse scan)、`ppo_policy_loss`、`bpo_policy_loss`、`surrogate_loss`(分发)、`value_loss`(支持 value clip)、`anchor_loss`(stop-grad)、`normalize`。
- [x] `expo_ft/data/rollout_buffer.py`：`RolloutBuffer`（存 **latent z**、obs(pytree)、action、logp_old、value、reward、done；`finalize`→GAE；`iterate_minibatches`）。
- [x] `expo_ft/agents/alg/dbpo.py`：`DBPOLearner`(struct.PyTreeNode) `create / sample_actions / update`。**drift backbone 以抽象 `drift_apply_fn(params, obs, z)→(mean, cond_emb, value_feat)` 持有**——同一 learner 现在跑 mock、Phase 3 换真实 pi0.5（只改 `create`）。`update` 复用存储的 z 重算 prefix log-prob→ratio→surrogate(+value+entropy+anchor)→对 (θ,ψ,φ) 求梯度，冻结 θ̄。
- [x] dry-run（mock drift backbone，无 pi0.5）：rollout→GAE→PPO/BPO 多 epoch，**26/26 测试全过**（Phase1 10 + core 11 + dryrun 5）。验证：loss 全程有限、**首次更新 ratio≈1（z 复用正确性命门）**、参数确实更新、critic 能拟合 returns、BPO 模式跑通。
- **交付**：✅ 假数据上稳定 `update` 的 `DBPOLearner`（PPO + BPO 双模式）。
- **关键发现（GPU fp32 精度）**：GPU 默认对 fp32 matmul 用 TF32（~1e-3 相对误差），使 batch 不同的 rollout/update 前向略有差异，首次 ratio 漂到 ~1.004（设 `jax_default_matmul_precision="highest"` 后精确到 1e-5）。实践中被 PPO/BPO clipping 吸收；ratio 保真度 vs 精度是 **Phase 3/4 调参点**（真实 pi0.5 前向本就 bf16）。
- **接口说明**：`DBPOLearner` 暂为纯 `struct.PyTreeNode`（未继承 `AgentLearner`），因 on-policy `update(minibatch)` 签名与 off-policy `update(agent,batch,utd_ratio)` 不同；train-loop 适配器留到 Phase 3。

### Phase 3 — 真实 pi0.5 drift 集成 + 配置 + 注册 ✅ 完成
（BPO surrogate 已在 Phase 2 完成；本阶段聚焦把抽象 `drift_apply_fn` 接到真实 pi0.5 nnx，并让其在真实 3B 规模可训。）
- [x] `expo_ft/agents/alg/dbpo_pi05.py`：
  - `make_drift_apply_fn(model_def)` → `drift_apply_fn(params, obs, z, frozen)`：`nnx.merge(model_def, [trainable, frozen]) → preprocess_observation → _sample_actions_drifting(return_hidden=True)`，对 `params` 可微。
  - `build_dbpo_from_pi05(...)`：从 pi0.5 nnx (model_def + 全量 State) 组装 learner；actor = `flax TrainState(params=nnx.State, tx=optax)`（nnx.State 是纯数组 pytree，optax/jax.grad 直接可用 → **Phase 2 update 路径零改动复用**）。
  - **`trainable_filter`（关键）**：按 nnx Filter 把 State 切成 trainable（进优化器+求梯度）与 frozen（carried，actor+anchor 共享、不动）。**真实 3B 必需**——adam over 全量 = ~24GB；冻结 VLM/SigLIP、只训 action expert/投影才可行。
- [x] frozen 线程化：`drift_apply_fn` 升为 4 参 `(params, obs, z, frozen)`；`DBPOLearner` 加 `actor_frozen` 字段；mock 路径 `frozen=None` 不受影响。
- [x] `run_ppo_iteration(learner, buffer, ppo_epochs, num_minibatches, seed)`：on-policy 迭代驱动（train_pi_robo 的入口）。
- [x] `configs/model/dbpo_pi_config.py`（`model_cls="DBPOLearner"`，含 actor_trainable_regex + DBPO/BPO 超参，§6.3-6.4）。
- [x] `alg/__init__.py` 导出 `DBPOLearner / build_dbpo_from_pi05 / run_ppo_iteration`。
- [x] 集成测试 `dbpo_pi05_test.py`（dummy gemma `"dummy"` 变体 width=64，CPU 数秒）：**5/5 通过** —— drift 契约形状、**梯度真实流入 pi0.5 actor**、ratio≈1（z 复用，highest 精度）、anchor 冻结不变、**trainable_filter 冻结 backbone（1096 trainable vs 429M frozen）**、BPO 跑通。
- **交付**：✅ 真实 pi0.5 drift 可被 DBPO 训练（dummy 验证），PPO/BPO 可切，配置/注册就绪。**全套 31 测试通过**。
- **关键发现**：① plain `jax.grad` 可穿 `nnx.merge + _sample_actions_drifting` 求出 pi0.5 actor 梯度（nnx.State 叶子是 ArrayImpl）；② 全量 finetune adam = 24GB 不可行 → **trainable_filter 非可选**；③ value 用 `cond_emb`（池化观测特征）。
- **未尽（→ Phase 4）**：把 `run_ppo_iteration` + rollout 采集接进 `train_pi_robo*.py`（真实 DBP checkpoint 加载 + env_client）；LoRA 感知的 trainable_filter 在真实 config 上核对；ratio 保真度的 matmul 精度设置。

### Phase 4-EXPO — EXPO-FT 原算法 × drift × RoboTwin（**当前第一优先级，2026-06-21 插入**）
> **先于 Phase 4（DBPO）。** 让现成 `EXPOLearner`/`BCLearner` 在 drift pi0.5 + RoboTwin 上闭环——风险最低、
> 最快验证 env/checkpoint/数据管线，给 DBPO 打底。核心事实：drift 对 EXPO 是**纯 config 切换**（`pi0.py`
> `sample_actions:347`/`compute_loss:254` 已按 `use_drifting_loss` 分发），obs/action 走 openpi 同套 transforms，
> EXPO `sample_actions:532` 已含 `process_transformed_outputs` 反归一化，**不用改算法、不用写新训练入口**。

- [x] **新 model config** `configs/model/expo_ft_pi_drift_config.py`（拷 `expo_ft_pi_config.py` 改）：
      `pi05_config_name="pi05_aloha_robotwin_drifting_stack_blocks_two"`、`residual_action_xyzg=False`、
      `freeze_pi05_encoder=True`、`actor_success_only=True`。**不改** `expo_ft_pi_config.py`。
      🔴 仍需用户填 `pi05_weight_loader_path`(DBP drift ckpt) + `pi05_assets_dir/asset_id`(RoboTwin norm_stats)。
- [x] **RoboTwin 数据 loader** `expo_ft/env/robotwin_utils.py::process_robotwin_dataset`（新文件，**不改** DROID 路径）：
      读 RoboTwin demo hdf5（`joint_action/vector`(T,14) + `observation/{head,left,right}_camera/rgb` JPEG），
      产出与 `RoboTwinEnv.get_observation` **完全相同的扁平键**。`train_pi_robo_async.py` 按
      `config_task.dataset_loader`(="robotwin") 分发（DROID 默认 'droid' 不变）。**已在真实 episode 冒烟通过**
      （253 帧、键一致、图像 CHW、state(14)、actions(14)、终局 done/reward 正确）。
- [x] **🟡→✅ 图像 HWC/CHW 已定论 = CHW**：`aloha_policy.py:171` `_decode_aloha.convert_image` 做
      `einops.rearrange(img, "c h w -> h w c")` → **AlohaInputs 期望 CHW 输入**。`RoboTwinEnv._chw` 与 loader 均产 CHW，正确一致。
- [ ] **运行**：启 `client_robotwin/run_robotwin_client.py`（RoboTwin venv）→ 跑现成
      `train_pi_robo_async.py --config configs/model/expo_ft_pi_drift_config.py --config_task configs/task/robotwin_stack_blocks.py --dataset_path <RoboTwin demo dir>`。
- [ ] 前置：greedy（drift mean）评 DBP baseline 成功率>0；核对 `control_hz`/`replan_steps`/`max_decision_steps`（EXPO 按单 action 计数）。
- [ ] **🟡 端到端集成**：用真实 DBP ckpt + norm_stats 验证 obs dict 流经 replay buffer robotwin transform（`_preprocess_single_transition`）
      与 `process_raw_inputs` 不报错；drift 多候选采样（EXPO `N=8`）noise 形状路径；`freeze_pi05_encoder=True` 与 drift `noise_samples` 分支。
- [ ] 验证不回归：DROID/EXPO 既有路径行为不变（dispatch 默认 'droid' 字节等价）；DBPO 31 测试仍过。
- **交付**：EXPO-FT（off-policy actor-critic）在 RoboTwin drift pi0.5 上的可训练闭环 + 初步学习曲线。
- **运行时待确认**：drift 多候选采样（EXPO `N=8`）的 noise 形状路径；`freeze_pi05_encoder=True` 与 drift `noise_samples` 分支。

### Phase 4 — RoboTwin 仿真闭环（独立 client 包）+ 训练脚本接线 + 调参（DBPO，**Phase 4-EXPO 之后**）
> 本阶段以 **RoboTwin 仿真**（stack two blocks）为首个验证环境，替代原计划的真机；真机迁移留到 Phase 4+。

**4a — RoboTwin 独立 client 包（零污染 DROID client）**
- [ ] 新建独立包 **`client_robotwin/`**（与 `client/` 平级，**绝不进 `client/`**，不改任何 DROID 文件）：
  - [ ] `client_robotwin/envs/robotwin_env.py`：env 适配器，接口同 `droid_env.py`
        （`reset`/`get_observation`/`step(action)→{executed_action}`/`get_info_for_step()→(done,success,reward,mask)`），
        包住已改造的 RoboTwin。**参考 `RLinf/rlinf/envs/robotwin/robotwin_env.py`**：VectorEnv 并行、
        稀疏 0/1 终局 reward（`_calc_step_reward`/无 shaping）、success_seeds reset、auto_reset、
        chunk 整段 vs 逐 action 执行（默认**方式 2 逐 action TOPPRA**）。详见 `docs/RLINF_ROBOTWIN_REFERENCE.md` §2。
  - [ ] `client_robotwin/run_robotwin_client.py`：server，speak 与 `client/run_client.py` **完全相同的 websocket 协议**
        （`create_env`/`reset`/`step`/`get_observation`/`get_info_for_step`，复用 `openpi_client.msgpack_numpy`）。
  - [ ] RoboTwin 重 sim 依赖（sapien/robotwin）隔离在该包自己的环境；learner 端 `expo_ft/env/env_client.py` **不动**。
**4b — 训练入口（不破坏 EXPO/BC per-step 路径）**
- [ ] 新建 `train_pi_robo_dbpo.py`（或在 `train_pi_robo.py` 加**独立** `DBPOLearner` 分支）：`build_dbpo_from_pi05`
      加载 RoboTwin DBP checkpoint → **决策步（macro-action）粒度** on-policy 循环：每决策步 `sample_actions`→执行
      `H_e` 步→聚合 reward（稀疏终局，参考 RLinf `_cal_chunk_rewards`/chunk_level sum）→存 `(o,z,x_exec,logp_old,value,reward,done)`→
      采满 `rollout_size`→`buffer.finalize` GAE→`run_ppo_iteration`→发布参数。
- [ ] `H_e` ≡ env replan 间隔（红线）；config 改双臂（`pi05_aloha_robotwin`/`action_dim=14`/`n_real_dims=14`/`H=50`）。
- [ ] 前置：greedy（mean，无噪声）评 DBP baseline 确认 stack-two-blocks 成功率 >0。
- [ ] 监控：reward 曲线、**ratio 分布/approx-KL**（chunk_level 700 维易爆，见 RLinf `bpo_ratio_clamp`）、log-std 演化、anchor loss、value 拟合。
- [ ] 定 D8 的 rollout/epoch；定 §6.4 未定超参。
- **交付**：RoboTwin 仿真可训练的 DBPO×BPO，初步学习曲线。

### Phase 5（可选）— 增强
- [ ] BPO median 头 `μ_ψ`（`bpo_use_median=True`）。
- [ ] heatmap head 的 RL 变体探索。
- [ ] 推理服务（websocket）暴露 drift+stochastic 部署（部署去噪声，纯 mean，1-NFE）。

---

## 5. DBPO 数学 → 代码映射（实现备忘）

| 论文 | 公式 | 代码落点 |
|---|---|---|
| 高斯 actor | Eq. 46 `N(μ_θ(o,z), σ̃_ψ(o)²)` | `dbpo_utils.gaussian_sample` |
| log-std 头 | Eq. 44–45 `g_ψ(c_θ(o))` + clip | `dbpo_heads.LogStdHead` |
| latent 复用 | Eq. 47, 50 | `RolloutBuffer.latent_z` + `update` 传同一 `z` 进 `actor.apply` |
| prefix log-prob | Eq. 48/57 | `dbpo_utils.gaussian_logprob_prefix(..., H_e)` |
| ratio | Eq. 50 `exp(logp_new - logp_old)` | `update` 内 |
| PPO clip | Eq. 51 | `surrogate_loss(mode="ppo")` |
| value loss | Eq. 52 `0.5·(V_φ(o)-R̂)²` | `dbpo_utils.value_loss` |
| entropy | Eq. 53 对角高斯闭式 | `dbpo_utils.gaussian_entropy` |
| **anchor** | Eq. 54 `‖μ_θ-μ_{θ̄}‖²` | `update`：用 `anchor_params` 再前向一次 drift mean |
| 总目标 | Eq. 55 | `update` 汇总 |
| BPO ATV | BPO Eq. 8/16 | `surrogate_loss(mode="bpo")` |

**实现红线（最易错，务必守住）**：
1. **必须复用 rollout 时的 `z`** 算 ratio，否则 ratio 失真（DBPO 全部前提）。
2. log-prob **只在 `H_e` 个执行步**上求和，不是整个 `H`。
3. log-std 是 **state-conditioned 头**，记得 clip。
4. **anchor 用冻结的初始 drift 参数**，与 `z`/`o` 同输入再前向。
5. value **只输入观测特征**。

---

## 6. 配置与超参

### 6.1 drift pi0.5 配置（继承改进版 openpi）
`use_drifting_loss=True`，`drifting_gen_per_label=4`，`drifting_temperatures=(0.2,)`（论文单温度最佳），`drifting_per_timestep_loss=False`（chunk mode，论文默认更优）。

### 6.2 动作时序（对齐 EXPO-FT）
`action_horizon (H)`、`replan_steps (H_e)`：沿用 EXPO-FT 现有 pick 配置值，log-prob 窗口 = `H_e`。

### 6.3 BPO 超参（论文推荐）
`bpo_lambda=0.001`，`alpha1=0`（TV 权重），`w_value=0.5`，clip `eps` 比 PPO 调大 0.1，`ppo_epochs` 翻倍，`entropy_coef` 取 PPO 0.1×。

### 6.4 DBPO RL 超参（论文未给，**初始默认值，Phase 4 调**）
| 超参 | 初值 | 来源 |
|---|---|---|
| `clip_eps` | 0.2 | PPO 通用 |
| `gae_lambda` | 0.95 | PPO 通用 |
| `discount` | 0.99 | PPO 通用 |
| `c_value` | 0.5 | PPO 通用 |
| `c_entropy` | 0.0（BPO 敏感，先关）| BPO 建议 |
| `lambda_anchor` | 1.0（待调，消融重要）| DBPO repo 待核 |
| `logstd_init` | -2.0 | 经验 |
| `logstd_min/max` | -5.0 / 0.0 | 经验 |
| `ppo_epochs` | 10 | BPO 翻倍建议 |

> ⚠️ `lambda_anchor`、`logstd` 范围、rollout 大小须以 `github.com/YuxuanGao0822/DBPO` 实际值核对/覆盖（Phase 0/4 的 TODO）。

---

## 6.5 Phase 0 结论 — openpi API 兼容性核对（2026-06-17）

**核对范围**：EXPO-FT 全仓对 openpi 的真实依赖面（排除自带 openpi 目录）+ 两 fork 逐接口 diff + checkpoint/config 兼容性。

### 依赖面（EXPO-FT 服务端真正用到的 openpi 模块）
`pi05.py` / `vla_base.py` / `replay_buffer.py` / `train_utils.py` / `alg/*` 共依赖：
`models.model`、`policies.policy`、`training.{config, utils, optimizer, sharding, weight_loaders}`、`transforms`、`shared.{array_typing, nnx_utils}`。客户端仅用 `openpi_client.{msgpack_numpy, image_tools}`（与 drift 无关）。

### 两 fork 关系
- 自带 = `pd-perry/openpi` @ `expo_ft`（234 提交）；改进版 = `N0ne1eft/openpi` @ `feat/jax-return-hidden`（221 提交）。
- 共同 merge-base = `54cbaee`（上游 openpi）；自带在其后 28 提交、改进版 15 提交 → **分叉浅，合并易**。

### 关键模块 diff（自带 vs 改进版）
| 模块 | 差异 | 结论 |
|---|---|---|
| `training/utils.py`(TrainState)、`sharding`、`optimizer`、`weight_loaders`、`transforms`、`shared/nnx_utils`、`shared/array_typing` | **0 行** | ✅ 基础设施字节相同 |
| `models/model.py` | 13 行 | 自带禁用 pytorch 导入（"will lead to errors"），改进版启用。**不在冲突集**，merge 自动保留自带版（pytorch 禁用）→ 对 JAX learner 安全 |
| `policies/policy.py` | 90 行 | 自带加 `infer(is_batch, for_training)`；改进版加 `infer_with_hidden`。**需合并双方** |
| `training/config.py` | 544 行 | 扁平 `_CONFIGS=[TrainConfig(...)]` 列表，两边各 append 自己的具名配置。**需拼接双方** |

### 合并冲突集 = 仅 4 个文件
| 文件 | 难度 | 处理 |
|---|---|---|
| `models/pi0.py` | 平凡 | 自带的"改动"只是注释掉一段重复 `sample_actions` 死代码；取改进版（含 drift）不丢功能 |
| `scripts/serve_policy.py` | 平凡 | 仅 serving 脚本，learner 不用 |
| `training/config.py` | 机械 | 拼接两组 `_CONFIGS` 条目（保留 `expo_pi05_*` + drift/heatmap/robotwin 配置）|
| `policies/policy.py` | 机械 | 保留 EXPO-FT `infer(is_batch, for_training)` + `module_jit(static_argnames=...)`，并加 `infer_with_hidden` |

### checkpoint / config 兼容性
- **drift 不新增模型参数**：复用 `action_in_proj`/`action_out_proj`/同 backbone（`pi0.py:226,274`）→ 现有 flow-matching pi0.5 checkpoint **结构上可直接加载为 drift 模型**。DBPO 的 log-std/value 头是 learner 新增全新参数，不在 openpi checkpoint 内；anchor = 加载后 drift backbone 的快照。
- ⚠️ 语义上：drift 单步生成需 **Stage-1 (DBP) 训练过的 checkpoint** 才有好效果；flow-matching 权重虽可加载，但 `action_out_proj` 输出的是 velocity 而非 one-step action。
  - **现状（用户确认 2026-06-17）**：手上已有一个针对 **RoboTwin benchmark 某任务微调的 drift checkpoint**，可作为 RL 起点与联调验证用。
  - **流程约定**：**每个新任务，先用改进版 openpi 的 `pi05_*_drifting` 配置做一轮 DBP 离线训练**得到 Stage-1 checkpoint，再进入 DBPO 在线 RL。即 DBPO 的 actor 初始权重 = 该任务的 DBP checkpoint，anchor `θ̄` = 同一份快照。
- **向后兼容**：`use_drifting_loss`/`use_heatmap_loss` 默认 `False` → 合并后现有 EXPO-FT flow-matching 配置不受影响。
- **依赖**：两 fork pyproject 仅差一个 `chex==0.1.90` 固定（合并保留自带 pyproject 即可）；改进版未引入新第三方依赖。server venv 已装 torch 2.7.1+cu126。editable 安装 → 合并后代码即时生效，**无需为新依赖重装**。
- **实锤**：当前 server venv 装的就是自带 fork（`Pi0Config` 无 `use_drifting_loss` 属性）→ 必须合并才能拿到 drift。

### 合并执行步骤（Phase 0 收尾）
```bash
cd expo_ft/agents/vla/openpi          # 自带 fork, branch expo_ft, editable
git remote add improved /home/xukainan/openpi
git fetch improved
git checkout -b expo_ft_drift          # 安全分支，便于回滚
git merge improved/feat/jax-return-hidden
#   解决 4 个冲突：pi0.py/serve_policy.py 取改进版；config.py 拼接两组 _CONFIGS；
#   policy.py 保留 infer(is_batch,for_training) 并加 infer_with_hidden
# 验证：python -c "import openpi.models.pi0_config as c; print(c.Pi0Config().use_drifting_loss)"  → False
#       python -c "import openpi.models.model"  # 确认 JAX 导入链不被 pytorch 拖累
```

---

## 7. 进展日志 / Changelog

> 倒序追加，每条：日期 · 阶段 · 改动 · 涉及文件 · 验证结果 · 下一步。

### 2026-06-21 · Phase 4-EXPO · 修复 language instruction 不符（rollout 0% 元凶）· 参考 RLinf
- **现象**：填好 drift ckpt + norm_stats、用 `--actor_only_base_actions` 且 `replan_steps=50`（动作选择/执行粒度已对齐 eval）后，
  EXPO rollout 仍 **0% 成功**，而 RoboTwin 自带 `eval_policy_client.py` 同 ckpt 跑 50-60%。逐项排除（图像 CHW/相机/动作反归一化/exec backend 均一致）后，
  **用户定位到根因 = 语言 instruction 不符**。
- **根因（两层）**：
  - ① **EXPO env 没生成真实指令**：`RoboTwinEnv.reset` 里 `_resolve_instruction()` 调 `get_instruction()` 时 RoboTwin 还没注入指令（`_base_task.py:577` 返回 `self.instruction=None`）→ 回退到固定串 "stack the two blocks"。从未调用 RoboTwin 的指令生成。
  - ② **repack 丢 prompt**：robotwin 的 openpi repack 结构无 `prompt` 键（`config.py:654-664`）→ 推理/replay buffer 跑 repack 时把 obs 的 prompt 丢掉 → `InjectDefaultPrompt` 塞回 config 默认串 `_ROBOTWIN_DRIFTING_PROMPTS["stack_blocks_two"]`="stack the two blocks"。（DROID repack 含 `"prompt":"prompt"` 故无此问题；标准 openpi **推理不跑 repack**，故 eval 没事。）
  - **真实指令**是**每集随机的模板**（`description/task_instruction/stack_blocks_two.json` 的 `{A}=red block/{B}=green block/{a}{b}=arm` 占位），如 *"Place red block and green block centrally, then stack green block on red block."*——**不是** "stack the two blocks"。
- **权威参考**：① `eval_policy_client.py:403/435-438`（用户跑通的 50-60% 路径）：`play_once()` 拿 `episode_info` → `generate_episode_descriptions(task, [info], N)` → `np.random.choice(results[0][instruction_type])` → `set_instruction`。
  ② **RLinf** `RoboTwin-rlinf/robotwin/envs/vector_env.py:107/115/117`：`task.get_info()` 拿 info（**只取一次、缓存**）→ 每集 `create_instruction()` 随机选 → `obs["instruction"]`。RL loop 里**不跑专家**。
- **修复（RLinf 对齐，纯增量）**：
  - ① `client_robotwin/envs/robotwin_env.py`：`_ensure_episode_info()` 启动跑**一次** `play_once` 缓存 `episode_info`（本仓 RoboTwin 无 `get_info`，info 只在 play_once 填）；`_create_instruction()` 每集用缓存 info 经 `generate_episode_descriptions` + `np.random.choice` 随机选模板，`set_instruction` → `obs["prompt"]`。`instruction_type` 默认 "seen"，可配。
  - ② `expo_ft/utils/train_utils.py::build_pi05_config`：给缺 `prompt` 的 aloha/robotwin repack **补 `"prompt":"prompt"`**（DROID 已含→no-op），让每集真实指令流到策略；`pi05.py::process_raw_inputs` 与 `replay_buffer.py::insert` 加 `setdefault("prompt", ...)` 兜底（无 prompt 输入不致 repack KeyError，DBPO 安全）。
  - ③ `expo_ft/env/robotwin_utils.py`：离线 loader 从 `data/<task>/<config>/instructions/episode{N}.json` 读每集真实指令存进 `obs["prompt"]`（与在线 rollout 同分布），新增 `config.instruction_type`。
- **验证**：① 本地 `generate_episode_descriptions` 产出真实指令（seen/unseen 各 100 条，样例如上）；② `build_pi05_config` 后 robotwin repack 结构含 `prompt`；③ 6 文件语法 OK；④ DBPO 12 测试回归。**待云端验证**：填对 `instruction_type` 后 rollout 成功率应回升到 ~50-60%。
- **涉及文件**：`client_robotwin/envs/robotwin_env.py`、`expo_ft/utils/train_utils.py`、`expo_ft/agents/vla/pi05.py`、`expo_ft/data/replay_buffer.py`、`expo_ft/env/robotwin_utils.py`、`configs/task/robotwin_stack_blocks.py`、`docs/CLOUD_RUNBOOK.md`。
- **下一步（可选优化）**：若给本仓 RoboTwin 加 `get_info()`（仿 RLinf）即可免去启动那次 `play_once`；arm tag 现用首集缓存值（块颜色永远对），需更精确可改每集取。

### 2026-06-21 · Phase 4-EXPO · 修复 aloha/robotwin repack 的 `action` 键接缝 + 云端 runbook 扩 EXPO track
- **背景**：审计「obs dict 流经 replay buffer robotwin transform / `process_raw_inputs` 不报错」这道之前未勾选的接缝时，
  发现 **真实 bug**：robotwin/aloha 的 openpi repack 用 LeRobot 约定 `{"actions": "action"}`（**单数** `action`，
  `config.py:250/662`、`action_sequence_keys=("action",)`，DBP 训练依赖、**不能改**），而 EXPO 数据管线（为 DROID 写）
  两处都只提供**复数** `actions`：① `replay_buffer.py::insert`（`offline_ratio=0` 默认 → `BatchProcessor.__init__`
  首次 `insert_dataset` 即触发）；② `pi05.py::process_raw_inputs`（在线采样的 dummy）。`RepackTransform.__call__` 做
  `flat_item["action"]` → **KeyError**，learner 在训练开始前就崩。DROID 因用自定义 `{"actions": "actions"}` 不受影响。
- **修复（纯增量、DROID 行为不变）**：两处各**追加**一个 `action` 别名键——
  ① `insert`：`obs_data_dict["action"] = action_chunk_raw`（(H,14) 真实 chunk，作 actor 训练 target）；
  ② `process_raw_inputs`：`raw_observations["action"] = np.zeros((action_horizon, action_dim))`（**2D** dummy，
  因 AlohaInputs `_encode_actions_inv` 做 `actions[:, [6,13]]` 需 2D；该 dummy 经 `Observation.from_dict` 后即丢，仅形状要过 transform）。
  DROID repack 读 `actions`、忽略 `action` 键，故 DROID/DBPO 路径字节不变。
- **另一定论**：`aloha_policy.py:171` `convert_image` **无条件** `c h w -> h w c` → AlohaInputs 期望 **CHW 输入**；
  `RoboTwinEnv._chw` 与 `process_robotwin_dataset._decode_rgb_chw` 均产 CHW，三方一致（图像轴序无 bug）。
- **验证**：DBPO 12 测试全过（`dbpo_dryrun_test` 7 + `dbpo_pi05_test` 5，覆盖共享 pi05/transforms → 修复无回归）；
  EXPO model config + robotwin loader 在 `.venv` 加载/导入 OK；learner venv 已含 cv2 4.11/h5py 3.16（loader 依赖经 openpi 传递）。
  **仍待云端验证**：真实 DBP ckpt + norm_stats 下端到端 transform（runbook gate 6-EXPO 设了诊断点）。
- **runbook**：`docs/CLOUD_RUNBOOK.md` 重构为 **EXPO-FT(track A，优先) / DBPO(track B)** 双轨——步骤 0–5 共用，
  3/6/7 分叉；补 ≥2 GPU 要求、`--dataset_path`(真实 RoboTwin demo)、`--max_steps`(非 `--max_iters`)、norm_stats 必填、
  EXPO 监控键（`training/critic_loss`/`residual_q`/...）、排查表 EXPO 行。
- **涉及文件**：`expo_ft/data/replay_buffer.py`、`expo_ft/agents/vla/pi05.py`、`docs/CLOUD_RUNBOOK.md`、`docs/DBPO_DEV.md`。
- **下一步**：用户填 DBP ckpt + norm_stats → 云端 `git pull` → 步骤 4 greedy gate → 步骤 5 env-server → 步骤 6-EXPO learner。

### 2026-06-21 · Phase 4-EXPO 起步 · 新 model config + RoboTwin demo loader + dispatch ✅（冒烟过）
- **改动（纯增量，零侵入）**：
  - 新增 `configs/model/expo_ft_pi_drift_config.py`：`EXPOLearner` + `pi05_aloha_robotwin_drifting_stack_blocks_two`
    （drift）+ `residual_action_xyzg=False` + `freeze_pi05_encoder=True`。**不改** `expo_ft_pi_config.py`。
  - 新增 `expo_ft/env/robotwin_utils.py::process_robotwin_dataset`：读 RoboTwin demo hdf5
    （`joint_action/vector` + 各相机 JPEG `rgb`），产出与 `RoboTwinEnv.get_observation` 同键的 transition。**不改** `droid_utils.py`。
  - `train_pi_robo_async.py`：离线 loader 按 `config_task.dataset_loader` 分发（默认 'droid' → DROID 路径字节不变；
    'robotwin' → 新 loader）。`configs/task/robotwin_stack_blocks.py` 加 `dataset_loader="robotwin"`。
- **关键定论 ①（图像轴序）**：= **CHW**（`aloha_policy.py:171` AlohaInputs 内部 `c h w -> h w c`，故期望 CHW 输入）——
  RoboTwinEnv 与 loader 均产 CHW，原先标的 🟡 HWC/CHW 风险**消除**。
- **关键定论 ②（动作时序）**：DBP 训练管线 `policy/pi05/scripts/process_data.py:103-124` 用
  **`action[t] = state[t+1]`**（下一帧绝对 qpos 目标，**非 delta**），`observation.state[t]=state[t]`，共 T-1 行。
  loader 已对齐（原先误用同帧 `vector[t]` → 修正为 `vector[t+1]`）——与在线 rollout 一致（策略输出"下一目标 qpos"，
  `RoboTwinEnv.step` 经 `take_action(qpos)` 下发绝对目标）。state 布局 `[左臂6,左夹爪1,右臂6,右夹爪1]` 三方一致。
- **数据管线分工**：`script/collect_data.py` 脚本自动采专家 demo（**无需键鼠手采**）→ 原始
  `data/{task}/{setting}/data/episode*.hdf5`（`joint_action/vector` + JPEG rgb）= **loader 直接读的**；
  `policy/pi05/process_data_pi05.sh`(→`scripts/process_data.py`+`compute_norm_stats.py`) 转 LeRobot **仅供 DBP 训练**
  产 drift ckpt + **norm_stats**（后者正是 EXPO replay buffer 必需的 `pi05_assets_dir/asset_id`）。EXPO buffer 自己跑 openpi transforms，不吃 LeRobot。
- **env A+B 已落地**（本仓当前 `robotwin_env.py`）：episode 预算用 RoboTwin 自带 `step_lim`
  （`_eval_step_limit.yml`：stack_blocks_two=**800**，非硬编 200）+ episode 间 `close_env(clear_cache)` 防 sapien 泄漏 + UnStableError 换 seed 重试。
- **验证**：`.venv` 下 ① py_compile 全过；② model/task config 学习器侧加载正常（RoboTwin 依赖 guard 跳过）；
  ③ `process_robotwin_dataset` 在真实 `shake_bottle` episode 跑通（253 帧，键/CHW/14-D/终局 reward 均正确）。
- **涉及文件**：新增 `configs/model/expo_ft_pi_drift_config.py`、`expo_ft/env/robotwin_utils.py`；
  改 `train_pi_robo_async.py`（dispatch）、`configs/task/robotwin_stack_blocks.py`（dataset_loader）。
- **下一步**：用户填 DBP drift ckpt + RoboTwin norm_stats（`pi05_weight_loader_path`/`pi05_assets_dir`/`pi05_asset_id`）→
  启 robotwin server → 端到端集成（obs 流经 replay buffer robotwin transform；drift N=8 多候选采样路径）。

### 2026-06-21 · 优先级调整 · EXPO-FT × drift × RoboTwin 提到第一优先（先于 DBPO）
- **决策**：先让 EXPO-FT 原算法（`EXPOLearner`/`BCLearner`）在 drift pi0.5 + RoboTwin 上跑通，再回到 DBPO。
  新增 **Phase 4-EXPO**（见 §4），插在 Phase 4（DBPO）之前。
- **关键调研结论（本轮代码核对）**：
  - **drift 对 EXPO 是纯 config 切换**：合并后 `pi0.py` 的 `sample_actions:347` / `compute_loss:254` 已按
    `use_drifting_loss` 自动分发到 `_sample_actions_drifting` / `_compute_loss_drifting`；EXPO 的采样
    （`_jitted_infer→Policy.infer→model.sample_actions`）与 actor 更新（`train_step→compute_loss`）天然吃到，**零算法代码改动**。
  - **obs/action 天然兼容 RoboTwin**：replay buffer 的 3 路图像槽（`base_image`/`left_wrist_image`/`right_wrist_image`）
    对上 aloha 三相机；`RoboTwinEnv.get_observation` 返回的 `observation.images.cam_high/...` 正是 robotwin RepackTransform 读的键；
    buffer 与 `process_raw_inputs` 走同套 openpi robotwin transforms。
  - **EXPO 已含动作反归一化**：`sample_actions:532` 调 `process_transformed_outputs`（Unnormalize+AlohaOutputs）——
    这正是 DBPO rollout（`train_pi_robo_dbpo_async.py`）缺的步（DBPO 直接下发归一化 mean，Phase 4 待补）。
  - **不用写新训练入口**：EXPO/BC+drift+RoboTwin 直接复用 `train_pi_robo_async.py` 的 `EXPOLearner` 分支。
  - **待补接缝**：新 `expo_ft_pi_drift_config.py`、RoboTwin 数据 loader（`process_droid_dataset` 是 DROID 专用）、
    DBP ckpt 路径、图像 HWC/CHW 轴序核对（同 DBPO 隐患）。
- **涉及文件**：`CLAUDE.md`（标题/优先级块/第一原则/env 算法无关/新 config/新训练入口 多处）、`docs/DBPO_DEV.md`（本条 + Phase 4-EXPO）。
- **下一步**：起草 `configs/model/expo_ft_pi_drift_config.py` + RoboTwin 数据 loader/stub + 图像轴序冒烟核对。

### 2026-06-18 · Phase 4 前期 · RLinf 调研 + value 头改 suffix/stop_grad + RoboTwin client 计划
- **调研**：拆解 RLinf（`/home/xukainan/RLinf`）在 RoboTwin 上对 pi0.5 做 PPO 的实现，产出
  `docs/RLINF_ROBOTWIN_REFERENCE.md`（RoboTwin 适配 / chunk 级 macro-action MDP / critic head 设计 / 冻结策略）。
  另产出 `docs/BIMANUAL_RL_HORIZON_SURVEY.md`（双臂/长程 RL 的 H、H_e 惯例）。
  **注**：RLinf 里的 BPO/DBPO 代码是 polar823 自加的，非 RLinf 原生；可参考的是其 RoboTwin 适配 + PPO critic 设计。
- **改动（value 头，依 RLinf 实测设计）**：value 头输入从 `cond_emb` 改为 **mean-pooled `suffix_feat`**，并加
  **`stop_gradient`**（= RLinf `detach_critic_input`）。`networks/dbpo_heads.py`（`ValueHead.detach_input=True`）、
  `agents/alg/dbpo_pi05.py`（`value_feat=jnp.mean(suffix_feat,axis=-2)`，启用 suffix）。log-std 头仍用 `cond_emb`。
- **关键事实校正**：RoboTwin 2.0 默认 `pi0_step=50`（整段执行，非 open-loop 8）；DBPO repo `H_e=n_action_steps`（≤8）、
  `clip_ploss_coef=0.02`、`anchor=1.0`。**红线**：`H_e ≡ env replan 间隔`（不可解耦）。
- **计划新增**：Phase 4 拆为 4a（**独立 `client_robotwin/` 包**，零污染 DROID client）+ 4b（`train_pi_robo_dbpo.py`
  决策步 on-policy 循环）。CLAUDE.md「新 client」约定改为独立包。
- **验证**：DBPO 测试套件（dbpo_pi05/dryrun/phase1）重跑确认 value 头改动不回归（见本条提交后状态）。
- **下一步**：Phase 4a —— 起草 `client_robotwin/`（参考 RLinf `robotwin_env.py`）。
- **涉及文件**：`dbpo_heads.py`/`dbpo_pi05.py`（改）；`CLAUDE.md`/`docs/DBPO_DEV.md`/新增 2 份 docs（改/增）。

### 2026-06-17 · Phase 3 · 真实 pi0.5 drift 集成完成 ✅
- **改动**：把抽象 `drift_apply_fn` 接到真实 pi0.5 nnx drift，并使其在 3B 规模可训。
  - `expo_ft/agents/alg/dbpo_pi05.py`：`make_drift_apply_fn`（nnx.merge+preprocess+drift sampler）+ `build_dbpo_from_pi05`（含 `trainable_filter`）。
  - `dbpo.py`：drift_apply_fn 升为 `(params,obs,z,frozen)`，加 `actor_frozen` 字段 + `run_ppo_iteration` 驱动；mock 路径兼容。
  - `configs/model/dbpo_pi_config.py`（`model_cls="DBPOLearner"`）；`alg/__init__.py` 导出。
  - 测试：`dbpo_pi05_test.py`（dummy pi0.5）。
- **关键决策/发现**：① actor = `flax TrainState(params=nnx.State)` → Phase 2 update 零改动复用（实测 jax.grad 穿 nnx.merge 求出 actor 梯度）；② **trainable_filter 非可选**（全量 adam=24GB）——冻结 VLM/SigLIP 只训 action expert/投影；③ value 用 cond_emb。
- **验证**：**全套 31 测试通过**（Phase1 10 + core 11 + dryrun 5 + 真实集成 5）。集成测试确认：梯度入 pi0.5 actor、ratio≈1、anchor 冻结、trainable_filter 切分（1096 vs 429M）、BPO 跑通。
- **涉及文件**：新增 `dbpo_pi05.py` / `dbpo_pi05_test.py` / `configs/model/dbpo_pi_config.py`；改 `dbpo.py` / `alg/__init__.py`（expo-ft 主仓，未提交）。
- **下一步**：**Phase 4** —— `train_pi_robo*.py` 加 DBPOLearner 分支（真实 DBP checkpoint + env_client + on-policy 循环），真机小跑调参。

### 2026-06-17 · Phase 2 · DBPOLearner + RolloutBuffer + GAE 离线 dry-run 完成 ✅
- **改动**：把 Phase 1 的件组装成完整 on-policy learner，三层解耦（算法核 / buffer / learner），drift backbone 抽象为可替换 callable。
  - `expo_ft/agents/alg/dbpo_core.py`：GAE + PPO/BPO surrogate + value/anchor loss（纯函数）。
  - `expo_ft/data/rollout_buffer.py`：`RolloutBuffer`（**存 latent z**）。
  - `expo_ft/agents/alg/dbpo.py`：`DBPOLearner.{create,sample_actions,update}`，`drift_apply_fn(params,obs,z)→(mean,cond_emb,value_feat)` 抽象。
  - 测试：`dbpo_core_test.py`、`dbpo_dryrun_test.py`。
- **关键决策**：① 算法核与 pi0.5 解耦（mock 现测、Phase 3 换真模型只改 `create`）；② surrogate 可插拔（ppo/bpo）；③ DBPOLearner 暂为纯 struct.PyTreeNode（on-policy 签名异于 AgentLearner，适配器 Phase 3）；④ 发现 GPU TF32 致 ratio 漂移 ~0.4%，highest 精度可消除，被 clipping 吸收。
- **验证**：**26/26 DBPO 测试全过**——首次更新 ratio≈1（z 复用）、loss 有限、参数更新、critic 拟合 returns、BPO 跑通、GAE 对拍 numpy。
- **涉及文件**：5 个新文件（expo-ft 主仓，未提交）。
- **下一步**：**Phase 3** —— 把 `drift_apply_fn` 接到真实 pi0.5 JAX drift（`_sample_actions_drifting(return_hidden=True)`，复用 `pi05.py` 的 nnx/TrainState 加载）；`dbpo_pi_config.py`；on-policy 训练循环接 `train_pi_robo*.py`（采满 rollout→多 epoch 更新→发布）；PPO/BPO 在仿真/假环境端到端对拍。

### 2026-06-17 · Phase 1 · 随机适配头 + 高斯工具集完成 ✅
- **改动**：实现 DBPO 把确定性 drift 变随机 actor 所需的全部底层件（纯新增，零侵入现有算法）。
  - `expo_ft/agents/alg/dbpo_utils.py`：`gaussian_logprob`（executed-prefix Eq.48/57，支持 `prefix_start/prefix_len/dim_mask`）、`gaussian_entropy`、`gaussian_sample`、`step_mask`。
  - `expo_ft/networks/dbpo_heads.py`：`LogStdHead`（state-conditioned logσ，clip，初始恒等 `log_std_init`）、`ValueHead`（V(o)）。
  - `expo_ft/agents/alg/dbpo_phase1_test.py`：自包含测试（venv 无 pytest）。
- **关键决策**：heads 放 `networks/`（与现有 `nn.Module` 同处）而非计划中的 `vla/`；执行窗口对齐 EXPO-FT 的 `replan_steps`（start=0，无 T_o 偏移）；padded 动作维用 `dim_mask` 处理。
- **验证**：`py_compile` 通过；**Phase-1 单测 10/10 通过**（log-prob/entropy 对拍 tfp、prefix masking、sample 统计、头初始化/clip、jit）。
- **涉及文件**：3 个新文件（expo-ft 主仓，未提交）。
- **下一步**：**Phase 2** —— `DBPOLearner.{create,sample_actions,update}` + `RolloutBuffer`（存 latent z）+ GAE，先用 PPO surrogate 在假数据上 dry-run（loss 下降、无 NaN、ratio≈1 起步）。

### 2026-06-17 · Phase 0 收尾 · drift 合并执行完成 ✅
- **改动**：在自带 openpi checkout（`expo_ft/agents/vla/openpi`）建 `expo_ft_drift` 分支，`git merge improved/feat/jax-return-hidden`（improved = 本地 remote 指向 `/home/xukainan/openpi`）。合并提交 **`e500a21`**（双父 `46407a4` EXPO-FT + `e2258f7` drift）。
- **冲突解决**：git 自动合并了 `config.py`/`policy.py`/`serve_policy.py`；仅 `pi0.py` 手工解决：① 块1 取 drift 方法（HEAD 侧是死代码注释）；② 块2 保留 EXPO-FT 的 `preprocess(train=train)` + `noise_rng`，去重 `dt`。
  - **额外修复**：合并使 `sample_actions` 的 drift dispatch 收到 EXPO-FT 的 4D `noise (b,ns,ah,ad)`，而 `_sample_actions_drifting` 期望 3D → 在 dispatch 处 flatten `(b ns)` 并 repeat observation（对齐 flow-matching 的 num_samples 模式）。`model.py` 由 merge 自动保留自带版（pytorch 禁用）。
- **验证（server venv，全绿）**：
  - `py_compile` 4 个合并文件全过；`use_drifting_loss/heatmap` 默认 False（向后兼容）。
  - JAX 导入链 OK：`openpi.models.{model,pi0,pi0_config}` + `training.config`（91 配置，含 `expo_pi05_droid_lora_finetune_sft_cartesian_state` + 52 个 drifting 配置）。
  - `Policy.infer` 同时含 `is_batch/for_training`（EXPO-FT）与 `infer_with_hidden`（drift）；`Pi0._sample_actions_drifting` 在册。
  - **终极集成**：`expo_ft.agents.vla.{vla_base,pi05}` + `expo_ft.agents.alg` 均成功导入合并后的 openpi。
- **涉及文件**：`expo_ft/agents/vla/openpi/`（合并提交 e500a21，主要 `models/pi0.py` 手工 + config/policy 自动）。
- **下一步**：进入 **Phase 1** —— 实现 `dbpo_heads.py`（`LogStdHead`/`ValueHead`，Flax）+ `dbpo_utils.py`（executed-prefix 高斯 log-prob / entropy / sample），并写单测。回滚方式：`git checkout expo_ft`（合并隔离在 `expo_ft_drift` 分支）。

### 2026-06-17 · Phase 0 · openpi API 兼容性核对完成（纯只读分析，未改环境）
- **改动**：完成依赖面提取、两 fork git 关系与逐接口 diff、checkpoint/config 兼容性核对；落定 **D3 = git merge（非整体替换）**。结论写入 §6.5。
- **关键结论**：
  - EXPO-FT 依赖的 openpi **基础设施模块全部字节相同**（TrainState/sharding/optimizer/transforms/...）→ 合并低风险。
  - 两 fork 双向分叉，**冲突仅 4 文件**（pi0.py、serve_policy.py 平凡；config.py、policy.py 机械拼接）。整体替换会崩，因 `pi05.py:218` 依赖 pd-perry 的 `infer(is_batch,for_training)` 且 EXPO-FT 具名 config 只在自带 fork。
  - drift **不新增模型参数** → flow-matching checkpoint 结构兼容；DBPO 头是 learner 新增。`use_drifting_loss` 默认 False → 向后兼容。server venv 已装 torch 2.7.1，当前装的是无 drift 的自带 fork（实锤需合并）。
- **涉及文件**：更新 `docs/DBPO_DEV.md`（D3 决策 + §6.5 Phase 0 结论 + 本条）。
- **下一步**：执行 §6.5 的 git merge（建 `expo_ft_drift` 分支合并 drift），解决 4 冲突文件，验证 JAX 导入链 + `use_drifting_loss` 可见；随后进入 **Phase 1**（log-std 头 / value 头 / 高斯 log-prob 工具集）。

### 2026-06-17 · 调研 & 文档 · 初始化
- **改动**：完成三方调研（EXPO-FT 框架 / 改进版 openpi drift head / DBPO+BPO 论文），确立 D1–D7 架构决策，产出本开发文档与 Phase 0–5 计划。
- **关键结论**：
  - EXPO-FT 全 JAX；新增算法走 `model_cls` 分支，低侵入。
  - 改进版 openpi 的 **JAX 路径已预埋 drift `return_hidden` DBPO 契约** `(mean, suffix_feat, cond_emb)` → 决定 **JAX 路线（D1）**。
  - DBPO = 给 drift 加 state-conditioned 高斯（log-std 头）+ **latent 复用** + executed-prefix log-prob + anchor 的 PPO；BPO = 把 PPO clip 换成 `|ρ-(1+ε·tanh(A/2λ))|·|A|` 的 ATV loss（最小改动）。
  - 论文未给 DBPO RL 超参 → §6.4 暂定 + Phase 0/4 核对 repo。
- **涉及文件**：新增 `docs/DBPO_DEV.md`（本文件）。
- **下一步**：Phase 0 —— 让 EXPO-FT `.venv` 用上含 drift 的 openpi，核对 API 兼容性，跑通 drift pi0.5 推理 + `return_hidden` 三元组形状。

---

## 附录 A — 关键文件索引

**EXPO-FT**
- 学习器：`expo_ft/agents/alg/expo_ft.py`（off-policy 参考）、`bc.py`、`agent.py`（基类/接口）
- VLA 封装：`expo_ft/agents/vla/pi05.py`（JAX/nnx，drift learner 参照它接 openpi）、`vla_base.py`
- 数据：`expo_ft/data/replay_buffer.py`、`batch_processor.py`
- 环境：`expo_ft/env/env_client.py`
- 训练入口：`train_pi_robo.py:139-147`（`model_cls` 分支）、`train_pi_robo_async.py`（async 三锁）
- 配置：`configs/model/expo_ft_pi_config.py`、`configs/task/pick.py`

**改进版 openpi（`/home/xukainan/openpi`）**
- JAX drift：`src/openpi/models/pi0.py`（`_compute_loss_drifting:193`、`_sample_actions_drifting:278`，`return_hidden` 三元组）
- PyTorch drift：`src/openpi/models_pytorch/pi0_pytorch.py`
- RL 契约（PyTorch serving）：`src/openpi/policies/policy.py:108 infer_with_hidden`
- 配置：`src/openpi/models/pi0_config.py:37-65`（drift/heatmap 开关）

**论文**
- DBPO：`/home/xukainan/Zotero/storage/EICUQRAD/Gao 等 - 2026 ...pdf`（Eq. 43–57 为 RL 核心，Algorithm 2）
- BPO：`/home/xukainan/Zotero/storage/XBEQZDUU/Ao 等 - 2026 ...pdf`（Eq. 8/16 为 surrogate，Algorithm 1）

## 附录 B — 待核对 / 开放问题
1. ~~D3 openpi 替换方案 + API diff~~ → **Phase 0 已解决**：git merge 合并，冲突仅 4 文件（见 §6.5）。
2. `lambda_anchor`、`logstd` 范围、rollout 大小：核对 DBPO repo 实际值（Phase 0/4）。
3. value 头输入特征 (a)`cond_emb` vs (b)`suffix_feat`（Phase 2 A/B）。
4. 真机 on-policy 数据效率 / rollout horizon / epoch（D8，Phase 4）。
5. `H_e` 与 `env_client` 执行语义的严格对齐（Phase 3）。
6. checkpoint 兼容：drift pi0.5 权重加载、anchor 参数快照、(θ,ψ,φ) 一致性保存（贯穿）。
