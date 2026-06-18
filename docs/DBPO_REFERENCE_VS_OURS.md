# DBPO 原实现全貌 + 与当前复现的逐项 diff

> 基于 **本地 clone 的 `github.com/YuxuanGao0822/DBPO`（`/home/xukainan/DBPO`）逐文件精读**，给出 DBPO 算法的全部实现细节，并与当前 `expo_ft` 复现逐项对照。
> 权威来源文件：`dbpo/methods/dbpo/ppo_adapter.py`（优化器/loss/采样/ratio）、`dbpo/methods/dbpo/rollout_buffer.py`（GAE）、`dbpo/methods/dbp/policies.py`（drift 采样）、`dbpo/workspace/dbpo_finetune_workspace.py`（训练循环）、`configs/finetune.yaml`（超参）。
> 日期：2026-06-18。

---

## 1. DBPO 原实现全貌（authoritative）

### 1.1 drift model 如何产 action chunk（回答你的 Q1）
**一次前向预测整段 chunk，不是逐 action×horizon 次。** `policies.py:147` `naction_pred = self.model(noise[B,H,d_a], global_cond)`：`DBPUNet1D` 是 horizon 维的 1D 时序卷积网，输入整段噪声 `z[B,H,d_a]` + 观测条件，**单次前向输出整段 `action_pred[B,H,d_a]`（1-NFE）**。执行片段 = `action_pred[:, n_obs_steps-1 : n_obs_steps-1+n_action_steps]`（`_slice_action`）。
→ 与你 pi0.5 的 `_sample_actions_drifting`（整段、1 前向）**一致** ✅。

### 1.2 actor 结构（双副本）
`ppo_adapter.py`：
- `actor_old`（`:132`）：加载 Stage-1 DBP 权重（EMA），**冻结**（`requires_grad=False`, eval），**同时充当 anchor θ̄**。
- `actor_ft`（`:142`）：`actor_old` 的 deepcopy，可训；包成 `DBPOActor` = drift policy + `StateConditionedLogStdHead`。
- `StateConditionedLogStdHead`（`:19`）：`nn.Linear(cond_dim, H·d_a)`，**zero weight + bias=init_log_std** → 初始 logσ 恒为 init_log_std，状态无关，训练学状态依赖。**输入 = cond_emb**（`predict_action(output_embedding=True)` 返回的 global_cond）。
- **critic**：`ValueCritic(obs_dim, n_obs_steps)`（finetune.yaml）——**独立网络，输入原始低维 obs/state**，不共享 actor backbone。

### 1.3 采样（get_actions, `:268`）
- `z = randn(B, H, d_a)`（整段噪声）。
- `mean, std = actor_ft(cond, z)`；`std = clamp(logσ, [log0.03, log0.10]).exp()`。
- **采样用单独的 `sampling_std = clamp(std, min=0.03)`**（探索下限），`sampling_dist=Normal(mean, sampling_std)`。
- 训练：`sample()` 后**截断到 `mean ± 3·sampling_std`（randn_clip_value=3）**，再 clamp 到 `[act_min,act_max]`。eval：直接取 `mean`（确定性部署）。
- `chains = stack([z, actions])`（**z 与采样动作都存**）。logp 用 sampling_dist 算。

### 1.4 ratio 怎么算（get_logprobs, `:224` + loss, `:349`）
- 复用存的 `z = chains[:,0]`、`actions = chains[:,1]`；用**当前参数**+同 z 重算 `dist`。
- `logp = _reduce_executed_prefix(dist.log_prob(actions))`：对 `[:, :H_e, :]` **求和**（step×dim）；**可选 `normalize_act_space_dimension` → ÷ (H_e·d_a)**（高维归一化开关）。
- **logp 截断**：`newlogprobs.clamp(min=logprob_min=-2.0)`（old 同样）→ 限制 logp 量级。
- `logratio = newlogp - oldlogp`；**`ratio = exp(logratio)`**。z 复用使 joint ratio = conditional ratio（Eq 50）。

### 1.5 PPO 更新（loss, `:318`）
- norm_adv：`(adv-mean)/(std+1e-8)`。
- **PPO clip**：`pg1=-adv·ratio`，`pg2=-adv·clamp(ratio,1-ε,1+ε)`，`pg_loss=max(pg1,pg2).mean()`，**ε=clip_ploss_coef=0.02**。
- **value**：`v_loss=0.5·mean((critic(obs)-returns)²)`，可选 value clip。
- **entropy**：`entropy_loss = -entropy.mean()`。
- **anchor**：同 z 下 `old=actor_old.predict_action(z).action_pred`（冻结）、`new=actor_ft.policy.predict_action(z).action_pred`，**`anchor=F.mse_loss(new,old)`（对整段 H×d_a 取 mean）**。
- 组合（workspace `:319`）：`loss = pg_loss + ent_coef·entropy_loss + vf_coef·v_loss + anchor_coeff·anchor_loss`。

### 1.6 GAE（rollout_buffer.py `:90`）
标准 GAE：`delta = r·scale + γ·V'·(1-terminated) - V`；`adv = delta + γ·λ·(1-terminated)·lastgae`；`returns = adv + V`。**γ=0.99 直接用在决策步上（不是 γ^H_e）**；λ=0.95；`non_terminal` 只看 terminated（truncation 不阻断 bootstrap）。每决策步存**一个**标量 reward（reward 聚合在 env 内完成）。

### 1.7 训练循环（workspace + finetune.yaml）
| 项 | 值 |
|---|---|
| n_steps（每 iter 决策步数）| 500 |
| n_train_itr | 1500 |
| batch_size（minibatch，over n_steps×n_envs 展平）| 50000 |
| **update_epochs** | **5** |
| **target_kl 早停** | **0.02**（`approx_kl>target_kl` 即 break） |
| **n_critic_warmup_itr** | **5**（前 5 iter 只更新 critic，actor 冻结） |
| ent_coef / vf_coef / anchor_coeff | **0.01** / 0.5 / 1.0 |
| γ / λ | 0.99 / 0.95 |
| actor_lr / critic_lr | 1e-5 / **1e-4**，分开 AdamW + Cosine 调度 |
| grad clip | `clip_grad_norm_(max_grad_norm)` |
| **logσ clamp** | **[log0.03, log0.10] = [-3.51, -2.30]** |
| init logσ / min_sampling_std / randn_clip / logprob_min | log0.03 / 0.03 / 3.0 / -2.0 |
| clip_ploss_coef（PPO ε）| 0.02 |
| H（horizon）/ H_e（executed）| robomimic 4/4；kitchen/pusht 16/8 |
| surrogate | **纯 PPO**（DBPO 无 BPO；BPO 是你的扩展）|

---

## 2. 逐项 diff（DBPO 原实现 ↔ 你的复现）

| # | 维度 | DBPO 原实现 | 你的复现 | 一致? | 影响 |
|---|---|---|---|---|---|
| 1 | drift 整段预测 | 1 前向出整段 | pi0.5 同 | ✅ | — |
| 2 | z 复用算 ratio | ✅ | ✅ | ✅ | — |
| 3 | executed-prefix logp 求和 | sum over [:H_e]×d_a | sum over prefix_len×dims | ✅ | — |
| 4 | **logp 维度归一化** | **`normalize_act_space_dimension` 默认=True（÷N）** | 无（只 sum）→ **已修：加 `normalize_dims` 并 config 置 True** | ✅(已修) | 🔴🔴 **这是 700 维不稳的真正根因**：DBPO 默认归一化使 `log r=(1/N)Σδ_j`、Var~1/N；你之前 raw sum 才会爆。修正后高维告警基本化解 |
| 5 | **PPO ε** | **0.02** | 0.2 | ❌ | 🔴 大 10×，700 维下灾难 |
| 6 | γ 用法 | **0.99 直接/决策步** | 注释建议 0.99^H_e | ⚠️ | 中：应直接 0.99 |
| 7 | GAE 公式 | 标准 | 逐行一致 | ✅ | — |
| 8 | **value 输入** | **独立 critic(原始 obs/state)** | suffix_feat（共享 backbone）| ❌ | 中：VLA 必须用编码特征；最贴论文应用 **cond_emb**（V(o)），suffix 是 RLinf 选择 |
| 9 | value clip | 可选 | 可选 | ✅ | — |
| 10 | logstd 头输入 | cond_emb | cond_emb | ✅ | — |
| 11 | **logσ clamp 范围** | **[0.03, 0.10]**（σ 很窄）| **[exp(-5),exp0]=[0.0067, 1.0]** | ❌ | 🔴🔴 **单项最致命**：σ_max=1.0 vs 0.10，logp 方差/ratio 爆炸根源 |
| 12 | init σ | 0.03 | 0.135（logσ=-2）| ❌ | 中：起步 σ 大 4.5× |
| 13 | **采样 std vs logprob std** | **分离**（采样 clamp≥0.03，logprob∈[0.03,0.10]）| 同一个 | ❌ | 中 |
| 14 | **样本截断** | **±3σ（randn_clip=3）** | 无 | ❌ | 🔴 截断使每坐标 (a-μ)/σ≤3 → δ_j 有界 → ratio 可控 |
| 15 | **logp 截断** | **clamp(min=-2.0)** | 无 | ❌ | 🔴 限制 logp 量级 → ratio 有界 |
| 16 | anchor 归约/范围 | **mse（mean over H×d_a），整段 H** | mean over (B,H_e) of sum over d_a，**仅 prefix** | ❌ | 中：你的 anchor ≈ d_a(=14)× 强且只约束前缀 → λ_anchor 失标 |
| 17 | anchor 同 z、对 mean、stop-grad θ̄ | ✅ | ✅ | ✅ | — |
| 18 | 双副本 actor_old=anchor / actor_ft | ✅ | anchor_params 快照 + actor | ✅ | — |
| 19 | norm_adv | True | True | ✅ | — |
| 20 | **update_epochs** | **5** | 10 | ❌ | 中：你 2×（BPO 建议），DBPO 用 5 |
| 21 | **target_kl 早停** | **0.02** | 无 | ❌ | 🔴 高维 ratio 必备的安全阀 |
| 22 | **critic warmup** | **前 5 iter 只训 critic** | 无 | ❌ | 中：value 没拟合前不动 actor，稳训关键 |
| 23 | ent_coef | **0.01** | 0.0 | ❌ | 中：你关了（BPO 建议），DBPO 开 0.01 |
| 24 | vf_coef | 0.5 | 0.5 | ✅ | — |
| 25 | actor/critic lr | 1e-5 / **1e-4** + Cosine | 1e-5 / **3e-4**，无调度 | ⚠️ | 低-中：critic lr 3× + 无调度 |
| 26 | grad clip | clip_grad_norm | **无** | ❌ | 中：高维梯度需裁 |
| 27 | surrogate | 纯 PPO | PPO + BPO | ⚠️ | BPO 是你的扩展（合理）|
| 28 | reward 聚合 | env 内 → 每决策步 1 标量 | 计划同 | ✅ | — |

---

## 3. 影响分级

**🔴 必改（直接决定 700 维下能否稳训）**：
- **#11 logσ clamp 收紧到 ~[0.03, 0.10]**（你现在 σ_max=1.0）。这是 DBPO 控制 ratio 的第一道闸。
- **#5 PPO ε 0.2→0.02**；BPO ε 另设。
- **#14 样本截断 ±3σ + #15 logp clamp(min≈-2)**：DBPO 靠这两条把**每坐标 δ_j 限有界**，N 个求和才不炸（正对应数学推导里 r=exp(Σδ_j) 的控制）。
- **#21 target_kl≈0.02 早停 + #26 grad clip**：高维 ratio 的安全阀。
- **#4 logp 维度归一化开关**：路线 B（整段 50）几乎必须开（÷N 让 ratio 维度不变）。

**🟡 应对齐 DBPO**：#6 γ 直接用 0.99（非 ^H_e）；#20 epochs 10→5；#22 critic warmup=5；#23 ent_coef 0→0.01；#16 anchor 归约对齐（整段 + mse 或重标 λ_anchor）；#25 critic_lr 3e-4→1e-4 + 加调度。

**🟢 设计选择（记录即可）**：#8 value 输入（DBPO=独立 critic(obs)；VLA 下用 cond_emb 最贴论文，suffix 是 RLinf 经验，已选 suffix）；#27 BPO 扩展。

> 注：你当前的 #5/#20/#23 取值多来自 dev doc 里 **BPO 的调参建议**（clip+0.1、epoch×2、entropy×0.1），与 **DBPO base**（ε0.02、epoch5、ent0.01）不一致。做 DBPO+BPO 需明确以哪个为 base，别混用。

## 4. 与上一轮数学推导的呼应
数学推导（`DBPO_ALGORITHM_VERIFICATION.md` §推导）说 `ratio=exp(Σ_{N=H_e·d_a}δ_j)` 对 N 指数敏感。**DBPO 原实现正是用一组"每坐标 δ_j 限界"机制来压住它**：logσ∈[0.03,0.10]（#11）、样本±3σ 截断（#14）、logp clamp（#15）、KL 早停（#21）、归一化开关（#4）。你的复现这些**几乎全缺**，却同时把 N 从 ≤112 推到 700 → 比 DBPO 原本就脆弱得多。**先补齐这些 DBPO 自带的稳定器**，是路线 B（整段 50）可行的前提；路线 A（小 H_e）则让 N 回到 DBPO 验证区间。

## 5. 待办
- [ ] config 对齐 DBPO base：`clip_eps_ppo=0.02`、`logstd_min/max≈log0.03/log0.10`、`logstd_init≈log0.03`、`ent_coef=0.01`、`ppo_epochs=5`、`critic_lr=1e-4`、`target_kl=0.02`、`gamma`直接0.99。
- [ ] 实现缺失稳定器：样本 ±3σ 截断、logp clamp、grad clip、critic warmup、logp 维度归一化开关。
- [ ] anchor 归约对齐（整段 + mse 或重标 λ_anchor）。
- [ ] H_e 路线 A(≤8) vs B(50+全部稳定器) 决策（见 ROBOTWIN_ADAPTATION_PLAN §D1）。
