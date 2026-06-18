# DBPO×BPO 实现正确性核对 + pi0.5 超参分析

> 依据**论文原文 + 各自项目代码**，核对当前 `expo_ft` 实现是否符合 DBPO / BPO，并针对 **pi0.5 VLA（双臂 H=50, d_a=14）** 的特性分析超参。
> 维护人：polar823。日期：2026-06-18。
>
> **一手来源**：
> - DBPO 论文：`Zotero/.../EICUQRAD/Gao 等 - 2026 - Drift-Based Policy Optimization.pdf`（RL 核心 Eq 43-57，Algorithm 2，附录 A/D）
> - DBPO 代码：`github.com/YuxuanGao0822/DBPO`（`configs/finetune.yaml`、task configs）
> - BPO 论文：`Zotero/.../XBEQZDUU/Ao 等 - 2026 - Bounded Ratio Reinforcement Learning.pdf`（Eq 7/8/12/16，Thm 4.1，Alg 1）
> - BPO 代码：`github.com/bounded-ratio-rl/bounded_ratio_rl`（`cfgs/algo/bpo/*.yaml`、`src/sb3/BPO.py`）

---

## 0. 结论速览

当前实现的**数学骨架与两篇论文一致**（GAE / latent-z 复用 / executed-prefix logp / PPO clip / anchor / BPO ATV 公式 / 高斯适配头），**已通过 31 个单测**。但存在 **3 处与论文的偏差**和 **1 个针对 pi0.5 的根本性风险**：

1. ⚠️ **value 输入**：DBPO 论文 Eq 52 用 `V_φ(o_t)`（纯观测特征 = `c_θ(o)` = cond_emb）；当前改用 **suffix**（RLinf 做法）—— 偏离论文，但 RLinf 在 VLA 上验证过。
2. ⚠️ **clip_eps 取值无依据**：DBPO(PPO) 用 **ε≈0.02**，BPO 用 **ε=0.3**，当前 `clip_eps=0.2` 两者都不符。
3. ⚠️ **anchor 归约**：论文 Eq 54 是整 chunk 的 `||·||²`（对 H×d_a 求和）；当前对 H 取 mean → 与 λ_anchor=1.0 的标定差一个 ~H 因子。
4. 🔴 **根本风险（pi0.5 专有）**：DBPO/BPO 都为**低维短 chunk**调参（d_a≤7, H=8-16, ratio 维 ≤~112）。pi0.5 整段执行 ratio 维 = **H_e·d_a = 50·14 = 700**。**DBPO 论文 Limitations 明确警告 `H>32` 会"sacrificing training stability"**——pi0.5 H=50 正在此区间。clip_eps 等超参不能从论文直接搬。

---

## 1. 正确性核对（论文公式 ↔ 当前代码）

| 论文 | 公式/出处 | 当前实现 | 结论 |
|---|---|---|---|
| DBPO 高斯均值+对角 | Eq 46 `N(μ_θ(o,z), diag(σ̃²))` | `dbpo_utils.gaussian_sample` | ✅ |
| DBPO log-std 头 | Eq 44-45 `logσ=g_ψ(c_θ(o))`, clip | `dbpo_heads.LogStdHead`（cond_emb 输入, clip）| ✅ 输入正确（cond_emb=c_θ(o)）|
| DBPO latent 复用 | Eq 47/49/50（z 固定→joint ratio=conditional ratio）| `RolloutBuffer` 存 z，`update` 复用 | ✅ 正确性命门，已对 |
| DBPO executed-prefix logp | Eq 48/57 `Σ_{h∈[T_o,T_o+H_e)} Σ_m logN` | `dbpo_utils.gaussian_logprob(prefix_len, dim_mask)` | ✅ 对 H_e×d_a 求和 |
| DBPO PPO clip | Eq 51 `min(rÂ, clip(r,1±ε)Â)` | `dbpo_core.ppo_policy_loss` | ✅ |
| DBPO value | Eq 52 `0.5(V_φ(o_t)-R̂)²` | `dbpo_core.value_loss` | ✅ 形式对；⚠️ **输入 suffix≠论文 o**（见 §2.1）|
| DBPO entropy | Eq 53 | `dbpo_utils.gaussian_entropy` | ✅ |
| DBPO anchor | Eq 54 `E‖μ_θ-μ_θ̄‖²₂` (stop-grad θ̄) | `dbpo_core.anchor_loss` | ✅ 形式对；⚠️ **归约差 H 因子**（见 §2.3）|
| DBPO 总目标 | Eq 55 `-J_clip+c_v L_v-c_e H+λ_a L_a` | `dbpo.update` | ✅ |
| **DBPO 用 PPO（非 BPO）** | Eq 15/51 | 你额外加了 BPO（你的扩展）| ✅ DBPO 原版=PPO；BPO 是增量 |
| BPO ATV target | Eq 8 `ρ*=1+ε·tanh(Ã/2λ)` | `dbpo_core.bpo_policy_loss` `1+clip_eps·tanh(A/2λ)` | ✅ 公式恒等（与 repo `1-ε+2ε·sigmoid` 一致）|
| BPO 权重 | Eq 16 `(|R_φ-V_φ|+α₁)`=\|A\| | `\|advantages\|·\|ratio-target\|` | ✅ **=\|A\| 权重，与论文一致**（repo `abs((ratio-target)*(q-v))` 同义）|
| BPO median 网络 μ_ψ | Eq 15, `use_median=True`(默认) | 用 mean 近似（无 median 头）| ⚠️ 简化（论文 Remark4.3+消融证 mean≈median，可接受）|

---

## 2. 与论文的偏差（需决策）

### 2.1 value 输入：suffix（RLinf）vs cond_emb（DBPO 论文）
- **DBPO 论文**：Eq 43 backbone 出 `μ_θ(o,z)` 与 `c_θ(o)`；Eq 52 `V_φ(o_t)` 只吃**观测** → 即 `c_θ(o)`=cond_emb。
- **当前（依 RLinf）**：value 读 mean-pooled `suffix_feat`（action expert 输出，含 action 上下文）+ stop_gradient。
- **判断**：偏离 DBPO 论文，但 RLinf 在 RoboTwin/pi0 RL 上验证过；DBPO 论文是低维 U-Net。两者皆可，**保留 suffix 需知这是 RLinf 经验选择而非 DBPO 论文做法**。若想最贴论文可切回 cond_emb（A/B）。stop_gradient 与论文 detach 精神一致 ✅。

### 2.2 clip_eps 取值（当前最缺依据的一项）
| 来源 | ε 值 | 出处 |
|---|---|---|
| DBPO (PPO clip) | **0.02** | `DBPO repo finetune.yaml: clip_ploss_coef=0.02` |
| BPO (ATV ε) | **0.3** | `BPO repo bpo/mujoco.yaml,atari.yaml: clip_ratio=0.3`；论文 Cor 4.5「choosing a small ε」motivate 偏小 |
| PPO 通用 | 0.2 | — |
| **当前** | 0.2（同时用于 ppo 和 bpo）| `dbpo_pi_config.clip_eps` |
- **问题**：DBPO(PPO) 该用 ~0.02，BPO 该用 ~0.3，单一 0.2 两者都不符。**应按 surrogate 分别设 ε**（见 §4）。

### 2.3 anchor 归约差 H 因子
- 论文 Eq 54：`‖μ_θ-μ_θ̄‖²₂` 对**整 chunk 向量**（H×d_a）求和。
- 当前 `anchor_loss`：`mean(sum((μ-μ̄)², axis=-1))` → 对 d_a 求和、对 (B,H) 取 mean → 比论文小 ~H(=50) 倍。
- **影响**：λ_anchor=1.0（DBPO repo 标定于论文归约）在当前归约下相对 J_clip 偏弱 ~50×。**要么对 H 改 sum，要么把 λ_anchor 调大**（anchor 在 DBPO 消融里至关重要：去掉 0.90→0.75）。

---

## 3. 🔴 pi0.5 的根本风险：ratio 维度

**两篇工作的调参区间（一手）**：
- DBPO（附录 A）：`H` typical **8–16**；`d_a` 低（planar 例 d_a=2，robomimic 7）；`1≤H_e≤H-T_o+1`（**论文本身用 receding-horizon 部分执行**，Eq 57 只对 H_e 前缀算 ratio）。→ ratio 维 = H_e·d_a ≤ **~112**。
- BPO：mujoco 单步，动作维 ≤17。
- **DBPO 论文 Limitations 原文**：*"For tasks with very high action dimensionality (e.g., **d_a>20**) or **long prediction horizons (e.g., H>32)**, memory constraints may necessitate switching to step-wise mode or reducing batch size, **potentially sacrificing temporal coherence or training stability.**"*

**pi0.5 双臂**：d_a=14，H=50。若**整段执行**（当前决策 H_e=H=50）→ **ratio 维 = 700**，是论文区间的 6–40×，且 **H=50>32 正中论文警告**。

**为什么致命**：`ratio = exp(Σ_{700} Δlogp)`。同样的每坐标策略漂移 δ → 总 Δlogp≈700δ → ratio≈exp(700δ)。把标量 ratio 限制在 1±ε，只需每坐标漂移 ε/700 就触界 → **要么几乎全部 clip（有效梯度≈0），要么 ratio 在更新间爆炸**（这正是你在 RLinf 自加 `bpo_ratio_clamp=4.0` 要解决的现象）。论文的 ε（PPO 0.02 / BPO 0.3）是为 ≤112 维标定的，**不能线性外推到 700 维**。

---

## 4. 针对 pi0.5 的超参建议

> 核心矛盾：你已决定"整段执行 H_e=50"（对齐 RLinf），但这把 ratio 维顶到 700、正中 DBPO 论文警告。下面给**两条路线**。

### 路线 A（贴 DBPO 论文，推荐优先验证）：缩小 executed-prefix 维度
DBPO 论文 Eq 48/57 本就只对 **H_e 前缀**算 ratio 且支持 H_e<H。取 **H_e=8–16**（论文 H 区间），ratio 维降到 112–224，超参可近用论文值：
- PPO ε=0.02–0.1；BPO ε=0.2–0.3；λ_anchor 调到与 J_clip 同量级；其余照论文。
- 代价：与"整段执行"决策冲突；但**这是论文验证过的区间**，最可能直接收敛。

### 路线 B（坚持整段 H_e=50）：必须做高维稳定化
若坚持整段执行，单纯改 ε 不够，需组合：
| 超参 | 当前 | pi0.5 建议（整段）| 依据 |
|---|---|---|---|
| logp 聚合 | sum over 700 | 考虑 **action_level**（对 14 维求和/步，ratio 仍随 H_e 但更可控）或 per-dim 均值变体；RLinf 提供 token/action/chunk 三档 | 700 维 sum 是 ratio 失稳主因 |
| ratio clamp | 无 | **加 `clamp(max≈3–4)`**（=RLinf `bpo_ratio_clamp`）| RLinf 实证 |
| PPO ε | 0.2 | **0.02**（DBPO repo），甚至更小 | 700 维下 trust region 要更紧 |
| BPO ε | 0.2 | **0.1**（论文偏小 + 高维）+ clamp | BPO Cor4.5 + 高维 |
| logσ_max | 0.0(σ=1) | **≤ -1**（σ≤0.37）| σ 大→logp 方差大→ratio 爆；归一化关节动作不需 σ=1 |
| logσ_init | -2.0(σ≈0.14) | -2 ~ -3 | 保守起步 |
| λ_anchor | 1.0 | **≥1.0 且对 H 改 sum 或同步放大**（见 §2.3）| anchor 消融关键，高维更需强约束防 action expert 漂离 DBP 流形 |
| KL 早停 | 无 | **加 approx_kl 早停**（target_kl，如 0.01–0.05）| BPO/PPO/RLinf 都用；高维 ratio 更需 |
| ppo_epochs | 10 | 起步 ≤5 + KL 早停 | 高维多 epoch 易过冲 |
| bpo_lambda | 1e-3 | 1e-3（保持；注意归一化 adv 下 tanh 近 sign）| BPO repo |
| c_entropy | 0 | 0（700 维 entropy 量级大，勿开）| 两者默认 0 |

### 通用（两条路线都建议）
- **clip_eps 按 surrogate 分离**：config 加 `clip_eps_ppo` / `clip_eps_bpo`，别共用一个。
- **首轮监控**：ratio 分布、`approx_kl`、clipfrac、logσ 演化、anchor/J_clip 比值。ratio 中位数应≈1、尾部不爆。
- **matmul 精度**：`jax_default_matmul_precision="highest"`（Phase 2 已知 TF32 致 ratio 漂 0.4%；700 维下更敏感）。

---

## 5. 附：reward 聚合 = 什么（回答疑问）
chunk 级 macro-action MDP 里，"reward 聚合"= 把一个**决策步**内执行的 H_e 个 env 子步的 per-step reward **合成一个标量** macro-reward，使 rollout 成为决策步序列 `(o_t, a_t, R_t, o_{t+1})`，供**决策步粒度**的 GAE 与 value loss 使用。RLinf 的两处：
1. `robotwin_env.py:217 _cal_chunk_rewards`：把稀疏终局 reward 摊到 chunk 子步 → `chunk_rewards[B, chunk_step]`。
2. `algorithms/utils.py:79`：`rewards.sum(dim=-1)` → 每决策步一个标量 R_t；`dones.max(-1)`。
然后 `advantages.py:25` 在决策步序列上跑 GAE（每决策步一个 `V(o_t)`），value loss = `0.5(V(o_t)-R̂_t)²`（Eq 52）。
**稀疏 0/1 情形**：R_t ≈ 段内是否成功（除成功步外子步 reward=0，sum≈终局 success）。决策步 γ 用 `discount^H_e`（`dbpo_core` 已注明）。

## 6. 附：RoboTwin 数据需转格式（回答疑问）
**需要**。原始 hdf5（`/home/xukainan/RoboTwin/data/shake_bottle/.../episode0.hdf5`）：`joint_action/vector(T,14)` + `observation/*_camera/rgb` 为 **JPEG 编码字节（|S…）**，非解码数组 → 不能直接喂 pi0.5。RoboTwin `policy/pi05/scripts/process_data.py` 的转换（= DBP 训练用的格式）：
- **state = action = 14 维关节角** `[左臂6, 左夹爪1, 右臂6, 右夹爪1]`（`action=state`，关节位置目标）。
- **3 相机**：`head_camera`→主图、`left_camera`/`right_camera`→腕图（JPEG 解码 + resize）；**front_camera 不用**。
- instruction 从 `instructions/episode{i}.json`。
→ 离线训练已做（你有 DBP ckpt）。**RL env 的运行时 obs 必须产出同样格式**（sim 直接给解码数组，14 维关节 state，同样的相机选择/顺序）——这就是 ROBOTWIN_ADAPTATION_PLAN.md 的 D4，现已锁定。

---

## 7. 待办（基于本核对）
- [ ] clip_eps 按 surrogate 分离（`clip_eps_ppo=0.02` / `clip_eps_bpo=0.1–0.3`）。
- [ ] 决策路线 A（H_e=8-16，贴论文）vs B（整段50+稳定化）—— 与 §4 一并定。
- [ ] anchor 归约对齐论文（对 H sum）或上调 λ_anchor。
- [ ] logσ_max 收紧到 ≤-1；加 ratio clamp + approx_kl 早停（路线 B 必须）。
- [ ] value 输入 suffix vs cond_emb：保留 suffix（RLinf）已知偏离论文，记录在案。
