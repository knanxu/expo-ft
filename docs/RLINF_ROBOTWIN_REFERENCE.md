# RLinf 在 RoboTwin 上做 VLA-RL 的实现剖析（VLA-RL 参考）

> 目标：拆解 RLinf（`/home/xukainan/RLinf`）如何在 RoboTwin 双臂仿真上对 pi0.5 做 on-policy RL（PPO），
> 提炼可被本项目 drift pi0.5 RL（JAX）直接借鉴的部分。所有结论均引用 RLinf 实际源码（文件:行）。
> 剖析日期：2026-06-18。样例 config：`examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi_pi05.yaml`。

---

## 0. 一句话总览

RLinf = **PyTorch/FSDP + Ray** 的分布式 actor–rollout–env 三组件框架，对 **flow-matching pi0.5**（`num_steps=5`）
做 **chunk 级 macro-action 的 PPO**：双臂 chunk **整段 50 步执行**，reward/logprob/GAE 全在**决策步（chunk）粒度**，
稀疏 0/1 终局奖励。

> ⚠️ 归因说明：RLinf **原生只有 PPO**；本地 RLinf 里看到的 BPO 等变体是外部自加的、并非 RLinf 原生，本项目不再使用。
> 真正可参考的是 RLinf 原生的 **① RoboTwin 适配**（§2）、**② PPO 的 critic head 设计 + actor/critic 冻结策略**（§4）、
> **③ chunk 级 macro-action MDP**（§3）。

与本项目 drift pi0.5 RL 的**关键差异**（RLinf 原生 PPO 侧）：① RLinf 用 flow-matching（多步去噪）+ 固定探索噪声 `noise_level=0.3`，
我们用 **drift 单步 + 学习的 state-conditioned log-std 头**；② RLinf **整段执行 H_e=50**，我们倾向 **H_e 更短的前缀**。

---

## 1. 配置层（chunk / 执行 / 目标函数 / 奖励）

来源：`robotwin_adjust_bottle_ppo_openpi_pi05.yaml`

| 项 | 值 | 含义 |
|---|---|---|
| `actor.model.num_action_chunks` | **50** | chunk 长度 = env 执行步数（注释 "interface for the env"）|
| `actor.model.action_dim` | **14** | 双臂 aloha-agilex |
| `actor.model.num_steps` | 5 | **flow-matching 去噪步**（非 drift 单步）|
| `actor.model.add_value_head` / `openpi.detach_critic_input` | True / True | value 头挂在 VLA 上，**输入 detach**（与我们 value-on-obs 设计一致）|
| `openpi.noise_level` | 0.3 | **固定探索噪声**（RLinf 没有 state-conditioned log-std 头）|
| `algorithm.reward_type` | **chunk_level** | reward 在 chunk 内 50 步求和成每决策步一个标量 |
| `algorithm.logprob_type` | **chunk_level** | ratio = 对 50×14=700 个 log-prob 求和后的**单标量**/决策步 |
| `algorithm.entropy_type` | token_level | entropy 逐 token |
| `algorithm.adv_type` / `gamma` / `gae_lambda` | gae / 0.99 / 0.95 | **决策步粒度 GAE** |
| `clip_ratio_high/low` / `value_clip` | 0.2 / 0.2 | PPO clip ratio |
| `rollout_epoch` / `update_epoch` | 4 / 5 | 每轮采 4、每批更新 5 个 epoch |
| `env.train.total_num_envs` | **256** | 256 个并行 env（吞吐靠大规模并行）|
| `env.train.max_episode_steps` / `max_steps_per_rollout_epoch` | 200 / 200 | 每 episode 200 env-step |
| **决策步数/episode** | **200 // 50 = 4** | `env_worker.py:104` `n_train_chunk_steps`，每 episode 仅 **4 个 macro-action** |
| `group_size` | 1 | PPO；>1 即 GRPO |

> 实锤你看到的：**chunk 50 整段执行**，不做"预测 50 执行 25"。RoboTwin 部署惯例（`pi0_step=50`）与 RLinf 一致。

---

## 2. RoboTwin 仿真如何适配成 RL 环境

来源：`rlinf/envs/robotwin/robotwin_env.py`、`rlinf/envs/venv/venv.py`

- **向量化**：`RoboTwinEnv._init_env`(:78) 实例化 `robotwin.envs.vector_env.VectorEnv(n_envs=num_envs)`，
  把改造过的 RoboTwin 包成 `num_envs` 并行。底层物理执行（TOPPRA/插值/逐 action）在 **RoboTwin 包内部**，
  RLinf 只通过 `venv.step(chunk)` 调用。
- **整段 chunk 执行**：`step()`(:259) 接收 `[n_envs, horizon, action_dim]`，整条交给 `venv.step`；
  `_elapsed_steps += actions.shape[1]`(:299) → **以 chunk 长度推进时间**。
- **chunk_step → macro-action 奖励展开**：`chunk_step()`(:320) 调 `venv.step(chunk)` 后用
  `_cal_chunk_rewards()`(:217) 把**稀疏终局 reward 摊到 chunk 各子步**（终止步及之后置 reward），
  返回 `chunk_rewards[n_envs, 50]`、`chunk_terminations`（仅最后一步置位）。
- **稀疏 0/1 奖励**：`_calc_step_reward()`(:206) = `reward_coef * terminations`（终局 success），
  `use_rel_reward` 时取差分。**无 reward shaping**——与我们约定一致。
- **执行方式扩展**：`chunk_step_with_speed()`(:392) 走 `venv.step_with_speed(chunk, speed_actions)`，
  用**整段 TOPPRA + 每 env 速度元动作 (v, vel_scale, acc_scale)** 执行（speedtune）。
  **对应你的"方式1 整段 chunk TOPPRA"**；默认 `chunk_step` 对应 RoboTwin 包内的逐 action 执行。
- **reset/seeds**：`_init_reset_state_ids()`(:505) 从 `seeds/train_seeds.json` 读 **success_seeds**，
  只 reset 到"可解"的初始态；`auto_reset`、`ignore_terminations`（eval 下把 termination 当 truncation、
  记 `success_at_end`(:310)）、`success_once` 统计。

---

## 3. Rollout 与优势计算（决策步 MDP）

- **三组件 + Ray channel**：rollout worker 生成 chunk → 经 channel 发给 env worker（`recv_chunk_actions`:572）→
  `env_interact_step`(:381) 调 `chunk_step` → `EnvOutput(rewards=chunk_rewards, dones=chunk_dones, ...)` 回流。
  异步/流水线（`pipeline_stage_num`），actor 用 FSDP。
- **chunk_level reward**（`algorithms/utils.py:79`）：`rewards[n_chunk, bsz, 50].sum(dim=-1)` →
  每决策步一个标量 reward；`dones.max(dim=-1)`。**GAE 因此跑在 `num_chunk=4` 个决策步上、每决策步一个 V**。
- **chunk_level logprob**（`algorithms/utils.py:335`）：
  `logprobs.reshape(bsz,-1,action_dim).sum(dim=[1,2])` → **对 50×14 全求和 = 每决策步单标量**；
  `ratio = exp(logp_new - logp_old)`。另提供 `token_level`(每token)、`action_level`(对14维求和/步) 两档更细粒度。
- **GAE**：`algorithms/advantages.py:25` 标准 `gae = delta + gamma*lam*(~dones)*gae`（reverse scan）。
- **value**：每决策步一个 `V(o_t)`，detached 输入；critic 用 PPO Huber + value-clip。

> 这套「reward 子步求和 → 每决策步标量 reward + 一个 V → 决策步 GAE → chunk_level 单标量 ratio」
> **与 on-policy RL 的标准 rollout buffer / GAE 设计同构**，可作为正确性对照。

---

## 4. ⭐ critic head 设计 + actor/critic 冻结策略（RLinf 原生，最值得参考）

来源：`rlinf/models/embodiment/openpi/openpi_action_model.py`（config 实际走的路径）、
`rlinf/models/embodiment/modules/value_head.py`、`rlinf/models/embodiment/value_model/value_expert.py`。

### 4.1 critic head 与 action head 同级（共享 suffix，轻量方案 = 你 config 用的）
`OpenPi0ForRLActionPrediction` 里，**动作头与价值头从同一个 `suffix_out`（action expert 输出 hidden）分叉**：
```python
# get_velocity:854      动作头
v_t = self.action_out_proj(suffix_out)                       # [B, H, action_dim]
# _compute_value_from_suffix:875   价值头（与动作头同级）
suffix_out_value = suffix_out[:, :action_chunk].mean(dim=1)  # 池化成 [B, width]
if self.config.detach_critic_input:                          # ⭐ 关键
    suffix_out_value = suffix_out_value.detach()
value = self.value_head(suffix_out_value)[:, 0]              # ValueHead: MLP(1024,512,256)
```
`value_head` 只是个 MLP（`value_head.py`），**和 action_out_proj 平级**，都挂在 backbone 输出上。一次前向同时出 action mean 和 value，省算力。

### 4.2 训 critic 时是否冻结上层 VLM —— **靠 detach，不靠 requires_grad=False**
- config `detach_critic_input: True` → 价值头输入被 `.detach()` → **value loss 的梯度不回传到共享 backbone（VLM + action expert），只更新 value_head 自己的 MLP**。
- 等价效果 = "对 critic 而言上层全冻结"，但实现是 **截断输入梯度**而非设 `requires_grad=False`。
- **为什么必须隔离**：① value loss 早期噪声大、量级大（config `value_lr=1e-4` vs `actor_lr=5e-6`，差 20×），不 detach 会经共享 backbone 把 actor 的表征带偏；② 解耦 actor/critic 学习率与收敛节奏；③ 论文里 value 只依赖观测，不该反向塑造策略表征。

### 4.3 另一种更重的设计：value 作为独立并行 expert（可显式 freeze_vlm）
`value_expert.py` 的 `ValueExpert`/`ValueCritic` 把 value 做成**与 action expert 并列的 transformer expert**，共享 VLM 的 KV：
- `freeze_vlm=True`：Gemma3(VLM) `requires_grad=False` + `.eval()`，**KV cache detach** → value expert 在冻结 VLM 特征上训（`_set_requires_grad:240`）。
- `freeze_vlm=False`：VLM 可训，梯度经 KV 流回 Gemma3。
- `freeze_vision_encoder` 同理管 SigLIP；`trainable_experts` 控制哪些 expert 可训。
- 比 4.1 重，但 value 表征更独立。你 config 用的是 4.1 轻量方案，不是这个。

### 4.4 落到本项目 drift pi0.5 RL（JAX）的结论
**问：训 critic / actor 时要不要冻结上层 VLM？**
1. **critic 侧**：不需要单独把 VLM 设成不可训——只要在 value/critic head **输入处 `jax.lax.stop_gradient`**（= RLinf 的 `detach_critic_input=True`，如 speedtune 的 `rainbow_dqn` 的 `detach_input`）。这样 value loss 只更新 head 自己，不动 backbone。head 本就是独立模块时，补一个输入处 stop_gradient 即可。
   - 若 head 读 **`cond_emb`（VLM prefix 池化，纯 obs）**且 VLM 已被 `trainable_filter` 冻结 → 梯度本就到不了可训参数，detach 与否对 VLM 无影响；但若 head 改读 **`suffix_feat`（action expert 输出，可训，actor 也在训它）→ 必须 stop_gradient**，否则 critic loss 干扰 actor（这正是 RLinf detach 的理由）。
   - 若希望 value 只输入 obs → 选 `cond_emb` 更贴该取向；RLinf 选 `suffix_out`（含 action 上下文）+ detach。可 A/B。
2. **actor 侧**：是否冻结 VLM 是**独立问题**，由 `trainable_filter`/LoRA 决定（已冻结 VLM/SigLIP 只训 action expert + 投影）。与 critic 无关。
3. **一次前向出 mean+value**（4.1 共享 suffix）比独立 value expert 省算力，推荐先用共享方案 + stop_gradient。

---

## 5. 可直接借鉴到本项目 drift pi0.5 RL（JAX）的清单

| 借鉴点 | RLinf 出处 | 用法 |
|---|---|---|
| **critic head 与 action head 同级 + detach** | `openpi_action_model.py:875` + `value_head.py` | value head 挂 backbone 输出、与 action 头平级；输入 `stop_gradient`（=`detach_critic_input`）隔离 critic 梯度（见 §4）|
| **chunk_level macro-action MDP** | `utils.py:79/335` + `advantages.py:25` | on-policy RL 标准 rollout buffer 对照：reward 子步求和→每决策步标量+一个 V→决策步 GAE→单标量 ratio |
| **chunk_level ratio 数值风险** | `utils.py:335`（700 维求和单标量）| H×14 维 ratio 易失稳 → 用短 H_e 前缀 / ratio clamp 应对 |
| **logprob 粒度旋钮** | `utils.py:310/324/335` | 若 chunk_level ratio 太脆可借鉴 `action_level`（对 14 维求和/步）做更稳的变体实验 |
| **RoboTwin env 适配模板** | `robotwin_env.py` 全文 | 写 RoboTwin env 适配时可照搬：VectorEnv 并行、chunk 整段/逐 action、稀疏终局 reward、success_seeds reset、auto_reset、eval success_at_end |
| **TOPPRA/速度执行接口** | `robotwin_env.py:392 chunk_step_with_speed` + `venv.step_with_speed` | 对应"整段 TOPPRA + 速度元动作"执行方式；可参考其 `(v,vel_scale,acc_scale)` 元动作接口（与 speedtune 相关）|
| **value 头 detach 输入** | config `detach_critic_input:True` + `add_value_head` | 与 value-on-obs、stop-grad 设计一致 |
| **超参对照** | config `algorithm.*` | gamma0.99/λ0.95/clip0.2/actor_lr5e-6/value_lr1e-4/update_epoch5/rollout_epoch4 |

**不照搬/需注意**：
- RLinf 是 **flow-matching(5步) + 固定 noise 0.3**；我们是 **drift 单步 + 学习 log-std 头** → logprob/采样机制不同，GAE/buffer 机制可复用，stochastic adapter 走我们自己的 state-conditioned log-std。
- RLinf 靠 **256 并行 env** 喂数据；我们 RoboTwin 若并行度低，样本效率会是瓶颈（D8 开放问题）。
- RLinf **整段执行 H_e=50 + clamp**；我们仍倾向 **H_e 短前缀**（红线：H_e == env 实际执行步），二者可 A/B：要么"短 H_e"要么"长 H_e + ratio clamp"。

---

## 附：关键文件索引（RLinf）
- 样例 config：`examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi_pi05.yaml`
- RoboTwin env 适配：`rlinf/envs/robotwin/robotwin_env.py`；向量环境：`rlinf/envs/venv/venv.py`
- env worker（rollout 交互）：`rlinf/workers/env/env_worker.py`（`env_interact_step:381`、`n_train_chunk_steps:104`）
- chunk_level reward/logprob：`rlinf/algorithms/utils.py`（reward `:79`、logprob `:335`）
- GAE：`rlinf/algorithms/advantages.py:25`
- **PPO loss（actor/critic，RLinf 原生）**：`rlinf/algorithms/losses.py`（`compute_ppo_actor_loss:167`、`compute_ppo_critic_loss:312`）
- actor 优势/更新：`rlinf/workers/actor/async_ppo_fsdp_worker.py`（`compute_advantages_and_returns:64`）
