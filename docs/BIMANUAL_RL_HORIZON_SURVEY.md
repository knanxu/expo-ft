# 双臂 / 长程 机器人 RL 的 action horizon 与执行步数调研

> 调研动机：DBPO 与 EXPO-FT 都用**单臂、短程**机器人，action chunk 很短（DBPO 仿真 H≤16/H_e≤8，
> robomimic 档 H=H_e=4；EXPO-FT `replan_steps=8`）。但本项目在 **RoboTwin 用双臂机器人**，
> pi0.5 默认 `action_horizon=50`。本表回答：**双臂 / 长程 RL（含真机）项目实际用多大的预测 horizon H
> 和执行步数 H_e**，以对标我们 DBPO×pi0.5 双臂的取值。
>
> 所有数值均经 **GitHub 实际代码/配置文件**核对（标注来源文件:字段）；未能核对者标 `unconfirmed`。
> 诚实区分**真机 / 仿真**。调研日期：2026-06-17。
>
> 术语：**H** = 模型一次预测的 action chunk 长度（prediction horizon）；
> **H_e** = 实际在环境里执行、并参与 RL credit 的步数（executed / replan 间隔）。

---

## 表 A — VLA + RL（与本项目最相关）

| 项目 | 仓库URL | backbone | 机器人(臂数/真机or仿真/benchmark) | H | H_e | 数值来源(文件:字段) | 备注 |
|---|---|---|---|---|---|---|---|
| **SimpleVLA-RL** | https://github.com/PRIME-RL/SimpleVLA-RL | OpenVLA-OFT / pi0 | **双臂** RoboTwin1.0/2.0(仿真) + LIBERO(单臂仿真) + **真机长程灵巧(2026-01)** | OFT: LIBERO=8, **ALOHA=25**；pi0(RoboTwin)=50 | **= H（整段开环）** | OFT chunk: `moojink/openvla-oft prismatic/vla/constants.py` `LIBERO NUM_ACTIONS_CHUNK=8 / ALOHA=25, ACTION_DIM=14`；执行整段: `GenerateConfig num_open_loop_steps = NUM_ACTIONS_CHUNK`；pi0(RoboTwin)=50 见表C | README 明确「real-world RL on long-horizon dexterous tasks, ~300% over SFT, auto-recovery」(blog pending，臂数/horizon 细节未公开) |
| **RLinf-VLA** | https://github.com/RLinf/RLinf | OpenVLA/-OFT / pi0 | 双臂(RoboTwin) + 单臂(LIBERO/ManiSkill)；仿真 | `unconfirmed`（沿用各 backbone chunk） | `unconfirmed` | arXiv 2510.06710；未逐字段核对代码 | 统一 VLA-RL 框架，支持多 backbone |
| **RIPT-VLA** | https://github.com/Ariostgx/ript-vla | OpenVLA-OFT / QueST | 单臂为主(LIBERO/仿真) | `unconfirmed`（OFT 默认 8） | `unconfirmed` | 未逐字段核对 | 交互式 RL 微调 |

> OpenVLA-OFT 的执行语义是 **chunk 内不重规划、整段开环执行**（`num_open_loop_steps = NUM_ACTIONS_CHUNK`），
> 因此挂它的 RL（SimpleVLA-RL）天然 **H_e = H**。其**双臂 ALOHA 档 chunk = 25**（单臂 LIBERO 才 8）。

---

## 表 B — 真机 RL（real-robot RL）/ 带 action chunking 的 RL

| 项目 | 仓库URL | 方法类型 | 机器人(臂数/真机or仿真) | H | H_e | 数值来源(文件:字段) | 备注 |
|---|---|---|---|---|---|---|---|
| **HIL-SERL** | https://github.com/rail-berkeley/hil-serl | HIL off-policy AC (RLPD/SAC + 人工干预) | **单臂+双臂**；**真机** Franka | **1** | **1** | `serl_launcher/.../wrappers/chunking.py` `ChunkingWrapper(obs_horizon=1, act_exec_horizon=None)`；任务 config `act_exec_horizon=None`→单步 | **唯一确认有完整公开代码的双臂真机 RL**：object_handover(`DualFrankaEnv`, `dual-arm-learned-gripper`)。无 chunk、短程(MAX_EP=200) |
| **SERL** | https://github.com/rail-berkeley/serl | off-policy AC | 单臂；真机 Franka | 1 | 1 | 同上 `ChunkingWrapper` | HIL-SERL 基础框架 |
| **ConRFT** | https://github.com/cccedric/conrft | 一致性策略 RL 微调 (Cal-QL, 基于 Octo) | 单臂；真机 Franka | **1** | **1** | `train_conrft_octo.py: act_exec_horizon=None`；actor 输出 `(B, action_dim)`；obs 窗口=2 | 一致性策略本可出 chunk，实现配置为单步 |
| **DPPO** | https://github.com/irom-princeton/dppo | Diffusion-policy PPO（**DBPO 同源**） | 单臂(+transport双臂)；**全仿真** | **4 / 8** | **= H（整段开环）** | `cfg/.../ft_ppo_diffusion_mlp.yaml` `horizon_steps`(Gym=4, transport=8, Furniture=8) / `act_steps` / `cond_steps:1` | 无真机配置；transport=双臂、Furniture=长程 |
| **Q-chunking / ACRLPD** | https://github.com/ColinQiyangLi/qc | off-policy AC + action chunking | 单臂；**全仿真** OGBench/Robomimic | **5** | **5（整段开环抽干队列）** | `main.py: horizon_length=5`；`agents/ac{fql,rlpd}.py`；`evaluation.py` 队列排空 | 长程 sparse-reward 设计 |
| **PA-RL** | https://github.com/MaxSobolMark/PolicyAgnosticRL | Policy-agnostic RL（采样→critic排序→蒸馏） | 单臂；**真机 WidowX(Bridge)** + 仿真 | 无时间 chunk | **1** | `configs/real_config.py:146`；`chunking.py act_exec_horizon` | `num_actions_to_keep=4` 是候选数**非执行步** |
| **ResiP** | https://github.com/ankile/robust-rearrangement | Residual PPO 叠冻结 chunked diffusion BC | 单臂；**真机 Franka**+仿真 FurnitureBench | BC `pred_horizon=32`,`action_horizon=8` | **residual=1（逐步闭环）** | `src/config/base.yaml: pred_horizon:32 / action_horizon:8`；`residual.py` 单步残差 | **唯一显式分离 predict-H / execute-H_e** 的配置；但分离在冻结 BC 上、RL 残差仍逐步 |
| **ResFiT** | https://github.com/amazon-far/residual-offpolicy-rl | Residual off-policy RL (TD3/RLPD) 叠冻结 ACT/Diffusion BC | **双臂**(TwoArm)；**公开代码仅仿真**（论文为 29-DoF 双臂人形真机） | ACT base `chunk_size=50` | BC=50；**residual=1（逐步）** | `resfit/lerobot/policies/act/configuration_act.py: chunk_size=50, n_action_steps=50`；`residual_td3.py` | 双臂真机仅在论文，公开代码 sim-only |
| **BiKC** | https://github.com/ManUtdMoon/BiKC | **模仿学习(非RL)** keypose+一致性策略 | **双臂**真机 ALOHA(14-DoF)+仿真 | **16** | **8** | `diffusion_policy/config/...consistency_unet_workspace.yaml: horizon=16, n_action_steps=8` | 不是 RL，仅作双臂 chunk 取值参考 |

---

## 表 C — 双臂 policy backbone 的部署 horizon 惯例（多为 BC，用于对标"双臂该用多大 H"）

| 项目 | 仓库URL | 模型类型 | 机器人(双臂/真机or仿真) | H | H_e(执行/replan) | 数值来源(文件:字段) | 备注 |
|---|---|---|---|---|---|---|---|
| **RDT-1B** | https://github.com/thu-ml/RoboticsDiffusionTransformer | Diffusion Transformer(1B) | 双臂 Agilex/ALOHA；真机 | **64** | **64（整段后 replan）** | `configs/base.yaml: action_chunk_size: 64`；`scripts/agilex_inference.py: if t % chunk_size == 0` | ~25–30Hz |
| **openpi pi0 (ALOHA)** | https://github.com/Physical-Intelligence/openpi | Flow-matching VLA | 双臂 ALOHA；真机 | **50** | **25（执行前缀再 replan）** | H: `src/openpi/models/pi0_config.py: action_horizon=50`(默认)；H_e: `examples/aloha_real/main.py: action_horizon=25` + `ActionChunkBroker` | **真机双臂的 receding-horizon：预测50/执行25** |
| **openpi pi0.5 (ALOHA)** | https://github.com/Physical-Intelligence/openpi | Flow VLA (pi05) | 双臂 ALOHA；真机 | **50** | 25(broker) | `config.py: pi05_aloha → Pi0Config(pi05=True)` 继承默认 `action_horizon=50` | 单臂 DROID 才覆写为 15 |
| **ACT / ALOHA** | https://github.com/tonyzhaozh/act | Transformer-CVAE | 双臂 ALOHA；真机+仿真 | **100** | 无 temporal_agg: 100；有: 每 **1** 步(集成) | `README: --chunk_size 100`；`imitate_episodes.py: query_frequency=num_queries; if temporal_agg: =1` | 50Hz；temporal ensemble 融合重叠预测 |
| **Mobile ALOHA** | https://github.com/MarkFzp/act-plus-plus | Transformer-CVAE | 双臂+底盘；真机 | **100** | temporal_agg 每 1 步 + `BASE_DELAY` 错位 | `imitate_episodes.py: query_frequency / BASE_DELAY` | 补偿臂/底盘延迟 |
| **RoboTwin – pi0** | https://github.com/RoboTwin-Platform/RoboTwin | Flow VLA (openpi) | 双臂 ALOHA；仿真 | **50** | **50（整段）** | H: `policy/pi0/src/openpi/models/pi0.py: action_horizon=50`(默认)；H_e: `policy/pi0/deploy_policy.yml: pi0_step=50` + `deploy_policy.py: get_action()[:pi0_step]` | **「RoboTwin 双臂 H=50」的出处**；执行整段(异于 openpi 真机的前缀25) |
| **RoboTwin – ACT** | https://github.com/RoboTwin-Platform/RoboTwin | Transformer-CVAE | 双臂(action_dim=14)；仿真 | **50** | **50** | `policy/ACT/deploy_policy.yml: chunk_size:50, action_dim:14, temporal_agg:false` | 三相机=双臂 |
| **RoboTwin – RDT** | https://github.com/RoboTwin-Platform/RoboTwin | Diffusion Transformer | 双臂；仿真 | **64** | **64** | `policy/RDT/configs/base.yaml: action_chunk_size:64`；`deploy_policy.yml: rdt_step:64` | 整段后 replan |
| **DexMimicGen** | https://github.com/NVlabs/dexmimicgen | BC-RNN(robomimic) | 双臂灵巧手；仿真 | 无 chunk(RNN ctx=10) | 单步 | `config_utils.py: seq_length=10`(RNN 上下文非 chunk) | 数据生成；非 chunk 策略 |

---

## 综合结论

### 1. 双臂 backbone 的预测 horizon H 普遍是 50–100
- Flow VLA **pi0 / pi0.5**：双臂 ALOHA / RoboTwin 一律 **H=50**（`Pi0Config.action_horizon` 默认；单臂 DROID 才降到 10–15）。**这就是你 RoboTwin 双臂 H=50 的来源，且与同类双臂部署一致——50 不是异常值，是双臂标准档。**
- Diffusion **RDT-1B**：H=64。ACT/ALOHA：H=100（RoboTwin-ACT 仿真用 50）。
- 对照：单臂 DBPO/EXPO/DROID 的 4–16 明显更短——horizon 随机器人自由度与任务时长增长，符合直觉。

### 2. 执行步数 H_e 有三种主流做法
- **整段开环 H_e=H**：RoboTwin 全家(pi0=50 / ACT=50 / RDT=64)、RDT-1B 真机(64)、OpenVLA-OFT / **SimpleVLA-RL**（chunk 内不重规划）、DPPO/QC 仿真。吞吐高。
- **执行前缀 H_e≈H/2（receding-horizon）**：**openpi 官方真机双臂 ALOHA 预测 50 / 执行 25**——闭环更鲁棒。OpenVLA-OFT 的**双臂 ALOHA 常量也恰是 25**。**「双臂执行 ≈25」在两处真机/部署独立出现，是双臂 receding-horizon 的事实基准。**
- **每步重推理 + temporal ensemble H_e=1**：ACT / Mobile ALOHA（`temporal_agg`）。

### 3. 诚实的空白：没有「双臂 + 长程 + 多步 chunk(H_e>1)」三者齐全的公开真机 RL
- **多步 chunk 的 RL 目前都在仿真**（DPPO/QC/ACRLPD/SimpleVLA-RL-RoboTwin），执行多为整段短 chunk（4–8）或整段（OFT 8–25 / pi0 50）。
- **双臂真机 RL 的唯一完整公开代码 = HIL-SERL** 的 object_handover，但它 **H=H_e=1、无 chunk、短程**。
- **残差类**（ResiP 真机 / ResFiT 论文双臂）把多步 chunk 留在**冻结 BC 底座**，RL 残差本身逐步（H_e=1）。
- **SimpleVLA-RL 的真机长程 RL（2026-01，~300% 提升）** 是目前最接近"双臂+长程+chunk 真机 RL"的工作，但细节（臂数、H、是否真双臂）尚未公开。
- ⇒ **你的 DBPO×drift-pi0.5 做双臂 / 长程 / 多步 chunk 执行的真机 RL，正落在该表尚无人占据的空白点。** 先在 RoboTwin 仿真跑通是合理路径。

### 4. 对本项目 H / H_e 取值的对标建议

> ⚠️ **决策更新（2026-06-18，以此为准）**：下表是调研期的对标建议；用户已最终决定 **DBPO rollout 执行整段 chunk
> `H_e = H = 50`（对齐 RLinf，不部分执行）**，EXPO-FT 原算法保留部分执行。本表的"起步 H_e=8 / 前缀"建议**已被覆盖**，
> 仅留作对标参考。残留项：整段 700 维 ratio 易爆 → 监控后按需加 ratio clamp。详见 `docs/ROBOTWIN_ADAPTATION_PLAN.md` §1/§D1。

| | 取值 | 依据 |
|---|---|---|
| **预测 H** | **50** | pi0/pi0.5 双臂默认；RoboTwin/openpi/RDT 同档，无需改 |
| **执行/credit H_e（起步）** | **8–25** | DBPO 同源 DPPO 用整段 4–8；DBPO repo 最大 H_e=8；而双臂 receding-horizon 事实基准是 **25**（openpi 真机 ALOHA + OpenVLA-OFT ALOHA）。折中：DBPO 的 joint-Gaussian ratio 对维度敏感 → **起步 H_e=8（贴 DBPO/DPPO），稳定后向双臂基准 25 靠拢**；上限不超过整段 50 |
| **执行整段 vs 前缀** | **前缀(receding-horizon)** | 既是 openpi 真机双臂做法(50→25)，也利于 DBPO ratio 稳定；整段 50（RoboTwin/SimpleVLA-RL 风格）对 DBPO 单标量 ratio 风险最大 |

> 红线复述（见 CLAUDE.md）：**H_e（RL credit 的执行步数）必须 == env 实际执行/replan 步数**（RoboTwin `pi0_step`）。
> 选 H_e=8 就把 `pi0_step` 也设 8，并在该 replan 间隔下重新 greedy 评 DBP baseline。

---

## 来源仓库索引
- VLA-RL: [SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL) · [OpenVLA-OFT](https://github.com/moojink/openvla-oft) · [RLinf](https://github.com/RLinf/RLinf) · [RIPT-VLA](https://github.com/Ariostgx/ript-vla)
- 真机/chunked RL: [HIL-SERL](https://github.com/rail-berkeley/hil-serl) · [SERL](https://github.com/rail-berkeley/serl) · [ConRFT](https://github.com/cccedric/conrft) · [DPPO](https://github.com/irom-princeton/dppo) · [Q-chunking](https://github.com/ColinQiyangLi/qc) · [PA-RL](https://github.com/MaxSobolMark/PolicyAgnosticRL) · [ResiP](https://github.com/ankile/robust-rearrangement) · [ResFiT](https://github.com/amazon-far/residual-offpolicy-rl) · [BiKC](https://github.com/ManUtdMoon/BiKC)
- 双臂 backbone: [RDT-1B](https://github.com/thu-ml/RoboticsDiffusionTransformer) · [openpi](https://github.com/Physical-Intelligence/openpi) · [ACT](https://github.com/tonyzhaozh/act) · [Mobile ALOHA](https://github.com/MarkFzp/act-plus-plus) · [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) · [DexMimicGen](https://github.com/NVlabs/dexmimicgen)
