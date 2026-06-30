# EXPO-FT × drift × RoboTwin × SpeedTune 项目约定

> 本仓库在 EXPO-FT 真机在线 RL 框架上**增量集成**：① drift pi0.5 动作头（单步生成，替换 flow-matching）；
> ② RoboTwin 仿真环境（双臂 aloha-agilex）；③ SpeedTune 加速模块（冻结 VLA 之上的 branching Rainbow-DQN，
> 学「该不该加速 / 加速多少」）。
> 以 **RoboTwin 仿真**（stack two blocks，双臂 aloha-agilex）作为首个跑通验证环境，后续再迁真机。
> 维护人：polar823。

> **运行环境（重要）**：所有训练 / rollout / eval **均在云端服务器**完成（见 `scripts/run_speedtune_*.sh`
> 里 `EXPO_ROOT=/home/chenlu/...`、`SPEEDTUNE_VLA_CKPT` 等环境变量）；**本仓库本地只提供代码**。
> 因此本地 **没有** `logs/`、wandb 记录、模型 checkpoint、norm_stats、RoboTwin demo 数据，也**跑不了**训练/仿真/**测试**
> （**显存不足**——连 `pytest`、`import jax/sapien/torch` 都不要在本地执行）。
> **工作流 = Claude 本地写代码 → 维护人上传云端服务器 → 云端调脚本跑训练/测试**；Claude 本地只做**静态检查**
> （`ast.parse` 语法、逻辑 review；`*_test.py` 照写但**不本地执行**），所有 `pytest`/训练/rollout/eval 验证交云端。
> 调试时不要假设本地能复现实验或读到训练产物——需要运行结果 / 实际超参 / success 曲线时，**问维护人**（或让其在云端取），
> 不要从本地空目录推断"未配置"。本地 `/home/xukainan/RoboTwin` 仅供**读代码对照**（fork 自 knanxu/RoboTwin）。

> **云端绝对路径（写脚本 / 给云端命令时直接用这些；本地无对应物，勿据本地推断）**：
> - 项目代码：`EXPO_ROOT=/home/chenlu/expo-ft`、`ROBOTWIN_ROOT=/home/chenlu/RoboTwin`。
> - 冻结 drift pi0.5 ckpt（`stack_blocks_two`，EXPO/SpeedTune 共用）根目录：
>   `/home/chenlu/openpi/checkpoints/pi05_aloha_robotwin_drifting_stack_blocks_two/drifting_v1/29999/`；
>   其下 `params/`（权重）与 `assets/`（norm_stats）**平级**，norm_stats 在 `assets/trossen/`（asset_id=`trossen`）。
> - SpeedTune 三个环境变量（`expo_ft/utils/train_utils.py` 拼成 openpi `weight_loader` + `AssetsConfig`）：
>   `SPEEDTUNE_VLA_CKPT=<根目录>/params`、`SPEEDTUNE_VLA_ASSETS=<根目录>/assets`、`SPEEDTUNE_VLA_ASSET_ID=trossen`。
>   坑：① `assets/` 与 `params/` **平级**而 CKPT 指向 `params/`，所以 `$SPEEDTUNE_VLA_CKPT/assets`（=`params/assets`）是错的，必须用 `<根目录>/assets`；
>   ② asset_id 实际是 `trossen`（`speedtune_dqn_config.py` 注释里的 "robotwin" 只是占位例子，用错→norm_stats 留空坑，见 memory）。

> **当前主线**：让 EXPO-FT 原算法（`EXPOLearner`/`BCLearner`）在 drift pi0.5 + RoboTwin 上跑通。
> drift 对 EXPO-FT 是**纯 config 切换**——合并后的 `pi0.py` 里 `sample_actions`/`compute_loss` 已按
> `use_drifting_loss` 自动分发到 drift，EXPO 的采样（`_jitted_infer`）与 actor 更新（`train_step`）天然吃到，
> **算法代码零改动**；RoboTwin 的 obs/action 天然走 openpi 同一套 transforms（replay buffer 的 3 路图像槽
> base+left/right_wrist 对上 aloha 三相机；EXPO 的 `sample_actions` 已含 `process_transformed_outputs` 反归一化）；
> 训练入口直接用现成 `train_pi_robo_async.py` 的 `EXPOLearner` 分支。
> EXPO/BC + drift + RoboTwin 的关键接缝：`configs/model/expo_ft_pi_drift_config.py`（指向 drift robotwin
> openpi config + DBP drift ckpt + norm_stats）、RoboTwin 数据 loader（`process_robotwin_dataset`，非 DROID 专用的
> `process_droid_dataset`）、delta 动作反归一化要加回当前 state、norm_stats 必须指 ckpt 自带那份（两个隐蔽坑详见 memory）。

## 第一原则：零侵入、纯增量

新增 drift / RoboTwin / SpeedTune 相关功能时，**不得修改 EXPO-FT 既有功能**。既有的 `EXPOLearner`、`BCLearner`、
它们的 config（`expo_ft_pi_config.py` 等）、`droid_env.py`、原 `train_pi_robo*.py` 的 EXPO/BC 代码路径必须保持行为不变。
扩展一律走"新增文件 / 新增分支"，不走"改原文件逻辑"。
**特别地**：让 EXPO/BC 适配 drift+RoboTwin 也走零侵入——新增 `configs/model/expo_ft_pi_drift_config.py`（指向
drift robotwin openpi config + DBP ckpt），**不改** `expo_ft_pi_config.py`；新增 `process_robotwin_dataset`（或在
`train_pi_robo_async.py` 按 `env_name` 分发 loader），**不改** `process_droid_dataset` 的 DROID 路径。

落地约定：

- **新 client（独立包，零污染 DROID client）**：RoboTwin 适配写成**全新独立包** `client_robotwin/`（与 `client/` 平级，
  不进 `client/`），含 ① `client_robotwin/run_robotwin_client.py`（RoboTwin server，与 `client/run_client.py`
  **完全相同的 websocket 协议**：`create_env`/`reset`/`step`/`get_observation`/`get_info_for_step`，复用
  `openpi_client.msgpack_numpy`）；② `client_robotwin/envs/robotwin_env.py`（env 适配，接口同 `droid_env.py`：
  `reset` / `get_observation` / `step(action)→{executed_action}` / `get_info_for_step()→(done,success,reward,mask)`），
  包住已改造的 RoboTwin。**绝不改 `client/` 内任何 DROID 文件**（`droid_env.py`/`run_client.py`/…）；
  RoboTwin 重 sim 依赖（sapien/robotwin）隔离在 `client_robotwin/` 自己的环境里。learner 端 `expo_ft/env/env_client.py` 通用、**不动**，
  连哪个 server 由启动的 server 决定。可适当参考 `RLinf/rlinf/envs/robotwin/robotwin_env.py` 的 RoboTwin RL 适配实现。
- **env 算法无关**：`client_robotwin/` 的 RoboTwin env 对所有 `model_cls` 通用（per-action `step` + 稀疏 0/1
  终局奖励），不为某一算法定制。EXPO/BC 走现成 `train_pi_robo_async.py`，obs/action 经 pi0.5
  `process_raw_inputs`/`process_transformed_outputs` 与 replay buffer 的同套 openpi transforms，已自带反归一化。
  SpeedTune 在同一 env 上加 `step_chunk` 协议（整段 chunk 执行 + 速度控制），**纯新增不改既有 per-action step**。
- **单 env，不做并行**：仿真只用**单个 env**（不引入 RLinf 的多 env 向量化，代码量大）。目的是在仿真里跑通**真机 RL 流程**，
  而真机本就无法并行 env。样本效率不足是已知代价。
- **SpeedTune 加速模块（独立、零侵入）**：`expo_ft/speedtune/`（`exec_backends.py`）+ `agents/alg/speedtune_dqn.py`
  + `networks/rainbow_dqn.py` + `data/speedtune_buffer.py`，入口 `train_speedtune_async.py`、评估 `eval_speedtune.py`，
  config `configs/model/speedtune_dqn_config.py`。VLA **全程冻结**（只前向产 detached `suffix_feat`），branching
  Rainbow-DQN 纯 off-policy 学速度档；与 EXPO/BC 平级，不进 EXPO/BC 代码路径。pi0.5 drift 前向适配器
  `make_drift_apply_fn` 在 `expo_ft/agents/vla/drift_adapters.py`（**算法无关**：从 nnx pi0.5 取
  `(mean, cond_emb, value_feat)`，供任何需要 drift 特征的下游模块复用）。
- **新 config**：新增 `configs/model/expo_ft_pi_drift_config.py`（EXPO+drift+RoboTwin）、
  `configs/model/speedtune_dqn_config.py`（SpeedTune）、`configs/task/robotwin_*.py`（算法无关，EXPO/SpeedTune 共用）；
  **不改** `pick.py` / `real_base.py` / `expo_ft_pi_config.py` 等既有 config。
- **新训练入口**：**EXPO/BC + drift + RoboTwin 直接复用现成 `train_pi_robo_async.py`**（`EXPOLearner`/`BCLearner`
  分支，无需新入口）；SpeedTune 用独立 `train_speedtune_async.py`，不在原脚本里破坏 EXPO/BC 的 per-step 路径。
- **验证不回归**：改动后 既有测试（集中在 `test/`，含 SpeedTune 与 RoboTwin）保持通过，且 `expo_ft.agents.{vla,alg}` 与
  EXPO/BC 导入/构造不受影响。

> 历史：曾在本框架上尝试 **DBPO**（用 BPO/PPO 对 drift policy 做 on-policy RL 微调），因与现框架冲突，已于
> 2026-06-22 整体移除（`dbpo*.py` / `train_pi_robo_dbpo_async.py` / `dbpo_pi_config.py` / 相关测试与文档全部删除），
> 现仓库只保留 **EXPO-FT / BC** 与 **SpeedTune**。需要时可从 git 历史（`dbpo-robotwin` 分支早期提交）找回。


## 语言

所有解释/文档用中文；代码注释，技术术语与代码标识符保留英文原文。
