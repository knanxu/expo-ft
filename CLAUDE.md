# EXPO-FT × drift × RoboTwin × DBPO 项目约定

> 本仓库在 EXPO-FT 真机在线 RL 框架上**增量集成**：① drift pi0.5 动作头；② RoboTwin 仿真环境；
> ③ 在此基础上的 DBPO/BPO on-policy RL。
> 以 **RoboTwin 仿真**（stack two blocks，双臂 aloha-agilex）作为首个跑通验证环境，后续再迁真机。
> **权威活文档：`docs/DBPO_DEV.md`**（决策记录、数学→代码映射、分阶段计划、Changelog）。
> 维护人：polar823。

> **当前优先级（2026-06-21 调整）**：**先让 EXPO-FT 原算法（`EXPOLearner`/`BCLearner`）在 drift pi0.5 + RoboTwin
> 上跑通，再回到 DBPO on-policy RL。** 理由：① drift 对 EXPO-FT 是**纯 config 切换**——合并后的 `pi0.py`
> 里 `sample_actions`(`:347`)/`compute_loss`(`:254`) 已按 `use_drifting_loss` 自动分发到 drift，EXPO 的采样
> （`_jitted_infer`）与 actor 更新（`train_step`）天然吃到，**算法代码零改动**；② RoboTwin 的 obs/action 天然走
> openpi 同一套 transforms（replay buffer 的 3 路图像槽 base+left/right_wrist 对上 aloha 三相机；EXPO 的
> `sample_actions` 已含 `process_transformed_outputs` 反归一化——这是 DBPO rollout 缺的那步）；③ 连新训练入口都不用写，
> 直接用现成 `train_pi_robo_async.py` 的 `EXPOLearner` 分支。这是风险最低、最快闭环的路径，也为 DBPO 验证
> env/checkpoint/数据管线打底。**DBPO 产物（`dbpo*.py` / `train_pi_robo_dbpo_async.py`）保留不动。**
> EXPO-FT+drift+RoboTwin 的待补接缝（详见 `docs/DBPO_DEV.md` Phase 4-EXPO）：新建
> `configs/model/expo_ft_pi_drift_config.py`、RoboTwin 数据 loader（`process_droid_dataset` 是 DROID 专用，
> 需 `process_robotwin_dataset` 或最小 stub）、填 DBP drift ckpt、核对图像 HWC/CHW 轴序。

## 第一原则：零侵入、纯增量

新增 drift / RoboTwin / DBPO 相关功能时，**不得修改 EXPO-FT 既有功能**。既有的 `EXPOLearner`、`BCLearner`、
它们的 config（`expo_ft_pi_config.py` 等）、`droid_env.py`、原 `train_pi_robo*.py` 的 EXPO/BC 代码路径必须保持行为不变。
扩展一律走"新增文件 / 新增分支"，不走"改原文件逻辑"。
**特别地**：让 EXPO/BC 适配 drift+RoboTwin 也走零侵入——新增 `configs/model/expo_ft_pi_drift_config.py`（指向
drift robotwin openpi config + DBP ckpt），**不改** `expo_ft_pi_config.py`；新增 `process_robotwin_dataset`（或在
`train_pi_robo_async.py` 按 `env_name` 分发 loader），**不改** `process_droid_dataset` 的 DROID 路径。

落地约定：

- **新算法**：`DBPOLearner` 以独立 `model_cls="DBPOLearner"` 注册（`agents/alg/dbpo*.py`），
  在 `train_pi_robo*.py` 里**新增** `elif model_cls == "DBPOLearner"` 分支；绝不改 EXPO/BC 分支。
- **新 client（独立包，零污染 DROID client）**：RoboTwin 适配写成**全新独立包** `client_robotwin/`（与 `client/` 平级，
  不进 `client/`），含 ① `client_robotwin/run_robotwin_client.py`（RoboTwin server，speak 与 `client/run_client.py`
  **完全相同的 websocket 协议**：`create_env`/`reset`/`step`/`get_observation`/`get_info_for_step`，复用
  `openpi_client.msgpack_numpy`）；② `client_robotwin/envs/robotwin_env.py`（env 适配，接口同 `droid_env.py`：
  `reset` / `get_observation` / `step(action)→{executed_action}` / `get_info_for_step()→(done,success,reward,mask)`），
  包住已改造的 RoboTwin。**绝不改 `client/` 内任何 DROID 文件**（`droid_env.py`/`run_client.py`/…）；
  RoboTwin 重 sim 依赖（sapien/robotwin）隔离在 `client_robotwin/` 自己的环境里。learner 端 `expo_ft/env/env_client.py` 通用、**不动**，
  连哪个 server 由启动的 server 决定。可适当参考 `RLinf/rlinf/envs/robotwin/robotwin_env.py` 的 RoboTwin RL 适配实现。
- **env 算法无关（且 EXPO/BC 适配为当前第一里程碑）**：`client_robotwin/` 的 RoboTwin env **不止服务 DBPO**——
  EXPO-FT 原算法（`EXPOLearner`/`BCLearner`）在新任务 env 上的适配同等重要，且 **2026-06-21 起为当前第一优先级、先于 DBPO**。
  env 接口对所有 `model_cls` 通用（per-action `step` + 稀疏 0/1 终局奖励），不为某一算法定制；DBPO 与 EXPO/BC 只在
  **训练入口的 rollout/replan 逻辑**上不同（见「动作时序」）。EXPO/BC 走现成 `train_pi_robo_async.py`，obs/action 经
  pi0.5 `process_raw_inputs`/`process_transformed_outputs` 与 replay buffer 的同套 openpi transforms，已自带反归一化。
- **单 env，不做并行**：仿真只用**单个 env**（不引入 RLinf 的多 env 向量化，代码量大）。目的是在仿真里跑通**真机 RL 流程**，
  而真机本就无法并行 env。样本效率不足是已知代价（D8）。
- **动作时序（rollout 执行粒度）**：**DBPO rollout 执行整段 action chunk**（`H_e = H = 50`，对齐 RLinf，不做部分执行/前缀截断）；
  **EXPO-FT 原算法保留其部分执行**（per-step + `replan_steps`）。两者复用同一个 per-action 的 RoboTwin env，差异只在训练入口循环。
- **新 config**：新增 `configs/model/expo_ft_pi_drift_config.py`（EXPO+drift+RoboTwin，**当前优先**）、
  `configs/model/dbpo_pi_config.py`（已有）、`configs/task/robotwin_*.py`（已有，算法无关，EXPO/DBPO 共用）；
  **不改** `pick.py` / `real_base.py` / `expo_ft_pi_config.py` 等既有 config。
- **新训练入口**：**EXPO/BC + drift + RoboTwin 直接复用现成 `train_pi_robo_async.py`**（`EXPOLearner`/`BCLearner`
  分支，无需新入口）。仅 DBPO 的 on-policy 循环与 per-step off-policy 冲突时才用独立 `train_pi_robo_dbpo_async.py`，
  不在原脚本里破坏 EXPO/BC 的 per-step 路径。
- **验证不回归**：改动后既有 31 个 DBPO 测试保持通过，且 `expo_ft.agents.{vla,alg}` 与 EXPO/BC 导入/构造不受影响。


## 语言

所有解释/文档用中文；代码注释，技术术语与代码标识符保留英文原文。
