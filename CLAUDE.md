# EXPO-FT × DBPO 项目约定

> 本仓库在 EXPO-FT 真机在线 RL 框架上**增量集成** DBPO/BPO（drift pi0.5 + on-policy RL），
> 并以 **RoboTwin 仿真**（stack two blocks）作为首个跑通验证环境，后续再迁真机。
> **权威活文档：`docs/DBPO_DEV.md`**（决策记录、数学→代码映射、分阶段计划、Changelog）。
> 维护人：polar823。

## 第一原则：零侵入、纯增量

新增 DBPO/RoboTwin 相关功能时，**不得修改 EXPO-FT 既有功能**。既有的 `EXPOLearner`、`BCLearner`、
它们的 config、`droid_env.py`、原 `train_pi_robo*.py` 的 EXPO/BC 代码路径必须保持行为不变。
扩展一律走"新增文件 / 新增分支"，不走"改原文件逻辑"。

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
- **env 算法无关（同等重要）**：`client_robotwin/` 的 RoboTwin env **不止服务 DBPO**——EXPO-FT 原算法
  （`EXPOLearner`/`BCLearner`）在新任务 env 上的适配同等重要、同等优先。env 接口对所有 `model_cls` 通用
  （per-action `step` + 稀疏 0/1 终局奖励），不为某一算法定制；DBPO 与 EXPO/BC 只在**训练入口的 rollout/replan 逻辑**上不同（见「动作时序」）。
- **单 env，不做并行**：仿真只用**单个 env**（不引入 RLinf 的多 env 向量化，代码量大）。目的是在仿真里跑通**真机 RL 流程**，
  而真机本就无法并行 env。样本效率不足是已知代价（D8）。
- **动作时序（rollout 执行粒度）**：**DBPO rollout 执行整段 action chunk**（`H_e = H = 50`，对齐 RLinf，不做部分执行/前缀截断）；
  **EXPO-FT 原算法保留其部分执行**（per-step + `replan_steps`）。两者复用同一个 per-action 的 RoboTwin env，差异只在训练入口循环。
- **新 config**：新增 `configs/model/dbpo_pi_config.py`（已有）、`configs/task/robotwin_*.py`；
  **不改** `pick.py` / `real_base.py` / `expo_ft_pi_config.py` 等既有 config。
- **新训练入口**：若 on-policy 循环与现有 per-step off-policy 循环冲突，新建 `train_pi_robo_dbpo.py`，
  不在原脚本里破坏 EXPO/BC 的 per-step 路径。
- **验证不回归**：改动后既有 31 个 DBPO 测试保持通过，且 `expo_ft.agents.{vla,alg}` 与 EXPO/BC 导入/构造不受影响。


## 语言

所有解释/文档用中文；代码注释，技术术语与代码标识符保留英文原文。
