# TOPPRA 起点对齐与关节时序诊断设计

## 目标

把 whole-chunk TOPPRA 的位置起点与 per-action TOPPRA 每次 chunk 开始时的行为对齐：使用
SAPIEN articulation 的真实左右臂 qpos，而不是 PD drive target。修改后在同一条 RoboTwin
专家 episode 上，对比 per-action TOPPRA 与 whole-chunk TOPPRA 的真实关节速度、加速度、
规划速度和跟踪误差。

## 当前语义

- per-action TOPPRA：每次 `take_chunk_action_per_action` 的第一段从真实 qpos/qvel 开始；同一
  chunk 的后续段使用上一段目标 qpos 和规划末速度作为开环前馈。
- whole-chunk TOPPRA：`retime_chunk` 会 prepend 一个位置起点，但调用方当前传入
  `get_*_arm_jointState()`，该值是 drive target，不是真实 qpos。
- `get_obs()` 与专家数据字段保持现状。本改动只修执行器求解起点，不改变 VLA 输入或数据格式。
- `k_skip` 只用于 VLA 在线推理。本次专家 episode 回放不裁剪 action，固定 `max_actions=None`。

## 方案选择

采用局部替换方案：仅在 `take_chunk_action` 组装 `current_state_arm` 时改读
`get_left_arm_real_jointState()` 和 `get_right_arm_real_jointState()`。不全局修改
`get_*_arm_jointState()`，避免影响数据采集、VLA observation 和其他执行后端。

whole-chunk 仍保持 TOPPRA 的静止边界速度 `sd_start=0, sd_end=0`。本次“起点对齐”特指位置
起点 qpos；引入非零 whole-path 初始速度需要基于 spline 起始切向重新设计边界条件，不与这个
单一根因修复混合。

## 测试设计

### 自动化回归

构造最小 fake robot：drive target 与真实 qpos 使用不同数值，拦截 `retime_chunk` 输入并令其
立即 fallback。测试必须先证明旧实现把 drive target 传成 `current_state_arm`，修改后证明传入
真实 qpos。该测试不启动 SAPIEN 场景。

### 仿真对比

固定条件：

- task/episode/seed：`stack_blocks_two` / episode 0 / seed 0；
- 专家 action：完整 309 帧，按 50 帧 chunk 回放；
- 两后端：当前 per-action TOPPRA、修复后的 whole-chunk TOPPRA；
- 参数：`v=1.0`、`vel_limit=5 rad/s`、`acc_limit=8 rad/s²`；
- 物理频率：250 Hz；`k_skip=None`；
- 双臂力矩限制：每臂 `[30, 40, 30, 15, 10, 10] N·m`。

在每次 `scene.step()` 后采集：

- articulation 真实 qpos；
- articulation 真实 qvel；
- `qacc[t] = (qvel[t] - qvel[t-1]) / 0.004`；
- PD drive position target；
- PD drive velocity target，作为当步规划/前馈速度；
- `tracking_error = drive_qpos - real_qpos`。

## 输出

每个后端输出完整 CSV 和压缩 NPZ；汇总 CSV 给出每关节速度、加速度、规划速度、跟踪误差的
P50/P95/P99/峰值及速度/加速度越限比例。图表至少包含：

1. 两后端真实 qvel 随仿真时间变化，并标出速度限制；
2. 两后端真实 qacc 随仿真时间变化，并标出加速度限制；
3. 逐关节 qvel/qacc 峰值对比。

## 验收标准

- 自动化测试在修改前因 whole-chunk 使用 drive target 而失败，修改后通过；
- whole-chunk 求解器收到的首个位置点等于调用时真实左右臂 qpos；
- 专家回放未启用 `k_skip`，两后端使用完全相同的 episode 和约束参数；
- CSV/NPZ 的样本数与各后端 `dense_steps` 一致；
- 所有汇总数值可以从 NPZ 重新计算并通过一致性断言；
- 不修改 `get_obs()` 或专家数据格式；per-action 的后续变更以本文下方修订为准。

## 2026-06-30 修订：per-action 恢复零速度边界

### 范围

- 仅修改 `take_chunk_action_per_action`：每个相邻 action 的两点 TOPPRA 求解固定传入
  `sd_start=0.0, sd_end=0.0`。
- 保留当前 qpos 起点语义：每个 chunk 第一段从真实 qpos 开始；后续段仍以上一 action 目标
  qpos 作为几何起点。
- 不修改 whole-chunk；其整段 TOPPRA 已经使用 `sd_start=0.0, sd_end=0.0`。
- 不修改 `get_obs()`、fixedtime、`k_skip`、速度/加速度网格或力矩限制。

### 回归测试

增加 fake-robot 单元测试，拦截 per-action 对 `retime_chunk` 的每次调用。测试先在旧实现上证明
至少一个 `sd_end` 非零，再验证修改后所有段的 `sd_start` 和 `sd_end` 都严格为零。继续运行
whole-chunk 起点测试和 `toppra_chunk_executor_test.py`。

### 仿真复测

使用与上一轮完全相同的专家 episode 和参数：episode 0、seed 0、`v=1`、`k_skip=None`、
每臂力矩限制 `[30,40,30,15,10,10] N·m`，依次运行 `(vel,acc)=(1,1)、(2,2)、
(3,4)、(5,8)`。per-action 与 whole-chunk 均逐 4 ms 记录真实 qpos/qvel/qacc、规划速度和
跟踪误差，写入新的 `toppra_episode_dynamics_zero_boundary/`，保留上一轮结果用于前后对比。

### fixedtime 加速度产物

不修改 fixedtime 执行器。直接使用已采集的 episode 0、hold=15、v=1 与 v=4 的真实 qvel，
按 `qacc[t]=(qvel[t]-qvel[t-1])*250` 生成独立加速度时序图，并输出逐关节 P95/P99/峰值 CSV。

### 修订验收标准

- per-action 每次 `retime_chunk` 的两个边界速度参数均为零；
- 四档双后端复测产生 8 份完整 CSV/NPZ/JSON 与对应速度、加速度图；
- 新结果的样本数、qacc 和 tracking error 可以从 NPZ 独立重算并通过一致性断言；
- fixedtime v=1/v=4 加速度图和逐关节统计可直接用于判断尖峰位置与幅值；
- whole-chunk、fixedtime 和 `get_obs()` 源码不发生行为修改。
