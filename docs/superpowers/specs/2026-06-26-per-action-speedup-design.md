# SpeedTune 执行后端重设计：vel/acc 绝对值约束 + 真机力矩物理底座

> 日期：2026-06-26 ｜ 维护人：polar823 ｜ 分支：dbpo-robotwin
> 范围：SpeedTune `per_action_toppra`（连带共用 executor 的 `chunk_toppra`）提速 + `fixed_time(streaming)`
> 加真机力矩约束验证。**零侵入** EXPO/BC。
> 运行环境：训练/rollout/eval 均在云端；本地仅静态检查（不跑 pytest/仿真/sapien）。
> 实现分两阶段：**阶段 1 = per_action 提速 + 力矩底座（先做）**；**阶段 2 = fixed_time 力矩验证**。

---

## 1. 问题

eval 对比（`speedtune_compare_0622_2143`，fixed_time vs per_action_toppra）：per_action `mean_dense_steps_success`=12003 vs fixed_time 1737（**慢 6.9×**），而 DQN 选的激进度档位几乎相同（aggr 0.57 vs 0.52）→ **慢来自执行机制本身，非档位选择**。

两个待解问题：
1. **per_action 过慢**：vel_scale/acc_scale 之外一堆隐藏约束压低了速度。
2. **fixed_time 无约束作弊**：不过 TOPPRA、PD `force_limit=∞`，每 action 强制固定时长，等效速度可达 ~6.8 rad/s（超真机 5.5、超天花板 3.0），任何激进指令都被完美跟踪——不暴露真机问题。

> 数据可信度：该批两个 `server_*.log` 全是 `address already in use`，疑连残留旧 server（commit `9713d8a` 端口预检要修的 silent 问题）。**倍数需在 `9713d8a` 后重跑确认**；下文根因均为代码静态事实，不依赖该批数据。

---

## 2. per_action 慢的根因

逐 action 两点 TOPPRA，执行速度被一串 **`vel_scale`/`acc_scale` 之外的隐藏因素**压低：

1. **`DEFAULT_CRUISE_SD=1.0` 段间巡航钳**（`toppra_chunk_executor.py:30,277`）：`sd_end=min(v_cruise=1.0, safety·sd_max)`，vel_scale 只放大瞬时峰值、段边界死钉 1.0 rad/s。**首要根因**。
2. **base 倍率制**：`vel_lim=base×vel_scale`（`:123`），base=mplib 默认（≈1.0，维护人已核实；URDF `velocity=1000` 是占位，mplib 不取）。倍率制依赖隐式 base，不透明。
3. **`is_last → sd_end=0`**（`:269-270`）：每 `k_skip=10` action 减速到 0，周期 stop-and-go。
4. **`safety=0.9`**（`:277`）：段末速度额外 10% 折扣。
5. **段间方向突变**（`_base_task.py:1997`）→ `sd_start` 投影掉（`:268`）：逐段两点直线架构**固有，不可消除**。
6. **加速度约束与真机物理脱节**：固定 `acc_limit` 是运动学量；真机加速度是力矩导出（`q̈=M(q)⁻¹(τ−Cq̇−g)`，随位形/负载变）。且执行层 PD `force_limit=∞`，TOPPRA 规划的 acc 被无限力矩完美跟踪 → 加速度**实际不受任何力矩约束**。

> 概念澄清：`take_action`(:1568，RoboTwin 原生，EXPO/BC 用) **有** mplib TOPP 的 vel/acc 约束（两点零速）；§1 说"无约束"的是 `take_chunk_action_streaming`(:2178，fixed_time)，两者是不同函数，不矛盾。

---

## 3. 设计目标与原则

1. **per_action 执行速度由 DQN 输出的 `vel_limit`/`acc_limit`（绝对值）唯一决定**——清理 §2 #1–#4 隐藏因素；#5 架构固有，PD 兜底，诚实接受。
2. **加速度回到真机力矩（任务无关）**：`acc_limit` grid 由真机力矩 `τ_max` + **无负载**惯量**离线标定一次**（机器人固有，换任务不重标）；执行层挂 `force_limit=τ_max` 作物理底座,**自动吸收抓取负载与 per-joint 差异**,DQN 经 RL 学抓取后收敛。
3. **统一力矩底座**：两个后端（per_action / fixed_time）执行层共用同一真机 `τ_max`：
   - per_action：grid 已按 `τ_max` 标定 → 正常 **不饱和**（可控且快）。
   - fixed_time：无 TOPPRA、盲目激进 → `force_limit` **饱和** → 跟踪落后 → 失败。
   - **同一底座下高下立判**，对比纯粹反映「TOPPRA 预规划 vs 无规划」的价值，而非「有无力矩约束」。
4. **零侵入 EXPO/BC**：`force_limit` 默认 `None`（=∞，原行为），仅 SpeedTune env 开启。

per_action 三变量 grid 物理依据：`vel_limit`←真机最大速度（5~5.5，`PHYS_VEL_CEIL=5.0`）；`acc_limit`←力矩标定 q̈_max；`v`(压缩比)←MDP horizon，不变。

---

## 4. 阶段 1 改动：per_action 提速 + 力矩底座

数据流：`exec_backends grid（绝对 rad/s, rad/s²）→ decode → speed_params{vel_limit,acc_limit,v} → robotwin_env.step_chunk → take_chunk_action_per_action / take_chunk_action → compute_segment_sd_bounds / retime_chunk`。

| # | 改动 | 文件:函数 | 细节 |
|---|---|---|---|
| 1 | grid 改绝对值 + 改名 | `exec_backends.py::_default_specs` | `vel_scale→vel_limit`(rad/s)、`acc_scale→acc_limit`(rad/s²)；vel grid 上界≤真机速度，acc grid 由 §5 标定。`v` 不动。`aggressiveness`/reward 归一化逻辑**不变** |
| 2 | retime_chunk 绝对值化 | `toppra_chunk_executor.py::retime_chunk` | 去 `×base`；直接收绝对 `vel_lim/acc_lim`（标量广播到 dof），仅钳 `PHYS_CEIL` 安全网。签名去 `joint_vel_limits/joint_acc_limits/vel_scale/acc_scale` |
| 3 | sd 边界绝对值 + 去隐藏约束 + **可控性钳** | `compute_segment_sd_bounds`（加 `acc_limit` 参数）| `sd_end = safety·min(sd_max, sd_reachable)`：`sd_max=vel_limit/max\|tangent\|`；**`sd_reachable=√(sd_start²+2·acc_path·seg_len)`**（`acc_path=acc_limit/max\|tangent\|`）。⚠️ sd_reachable 钳**必不可少**——否则短段 `sd_end` 过大 → 两点 TOPPRA `FailUncontrollable` → 回退 take_action 退化 stop-and-go 且 vel_limit 失效（本地实测踩到）。去 `v_cruise`/删 `DEFAULT_CRUISE_SD`；去 `is_last` 归零；`safety 0.9→0.99` |
| 4 | PHYS_CEIL 安全网 | `toppra_chunk_executor.py:25-26` | `PHYS_VEL_CEIL 3.0→5.0`（真机速度）；`PHYS_ACC_CEIL` 抬到**不低于 §5 标定的 q̈_max 峰值**（原 3.0 是 curobo 专家轨迹用，不应再钳已力矩标定的 acc grid）。纯 backstop。**`planner.py:17-18` 的 PHYS_CEIL 不动**——它仅在 `take_action/scaled_limits` 路径，新设计下 scale=1 即 no-op，改了徒增对 EXPO/BC 的扰动风险 |
| 5 | 调用方传绝对值 | `_base_task.py::take_chunk_action_per_action`(:1989) + `take_chunk_action`(:1787) | 不再取 mplib base；参数/透传改 `vel_limit/acc_limit`；dof 从 `chunk_arm.shape[1]` 取；主循环结构不变 |
| 6 | step_chunk 透传 | `robotwin_env.py::step_chunk`(:337) | `speed_params.get("vel_limit"/"acc_limit")` |
| 7 | **执行层 force_limit** | `robotwin_env.py`（SpeedTune env 层）| RoboTwinEnv 新增可选 `force_limit`（per-joint，默认 None=∞=原行为）。setup 后对 arm 关节 `set_drive_property(..., force_limit=τ_max)`。**自动处理抓取负载**（sapien 含接触力动力学，无需 acc grid 标定负载）。**不改 `robot.py`**，保 EXPO/BC 零侵入 |
| 8 | TOPP 失败回退适配 | `take_chunk_action_per_action`(:2118) | 回退 `take_action` 时绝对值→倍率映射：最简用 `vel_scale=acc_scale=1.0`（边界情况，TOPP 失败才触发）|

标量广播：DQN 每 head 输出一个标量上限，广播到所有 12 dof。per-joint 上限为后续扩展。

> 注：grid 档数若变（如 vel 4→5 档），`ExecBackend.head_sizes` 变 → RainbowDQN 输出维度变 → 必须重训（本就是新 exec_backend run）。

---

## 5. acc_limit grid 力矩标定（核心新增，离线脚本，**任务无关**）

真机无负载加速度上界：

```
q̈_j,max(q) = (τ_j,max − g_j(q)) / M_robot_jj(q)
```

**重力 `g(q)` 必须计入**——双臂大臂抗重力占掉相当力矩，`q̈_max=(τ_max−g)/M`，忽略 g 会高估。**抓取负载不在此标定**：由执行层 `force_limit` 在物理层自动施加（sapien 含接触力动力学，见 §3#3、§4#7）+ RL 学阶段适配。

**标定流程（独立脚本，云端跑一次；机器人/τ_max 不变则所有任务复用）**：
1. 扫机器人工作空间代表位形（或跑一条覆盖性轨迹），逐点：
2. `M_robot(q)` ← sapien `compute_generalized_mass_matrix`；
3. `g(q)` ← `compute_passive_force(gravity=True)`（`robot._entity_qf` 已在用）；
4. `q̈_j,max(q) = (τ_j,max − g_j(q)) / M_robot_jj(q)`，扫遍位形取范围；
5. 输出 acc grid（标量广播）：上界取 per-joint q̈_max 的较高分位（弱关节由 per-joint `force_limit` 在执行层兜底），并给小档供 DQN 在抓取后收敛。

**任务无关性**：q̈_max 只依赖机器人位形与 τ_max（机器人固有），**不依赖物体/任务**。换任务/换物体**不重标**（机器人与 τ_max 不变即可）；负载与位形差异由 `force_limit`（硬底座）+ RL（学阶段适配）吸收。

输入：`τ_max`（per-joint，ARX5 10~40 N·m，**维护人按 datasheet/真机定**）。
待云端确认 API：sapien `compute_generalized_mass_matrix`、`compute_passive_force`。

---

## 6. 阶段 2 改动：fixed_time 力矩验证

- 执行层挂同一 `force_limit=τ_max`（复用 §4#7 机制）。streaming 不过 TOPPRA、`v` 压缩比大时相邻 action 间隔大 → 要求力矩超 `τ_max` → PD 饱和、跟踪落后。
- **验证目标**：探查 `v` 多大时 force_limit 下 `success_rate` 掉下来（无约束流式的真机问题）。主要是 eval + 分析，代码改动小（force_limit 机制阶段 1 已建）。

---

## 7. 零侵入边界

- **EXPO/BC**：走 `take_action`（`scaled_limits(1,1)` no-op），不碰 retime_chunk；`force_limit` 默认 None（∞）不施加。✓ **行为不变**。
- **force_limit 隔离在 RoboTwinEnv 层**（可选 config），不改 `robot.py`；SpeedTune env 开、EXPO/BC env 不开。
- **whole_chunk(chunk_toppra)** 与 per_action 共用 executor，一起改绝对值制（语义统一）。

---

## 8. 无法消除的偏差

1. **段间方向突变**（`:1997`）→ `sd_start` 转折处投影掉、自然减速。
2. **短段 acc 主导、vel_limit 够不着**：专家逐 action 段很短（~0.01–0.1 rad），`sd_reachable=√(2·acc_path·seg_len)` < `sd_max` → 速度由 `acc_limit` 主导，`vel_limit` 仅在长段/连续高速段（`sd_start` 累积后）生效。本地 bench 实测：短段 `vl2` 与 `vl4` 同速。

要完全连续 + vel 充分生效只有整段 `whole_chunk`；per_action 保留逐 action 可中断是有意取舍。

**本地验证**（shake_bottle 专家回放，`scripts/bench_exec_backends.py`）：per_action `dense_steps` 修复 sd_reachable 前 9697（全 fallback）→ 后 888（`vl4 al8`）/1326（`vl2 al4`），与 fixed_time（762–1484）、whole_chunk（695）同量级，档位单调生效，success 全 100%。

---

## 9. 测试与云端验证

- **本地（仅静态）**：`ast.parse`；更新 `toppra_chunk_executor_test.py`、`exec_backends_test.py`（照写，**不本地跑**）。
- **云端**：
  1. 确保 `9713d8a` 端口预检生效。
  2. 跑 acc 标定脚本，产出**无负载** q̈_max 范围 + acc grid（机器人固有、任务无关；佐证重力计入后大臂关节 q̈_max 偏低）。
  3. `print(planner.joint_vel_limits)` 复核 base 假设。
  4. 重跑 `eval_speedtune_compare`：① per_action `dense_steps` 大幅下降 ② `success_rate` 不降 ③ `vel_limit/acc_limit` 档位与实测速度/加速度单调对应 ④ per_action force_limit 正常不饱和。
  5. 阶段 2：fixed_time 扫 `v`，确认超限 `success_rate` 下降。

---

## 10. 输入参数汇总（维护人提供）

| 参数 | 来源 | 用途 |
|---|---|---|
| `τ_max`（per-joint, N·m） | ARX5 datasheet/真机（10~40） | force_limit（两后端执行层）+ acc 标定 |
| 真机最大关节速度（rad/s） | ARX5 实测（5~5.5） | vel_limit grid 上界 + `PHYS_VEL_CEIL` |

> 方块质量等负载参数**不再需要**——B 哲学下抓取负载由 `force_limit` 在物理层自动处理，acc 标定只用机器人自身无负载惯量。

---

## 11. 执行阶段修订记录（本地实测发现，已落地）

1. **本地可跑验证**：本地 RTX 4060 8GB + `sapien`/`toppra`/`curobo`/`torch-cuda` 全可用，执行后端验证（专家回放、标定、bench、视频）本地闭环，不必云端（仅完整 SpeedTune DQN 训练需云端）。RoboTwin env：`/home/xukainan/miniforge3/envs/RoboTwin`。

2. **真机 ARX5 实参**（arx5-sdk `include/app/config.h`，real-stanford；已用于 `bench_exec_backends.py`）：
   - `τ_max`(per-joint)=`[30,40,30,15,10,10]` N·m（force_limit）；`vel_max`=`[5,5,5.5,5.5,5,5]` rad/s。
   - 真机 PD `kp`=`[80,70,70,30,30,20]`、`kd`=`[2,2,2,1,1,0.7]`。⚠️ 与 RoboTwin sim PD(1000/200)差 12~285×——sim PD 远硬，是 fixed_time「无约束完美跟踪」的另一根源。

3. **sd_reachable 可控性钳**（`compute_segment_sd_bounds`）：去 v_cruise 后 `sd_end=safety·min(sd_max, sd_reachable)`，`sd_reachable=√(sd_start²+2·acc_path·seg_len)`。否则短段 sd_end 过大 → 两点 TOPPRA FailUncontrollable → 全回退 stop-and-go。

4. **per_action 段内开环前馈**（`take_chunk_action_per_action`，关键修订）：闭环用实测 q_current 在真机软 PD 下逐段滞后累积（seg/T 暴增、dense_steps 3172）。改为：首段从实测起（chunk 开头闭环），段间用**命令 q_current + 上段规划末速度向量 prev_qd 投影**算 sd_start，并钳 `sd_start≤safety·sd_max`。真机 PD 下 dense_steps 3172→267(↓12×)、真 fallback ~0；chunk 间由 VLA replan 闭环。诊断 `scripts/diag_per_action.py`。

5. **acc 标定 API**（`scripts/calibrate_acc_grid.py`）：用 `art.create_pinocchio_model().compute_generalized_mass_matrix`（**不是** articulation 直接方法）+ `compute_passive_force(gravity)`。

6. **完整对比测试**：`scripts/bench_exec_backends.py`（50 ep stack_blocks_two、三后端、fixed_time 扫 v、per_action/whole_chunk 控制变量、真机 PD+force_limit、不 k_skip、ep0 录视频）。
