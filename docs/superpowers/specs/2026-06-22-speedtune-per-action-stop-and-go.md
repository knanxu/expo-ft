# SpeedTune per_action「stop-and-go」修复 — Handoff（问题2）

> **用途**：本文件是问题2 的**上下文恢复点**。`/clear` 后新会话读本文件即可无损继续，不依赖对话历史。
> **运行环境**：本地只写代码 + `python3 -m py_compile` 静态检查（显存不足，跑不了 pytest/sapien/训练）；
> 所有运行验证由维护人在**云端**跑。详见 CLAUDE.md「运行环境」。
> **维护人**：polar823。日期：2026-06-22。

## 0. 当前状态

- **问题1（EXPO rollout 0% / 自动专家介入）已完成**：commit `99878fd`；设计 `docs/superpowers/specs/2026-06-22-robotwin-auto-expert-intervention-design.md`、计划 `docs/superpowers/plans/2026-06-22-robotwin-auto-expert-intervention.md`。**本文档只管问题2，不要再动问题1。**
- **问题2 根因已 100% 坐实并核实到代码行（见 §2）；修复尚未动手**——下一步要先与维护人敲定一个设计选择（§5），再实现。

## 1. 问题背景

SpeedTune = 冻结 VLA 上的 branching Rainbow-DQN 学速度档，三种动作执行后端各一个独立 run：
- `fixed_time` → RoboTwin `streaming`：每个 action 执行**固定** `hold_steps`(默认15) 物理步（论文式固定时长）。
- `per_action_toppra` → RoboTwin `per_action`：逐 action 双端 TOPPRA 重定时，动态执行步数。**← 本问题主角**
- `chunk_toppra` → RoboTwin `whole_chunk`：整段一次 TOPPRA。（用户尚未训练这种）

SpeedTune 侧映射在 `expo_ft/speedtune/exec_backends.py:7-9`（`fixed_time→streaming`、`per_action_toppra→per_action`、`chunk_toppra→whole_chunk`）；`client_robotwin/envs/robotwin_env.py::step_chunk` 的 `_RT_BACKEND` 透传 v/vel_scale/acc_scale 给 RoboTwin。

**用户报告**：已训 `fixed_time` + `per_action` 两种。**`per_action` 运行时间远小于 `fixed_time`**，原因是 TOPPRA 边界条件为 0 初始速度/加速度，运动一直 stop-and-go 走停。**期望**：`per_action` 训练后运行时间 ≈ 或快于 `fixed_time`（靠 TOPPRA 时间最优真加速），而不是 stop-and-go 伪快。

## 2. 根因（精确到代码行，已读源码核实）

**链路**：`per_action`（`/home/xukainan/RoboTwin/envs/_base_task.py:2039`）逐 action 调 `take_action` → `take_action` 用 `[current_qpos, target]` **两点路径**（`:1638` `np.vstack((left_current_qpos, left_arm_actions))`）→ 调 mplib `TOPP`（`:1647`）→ mplib `TOPP` 内 `instance.compute_trajectory()` **无参**（`/home/xukainan/openpi/.venv/lib/python3.11/site-packages/mplib/planner.py:372`）→ TOPPRA 默认 `sd_start=0, sd_end=0`（`inspect.signature` 实测）→ **每个 action 从 0 速度起、到 0 速度停 = 段间强制归零 = stop-and-go**。

**「为什么伪快」**：时间统计 `info["duration"] = dense_steps/250`（`_base_task.py:2048`，eval 累加在 `eval_speedtune.py:240` `t_acc += info["duration"]`）只数**执行物理步**。每个零速短段在 scale 后的高上限（vel/acc ≤3.0）下是短促三角速度曲线，time-optimal 时间极短，且 mplib 步数 `int(jnt_traj.duration/step)`（`planner.py:375`）**向下取整**系统性低估 → 总步数被压低 → 报告时间短，但机械臂实际「冲一下停一下」，不平滑也非真快。

> 注：`fixed_time`/`streaming` 的 `take_action_cnt_delta` 与 chunk 间 inference 延迟（`hold_and_render`，`_base_task.py:637`）三后端**共用且不计入 duration**，所以时间差**纯粹**来自执行步数差，坐实是 TOPP 零速边界导致。

**`docstring` 自证**：fork 作者在 `take_chunk_action_per_action`（`:1989`）注释里**自己写明**「每个 action 是 [current → target] 两点 TOPP, 首尾零速 = stop-and-go (保持 RoboTwin 原生行为)」。

## 3. 三后端机制对比（核实）

| | `streaming`(fixed_time) | `per_action`(per_action_toppra) | `whole_chunk`(chunk_toppra) |
|---|---|---|---|
| 入口(`_base_task.py`) | `take_chunk_action_streaming:2056` | `take_chunk_action_per_action:1983` | `take_chunk_action:1781` |
| 用 TOPP | 否（线性插值+前馈速度） | 是，逐 action mplib 两点 TOPP | 是，整段一次 TOPPRA |
| **TOPP 速度边界** | — | **每段两端 sd=0**（stop-and-go） | **仅全局首尾 sd=0**，内部连续 |
| 段间过渡 | 非零连续(`vel_ff`) | **强制 0** | 非零连续(natural spline) |
| 时长 | `M×hold_steps`，与速度曲线无关 | Σ各段步数(零速短段被低估) | 真实 TOPP 连续时长 |
| vel/acc_scale | 不生效 | 生效(scaled_limits 钳3.0) | 生效(`toppra_chunk_executor.py:117-120` 钳`PHYS_*_CEIL=3.0`) |

`whole_chunk` 实现：`retime_chunk`（`/home/xukainan/RoboTwin/envs/robot/toppra_chunk_executor.py:29`）整段拼航点→去重→累积弧长作 `path_s`→`SplineInterpolator(path_s, kept_arm, bc_type="natural")`(`:127`)→`TOPPRA(...).compute_trajectory(0, 0)`(`:139`，**全局首尾零速、内部连续**)→按 250Hz 采样。

## 4. 原版 RoboTwin 对比（GitHub RoboTwin-Platform/RoboTwin）

- 原版 **policy-eval 路径 = `take_action`**（原版 line 1479）与本地 fork 同构，fork 仅加 `scaled_limits(vel,acc)` + video + 返回步数；**核心两点零速 TOPP 完全一致** → **原版 policy-eval 本来就 stop-and-go**。`per_action` 没偏离原版设计，它就是「原版 take_action + 外露 vel/acc_scale」。
- 原版 **专家采集路径 = `move`/`together_move_to_pose`**（原版 ~line 884/794）→ mplib 规划出**整条多点轨迹** → `take_dense_action` 一次连续下发（相邻 waypoint 速度连续，**非零过渡**）。
- **含义**：原版专家轨迹本就是「整段连续执行」（类似 `whole_chunk`/`streaming`），只有 policy 复现退化成逐 action 零速。所以 **`whole_chunk` 才是 `per_action` 想要但没达到的正确形态**。三后端均为 fork 新增（原版无 `take_chunk_action*`），未触碰原版 `take_action`/`move`（零侵入）。

## 5. 修复方向 + ⚠️ 待定设计选择（**先与维护人定再动手**）

修复核心 = **消除相邻 action 间的零速边界**。但 `per_action` 与 `whole_chunk` 动作空间相同（v/vel_scale/acc_scale，`exec_backends.py:156-165`），若把 `per_action` 直接改成整段连续 TOPPRA，就**和 `whole_chunk` 重合**，三后端对比退化为两个。所以要先定：

> **设计选择 Q**：`per_action` 修好后与 `whole_chunk` 的区别定位是什么？
> - **(推荐) 保留逐 action 粒度但段间速度连续**：仍逐 action 切片（保留逐 action 的 success/contact 反馈粒度），但相邻 action 共享非零边界速度（本段 `sd_start` = 上段末速，`sd_end` 由下段方向预估）。与 `whole_chunk`(整段 natural spline) 形成「分段线性连续 vs 样条连续」对比。
> - **(替代) `per_action` 复用 `whole_chunk` 的 `retime_chunk`**：最省事但两后端几乎重合（仅逐 action 反馈不同）。
> - 其它由维护人定。

修复方向（按改动量，均**改 RoboTwin fork**，不要改 mplib 与原版 `take_action`）：
1. **【最小·配合推荐选项】** 改 `take_chunk_action_per_action`（`_base_task.py:1983`）不再逐 action 复用 `take_action`，改为自管 TOPP：绕过 mplib 写死的 `compute_trajectory()`，直接用 `toppra`（参考 `toppra_chunk_executor.py` 写法）传非零 `sd_start/sd_end`（段间速度连续）。
2. **【复用】** `per_action` 执行核换成调 `retime_chunk`（整段连续），仅按 action 边界切片做 success/contact 检查。
3. **【治标·可叠加】** 修 duration 向下取整低估：`info["duration"]` 改累加各段真实 `jnt_traj.duration`（需让 `take_action` 返回 duration），而非 `dense_steps/250`。
4. **（评估层）** `eval_speedtune.py` 增记 jerk/速度过零次数等平滑度指标，避免再被「伪快」误导。

## 6. 关键文件清单（绝对路径）

- `/home/xukainan/RoboTwin/envs/_base_task.py` — `take_action:1562`、`take_chunk_action(whole_chunk):1781`、`take_chunk_action_per_action:1983`、`take_chunk_action_streaming:2056`、`take_chunk_action_backend:2184`、两点路径`:1638`、mplib TOPP 调用`:1647`、per_action duration`:2048`。
- `/home/xukainan/RoboTwin/envs/robot/toppra_chunk_executor.py` — `retime_chunk:29`（`compute_trajectory(0,0):139`、scale 钳 ceil`:117-120`、natural spline`:127`）。
- `/home/xukainan/RoboTwin/envs/utils/chunk_accel.py` — `reconstruct_chunk`（v 压缩）。
- `/home/xukainan/openpi/.venv/lib/python3.11/site-packages/mplib/planner.py` — `TOPP:351`（零速根源 `compute_trajectory():372`，**第三方库，勿改**）。
- `expo_ft/speedtune/exec_backends.py` — 后端↔SpeedTune 映射`:7-9`、v/vel/acc grid`:138-141,156-165`。
- `client_robotwin/envs/robotwin_env.py::step_chunk` — `_RT_BACKEND` 映射 + 透传 speed_params。
- `eval_speedtune.py:240` — `t_acc += info["duration"]`（时间统计入口）。

## 7. 约束

- **改 RoboTwin fork**（`/home/xukainan/RoboTwin`，git remote `knanxu/RoboTwin`，**与 expo-ft 是不同 repo**）→ memory `speedtune-module` 记的「expo-ft 只对接、未改 RoboTwin」会被打破，这是有意的新动作，**commit 在 RoboTwin repo 里单独提**。
- expo-ft 侧若需配合（如 `exec_backends.py` grid、`eval_speedtune.py` 指标）按既有零侵入约定（新增/新分支，不破坏 fixed_time/whole_chunk 路径）。
- 不改 mplib、不改原版 `take_action`。
- 本地只 `py_compile` 静态检查；TOPPRA 行为/时间对比/平滑度在云端验证。

## 8. 下一步

1. 与维护人敲定 §5 的**设计选择 Q**（per_action 与 whole_chunk 的区别定位）。
2. 据此走 systematic-debugging→brainstorming(若需)→writing-plans→实现：改 `take_chunk_action_per_action` 消除零速边界 + （可选）修 duration 口径 + （可选）eval 平滑度指标。
3. 云端验证：per_action 执行时间 ≈/≤ fixed_time 且运动连续（jerk/速度过零下降），任务完成率不降。
