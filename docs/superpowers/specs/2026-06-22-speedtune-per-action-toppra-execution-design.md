# SpeedTune per_action 执行核重构 — 方法 B（两点非零边界 TOPPRA）设计

> **用途**：本文档是 per_action「stop-and-go」修复的**方案设计**（brainstorming 产出），是问题分析
> handoff `2026-06-22-speedtune-per-action-stop-and-go.md` 的下游。根因已在 handoff 坐实，本文档定**怎么改**。
> **运行环境**：本地只写代码 + `python3 -m py_compile` 静态检查（显存不足，跑不了 pytest/sapien/toppra/训练）；
> TOPPRA 行为、时间/连续性对比、success 率均由维护人在**云端**验证。详见 CLAUDE.md「运行环境」。
> **维护人**：polar823。日期：2026-06-22。**已与维护人确认方法 B 及全部默认选择（见 §3.2 / §3.5）。**

## 1. 目标与约束

**目标**：让 `per_action` 后端消除「逐 action 两点零速 TOPP → 段间速度归零 → stop-and-go 伪快」，改为
**逐 action 粒度 + 段间速度不归零**的真加速执行，运行时间 ≈/≤ `fixed_time`（靠 TOPPRA 时间最优，而非走停伪快）。

**硬约束（维护人明确）**：必须保留**逐 action 执行**。后续要在此之上做 **partial chunk execution（部分执行
action chunk）+ 异步推理（async inference）** 等 VLA 常用技巧，这些都依赖「一个 action 一个执行单元、可随时中断/
替换」的粒度。**因此整段一次规划的方案（whole_chunk 式 P3/P4）被排除**——它无法支持中途换 chunk / 部分执行。

**仿真约束**：满足关节速度/加速度上限。注意（见 §2）仿真**不在物理层 clamp** 速度/加速度，约束完全靠
TOPPRA 生成的参考轨迹本身满足上限来保证。

## 2. 已核实的关键事实（实现依据，均读源码到行）

1. **原版 stop-and-go 是固有设计，非 fork bug**：原版 `take_action`（`RoboTwin-Platform/RoboTwin`
   `upstream/main`，与 fork 基线 `0aeea2d` 逐字一致）用两点路径
   `np.vstack((current_qpos, action))`（`envs/_base_task.py:1535`）+ mplib `TOPP`（`:1543`），mplib 内部
   `compute_trajectory()` 无参 → `sd_start=sd_end=0` → 每个 action 0 速起 0 速停。fork 的 `per_action`
   后端（fork `take_action:1562`，两点 `:1638`、TOPP `:1647`）= 原版 + `scaled_limits(vel,acc)` + 实时
   video，**核心两点零速 TOPP 与原版完全一致**，忠实复刻。原版唯一的连续平滑执行只在**专家采集路径**
   （`move:884`→`together_move_to_pose:794`→`take_dense_action:1407`，一次规划整条轨迹、逐步带前馈速度下发）。

2. **TOPPRA 输出语义**：mplib `TOPP`（`/home/xukainan/openpi/.venv/.../mplib/planner.py:351`）输入一串
   **几何航点 q_pos**（无时间）+ vel/acc 上限，求**时间最优的 s(t)**，输出按 step 采样的
   `(ts, qs=q(t), qds=q̇(t), qdds=q̈(t), duration)`（`:375-379`）。**q_pos 路径是输入（policy 的 action 即目标
   航点），TOPPRA 算的是 timing（沿路径每点何时到达、逐时刻应在路径哪个位置），不是 q_pos 本身。**

3. **执行端 = SAPIEN 关节 PD 驱动**：`set_arm_joints`（`/home/xukainan/RoboTwin/envs/robot/robot.py:609`）
   对每个关节 `set_drive_target(q(t))`（`:616`）+ `set_drive_velocity_target(q̇(t))`（`:617`）——位置设定点
   + 速度前馈，每个 `scene.step()` 物理引擎算力矩跟踪参考。**仿真不 clamp 速度/加速度**，约束靠参考轨迹（fork
   的 `PHYS_*_CEIL=3.0` 是参考层天花板）。

4. **弧长参数化下 `sd` = 关节空间速度幅值**：`retime_chunk`（`/home/xukainan/RoboTwin/envs/robot/
   toppra_chunk_executor.py:29`）用**累积弧长**当 path_s（`:107`，不是 mplib 的归一化 [0,1]），此时
   `dq/ds` 是单位切向量、`|dq/ds|=1`，于是路径参数速度 `sd` 直接 = 关节空间速度幅值 (rad/s)。这让方法 B 的
   核心参数 `sd_end` 物理意义清晰（= 经过该 action 时的速度幅值），好设、好钳。

5. **两点直线段间速度方向必突变（几何必然）**：相邻 action 段是两条方向不同的直线，即使强行让 `sd` 在交界
   非零，关节速度 `q̇=(dq/ds)·sd` 的**方向**仍在交界突变。所以方法 B 只能做到**速度幅值不归零**，方向突变靠
   PD 兜底——这是 B 相对方法 A（前瞻样条）的已知代价，维护人已知情并接受（换取最小改动 + 逐 action 粒度）。

## 3. 方案设计：方法 B（两点 + 非零边界拼接）

### 3.1 为什么是 B（而非 A / C）

- **方法 A（滚动前瞻样条）**：更平滑（段间真连续），但实现复杂（前瞻窗口 + sd 衔接 + 只取首段切片）。
- **方法 C（整段预规划 + 逐段下发）**：违背逐 action 硬约束（不重读状态、async 换 chunk 要重算）。**排除。**
- **方法 B**：最小改动达成核心目标（逐 action + 不归零 + 守约束），完全兼容 partial chunk / async。**采用。**
  若后续 B 的平滑度不足，升级路径见 §8。

### 3.2 执行循环（重写 `take_chunk_action_per_action` 的执行核）

对 `reconstruct_chunk(v)` 压缩后的 chunk 逐 action 执行（`i = 0 … M-1`）：

1. **实读当前关节状态** `q_current`（左右臂 12 dof）；
2. 路径 = 两点 `[q_current, aᵢ]`，弧长参数化建样条（复用 `retime_chunk` 写法，见 §4）；
3. `compute_trajectory(sd_start_i, sd_end_i)` → 取 `q(t)/q̇(t)` 按 250Hz 采样（采样从 `t=dt` 起、不含起点，
   避免重复下发 `q_current`）；
4. 逐采样步 `set_arm_joints(q(t), q̇(t))` + `scene.step()`；每步检查 `eval_success` / 中断条件（支持
   partial chunk / async：执行到第 k 个 action 即可停或替换后续 chunk）；
5. 累加本段真实 `jnt_traj.duration`；进入下一 action。

### 3.3 `sd_end`（过点速度幅值）：B-固定（已确认）

- `sd_end_i = v_cruise`（一个固定 rad/s 常数），**最后一个 action `sd_end = 0`**（停稳）。
- `v_cruise` 须 ≤ 该段 per-joint 速度约束允许的最大路径速度，否则 toppra infeasible。实现时钳：
  `sd_end = min(v_cruise, SAFETY · sd_max)`，其中 `sd_max = min_j(vel_lim_j / |tangent_j|)`（弧长参数化下
  `tangent` 为单位向量），`SAFETY` 取保守比例（如 0.9）。
- **量纲提醒**：双臂 12 dof 一起求解（§3.5），弧长是 12 dof 联合弧长，故 `v_cruise` 是 **12 dof 关节速度
  向量的范数 (rad/s)**，与 per-joint 的 `PHYS_VEL_CEIL=3.0` 量纲不同，勿混。
- `v_cruise` 初值给保守常数，**云端实测调**（见 §8）。设为可配置参数（§4）。

### 3.4 `sd_start`（段起点路径速度）：闭环读实际（已确认）

- 每段起点 `sd_start_i = max(0, q̇_actual · t̂_i)`：读**实测**关节速度 `q̇_actual`，点乘本段单位切向
  `t̂_i = (aᵢ - q_current)/|aᵢ - q_current|`（弧长参数化下投影即点积）；`max(0, ·)` 防实测速度反向时取负。
- 与 §3.3 的 `sd_end` 不强制相等——B 不追求数学连续（方向本就突变），闭环读实测让参考与 PD 实际状态一致、
  更鲁棒。
- 首个 action（chunk 起始）：同样读实测 `q̇_actual` 投影——若上一 chunk 刚结束机臂仍在动则自然非零衔接，
  若静止则 ≈0。
- 需要一个**arm 关节速度 getter**（SAPIEN `get_qvel` 取 arm joints 分量）；fork 若无现成 getter 则在
  `robot.py` 新增（零侵入小增，不改既有方法）。具体 API 实现时确认。

### 3.5 默认参数（已确认）

- **双臂**：12 dof 一起求解（时间对齐，复用 `retime_chunk`），不左右分开；
- **vel/acc 约束**：沿用 `vel_scale/acc_scale` 放大 base 约束、钳 `PHYS_VEL_CEIL=PHYS_ACC_CEIL=3.0`
  （与 whole_chunk 一致，`toppra_chunk_executor.py:117-120`）；
- **gripper**：不参与 TOPP，按路径参数 `s` 在两点 gripper 值间插值下发（同 `retime_chunk` `:170-187`）；
- **duration 口径**：累加各段**真实** `jnt_traj.duration`，**不再用 `dense_steps/250`**（消除 `int()` 向下
  取整的"伪快"统计偏差，handoff §2）；
- **fallback**：见 §6。

## 4. 组件与改动（零侵入）

**改 RoboTwin fork**（`/home/xukainan/RoboTwin`，git remote `origin=knanxu/RoboTwin`）：

1. **扩展 `retime_chunk`**（`toppra_chunk_executor.py:29`）：新增可选参数 `sd_start=0.0, sd_end=0.0`，内部
   `instance.compute_trajectory(0, 0)`（`:139`）改为 `compute_trajectory(sd_start, sd_end)`。**默认值 (0,0)
   → `whole_chunk` 现有调用行为完全不变**（向后兼容，零侵入）。方法 B 以 `chunk=[aᵢ]`（单 target，内部
   vstack 成两点 `[current, aᵢ]`）+ 非零边界调用它。
   - 备选：若单 target 复用 `retime_chunk` 有歧义，另抽 `retime_segment(...)`；优先扩 `retime_chunk`。
2. **重写 `take_chunk_action_per_action`**（`_base_task.py:1983`）的执行核：不再逐 action 调 `take_action`
   （`:2039`），改为 §3.2 循环（自管 toppra 两点非零边界）；duration 改 §3.5 口径。
3. **`robot.py` 新增 arm 关节速度 getter**（若无现成，§3.4），供 `sd_start` 闭环读实测。

**不改**：mplib（`planner.py`，第三方）、原版/fork `take_action`（policy-eval 逐帧 eval 仍用它，保留不动）、
`take_chunk_action`（whole_chunk）、`take_chunk_action_streaming`（fixed_time）。

**expo-ft 侧**：**执行路径起步零改动**（验证用的 eval 平滑度指标为新增，见 §7，属"新增不破坏既有"）——`exec_backends.py` 的 `per_action_toppra→per_action` 映射（`:7-9`）不变，
`v_cruise` 用 RoboTwin 侧默认常数。需云端调参便利时，再扩 `client_robotwin/envs/robotwin_env.py::step_chunk`
的 `_RT_BACKEND` 透传 `v_cruise`（按既有透传 v/vel_scale/acc_scale 的方式，零侵入扩展）。

**提交**：RoboTwin 改动 commit 在 **RoboTwin repo 单独提**；本 design doc 与后续 expo-ft 侧改动 commit 在
expo-ft repo（handoff §7）。

## 5. 数据流

```
action_chunk (policy, M actions)
  → reconstruct_chunk(v)                      # v 压缩，既有
  → for i in range(M):
       q_current ← 实读 12dof 关节位置
       q̇_actual ← 实读 12dof 关节速度          # §3.4 闭环
       sd_start  = max(0, q̇_actual · t̂_i)
       sd_end    = (i==M-1) ? 0 : min(v_cruise, 0.9·sd_max)
       result    = retime_chunk(q_current, [aᵢ], gripper, sd_start, sd_end, ...)  # 简化; 完整签名见 §4
       for (q, q̇) in result.dense(samples from t=dt):
            set_arm_joints(q, q̇); scene.step()
            if eval_success or 中断(partial/async): break
       duration += result.duration            # §3.5 真实口径
```

## 6. 错误处理 / fallback

- **路径退化**（`q_current ≈ aᵢ`，去重后 <2 点）或 **toppra 求解失败**：该段先回退 `sd_end=0` 重解；仍失败
  → 回退到原 fork `take_action`（mplib，stop-and-go）执行该单 action，保证不崩（最坏退化为该段走停）。
- `sd_start` 实测速度反向 → `max(0, ·)` 钳为 0。
- `sd_end` 超约束 → §3.3 钳到可行范围；toppra 仍 infeasible 走上面 fallback。
- 所有 fallback 在 `info` 里记 `fallback_reason`（沿用 `take_chunk_action_per_action` 既有 info 字段）。

## 7. 验证（全部云端）

- **时间**：`per_action` 执行时间 ≈/≤ `fixed_time`（不再被零速短段 `int()` 低估的"伪快"误导，因 §3.5 已改真实
  duration 口径）。
- **连续性**：相对 P0（现状）jerk / 速度过零次数显著下降。**新增 eval 平滑度指标**（jerk、速度过零计数）——
  在 `eval_speedtune.py` 加记（expo-ft 侧零侵入新增，不改既有统计）。
- **正确性**：task success 率不低于 P0。
- **逐 action 兼容性**：验证 partial chunk（执行 k 个 action 中断）、async（中途替换后续 chunk）路径正常。
- **向后兼容**：`whole_chunk` 走 `retime_chunk` 默认 `(0,0)`，行为与改动前逐位一致（回归）。

## 8. 待定 / 后续

- **`v_cruise` 初值**：保守常数起步，云端按"时间↓、success 不降、jerk 可接受"实测调。
- **平滑度升级路径**（若 B-固定不够）：① `sd_end` 升级为 B-自适应（`v_cruise·max(0,cosθ)`，转弯按夹角
  衰减）；② 进一步可切方法 A（滚动前瞻样条）。本设计的执行核结构（自管 toppra + 可配 sd 边界）已为这两条
  升级预留接口。

## 9. 约束小结

- 改 RoboTwin fork；**不改** mplib、原版/fork `take_action`（policy-eval 保留）、whole_chunk、streaming。
- expo-ft 侧零侵入（起步），不破坏 `fixed_time`/`whole_chunk` 路径。
- 本地仅 `py_compile`；TOPPRA 行为 / 时间 / 平滑度 / success 全部云端验证。
- 详见 [[speedtune-per-action-stopandgo]]、[[speedtune-module]]。
