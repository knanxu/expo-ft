# EXPO-FT × RoboTwin：Per-action TOPPRA 问题分析与决策记录

> 更新日期：2026-06-29  
> 当前状态：仅调研与设计，**尚未修改算法代码**。  
> 范围：`fixed_time`、`per_action_toppra`、`chunk_toppra` 的轨迹生成、执行约束和仿真验证。

## 1. 当前结论

1. 当前问题不是 TOPPRA 算法或 `toppra` 库本身存在已确认的 bug，而是
   **把一条轨迹拆成相互独立的两点路径，并给相邻路径设置非零标量边界速度**的使用方式，
   无法在一般转角处保证关节速度连续。
2. 当前实现向 `compute_trajectory(sd_start, sd_end)` 传入的是路径参数速度
   `sd = ds/dt`，不是完整的关节速度向量，也不是“初始速度和初始加速度”两个参数。
   加速度通过 `JointAccelerationConstraint` 约束；当前 API 没有直接传入初始关节加速度。
3. TOPPRA 只负责在一条已经给定的、足够光滑的几何路径 `q(s)` 上求最快时间参数 `s(t)`。
   它不能修复几何路径本身在 action waypoint 处的尖角。合理流程必须是：

   ```text
   action waypoints
       → 构造至少 C1 连续的关节空间几何路径 q(s)
       → TOPPRA 在速度/加速度/动力学约束下求 s(t)
       → 以固定控制周期采样 q、qdot、qdd
       → 执行时逐 action 检查 success/contact，并可中断或重规划
   ```

4. 推荐方向是：**规划时使用包含前瞻 action 的连续路径，执行时仍保留 per-action 粒度**。
   “规划粒度”和“执行/中断粒度”应分开，整段或短窗口规划不等于不能逐 action 中断。
5. 不推荐继续调 `v`、`vel_limit`、`acc_limit` 来掩盖当前段间不连续；参数调小只能降低故障概率，
   不能从数学上消除边界速度跳变。

## 2. 为什么独立两点 TOPPRA 会在段间失效

设第 `i` 段为两点直线路径：

```text
q_i(s) = q_i_start + tangent_i * s
tangent_i = (target_i - q_i_start) / ||target_i - q_i_start||
```

关节速度与加速度为：

```text
qdot = q'(s) * sd
qddot = q''(s) * sd^2 + q'(s) * sdd
```

对于两点直线路径，段内 `q''(s)=0`。TOPPRA 可以约束该段内部的 `qdot` 和 `qddot`。
但是在相邻 action 的边界处：

```text
qdot_before = tangent_i     * sd_end_i
qdot_after  = tangent_(i+1) * sd_start_(i+1)
```

要使关节速度连续，必须满足完整向量等式：

```text
tangent_i * sd_end_i == tangent_(i+1) * sd_start_(i+1)
```

`sd_start` 和 `sd_end` 都只是一个非负标量。当两个 action 段的 12 维切向方向不同，除非两边速度都为
零，通常不存在两个标量能让上述向量相等。于是边界发生：

```text
Delta qdot != 0
qddot_boundary ≈ Delta qdot / Delta t
```

连续时间下 `Delta t → 0`，对应冲击型、理论上无界的加速度；离散 250 Hz 执行时虽然不会得到数学上的
无穷大，但第一帧会出现约为 `Delta qdot / 0.004` 的巨大参考加速度。SAPIEN PD、`force_limit` 和真机
控制器只能通过饱和、滞后或振动吸收这个不连续，因此可能导致：

- 位置跟踪落后，下一段从错误状态开始；
- 抓取/接触时冲击增大，物体滑落或被推开；
- 力矩饱和，仿真轨迹与规划轨迹不一致；
- success rate 随 `vel_limit`、`acc_limit` 或 `v` 增大而下降。

这与现有 benchmark 一致：

| 后端/配置 | success rate | 平均成功 dense steps |
|---|---:|---:|
| `fixed_time_v3.5` | 100% | 1310 |
| `fixed_time_v4.0` | 90% | 1141 |
| `per_action_v1_vel5_acc8` | 86% | 1750 |
| `per_action_v1_vel5_acc2` | 94% | 2975 |
| `chunk_v1_vel5_acc8` | 92% | 1822 |

降低 per-action 的加速度上限从 8 到 2 后成功率从 86% 回升到 94%，说明激进动态确实与失败相关；
但现有结果没有记录段间 `Delta qdot`、实际 `qddot`、跟踪误差和力矩饱和，因此它是支持证据，
还不是单独证明因果关系的完整实验。

## 3. 当前代码中的具体问题

### 3.1 `compute_trajectory` 参数语义被误解

当前 `retime_chunk` 调用：

```python
retimed = instance.compute_trajectory(sd_start, sd_end)
```

这里的两个参数分别是起点和终点的**路径速度**，不是初始速度和初始加速度。真实关节边界速度由
`q'(s) * sd` 决定。若希望衔接真实关节速度 `qdot_actual`，必须先让新路径起点切向 `q'(0)` 与
`qdot_actual` 共线，然后才能找到合适的标量 `sd_start`。

### 3.2 每段内部合规不代表拼接后合规

`JointVelocityConstraint` 和 `JointAccelerationConstraint` 只作用于传给单次 TOPPRA 的路径。
当前每个 action 都创建新的两点路径，上一段终点与下一段起点的速度跳变不属于任何一次求解的内部，
因此不会被 TOPPRA 检测。

### 3.2.1 把边界速度接口改成向量不能直接解决问题

这不是简单的“TOPPRA 接口只接受标量，所以换成向量接口即可”的问题。TOPPRA 做的是固定几何路径
`q(s)` 的时间参数化，其边界关节速度天然满足：

```text
qdot_boundary = q'(s_boundary) * sd_boundary
```

给定路径后，`q'(s_boundary)` 的方向已经固定，TOPPRA 只剩标量 `sd_boundary` 可以决定沿该方向走多快。
即使在外层新增 `qdot_boundary` 向量输入，也只能执行两件事：

1. 检查该向量是否与路径切向 `q'(s_boundary)` 共线；
2. 若共线，将它换算成标量 `sd_boundary` 后再调用 TOPPRA。

如果向量不与路径切向共线，它对该几何路径就是不可行边界条件。求解器不能在保持原路径不变的同时满足它。

对相邻两点直线段，非零连续边界速度必须同时平行于前一段和后一段：

```text
qdot_boundary ∈ span(tangent_i) ∩ span(tangent_(i+1))
```

一般转角处两个一维方向不共线，其交集只有零向量。因此：

- 若坚持每个 action 都是独立两点直线并严格经过尖角，唯一普遍可行的连续边界速度是零；
- 若要非零通过 waypoint，必须修改几何路径，使 waypoint 附近具有共享的连续切向，例如使用前瞻
  waypoint 构造 cubic spline/Hermite path，或在尖角处加入 blend/圆滑过渡段；
- 若必须直接指定任意完整的 `(q, qdot, qddot)` 边界状态，应使用支持全状态边界的轨迹生成/优化方法，
  而不是把任意速度向量直接塞进固定路径的 TOPPRA。

当前代码用 `qd_actual @ tangent` 将实测速度投影到新段切向，只保留平行分量。这个操作能生成 TOPPRA
接受的 `sd_start`，但被丢弃的正交速度分量会表现为重规划边界的参考速度跳变，所以它不是速度连续性修复。

### 3.3 开环使用上一个命令 target 会隐藏跟踪误差

当前 per-action 后续段使用：

```text
q_current = previous commanded target
```

而不是实际关节位置。这样避免了软 PD 滞后导致每段路径变长，但代价是规划起点可能不等于真实状态。
当存在 `force_limit`、接触或跟踪滞后时，这会引入位置设定点跳变。正确方案不应在“完全闭环导致每段
重算变慢”和“完全开环忽略实际状态”之间二选一，而应采用带前瞻的滚动轨迹，并只在明确的重规划点
使用实际 `(q, qdot)` 重建平滑前缀。

### 3.4 chunk 末端目前仍可能非零速度

现实现删除了 `is_last → sd_end=0`，因此 chunk 最后一段也可能以非零速度结束。如果随后等待 VLA 推理、
切换轨迹或没有立即下发连续参考，控制器会面对未定义的后续运动。末端策略必须显式区分：

- 已有下一 chunk 且连续拼接：按速度匹配进行滚动替换；
- 没有下一轨迹：规划受约束减速到零；
- 异步推理未完成：继续执行已规划的安全尾段，不能停在非零速度边界。

### 3.5 当前测试遗漏了最关键的跨段约束

`toppra_chunk_executor_test.py` 目前验证单段最大速度和单段非零末速，但没有把两段采样轨迹拼起来检查：

- waypoint 边界的 `||qdot[k+1]-qdot[k]||`；
- 由有限差分得到的 `max |qddot|`；
- chunk 末端速度；
- spline 是否越过关节位置限制；
- 实际状态对参考状态的跟踪误差；
- `force_limit` 饱和占比。

因此“每个单段测试通过”不能证明整个执行轨迹满足约束。

## 4. 三种可行方案

### 方案 A：前瞻窗口规划，逐 action 执行与滚动重规划（推荐）

每次使用：

```text
[actual q] + [action_i, action_(i+1), ..., action_(i+K)]
```

构造平滑路径并运行一次 TOPPRA，只执行到第一个或前几个 action 边界，然后读取实际状态并滚动窗口。

优点：

- 保留逐 action success/contact 检查、`k_skip` 和未来 async inference 的中断能力；
- TOPPRA 能看到 waypoint 转角，在连续几何路径上统一分配速度；
- 可利用实际状态闭环纠偏，而不是 50 个 action 全开环。

难点：

- 每次重规划的新路径必须用实际 `qdot` 构造方向匹配的起始切向，否则重规划点仍会速度跳变；
- spline 可能越过 waypoint 或关节位置限制，需要采样验证和 fallback；
- 窗口末端需准备安全制动尾段，避免异步推理未完成时无轨迹可执行。

### 方案 B：整 chunk 平滑规划，按 action/path 边界分段执行（第一阶段最稳妥）

对当前 chunk 一次构造连续路径并运行 TOPPRA，同时保存每个原始 action waypoint 对应的 `s` 或时间。
执行器仍逐物理帧下发，并在越过每个 action 边界时检查 success/contact；达到 `k_skip` 就停止。

优点：

- 改动和计算复杂度低；
- 单次轨迹内部速度、加速度连续；
- 可以直接复用现有 `retime_chunk` 的主体。

缺点：

- 中途替换 chunk 时仍需要速度匹配的重规划/过渡段；
- 若只执行 chunk 的前 `k_skip` 个 action，TOPPRA 的时间优化使用了更远的未来路径，需保证停止点有
  可执行的制动余量；
- 与 `chunk_toppra` 的规划核接近，区别主要应定义为“逐 action 反馈/可中断”和“整段无中断执行”。

建议先以方案 B 建立正确、可验证的基线，再在确实需要 async inference 时升级到方案 A。

### 方案 C：每个 action 严格 rest-to-rest

保持两点路径，但每段强制 `sd_start=sd_end=0`。

优点：数学上有效、实现简单、容易验证。  
缺点：必然 stop-and-go，无法达到连续轨迹的时间最优，只适合作为安全 fallback 和对照组。

### 明确不采用：独立两点路径 + 非零边界标量速度

该方法只有在相邻段切向完全相同或速度为零时才能保证速度连续，不适用于一般 VLA action chunk。

## 5. 推荐的正确 TOPPRA 使用方式

### 5.1 先生成合格的几何路径

路径至少应达到 `C1` 连续，使 `q'(s)` 在 waypoint 处连续。可选实现包括 cubic spline 或 cubic Hermite。
需要额外检查：

- 所有采样点在关节位置限制内；
- spline 对 dense VLA action 不产生明显过冲；
- 起点切向与实际 `qdot` 对齐；
- 需要停下的终点速度设为零；
- 接触阶段可通过降低约束或缩短窗口，而不是制造路径尖角。

如果必须严格经过每个 action 且不允许 spline 圆滑改变切向，那么尖角处物理可行的唯一选择是降速至零。

### 5.2 再用 TOPPRA 做时间参数化

在同一条连续路径上配置：

- per-joint `JointVelocityConstraint`；
- per-joint `JointAccelerationConstraint`；
- 起点/终点路径速度 `sd_start`、`sd_end`；
- 足够密的 gridpoints，并对最终密集采样结果再次数值验证。

TOPPRA 的结果是给定几何路径下的时间最优，不是任意起终状态之间的全局轨迹最优。几何路径、接触策略和
避障仍由上层决定。

### 5.3 执行层需要安全尾段和可控切换

执行器应保存：

- 规划参考 `q_ref/qd_ref/qdd_ref`；
- waypoint 到 `s/time/sample index` 的映射；
- 当前可安全执行到的 horizon；
- 若新轨迹未及时到达，可继续执行的减速尾段。

切换轨迹时应从实际 `(q, qdot)` 出发构造平滑过渡；如果不能构造速度匹配的新路径，先受约束制动到零，
再启动新轨迹。

## 6. ALOHA/ARX5 约束应如何迁移到 RoboTwin

当前 `aloha-agilex` embodiment 实际使用 ARX5 模型。ARX5 SDK 的 X5 参数为：

```text
joint_vel_max    = [5.0, 5.0, 5.5, 5.5, 5.0, 5.0] rad/s
joint_torque_max = [30, 40, 30, 15, 10, 10] N·m
joint controller = 500 Hz
```

SDK 没有给出一组可直接复制的固定 `joint_acc_max`。因此：

1. 速度上限可按真实的 per-joint 数组迁移，不应长期使用统一标量 `5.0`。
2. `acc_limit` 应作为保守的轨迹级限制，通过真机无负载/有负载跟踪实验标定，而不是仅由
   `tau_max / M_jj` 的单点公式决定。机械臂动力学包含耦合项、重力、科氏力、摩擦和控制器限幅。
3. `force_limit=tau_max` 可以让仿真暴露一部分力矩饱和问题，但不能自动复刻真机：
   - RoboTwin PD 为 stiffness=1000、damping=200；
   - ARX5 joint controller 的增益约为 `kp=[80,70,70,30,30,20]`、
     `kd=[2,2,2,1,1,0.7]`，并含重力补偿、插值和速度安全检查；
   - 仿真 250 Hz，官方控制器后台循环为 500 Hz。
4. 第一阶段不追求“完全复刻真机”，而应把与真机共用的命令级 safety filter 放在执行前：
   per-joint 速度限制、参考加速度限制、轨迹连续性检查和超限 fallback。仿真再叠加力矩饱和、延迟、
   控制频率和跟踪误差用于压力测试。
5. 若目标是真正的动力学可行时间最优，应进一步使用逆动力学力矩约束，而不仅是固定关节加速度盒约束：

   ```text
   tau = M(q) qddot + C(q, qdot) qdot + g(q)
   |tau_j| <= tau_max_j
   ```

   这需要把动力学约束表达成 TOPPRA 可接受的二阶约束，或使用支持 torque constraint 的轨迹优化器；
   当前实现尚未做到这一层。

## 7. 验证方案

### 7.1 不依赖 SAPIEN 的轨迹级测试

对同一组真实 demo/VLA action chunk，所有后端先只生成参考轨迹，检查：

```text
max |qdot_j|  <= vel_limit_j + tolerance
max |qddot_j| <= acc_limit_j + discretization_tolerance
max boundary ||Delta qdot|| 接近普通采样步的变化量
chunk 结束时 ||qdot|| = 0，除非已有速度匹配的下一轨迹
所有 q 在 joint position limits 内
无 TOPPRA fallback / fallback 原因被完整记录
```

尤其必须对**拼接后的完整轨迹**做有限差分，不能只检查每个单段返回值。

### 7.2 RoboTwin 执行级指标

每个物理步记录：

- `q_ref, qd_ref, q_actual, qd_actual`；
- 参考/实际有限差分加速度；
- 最大位置和速度跟踪误差；
- drive force/torque 及 `force_limit` 饱和比例；
- 接触发生前后固定窗口内的峰值速度、加速度和冲量；
- TOPPRA fallback 次数、重规划时间、dense steps、真实 wall time。

### 7.3 成功率实验

- 固定同一批 50 个 seed 和 demo；
- 比较 `fixed_time`、原 per-action、正确的平滑 TOPPRA 和 rest-to-rest fallback；
- 每组至少报告成功率置信区间，而不只报告点估计；
- “更快且不降成功率”应同时满足轨迹约束，而不能只比较 `dense_steps`。

## 8. 当前决策与待确认项

### 已确定

- 不把当前现象表述为 TOPPRA 库 bug，而表述为分段路径拼接方式错误。
- 不继续采用“独立两点路径 + 非零 `sd`”作为最终方案。
- 规划与执行粒度解耦：TOPPRA 需要看到连续前瞻路径，执行仍可逐 action 检查和中断。
- 约束测试必须覆盖跨段边界和实际跟踪，而不是只测单段 TOPPRA 输出。
- 在设计确认前不修改算法代码。

### 推荐但尚待确认

- 第一阶段采用方案 B：整 chunk/已执行 horizon 平滑规划，按 action 边界执行与检查；
- 第二阶段在需要 async inference 时升级方案 A：前瞻窗口 + 速度匹配滚动重规划；
- 方案 C 保留为无法安全重规划时的 fallback。

## 9. 相关代码与资料

本地代码：

- `expo_ft/speedtune/exec_backends.py`
- `client_robotwin/envs/robotwin_env.py`
- `scripts/bench_exec_backends.py`
- `scripts/bench_out/results.md`
- `/home/xukainan/RoboTwin/envs/robot/toppra_chunk_executor.py`
- `/home/xukainan/RoboTwin/envs/robot/toppra_chunk_executor_test.py`
- `/home/xukainan/RoboTwin/envs/_base_task.py`

外部一手资料：

- TOPPRA 官方文档：<https://hungpham2511.github.io/toppra/>
- TOPPRA quickstart：<https://hungpham2511.github.io/toppra/quickstart.html>
- ARX5 SDK：<https://github.com/real-stanford/arx5-sdk>
- ARX5 配置：<https://github.com/real-stanford/arx5-sdk/blob/main/include/app/config.h>

## 10. Whole-chunk TOPPRA 中 `v`、`vel_limit`、`acc_limit` 的作用

### 10.1 当前 benchmark 不支持“TOPPRA 没有工作”的判断

whole-chunk 结果为：

| 配置 | dense steps |
|---|---:|
| `v1 vel5 acc2` | 3622 |
| `v1 vel5 acc5` | 2298 |
| `v1 vel5 acc8` | 1822 |
| `v1 vel2 acc8` | 1962 |
| `v1 vel3.5 acc8` | 1826 |
| `v2 vel5 acc8` | 1825 |
| `v3 vel5 acc8` | 1814 |

对于从零速出发、零速结束、主要受加速度限制的轨迹，执行时间近似满足：

```text
T ∝ 1 / sqrt(acc_limit)
```

现有结果：

```text
T(acc=2) / T(acc=8) = 3622 / 1822 = 1.988
理论值 sqrt(8/2)                 = 2.000

T(acc=5) / T(acc=8) = 2298 / 1822 = 1.261
理论值 sqrt(8/5)                 = 1.265
```

两组数据几乎精确符合加速度受限轨迹的理论比例。这说明 TOPPRA 正在正确响应 `acc_limit`，当前轨迹大部分
时间处于 acceleration-limited，而不是 velocity-limited。

`vel_limit=2` 相比 `vel_limit=5` 仍把时间从 1822 增加到 1962，约增加 7.7%，所以 velocity constraint
并非完全无效；只是当 `vel_limit` 从 3.5 增加到 5 时，轨迹在达到速度上限前就需要减速，速度约束不活跃，
继续增大它不会缩短时间。这是约束优化的正常现象，不是 TOPPRA bug。

### 10.2 为什么 `v` 对 whole-chunk 时间几乎无影响

当前 `v` 的实现是对 action chunk 的 waypoint 做重采样：

```text
原 waypoint: q0, q1, q2, ..., q49
v=2 后:      q0, q2, q4, ..., q48
```

随后 `retime_chunk` 把这些 waypoint 当作**无时间戳的几何路径**，重新拟合 spline 并由 TOPPRA 决定时间。
如果原 action chunk 已经是平滑、密集采样，同一路径删掉一部分中间点后，拟合出的几何曲线、总弧长和
起终点几乎不变。TOPPRA 不知道这些点原来代表 16.7 Hz、也不知道 `v=2` 原意是“两倍播放速度”，因此求得
的物理最短时间基本不变。

所以：

- `v` 在 `fixed_time` 中有明确时间语义：waypoint 数减少，而每点仍 hold 固定步数，总时间缩短；
- `v` 在当前 whole-chunk TOPPRA 中主要改变几何路径分辨率，不是时间缩放变量；
- 先压缩 waypoint、再做无时间几何 TOPPRA，会主动丢掉 `v` 的时间语义。

这不是代码没有把 `v` 传进去，而是变量定义与 path parameterization 的语义不匹配。

### 10.3 为什么不能保证三个独立变量始终都影响时间

TOPPRA 求解的是满足所有约束的最短时间。只有成为 active constraint 的变量才影响最优时间：

- 短路径或频繁转向时通常由 `acc_limit` 主导；
- 足够长的直线/低曲率路径才可能加速到 `vel_limit`，此时速度约束主导一部分时间；
- `vel_limit` 高于该路径可达到的峰值速度后，继续增大没有效果；
- `acc_limit` 高到速度约束先饱和后，继续增大也可能几乎没有效果。

因此不能要求 `vel_limit` 和 `acc_limit` 在每条轨迹、每个状态下都独立且显著地改变时间。这不是优化器
缺陷，而是两个不等式约束存在 active/inactive 区域。若 DQN 同时控制多个经常不活跃或互相替代的变量，
还会产生动作不可辨识：多个 action bin 得到几乎相同的执行时间和成功率，学习信号很弱。

### 10.4 如果要求 `v` 真正控制 TOPPRA 执行速度

不能继续把 `v` 只定义为 waypoint 压缩比。可选定义如下。

#### 定义一：对 TOPPRA 结果做统一时间缩放

设 TOPPRA 在基础约束下得到 `q(t)`，选择速度倍率 `v_time` 后：

```text
q_new(t)     = q(v_time * t)
qdot_new     = v_time     * qdot
qddot_new    = v_time^2   * qddot
T_new        = T / v_time
```

但 `v_time > 1` 只能在缩放后仍满足关节约束时使用：

```text
v_time <= min_j(vel_limit_j / max|qdot_j|)
v_time <= min_j(sqrt(acc_limit_j / max|qddot_j|))
```

如果原 TOPPRA 已经在同一组 `vel_limit/acc_limit` 下求得时间最优，至少一个约束通常已经活跃，继续令
`v_time > 1` 会违反约束。因此该变量更适合在“先按硬件最大约束生成最快轨迹，再允许 RL 选择
`v_time ∈ (0,1]` 主动减速”的架构中使用。

#### 定义二：用一个 speed scale 同时缩放速度和加速度约束

令 RL 输出无量纲 `speed_scale = lambda`：

```text
vel_effective = lambda   * vel_hardware
acc_effective = lambda^2 * acc_hardware
```

然后重新运行 TOPPRA。这样在相似路径上，轨迹时间更接近按 `1/lambda` 缩放，变量对速度的作用清晰，
同时保持 `qdot`、`qddot` 的物理一致性。

该定义下再同时让 RL 独立选择 `vel_limit`、`acc_limit` 会存在明显冗余。更合理的三变量语义可以是：

1. `progress/horizon`：本次执行多少 path/action，控制闭环频率而不是物理速度；
2. `speed_scale`：统一控制轨迹快慢；
3. `contact/acc_scale`：接触阶段额外限制加速度或力矩，控制动作柔和度。

#### 定义三：保留独立 `vel_limit` 和 `acc_limit`，删除 whole-chunk 的 `v`

如果研究目标就是比较不同关节速度/加速度约束下的时间最优轨迹，那么 `vel_limit` 与 `acc_limit` 已经完整
定义了 TOPPRA 的速度。此时 whole-chunk 中的 waypoint 压缩 `v` 没有独立的物理时间含义，应该只作为
路径降采样/计算量参数固定，而不应作为 RL 速度动作。

### 10.5 推荐决策

1. 当前数据首先解释为 acceleration-limited，不判定 TOPPRA 存在 bug。
2. 在修改动作空间前，增加轨迹诊断：每次记录每关节峰值 `qdot/qddot`、约束利用率、TOPPRA return code、
   path length 和 duration，明确哪一个 constraint active。
3. whole-chunk 若要保留三个可学习变量，推荐重新定义为
   `progress/horizon + speed_scale + contact_scale`，不要继续把 waypoint 压缩比 `v` 当作 TOPPRA 时间倍率。
4. 若必须保留当前变量名，则至少把 `v` 改成 `time_scale/speed_scale` 的语义；`vel_limit/acc_limit` 作为
   硬上限，`v` 只允许在可行范围内减速，不能要求它突破 TOPPRA 已求出的约束下最短时间。
5. `vel_limit` grid 应根据真实轨迹的峰值速度分布选择。现有结果表明 3.5 和 5 大多处于 inactive 区域；
   若要在当前任务中提供学习信号，应把档位集中到实际峰值附近，而不是继续提高上界。

## 11. 保留 FixedTime 的 `v` 加速语义，同时加入 TOPPRA 约束

### 11.1 FixedTime 中 `v` 为什么有效

`reconstruct_chunk(v)` 从原始 action 序列按索引 `0, v, 2v, ...` 取样。FixedTime 对每个保留 waypoint
仍执行固定的 `hold_steps=H`，因此：

```text
M(v) ≈ N / v
T_fixed(v) = M(v) * H / control_hz ≈ T_fixed(1) / v
```

这等价于要求机器人用更短时间走过近似相同的几何路径。它自然会把速度放大约 `v` 倍、加速度放大约
`v²` 倍，但当前 FixedTime 不检查这些放大后的参考是否超出关节约束。

Whole-chunk TOPPRA 则丢弃原 action 的时间戳和 `hold_steps`，只保留几何路径并重新求约束下的最短时间：

```text
T_toppra = minimum feasible duration(path, vel_limit, acc_limit)
```

因此同一条路径无论先以 `v=1` 还是 `v=3` 降采样，只要几何形状近似不变，TOPPRA 都会得到近似相同的
`T_toppra`。FixedTime 的 `v` 加速优势不是来自路径变短，而是来自**期望执行时间按 `1/v` 缩短**。

### 11.2 不推荐通过故意缩短 spline 长度恢复 `v` 的影响

压缩 waypoint 后选择更激进的 spline、让曲线切角或偏离原 waypoint，确实可能降低几何弧长并缩短
TOPPRA 时间，但它改变的是机器人运动路径：

- 可能跳过抓取前对准、闭合夹爪、接触保持等关键 waypoint；
- spline 切角可能碰撞物体或工作台；
- 路径更短不代表曲率更小，较大的 `q''(s)` 反而会让 acceleration constraint 更严格；
- 与 FixedTime baseline 的比较不再只比较执行速度，而是同时比较不同空间路径。

可以把“路径简化/shortcut tolerance”作为独立研究变量，但不应把它当作 FixedTime `v` 的等价实现。

### 11.3 推荐组合：期望时长 + TOPPRA 可行时长下界

保留 FixedTime 的 `v` 定义，用它计算期望时长：

```text
T_desired(v) = M(v) * hold_steps / control_hz
```

几何路径原则上仍由完整 action chunk 构造；如果为了计算量做 waypoint 简化，应使用固定且经过误差验证的
简化规则，不让 `v` 同时改变时间和空间路径。这样实验中 `v` 的含义才保持单一。

对平滑几何路径运行 TOPPRA，得到硬约束下最快轨迹及最短时长：

```text
trajectory_fast, T_min = TOPPRA(path, vel_limit, acc_limit)
```

最终执行时长取：

```text
T_exec = max(T_desired(v), T_min)
```

- 当 `T_desired(v) >= T_min`：当前 `v` 在约束内可行，把 TOPPRA 最快轨迹统一放慢到
  `T_desired(v)`，因此 `v` 保持与 FixedTime 一样的时间控制效果；
- 当 `T_desired(v) < T_min`：FixedTime 要求的速度会违反约束，执行时钳到 `T_min`；此后继续增大
  `v` 不再加速，这是安全约束应有的饱和行为；
- `vel_limit/acc_limit` 决定 `T_min`，从而决定 `v` 最多能加速到什么程度。

若 TOPPRA 最快轨迹为 `q_fast(t)`，且 `gamma=T_exec/T_min >= 1`，可通过统一时间拉伸执行：

```text
q_exec(t)      = q_fast(t / gamma)
qdot_exec(t)   = qdot_fast(t / gamma) / gamma
qddot_exec(t)  = qddot_fast(t / gamma) / gamma²
```

时间拉伸只会降低速度和加速度，因此不会破坏 TOPPRA 已满足的上限。

这个组合明确分工：

| 变量 | 物理意义 | 何时影响时间 |
|---|---|---|
| `v` | FixedTime 风格的期望播放倍率 | 未触及约束下界时 |
| `vel_limit` | 关节速度硬/策略上限 | 速度约束成为 active constraint 时 |
| `acc_limit` | 关节加速度硬/策略上限 | 加速度约束成为 active constraint 时 |

无法也不应该保证三个变量在所有状态下都同时显著影响时间；达到约束后 `v` 必须饱和，inactive constraint
也不会改变最优解。

### 11.4 为什么当前 `vel_limit` 弱、`acc_limit` 强

TOPPRA 在固定路径 `q(s)` 上使用：

```text
qdot  = q'(s) * sd
qddot = q''(s) * sd² + q'(s) * sdd
```

速度限制给出路径速度上界：

```text
sd <= min_j(vel_limit_j / |q'_j(s)|)
```

加速度限制不仅限制沿路径加速 `sdd`，还限制 spline 曲率项 `q''(s) * sd²`。当前 whole chunk 具有以下
特征：

1. chunk 起点和终点均为零路径速度；
2. 轨迹由大量较短的关节空间段组成；
3. natural cubic spline 在 waypoint 附近存在非零曲率；
4. 机器人必须在有限路径长度内完成加速和减速。

若路径长度不足以达到速度上限，速度曲线是近似三角形而不是带匀速平台的梯形。以一维近似表示：

```text
peak_velocity_acc_limited ≈ sqrt(acc_effective * path_length)
```

当：

```text
peak_velocity_acc_limited < vel_limit_effective
```

速度上限从 3.5 提高到 5 不会改变结果，因为轨迹根本到不了 3.5；加速度上限则直接决定峰值速度和总时间。
当前 `acc=2/5/8` 的时间几乎精确符合 `1/sqrt(acc)`，已经证明主要处于这个区域。

`vel=2` 比 `vel=5` 慢约 7.7%，说明少量路径位置能够达到 2；`vel=3.5` 与 5 基本相同，说明 3.5 以上
没有 active velocity constraint。natural spline 的曲率项还会进一步强化 acceleration-dominated 特征。

### 11.5 应增加的诊断，而不是先修改 spline

每次 TOPPRA 规划应输出：

```text
T_min
T_desired(v)
T_exec
path_length
max_j,t |qdot_j| / vel_limit_j
max_j,t |qddot_j| / acc_limit_j
每个关节 velocity/acceleration constraint 的最大利用率
达到 >= 0.95 利用率的 active constraint 及持续时间比例
spline 最大 |q''(s)| 和关节位置过冲
```

据此调整变量 grid：

- 如果所有 velocity utilization 都低于 0.6，降低 `vel_limit` grid 才能产生可辨识作用；
- 如果 acceleration utilization 接近 1，说明当前确实由 `acc_limit` 控制；
- 如果 `T_desired(v) < T_min`，该 `v` 已超出物理可行范围，应报告被约束钳制，而不是继续缩短路径作弊；
- 如果路径简化后 `path_length` 明显下降，同时末端误差/碰撞/成功率恶化，应将其识别为空间 shortcut 的
  影响，而不是执行速度收益。

### 11.6 当前推荐决策

1. 几何路径先采用可验证的 `C1` 连续插值，不以“故意缩短路径”为主要加速机制。
2. `v` 保留 FixedTime 的期望时长语义；TOPPRA 负责计算约束下的 `T_min`。
3. 执行时间使用 `max(T_desired(v), T_min)`，既保留 FixedTime 的加速效果，又不会超过速度/加速度约束。
4. `vel_limit` 当前作用弱是因为 inactive，而非接口失效；先记录约束利用率，再把 grid 移到实际峰值附近。
5. 接触时降低 `acc_limit` 或整体 speed scale；不要通过改变 spline 切角来代替接触减速。

## 12. RoboTwin 原生执行、论文 FixedTime 与约束执行的研究定位

### 12.1 当前困境来自比较对象不在同一可行域

目前三个执行器实际解决的是三个不同问题：

| 执行器 | 时间语义 | 边界条件 | 显式速度/加速度约束 |
|---|---|---|---|
| RoboTwin 原生 per-action TOPPRA | 每个 action 单独求最快 | 每段零速起止 | 固定约束约 1 |
| 当前改进 per-action TOPPRA | 每个 action 单独求最快 | 非零标量边界，但转角不连续 | 可调 `vel/acc` |
| 论文式 FixedTime | waypoint 数 × 固定 hold time | 连续插值，无逐点停稳 | 当前没有显式约束 |

RoboTwin 原生执行因为每个 action 都 rest-to-rest，本身就是一个非常慢的 baseline。提高它的约束上限、取消
部分停顿后能够明显加速，但仍然保留“逐段规划”的结构成本，无法自然追上连续 FixedTime。

FixedTime 更快的主要原因不是它求出了更优的受约束轨迹，而是它直接要求机器人在更短时间内走完路径，
超出的速度、加速度和力矩由理想仿真 PD、跟踪误差或饱和吸收。RoboMimic 中论文使用的机器人模型、低层
控制器和 action frequency 与 RoboTwin 不同，因此只能复现其 high-level scheduling 思想，不能假设物理可行域相同。

### 12.2 判断 TOPPRA 是否真的“比 FixedTime 慢”的关键定理

如果满足以下条件：

1. 两个执行器走完全相同的几何路径；
2. 起点和终点边界条件相同；
3. FixedTime 的完整参考轨迹满足同一组 `vel_limit/acc_limit`；
4. TOPPRA 在该路径和约束上正确求得时间最优解；

那么必然有：

```text
T_TOPPRA_min <= T_FixedTime_feasible
```

因为 FixedTime 轨迹本身已经是 TOPPRA 优化问题的一个可行解，时间最优解不可能比它更慢。

因此若实验观察到 `T_TOPPRA > T_FixedTime`，至少有一项不成立：

- FixedTime 参考速度或加速度已经超过 TOPPRA 约束；
- FixedTime 与 TOPPRA 使用了不同路径或不同零速边界；
- FixedTime 的命令超限，但仿真 PD 没有真实跟踪，实际运动被隐式滤慢；
- per-action 两点规划不是同一条连续路径上的全局 TOPPRA；
- TOPPRA 实现、离散 grid、spline 或统计口径存在问题。

这给出了最重要的诊断实验：先生成 FixedTime 的 250 Hz 完整参考 `q_ref`，对其有限差分得到
`qd_ref/qdd_ref`，检查是否满足准备给 TOPPRA 的同一组约束。没有这一步，不能把 FixedTime 的时间作为
受约束算法必须追平的目标。

### 12.3 当前 per-action TOPPRA 不适合作为最终主方法

它可以保留为历史 ablation：

- RoboTwin 原生：两点、零边界速度、固定约束；
- 改进版本：两点、非零边界速度、可调约束；

用于证明“解除原生 stop-and-go 能加速”。但由于段间转角不连续，它不适合作为最终的安全约束执行器，
也不适合承担追平连续 FixedTime 的目标。

“per-action”仍可保留为执行语义：在每个 action/path 边界检查 contact、success 和是否替换后续动作；
规划核则应改为连续 chunk 或滚动前瞻路径。

### 12.4 推荐主线：Constraint-aware FixedTime / TOPPRA safety shield

不要把研究问题定义为“用纯 TOPPRA 替代论文 FixedTime”，而定义为：

> 保留论文 FixedTime 的请求速度和 chunk 压缩机制，用 TOPPRA 计算相同路径在机器人约束下的最短可行
> 时间；只有 FixedTime 请求超限时才做最小必要降速。

流程：

```text
VLA action chunk
  → 构造连续且经过验证的几何路径
  → v 给出论文 FixedTime 的 requested duration T_desired(v)
  → 固定硬件 vel/acc 上限下 TOPPRA 给出 T_min
  → T_exec = max(T_desired, T_min)
  → 接触/高风险状态额外降低 requested speed 或收紧 soft acc/torque limit
  → 按 action 边界执行、检查并可中断
```

这不是放弃 FixedTime 的优势，而是给它增加约束 shield：

- 安全区域内，行为和论文 FixedTime 一致，`v` 能直接加速；
- 请求速度超过物理可行域时，只降到最接近请求的可行速度；
- 相比统一保守降速，它仍是约束下最快；
- 相比无约束 FixedTime，它能报告“请求被哪个关节、哪个约束钳制”。

### 12.5 速度模块动作空间需要区分“控制量”和“硬约束”

当前同时让 DQN 选择 `v`、`vel_limit`、`acc_limit`，会混淆：

- `v` 是任务策略的 requested speed；
- 机器人最大关节速度、最大力矩是硬件 hard limit，不应由 RL 放大；
- 接触阶段的较低速度/加速度是 policy soft limit；
- 多个 inactive/redundant limit 会让 DQN 得到相同 transition 和 reward，动作难以辨识。

更清晰的变量设计：

```text
action 1: requested_speed / v       # 希望多快
action 2: executed_horizon          # 本次执行多少路径，控制闭环频率
action 3: contact_soft_scale        # 接触时的 acc/torque 柔和度

fixed config: hardware_vel_limit
fixed config: hardware_acc/torque_limit
```

如果研究必须保留三个原变量，也应规定：

- `vel_limit <= hardware_vel_limit`；
- `acc_limit <= calibrated_safe_acc_limit`；
- `v` 决定请求时长；
- TOPPRA 只在请求超限时钳制；
- 记录每个变量是否 active，避免把无效档位交给 DQN 学习。

### 12.6 建议实验矩阵

| 方法 | 目的 |
|---|---|
| RoboTwin native per-action TOPPRA | 原生环境基线，展示 stop-and-go 成本 |
| Paper-style FixedTime | 论文执行语义与速度上界，但允许报告约束违反 |
| FixedTime + uniform global slowdown | 简单安全基线 |
| Constraint-aware FixedTime + TOPPRA shield | 推荐主方法 |
| Whole-chunk fastest TOPPRA | 相同路径/约束下的速度理论下界 |
| 原非零边界 per-action TOPPRA | 结构性问题 ablation，不作为最终方法 |

每个方法同时报告：

```text
success rate
execution time
reference vel/acc violation rate
actual tracking error
torque saturation/contact impulse
requested v 被约束钳制的比例
```

论文式 FixedTime 可以保留为主要 baseline，但结果应分成两层解释：

1. unconstrained task performance：是否复现论文的速度收益；
2. constraint-matched performance：在相同机器人约束下，推荐方法能否比简单 FixedTime 降速更快、成功率更高。

### 12.7 当前研究决策建议

1. 不再要求有结构缺陷的独立 per-action TOPPRA 追平无约束 FixedTime。
2. 保留 RoboTwin 原生与当前 per-action 改进作为 ablation，说明现有环境为何慢及非零拼接为何不稳。
3. 以论文 FixedTime 为请求调度 baseline，以 whole-chunk/rolling TOPPRA 作为约束 shield，而不是替代其 `v` 语义。
4. 首先运行相同路径、相同边界、相同约束的可行性检查；若 FixedTime 本身满足约束，TOPPRA 却更慢，
   才进入实现 bug 调试。
5. 最终贡献应描述为“状态/接触感知的约束速度调度”，而不是单纯“把 RoboTwin 原生 TOPPRA 调快”。
