# Whole-chunk TOPPRA 单速度控制、专家回放 benchmark 与 SpeedTune 适配设计

- 日期：2026-07-02
- 维护人：polar823
- 分支：dbpo-robotwin
- 状态：设计已确认，待维护人复核后编写实施计划

## 1. 背景

whole-chunk TOPPRA 当前把 `v`、`vel_limit`、`acc_limit` 作为三个相互独立的
SpeedTune 动作变量。已有轨迹表明速度和加速度约束强耦合：同一条路径在
`vel_limit=3` 时，`acc_limit=3` 用时 12.70 s 且峰值速度仅 2.55 rad/s；
`acc_limit=8` 用时 7.85 s 且峰值速度达到 2.96 rad/s。与此同时，whole-chunk
中的 `reconstruct_chunk(v)` 只改变路径采样密度，基本不改变路径和 TOPPRA 时长，
因此 `v` 是无效自由度。

本设计把 whole-chunk 的控制量收敛为单一 `vel_limit`，先用
`acc_limit = 6 * vel_limit` 的简单规则验证加速效果，再把 SpeedTune 的
whole-chunk 动作空间改成单 head。fixed-time 保持现有执行方式，作为对照后端。

## 2. 目标与非目标

### 2.1 目标

1. 修改生产 whole-chunk 执行顺序为：截取前 `execution_steps` 个 action，再做
   spline 和 TOPPRA。
2. whole-chunk 只接收 `vel_limit`；执行层内部使用
   `acc_limit = 6 * vel_limit`，不设置额外加速度上限。
3. 用 50 条专家数据分别测试 7 个 whole-chunk 速度档与 7 个 fixed-time 速度档，
   记录成功率、执行时间、物理步数以及规划/实测速加速度轨迹。
4. benchmark 验证后，把 SpeedTune `chunk_toppra` 改成单 `vel_limit` head，继续使用
   Rainbow-DQN。
5. fixed-time 与 whole-chunk 使用一致的 raw-speed、success-gated reward 参数。

### 2.2 非目标

- 不修改 fixed-time 的插值与 hold 执行语义。
- 不修改 per-action TOPPRA 的动作空间、reward 或执行器。
- 第一轮不实现依路径长度搜索最小 `acc_limit`；固定尝试倍率 6。
- 专家回放 benchmark 不使用 k-skip，避免跳过专家动作造成任务失败。
- 不复用旧 three-head whole-chunk checkpoint。

## 3. 生产 whole-chunk 执行流程

### 3.1 接口

whole-chunk 生产接口只消费速度限制和执行窗口：

```text
take_chunk_action(
    action_chunk,
    vel_limit,
    execution_steps=None,
    video_save_freq=-1,
)
```

`execution_steps=None` 表示执行完整输入 chunk；正整数表示只执行输入 chunk 的前
`execution_steps` 帧。client 端把训练 config 中的 whole-chunk k-skip 映射到该参数。

### 3.2 数据流

```text
VLA action_chunk
    -> requested = len(chunk) if execution_steps is None else execution_steps
    -> N = min(len(chunk), requested, remaining action budget)
    -> action_chunk = action_chunk[:N]
    -> 拆分双臂和夹爪
    -> 在臂路径首部加入机器人当前关节状态
    -> natural cubic spline
    -> acc_limit = 6 * vel_limit
    -> whole-chunk TOPPRA，sd_start = sd_end = 0
    -> 250 Hz 密集采样并执行
```

whole-chunk 删除 `reconstruct_chunk(v)`，不再消费外部 `v` 或 `acc_limit`。fixed-time
继续使用 `reconstruct_chunk(v)`。

### 3.3 执行统计

whole-chunk `info` 除现有字段外，增加：

- `vel_limit`：调用方选择的速度限制；
- `acc_limit`：内部派生值 `6 * vel_limit`；
- `planned_cruise_fraction`：规划轨迹中满足
  `max_j(abs(planned_qvel_j)) >= 0.95 * vel_limit` 的采样时间占比；
- `execution_steps`：本次实际进入 spline/TOPPRA 的 action 数。

退化路径、非有限轨迹或 TOPPRA 求解失败仍返回明确 fallback，不静默切换为另一种
运动语义。

## 4. 专家回放 benchmark

### 4.1 数据与场景恢复

数据位于：

```text
/home/xukainan/RoboTwin/data/stack_blocks_two/demo_clean/data/episode0.hdf5
...
/home/xukainan/RoboTwin/data/stack_blocks_two/demo_clean/data/episode49.hdf5
```

每个 `episode_i.hdf5` 与 `seed.txt` 中第 i 个 seed 配对。HDF5 包含动作与观测，
但不包含完整 simulator/object 初始状态，所以必须用对应 seed 调用 `setup_demo()`
恢复录制时布局。seed 不参与回放时的额外随机化。

### 4.2 测试矩阵

| 后端 | 档位 | 每档 episode | k-skip |
|---|---|---:|---|
| whole-chunk TOPPRA | `vel_limit = {1, 1.5, 2, 2.5, 3, 3.5, 4}`，`acc=6*vel` | 50 | None |
| fixed-time | `v = {1, 1.5, 2, 2.5, 3, 3.5, 4}` | 50 | None |

总计 14 * 50 = 700 次回放。每 50 个专家 action 组成一个输入 chunk；末尾不足
50 但不少于 2 的 chunk 继续执行。fixed-time 使用 `hold_steps=15`。两后端使用相同
episode/seed 顺序和现有 ARX5 force limits。

### 4.3 逐 episode 产物

所有 700 次回放保存压缩 NPZ：

```text
test/exec_dynamics/out/wholechunk_vel_fixedtime/
  traces/<backend>/<setting>/episode_<N>.npz
```

每份 NPZ 包含：

- 实测 `qvel`、有限差分 `qacc`；
- 规划/drive-target `qvel`、有限差分 `qacc`；
- 250 Hz 时间轴；
- whole-chunk 每次 TOPPRA 调用的规划巡航时间比例。

逐 episode CSV 至少记录：

- `backend`、`setting`、`episode`、`seed`；
- `success`；
- `planned_duration_s`；
- `sim_duration_s = dense_steps / 250`；
- `wall_time_s`；
- `dense_steps`；
- fallback 次数与原因；
- whole-chunk 巡航比例的均值与最小值。

### 4.4 汇总与图表

- 每档任务成功率；
- 成功 episode 与全部 episode 的平均/中位执行时间和物理步数；
- 成功率、执行时间、物理步数随速度档变化的对比图；
- 规划与实测 qvel/qacc 分位数图；
- whole-chunk 规划巡航比例图，用于检查倍率 6 是否让规划速度充分触及上限；
- 每个档位只绘制 `episode0` 的完整时序图，其余 episode 保留 NPZ。

benchmark 在本机 RoboTwin conda 环境运行。脚本支持 `--quick` 小样本冒烟和正式
50-episode 模式；正式运行前先用 quick 模式检查输出结构、finite 值和 fallback。

## 5. SpeedTune whole-chunk 适配

本节在 benchmark 结果确认倍率 6 可接受后实施。

### 5.1 动作空间

`chunk_toppra` 改为单 head：

```text
head_sizes = (7,)
vel_limit grid = (1, 1.5, 2, 2.5, 3, 3.5, 4)
```

删除 whole-chunk 的 `v` 和 `acc_limit` head。冻结 VLA suffix feature、Rainbow-DQN
网络实现、PER 和 n-step replay 的基本结构不变；buffer 的 `n_heads` 自动变为 1。
旧 three-head checkpoint 不兼容，新结构从头训练。

### 5.2 Backend-specific k-skip

```text
fixed_time.k_skip = 10
chunk_toppra.k_skip = 20
```

fixed-time 需要满足 `k_skip * v_max <= VLA action horizon`。当前
`10 * 4 = 40 <= 50`，因此任一速度档重构后都至少有 10 帧供执行。whole-chunk
不做 `v` 重构，可以独立使用 20 帧执行窗口，在闭环重规划粒度与单次 TOPPRA 路径
长度之间折中。

### 5.3 Episode-end reward relabeling

rollout 期间先把 transition 放入 episode-local pending list，不立即写 replay。
episode 结束后计算：

```text
reward_success = task_success and not fixed_time_speed_violation
```

然后对本 episode 每个 SpeedTune 决策回填：

```text
x_t = v_t                    # fixed-time
x_t = vel_limit_t            # whole-chunk
r_t = x_t ** beta             if reward_success else 0
alpha = 1
beta = 2
```

不做 `(x-1)/3` 或其它 0-1 归一化，也不额外叠加 `r_task=1`。两个后端共用固定的
`alpha=1`、`beta=2`。原始动作值对应成功 reward：

| x | reward `x^2` |
|---:|---:|
| 1.0 | 1.00 |
| 1.5 | 2.25 |
| 2.0 | 4.00 |
| 2.5 | 6.25 |
| 3.0 | 9.00 |
| 3.5 | 12.25 |
| 4.0 | 16.00 |

任务失败 episode 的所有 transition reward 均为 0。

### 5.4 Fixed-time 规划超速门

fixed-time 每段已有规划跟踪速度：

```text
qvel_target = (q[i+1] - q[i]) * 250 / hold_steps
```

若 episode 内任一臂关节出现 `abs(qvel_target) > 4 rad/s`，记录
`fixed_time_speed_violation=True`。执行可以继续用于诊断，但即使物理任务最终成功，
该 episode 的所有 SpeedTune reward 仍回填为 0。日志同时保留 `task_success` 与
`reward_success`。

whole-chunk 由 TOPPRA 的 `vel_limit <= 4` 约束，不套用 fixed-time 超速门。

### 5.5 Bellman 更新

奖励在 episode 完成后确定并写入 replay，之后沿用现有 3-step Bellman target：

```text
R_t^(3) = r_t + gamma*r_(t+1) + gamma^2*r_(t+2)
y_t = R_t^(3) + gamma^3*(1-done)*Q_target(s_(t+3), a*)
```

`gamma=0.99`。失败或 fixed-time 超速 episode 的回填 reward 全为 0；成功 episode
按每个决策实际选择的 raw speed 平方回填。

### 5.6 C51 atoms 与 backend-specific support

C51 原子数由 51 增至 100：

```text
n_atoms = 100
```

所有 reward 非负，所以两个后端 `v_min=0`。stack_blocks_two 的 action budget 为
800。

fixed-time 使用 k-skip 10，最多 80 个 SpeedTune 决策：

```text
Q_max = 16 * (1 - 0.99^80) / (1 - 0.99) = 883.963
support = [0, 900]
```

whole-chunk 使用 k-skip 20，最多 40 个 SpeedTune 决策：

```text
Q_max = 16 * (1 - 0.99^40) / (1 - 0.99) = 529.645
support = [0, 550]
```

训练入口根据 `exec_backend` 选择 support，不能让两个独立训练 run 共用同一个
`v_max`。per-action 保持既有 support/config，不纳入本次调整。

## 6. 训练与评估日志

fixed-time 与 whole-chunk 每个决策统一记录：

- 选择的 raw speed 值；
- 派生 `acc_limit`（仅 whole-chunk）；
- `task_success`、`reward_success`；
- `fixed_time_speed_violation`（仅 fixed-time）；
- duration、dense steps、fallback；
- whole-chunk 规划巡航时间比例。

`eval_speedtune.py` 和 compare eval 不得假设 whole-chunk 有 3 个 head；应通过 backend
spec 解码单 head checkpoint，并输出速度档分布、成功率、执行时间与安全门统计。

## 7. 验证

### 7.1 静态与单元验证

- whole-chunk 在 spline 前截断到 `execution_steps`；
- whole-chunk 不调用 `reconstruct_chunk`；
- `acc_limit` 始终等于 `6 * vel_limit`；
- `planned_cruise_fraction` 计算使用规划速度；
- fixed-time 超过 4 rad/s 时设置 violation；
- success episode 按 raw speed 平方回填，failure/violation episode 全部回填 0；
- `chunk_toppra.head_sizes == (7,)`；
- fixed-time 与 whole-chunk 的 k-skip/support 分别为 10/[0,900] 与 20/[0,550]；
- C51 atoms 数量为 100，categorical projection 输出仍归一化且 finite。

### 7.2 本地仿真验证顺序

1. 单 episode、单档位 quick smoke；
2. episode0、14 个档位，检查所有 NPZ/CSV/PNG；
3. 正式运行 700 次回放；
4. 汇总 whole-chunk 与 fixed-time 的成功率、执行时间和物理步数；
5. 检查倍率 6 的规划巡航比例与 fallback；
6. benchmark 结果经维护人确认后启动新的单-head SpeedTune 训练。

## 8. 风险与处理

- `acc_limit=6*vel_limit` 是经验规则，不保证每个短路径窗口都达到 50% 巡航时间；
  benchmark 必须报告真实规划巡航比例，不能把倍率假设当成结论。
- TOPPRA 规划速度满足约束不代表受 force limit 的实测机器人完全跟踪；因此同时保存
  规划与实测 qvel/qacc。
- success-gated episode relabeling 不是论文原始即时 reward；其目的明确是让失败或超速
  episode 得不到任何速度收益。
- backend-specific support 使 fixed-time 与 whole-chunk checkpoint 的 distributional head
  数值语义不同，checkpoint/eval 必须携带 backend 配置，禁止混用。
- 700 次回放时间较长；输出逐 episode 写盘，支持中断后按已有完整文件跳过，避免整批重跑。

## 9. 最终决策记录

- whole-chunk：单 `vel_limit` 控制，`acc_limit=6*vel_limit`，无 `v` 重构；
- whole-chunk 执行顺序：先取前 `execution_steps`，再 spline/TOPPRA；
- benchmark：50 条专家数据、7 档 whole-chunk + 7 档 fixed-time、无 k-skip；
- SpeedTune：whole-chunk 单 head，fixed-time 与 whole-chunk raw speed grid 均为
  `(1,1.5,2,2.5,3,3.5,4)`；
- reward：episode-end success gate，raw speed 平方，`alpha=1`、`beta=2`；
- fixed-time：规划跟踪速度超过 4 rad/s 时整集 reward 置 0；
- k-skip：fixed-time=10，whole-chunk=20；
- C51：100 atoms；fixed-time support `[0,900]`，whole-chunk support `[0,550]`。
