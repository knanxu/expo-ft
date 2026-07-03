# SpeedTune 速度档位课程设计

- 日期：2026-07-03
- 状态：待用户复核

## 1. 目标

在现有 branching C51 Rainbow-DQN 和 epsilon-greedy 探索中加入速度档位课程。fixed-time 与
whole-chunk 均从最低速度开始，在任务表现达标后逐档开放高速，避免初期随机采到高速导致整个episode
奖励长期为零。

本轮不加入 NoisyNet，不修改Q网络结构、reward公式、k-skip或chunk执行方式，也不增加速度平滑约束。
策略仍可在相邻decision直接从最高档切换到最低档。

## 2. 课程规则

课程用于单速度head的 `fixed_time` 与 `chunk_toppra`。速度grid保持：

```text
[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
```

规则：

1. 初始只开放索引0，即速度1.0；
2. 统计最近20个完整episode的 `reward_success`；
3. 窗口成功率达到70%后只开放下一档；
4. 解锁后清空窗口，重新累计20个episode；
5. 未达标时保留最近20集滑动窗口，每完成一个episode重新判断；
6. 档位只解锁、不回锁；
7. 已开放范围内所有低速档始终可选。

`reward_success` 继续使用当前后端语义：

```text
fixed_time: task_success and not fixed_time_speed_violation
chunk_toppra: task_success
```

达到40000个环境decision时若episode尚未完成，该截断episode按失败写入replay，但不计入课程窗口，
不触发档位解锁。

## 3. 动作选择与Bellman更新

保留当前epsilon调度：

```text
epsilon_start=1.0
epsilon_end=0.05
epsilon_decay_steps=20000 decisions
```

课程mask同时作用于：

- greedy action：未开放档位的期望Q设为负无穷后再argmax；
- epsilon随机action：只在索引`[0, max_unlocked_idx]`内均匀采样；
- double-DQN next action：online Q只能从当前已开放档位选bootstrap action。

初始只有速度1.0，因此即使`epsilon=1.0`也不会随机到高速。新档位开放后由现有epsilon-greedy探索，
不增加额外actor或参数噪声。

## 4. 训练时序

每个完整episode：

```text
rollout with current curriculum mask
episode done
  -> compute task_success / fixed-time speed violation
  -> compute reward_success and relabel replay transitions
  -> retain the max-unlocked index used by this episode for its update block
  -> append reward_success to curriculum window
  -> possibly unlock exactly one speed bin
  -> run 6 update groups * 20 gradient updates using the retained index
```

同步与异步trainer共享同一课程状态机。异步课程只由actor主线程在完整episode结束时更新；learner线程
消费episode token时读取该token记录的开放档位，保证该episode对应的Bellman更新不会使用未来才开放的
动作。

## 5. 接触前减速能力

课程mask只限制当前最高可选档位，不限制相邻decision的速度变化。档位全部开放后，DQN仍可根据VLA
suffix feature在自由运动状态选择4.0，在接触前状态直接选择1.0。

能否提前减速仍取决于：

- suffix feature是否编码即将接触的信息；
- 接触是否发生在下一decision之前；
- episode级成功/失败奖励能否提供足够状态动作对比。

单一 `vel_limit` 仍无法在同一个whole-chunk TOPPRA内部改变速度。本轮只要求在包含即将接触动作的
chunk开始前选择低速，不实现chunk内分段速度约束。

## 6. 状态持久化与日志

checkpoint metadata或相邻课程状态文件保存：

- 当前最高开放档位和速度；
- 当前滑动窗口结果；
- 每档累计选择次数；
- 解锁发生的episode和decision；
- 窗口长度20、阈值0.7等配置。

W&B增加：

- `curriculum/max_unlocked_idx`；
- `curriculum/max_unlocked_speed`；
- `curriculum/window_success_rate`和`window_size`；
- `curriculum/unlock_count`；
- 各速度档位选择次数或比例。

eval读取checkpoint保存的最终开放档位，并在该范围内执行确定性greedy。只有训练期间已经解锁全部档位
时，eval才开放全部7档；不能让未训练、仍被锁定档位的随机Q值参与评价。

## 7. 测试

1. 课程在少于20集时不解锁；最近20集成功率达到70%时只解锁一档并清空窗口。
2. 未达标窗口正确滑动；最高档不会越界；档位不会回锁。
3. 大量epsilon随机采样和greedy采样均不能超过当前开放档位。
4. double-DQN bootstrap不会选择未开放动作。
5. fixed-time窗口使用安全成功，whole-chunk窗口使用任务成功。
6. 截断episode不推进课程。
7. 同步与异步训练都按episode更新课程，异步token携带对应的开放档位。
8. checkpoint保存课程状态，eval恢复Q网络时应用最终开放档位mask。

## 8. 非目标

- 不加入NoisyNet、actor、Boltzmann或参数空间探索。
- 不改变epsilon起止值和衰减步数。
- 不修改失败episode零奖励规则或fixed-time的4 rad/s超速门。
- 不为per-action TOPPRA的三个head设计多变量课程。
