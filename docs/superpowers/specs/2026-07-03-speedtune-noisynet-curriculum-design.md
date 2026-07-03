# SpeedTune NoisyNet 与速度档位课程设计

- 日期：2026-07-03
- 状态：待用户复核

## 1. 目标

在现有 branching C51 Rainbow-DQN 中加入 NoisyNet 探索，并让 fixed-time 与 whole-chunk
都使用单调速度档位课程。目标是避免初期均匀随机高速导致整集奖励为零，同时保留策略根据 VLA
suffix feature 在自由运动状态选择高速、在接触前状态直接切换到低速的能力。

本轮不修改 reward 公式、不增加速度平滑损失、不修改 k-skip，也不实现 whole-chunk 内部的分段
速度约束。网络仍然每个 decision 为当前 chunk 输出一个速度档位。

## 2. NoisyNet 网络

新增 factorized Gaussian `NoisyDense`，替换 dueling C51 的 value 与 advantage 输出层。共享 suffix
feature MLP 保持确定性，动作差异主要由 noisy advantage 流产生。

参数化遵循：

```text
W = mu_W + sigma_W * epsilon_W
b = mu_b + sigma_b * epsilon_b
epsilon_W = f(epsilon_in) outer f(epsilon_out)
f(x) = sign(x) * sqrt(abs(x))
```

默认 `sigma0=0.5`。online、target 和当前 batch loss 使用独立 noise key；每次 learner 梯度更新重新
采样噪声。

rollout 的参数噪声在 episode 开始时采样一次，并在该 episode 内保持不变。固定噪声并不约束动作
连续性：`Q_noisy(s_t, a)` 仍随状态变化，可在相邻 decision 直接从最高档跳到最低档。它只减少与状态
无关的逐 decision 随机抖动。

eval 使用 `use_noise=False`，仅使用 `mu` 参数确定性 greedy 推理。

## 3. 探索策略

NoisyNet替代当前线性 epsilon-greedy：

- `epsilon_start=epsilon_end=0`；
- 不再进行均匀随机档位替换；
- 训练动作由 noisy Q greedy 产生；
- eval由 deterministic Q greedy 产生。

保留 learner 的 epsilon 字段仅用于旧接口兼容，但默认训练配置关闭 epsilon。checkpoint contract
显式记录 `noisy_net=true` 与 `noisy_sigma0`，旧非 NoisyNet checkpoint 不兼容，必须重新训练。

## 4. 速度档位课程

课程仅用于单速度head的 `fixed_time` 与 `chunk_toppra`。两者速度grid均为：

```text
[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
```

规则：

1. 初始只开放索引0，即速度1.0；
2. 统计最近20个完整episode的 `reward_success`；
3. 窗口成功率达到70%后只开放下一档；
4. 解锁后清空窗口，重新累计20个episode；
5. 未达标时使用滑动窗口，每完成一个episode重新判断；
6. 档位只解锁、不回锁；
7. 已开放范围内所有低速档始终可选，允许最高档直接跳回1.0。

`reward_success` 的后端语义保持现有定义：

```text
fixed_time: task_success and not fixed_time_speed_violation
chunk_toppra: task_success
```

课程mask同时作用于 noisy greedy action 和任何兼容 epsilon 分支，禁止选择未开放档位。C51
double-DQN计算 next action 时也应用当时已开放范围，避免 bootstrap 到不可执行动作。

## 5. 训练时序

每个训练episode：

```text
reset NoisyNet rollout noise
while episode not done:
    suffix feature -> noisy Q -> masked greedy speed -> execute chunk
episode done:
    compute reward_success and relabel replay transitions
    update curriculum window; possibly unlock one bin
    run 6 update groups * 20 gradient updates
```

达到40000个环境decision时若episode尚未完成，该截断episode仍按失败写入 replay，但不计入课程的
20集窗口，也不触发档位解锁。

同步与异步trainer使用完全相同的 NoisyNet、mask与课程状态机。异步模式的课程只由actor主线程在
完整episode结束时更新；learner线程只消费已写入 replay 的episode update token。

## 6. 状态、奖励与接触前减速边界

NoisyNet和课程不会添加速度惯性。策略能否提前减速取决于 suffix feature 是否编码即将发生的接触，
以及接触是否跨越当前decision边界。训练日志继续保存速度、接触和decision时间序列，用于验证：

- 自由运动阶段是否逐步选择已开放的高档；
- 接触前一个decision是否直接切换到低档；
- fixed-time 超速是否集中在特定档位和状态；
- whole-chunk `k_skip=20` 是否导致接触发生在单个chunk内部。

单一 `vel_limit` 无法在同一个 whole-chunk TOPPRA内部临近接触时改变速度；本轮验收仅要求在包含
即将接触动作的chunk开始前选择低速。若 suffix feature无法预测下一chunk接触，后续单独评估可部署的
contact-risk feature或chunk内分段规划，不在本轮混入。

## 7. 状态持久化与日志

每个checkpoint保存：

- NoisyNet启用状态和`sigma0`；
- 当前最高开放档位及对应速度；
- 当前课程窗口的布尔结果；
- 每个档位的累计选择次数；
- 每次解锁发生的episode和decision计数。

W&B增加：

- `curriculum/max_unlocked_idx`、`curriculum/max_unlocked_speed`；
- `curriculum/window_success_rate`、`curriculum/window_size`；
- `curriculum/unlock_count`；
- 各速度档位选择率；
- NoisyNet sigma均值或范数。

eval读取新checkpoint contract，但不使用训练课程mask；全部7档开放，关闭noise后确定性选择。

## 8. 测试

1. `NoisyDense`在关闭noise时重复调用完全一致；相同key一致，不同key输出不同；sigma参数可获得梯度。
2. Noisy Rainbow输出shape、C51概率归一化、double-DQN更新和bandit可学习性保持正确。
3. action mask在大量不同noise key下都不能越过当前开放档位，且允许在开放范围内改变动作。
4. 课程在19集成功时不解锁，在20集且成功率70%时解锁一档并清空窗口；失败窗口滑动；最高档不越界。
5. fixed-time使用安全成功，whole-chunk使用任务成功；截断episode不推进课程。
6. 同步与异步trainer都只在episode边界重置rollout noise并更新课程。
7. checkpoint metadata拒绝旧非NoisyNet checkpoint，eval确定性恢复新checkpoint。

## 9. 非目标

- 不增加actor网络；动作仍由DQN直接选择。
- 不使用epsilon均匀随机探索。
- 不因NoisyNet或课程限制相邻decision的速度差。
- 不修改fixed-time超速阈值、reward的alpha/beta或失败episode零奖励规则。
- 不为per-action TOPPRA的三个head设计多变量课程。
