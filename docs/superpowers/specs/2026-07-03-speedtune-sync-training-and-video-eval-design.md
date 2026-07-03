# SpeedTune 同步训练与五集诊断视频设计

- 日期：2026-07-03
- 状态：待用户复核
- 依赖：`2026-07-02-acc4v2-paired-speedtune-eval-cloud-chain-design.md`

## 1. 目标与范围

1. 提交并推送 RoboTwin 中已经完成的 whole-chunk `acc_limit=4*vel_limit**2`、
   `execution_steps` 和 fixed-time `4 rad/s` 超速检测修改。
2. 为 SpeedTune 增加参考 `train_pi_robo.py` 的同步训练入口，同时保留可选异步训练。
3. `scripts/run_speedtune_train_eval.sh` 支持选择 `sync` 或 `async`，默认 `sync`。
4. eval 仍运行严格配对的 30 个 episode；计时 rollout 不录视频，之后重放并保存 5 个诊断 episode。
5. 增加 decision 数量诊断，验证 fixed-time 与 whole-chunk 对有效 action chunk 的消费语义。

本轮不修改 DQN 网络、速度离散档位、reward 公式、`k_skip` 默认值或 pi0.5 模型。

## 2. 训练步与更新语义

严格参考 `train_pi_robo.py`：训练终止由环境交互步数决定，不由梯度更新次数决定。

SpeedTune 中一次训练步定义为一次完整的：

```text
pi0.5 inference -> DQN speed decision -> env.step_chunk(...)
```

因此：

- `max_iters=40000` 表示最多执行 40000 次 `env.step_chunk()`；
- chunk 内部的 250 Hz dense simulation steps 不计入 `max_iters`；
- episode 结束且 replay 已满足预热条件时，触发 6 个 update group；
- 每个 update group 执行 `utd_ratio=20` 次 Rainbow-DQN 梯度更新；
- 梯度更新总数单独记录，但不控制训练结束；
- 训练在第 40000 个 decision 对应的 episode 中途结束时，必须把当前 pending episode
  作为截断 episode 正确写入 replay，不能静默丢弃；截断 episode 不伪造任务成功。

预热条件与 `train_pi_robo.py` 对齐为：至少完成 10 个 episode，且 replay 中 transition 数不少于
batch size。旧的 `learning_starts=1000` decision 门槛不再单独控制同步训练启动。

## 3. 同步训练数据流

新增同步入口，复用现有 VLA、DQN、backend、checkpoint 和 replay 构造逻辑：

```text
rollout one decision
  -> append transition to current episode
  -> if episode done:
       compute task_success and fixed-time speed gate
       relabel every transition reward
       insert complete episode into replay
       if warmup complete:
         run exactly 6 update groups × 20 gradient updates
       publish rollout/training metrics
       reset environment
```

同步模式没有后台 learner 线程；rollout 使用刚更新完成的 learner 参数执行下一 episode。checkpoint
名称继续使用累计梯度更新数，metadata contract 保持不变，并额外保存累计 decision 与 episode 计数，
便于恢复和审计。

## 4. 异步模式

异步入口保留用于实验对比，但修复 `threading.Event` 合并多个 episode 通知的问题。完成 episode
通过计数信号或队列逐个通知 learner；每个通知对应且仅对应 6 个 update group。异步模式仍以
40000 个 rollout decision 停止，退出时等待已经入队的 episode update 完成，再保存最终 checkpoint。

脚本接口：

```text
TRAIN_MODE=sync   # 默认
TRAIN_MODE=async
```

交互运行时允许选择；非交互/云端运行通过环境变量设置。非法值在启动 server 前立即失败。

## 5. fixed-time 与 whole-chunk decision 语义

- fixed-time：DQN 输出压缩倍率 `v`；先从原始 VLA chunk 重构为压缩后的新 chunk，再执行其中最多
  `fixed_time_k_skip=10` 个 action。
- whole-chunk：不重构；直接取原始 VLA chunk 前 `chunk_toppra_k_skip=20` 个 action，再进行 spline
  和 whole-chunk TOPPRA。

`20` 来自 fixed-time 训练结果中约为 `2` 的平均压缩倍率。实现正确时，两种后端完成相同任务所需的
VLA decision 数应处于相近量级，而不是机械地相差两倍。该关系不作为硬失败阈值，因为 DQN 速度选择、
任务失败和轨迹长度仍会造成差异；eval 必须报告：

- 每个 episode 的 `n_decision_steps`；
- 每个 decision 的输入 chunk 长度、截取前长度、实际执行 action 数和 dense simulation steps；
- 双方共同成功 episode 上 decision 数之比的均值、中位数和 p25/p75。

如果 decision 数明显分离，优先检查 fixed-time 重构顺序、whole-chunk `execution_steps=20` 是否真正
传入 RoboTwin，以及服务器是否运行了正确 commit。

## 6. 五集诊断视频

为了不污染端到端 wall-time，严格配对的 30 episode 主 rollout 保持不录视频。主 rollout 完成后，
使用相同 simulator seed 和 `(seed, episode, decision_step)` pi0.5 noise 重新顺序重放，并只为选中的
5 个 episode 开启视频：

1. 优先选择任一后端失败的 episode；
2. 其次选择 fixed-time 触发超速门的 episode；
3. 不足 5 个时，从双方共同成功 episode 按 episode id 补齐；
4. 两个后端使用相同的 5 个 episode id，各保存一份视频；
5. `video_manifest.json` 记录选择原因、episode id、seed、双方成功状态和视频路径。

诊断重放结果不参与 success rate、physics time 或 wall-time 聚合。为保证随机环境序列一致，诊断阶段
从 episode 0 顺序重放到最大的被选 episode，只对选中项调用 `start_video/stop_video`。

## 7. RoboTwin 版本可审计性

训练/eval 日志必须打印 expo-ft 与 RoboTwin 的 commit id。执行器 response 增加可验证字段或启动检查，
至少确认：

- whole-chunk 接受 `execution_steps`；
- whole-chunk 返回 `acc_limit=4*vel_limit**2`；
- fixed-time 返回 `max_planned_qvel` 与 `fixed_time_speed_violation`。

字段缺失时训练/eval 应立即报错，不能继续采用默认 `0/False`，以避免云端只更新 expo-ft 后静默运行旧
RoboTwin。

## 8. 测试与提交边界

采用测试先行：

- 同步循环的 40000 decision 终止、episode reward 回填、6×20 更新触发和末尾截断；
- 异步通知不丢 episode；
- `TRAIN_MODE` 默认值、override、非法值和 dry-run 命令展开；
- 5 个 episode 的确定性选择、同 id 双后端视频及 manifest；
- decision 统计只使用双方共同成功 episode；
- RoboTwin `execution_steps`、`4V²` 和 fixed-time 超速字段。

RoboTwin 仅提交六个执行器及测试文件；expo-ft 仅提交本功能涉及的训练、eval、脚本、配置、测试和文档。
现有 `CLAUDE.md` 修改、benchmark 文件删除、历史结果目录和其他用户改动不纳入提交。
