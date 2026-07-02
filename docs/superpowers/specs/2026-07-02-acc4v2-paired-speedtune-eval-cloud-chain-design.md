# Whole-chunk `4V²`、严格配对 SpeedTune Eval 与云端训练链路设计

- 日期：2026-07-02
- 状态：已确认，进入实施

## 1. 目标

1. whole-chunk TOPPRA 使用 `acc_limit = 4 * vel_limit ** 2`，不设置额外加速度上限。
2. 用 50 条专家轨迹、7 个速度档重放 350 次 whole-chunk，验证新执行器；该测试不使用 DQN 或 k-skip。
3. 云端从头训练 fixed-time 与 whole-chunk 两个单-head Rainbow-DQN。
4. 训练后用相同 30 个 seed 做严格配对 eval：冻结 pi0.5 产生 action chunk 和 suffix feature，对应 DQN 选择速度，执行各自 k-skip 后重新推理。
5. 完善字段传输、奖励门、checkpoint 兼容性校验、计时、错误处理与进程清理。

## 2. 两阶段测试边界

### 2.1 本地专家执行器验证

本地回放只回答 `4V²` 是否改善 whole-chunk 的速度、任务成功率和巡航比例：

```text
expert actions -> full 50-action chunk -> whole-chunk TOPPRA(V, 4V²)
```

不使用 DQN、不使用 k-skip。结果写入独立目录，且每条 summary/NPZ 写
`acc_rule="4*vel_limit^2"`，resume 必须校验该字段，不能复用旧 `6*vel_limit` 轨迹。

### 2.2 云端训练与策略 eval

训练和 eval 的真实闭环为：

```text
pi0.5(obs, deterministic noise key)
  -> action_chunk + suffix_feature
  -> backend-specific DQN
  -> fixed-time: v, k_skip=10
     whole-chunk: vel_limit, k_skip=20, acc=4V²
  -> new observation
  -> next pi0.5 inference
```

两个 DQN 均为单 head `(7,)`，但 support 不同，因此必须从头分别训练并通过 metadata 防止 checkpoint 混用。

## 3. Fixed-time 奖励正确性

RoboTwin streaming 执行器计算每段规划跟踪速度，任一关节 `abs(qvel)>4 rad/s` 即设置
`fixed_time_speed_violation=True`。字段必须经过：

```text
Base_Task info
 -> RoboTwinEnv.step_chunk
 -> websocket step_chunk response
 -> EnvClientWrapper
 -> train_speedtune_async pending episode
```

episode 结束后统一计算：

```text
reward_success = task_success and not fixed_time_speed_violation
reward_t = selected_raw_speed_t ** 2 if reward_success else 0
```

之后才写入 n-step replay buffer。whole-chunk 不应用 fixed-time 超速门。

## 4. Checkpoint contract

每个 checkpoint 除 `q_net/target/` 外保存 `speedtune_metadata.json`：

- `exec_backend`；
- `head_sizes`；
- `n_atoms`；
- `support=[v_min,v_max]`；
- `k_skip`；
- `speed_grid`；
- `acc_rule`（whole-chunk 为 `4*vel_limit^2`，fixed-time 为 `null`）。

eval 恢复前比较运行时 contract 与 metadata。由于 fixed-time 和 whole-chunk 参数形状均为
`(7,)`，metadata 不匹配必须报错，不能依赖 Orbax shape 检查。

## 5. 严格配对 eval

### 5.1 场景与 pi0.5 随机性

- 默认 `n_episodes=30`；
- 两个 env 在正式 rollout 前显式 `reseed(seed)`；
- episode `e` 使用同一 simulator seed；
- pi0.5 noise key 由 `(global_seed, episode, decision_step)` 纯函数生成，不能使用跨 episode 顺序消费的 RNG；
- 每条 episode 记录 requested seed、实际 backend、k-skip、support 和 acc rule。

### 5.2 成功与速度对比

同时报告：

- `task_success`；
- `reward_success/safe_success`；
- fixed-time 超速率；
- whole-chunk fallback 与巡航比例。

主速度指标只使用相同 episode index 且双方 `task_success=True` 的交集。对每个共同成功 pair 计算：

```text
physics_speedup_i = fixed_dense_steps_i / chunk_dense_steps_i
wall_speedup_i = fixed_e2e_wall_s_i / chunk_e2e_wall_s_i
```

报告 pair 数、均值、中位数、p25/p75，以及两个后端各自成功率。不能分别平均两组不同的成功样本后相除。

### 5.3 时间分解

物理执行时间是主要指标：

```text
physics_time_s = total_dense_steps / 250
```

部署指标记录：

- pi0.5 preprocess+forward+unnormalize wall time；
- DQN action selection wall time；
- env `step_chunk` RPC wall time；
- episode decision-loop end-to-end wall time。

compare 默认不录视频，避免 ffmpeg 改变 wall time；视频可通过 flag 单独开启作诊断。

## 6. 云端脚本

`scripts/run_speedtune_train_eval.sh` 默认：

```text
BACKENDS="fixed_time chunk_toppra"
N_EPISODES=30
```

仍允许显式 `BACKENDS` 选择 1–2 个合法后端。脚本启动前检查根目录、配置、VLA checkpoint、
conda env、GPU 数和端口占用；server 通过进程状态、端口和 readiness 日志确认后再启动训练。

server、train、eval 均进入独立进程组并登记 PID/PGID。任何训练失败、eval 失败、Ctrl-C、
SIGTERM 或正常 EXIT 均先发送 TERM，等待有限时间，再对残留进程组发送 KILL。退出摘要打印阶段、
返回码、server/train/eval 日志位置和 checkpoint 发现结果。

提供 `DRY_RUN=1`：执行参数解析、preflight 和命令展开，但不启动 GPU 进程，用于本地验证默认 backend、
端口/GPU 映射、config override、checkpoint-to-eval 参数及 cleanup 注册。

## 7. 验证

1. TDD 验证 `acc=4V²`、server 字段传输、episode reward gate、metadata mismatch、配对交集统计和确定性 noise key。
2. 350 条 whole-chunk 专家回放全部可读，trace 行数等于 dense steps，无错误/fallback 静默丢失。
3. shell `bash -n`、dry-run 默认/override/非法 backend/模拟中断清理测试通过。
4. 在无云端 VLA checkpoint 与四卡环境时不伪造完整训练成功；本地验收只证明链路静态和 mock/dry-run 正确。

## 8. 非目标

- 不把 pi0.5 改成独立网络服务；训练进程各自加载冻结 pi0.5，compare eval 进程加载一份并顺序复用。
- 不修改 per-action TOPPRA 的动作空间和 reward。
- 本轮不在线调整加速度；whole-chunk 固定使用 `4V²`。
