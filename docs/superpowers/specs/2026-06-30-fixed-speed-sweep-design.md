# 固定加速系数 sweep 测试脚本设计

> 日期：2026-06-30　维护人：polar823
> 目标文件：`scripts/bench_fixed_speed.py`（新建，零侵入）

## 1. 目的

在冻结 VLA（pi0.5 drift）**之后**外接一个**固定速度控制系数**的加速模块（替代 SpeedTune 的
branching Rainbow-DQN 动态选档），测量不同固定档位下的任务成功率，以隔离「执行层加速本身对
成功率的影响」。

**加速对象 = VLA 实时推理出的 action chunk**（经 `step_chunk` 的 `v/vel_limit/acc_limit`
压缩+限速执行），位置与 SpeedTune 完全相同，只是把 DQN 选档换成固定常数。**不针对专家数据**：
bench 全程走 `step_chunk`，而自动专家介入（takeover / `play_once` 录带）只在 per-action 的
`step()` 触发（`robotwin_env.py:308`），故**专家介入不可能发生**——成功率纯粹来自「VLA 动作 +
固定加速系数」，不掺任何专家纠正。

## 2. 复用与零侵入

- 复用 `eval_speedtune._build_vla` 加载冻结 VLA（3B，**只加载一次**，所有配置共用）。
- **不加载 DQN、不调 `backend.decode`**：直接构造固定 `speed_params` dict 传给
  `env.step_chunk(real_chunk, speed_params, exec_backend)`（`robotwin_env.py:387-389` 直接读
  `v/vel_limit/acc_limit`）。
- 新建 `scripts/bench_fixed_speed.py`，**不改任何既有文件**。一个 env server 即可，
  `step_chunk` 的 `exec_backend` 参数逐配置切 `fixed_time` / `per_action_toppra`。

## 3. Sweep 配置（去重后 26 点）

| backend | 变量 | 取值 | 其他变量 |
|---|---|---|---|
| `fixed_time` | v | 1, 1.5, 2, 2.5, 3, 3.5, 4 | streaming 不吃 vel/acc，只传 `{"v": x}` |
| `per_action_toppra` | v | 1, 1.5, 2, 2.5, 3, 3.5, 4 | vel_limit=1.0, acc_limit=1.0 |
| `per_action_toppra` | vel_limit | 1, 2, 3, 4 | v=1.0, acc_limit=1.0 |
| `per_action_toppra` | acc_limit | 1, 2, 3, 4, 5, 6, 7, 8, 9 | v=1.0, vel_limit=1.0 |

per_action 基线点 `(v=1, vel=1, acc=1)` 在三条 sweep 重复 → 按 `(backend, v, vel, acc)` 去重跑
一次、三条曲线共用（省 1×50 ep）。

## 4. 同批布局（控制变量，已确认）

每个配置开始前用 **`reseed` op**（`client_robotwin` 新增）重置 server 端 `RoboTwinEnv._ep_count=0`
并 `close_env(clear_cache=True)` 释放上一配置 scene → 下个配置从 seed 0 跑 `0..N-1` 的**同一批
布局**，成功率差异只来自加速系数。**不重建 env**（全程单个 sapien 实例），避免累积 scene 显存
——在 4 卡「同卡 server+bench」紧显存下尤其关键。VLA 只加载一次（与 env 无关）。

> 新增（零侵入，不改 `step/reset` 协议、不动 `env_client.py`）：`RoboTwinEnv.reseed()` +
> `run_robotwin_client` 的 `reseed` op；bench 用底层 `EnvClient._call_operation("reseed", ...)` 调。

## 5. 每配置内循环（复用 `run_backend_episodes` 骨架，去掉 DQN）

```
for ep in 0..N-1:                       # N=50（已确认）
    obs = env.reset()
    ep_success = False
    for step in range(max_decision_steps=400):
        obs_m     = preprocess(obs)
        z         = jax.random.normal(...)         # drift 单步生成
        mean,_,_  = drift_forward(obs_m, z)
        real_chunk= unnormalize(mean, obs_m)        # [H, 14] 绝对 qpos
        _, info   = env.step_chunk(real_chunk, FIXED_speed_params, backend)
        done, success, _, _ = env.get_info_for_step()
        ep_success |= success
        记录 dense_steps / exec_status(topp_fallback)
        if done: break
        obs = env.get_observation()
    n_success += ep_success
success_rate = n_success / N
```

## 6. 执行参数（与训练/SpeedTune 一致，保证可比）

- `k_skip=10`、`stream_hold_steps=15`：取 `speedtune_dqn_config.py` 默认。
- `instruction_type=seen`：取 `robotwin_stack_blocks.py`（与微调/已对齐的 eval 一致）。
- `force_limit`：默认 `30,40,30,15,10,10`（真机 ARX5 τ_max 底座）。**记录 `topp_fallback` 率**
  —— 力矩底座下高 `acc_limit` 可能压不出来，让用户看到哪些档位实际没加上去。
  CLI `--force_limit ""` 可关掉（∞，纯看速度系数、不受力矩天花板）。

## 7. 输出（`output_dir/`）

- `bench_summary.csv` / `.json`：每行
  `backend, sweep_var, v, vel_limit, acc_limit, n_episodes, n_success, success_rate,
  mean_dense_steps, mean_sim_time_s, fallback_rate`。
- 折线图（matplotlib Agg）：
  - `fixed_time_v.png`：success_rate vs v
  - `per_action_v.png` / `per_action_vel.png` / `per_action_acc.png`：success_rate（左轴）+
    mean_dense_steps 执行步数（右轴），直观看「加速 vs 成功率」权衡。
- sweep 默认**不录视频**（省时间）；`--record_video` 可开。

## 8. CLI

```
uv run python scripts/bench_fixed_speed.py \
    --config configs/model/speedtune_dqn_config.py \
    --config_task configs/task/robotwin_stack_blocks.py \
    --n_episodes 50 --seed 0 --max_decision_steps 400 \
    --backends both            # both | fixed_time | per_action \
    --config_subset 0/1        # i/N 分片（并行用，见下）\
    --force_limit <空=继承config默认> \
    --client_host localhost --client_port 8102 \
    --output_dir logs/bench_fixed_speed
```

## 9. 时长与并行

- 26 配置 × 50 ep ≈ **1300 episode**；双臂 stack 每 ep 约 0.5–2 min → **单卡串行约 10–22 小时**。
- 脚本内置 `--config_subset i/N`：把 26 个配置切成 N 片，第 i 片跑一部分。配合多卡多 server
  （不同 `CUDA_VISIBLE_DEVICES` + 不同 `--server_port`/`--client_port`）并行，几小时跑完。
- 各分片输出写各自 `output_dir`，最后人工合并 csv（或加一个 `--merge` 后处理，YAGNI 暂不做）。

## 10. 非目标（YAGNI）

- 不实现 DQN、不读 checkpoint（这是固定系数 baseline，不是 SpeedTune eval）。
- 不做接触时段激进度分析（那是 `eval_speedtune.py` 的职责）。
- 不自动合并分片结果（人工 cat csv 即可）。
