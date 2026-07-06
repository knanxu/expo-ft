# SpeedTune fixed-time vs whole-chunk 云端训练与配对评估

## 前置条件

- 4 张可用 GPU；RoboTwin server 使用 GPU 0/2，两个训练进程使用 GPU 1/3；
- RoboTwin conda 环境；
- 冻结 pi0.5 checkpoint，以及与其匹配的 norm assets；
- fixed-time 与 whole-chunk 必须从头训练，旧 three-head checkpoint 不兼容。

```bash
cd /home/chenlu/expo-ft
export SPEEDTUNE_VLA_CKPT=/absolute/path/to/pi05/params
export SPEEDTUNE_VLA_ASSETS=/absolute/path/to/assets
export SPEEDTUNE_VLA_ASSET_ID=trossen
```

先检查命令展开，不启动 GPU 进程：

```bash
DRY_RUN=1 bash scripts/run_speedtune_train_eval.sh
```

输出必须包含：

```text
backends: fixed_time chunk_toppra
max_iters: 20000
n_episodes: 30
chunk_toppra_k_skip: 40
backend_a=fixed_time ... backend_b=chunk_toppra
```

## 冒烟与正式运行

冒烟训练：

```bash
MAX_ITERS=2000 N_EPISODES=3 bash scripts/run_speedtune_train_eval.sh
```

正式训练与 30-seed 严格配对 eval：

```bash
bash scripts/run_speedtune_train_eval.sh
```

默认 `CHUNK_TOPPRA_K_SKIP=40`，训练预算默认 `MAX_ITERS=20000`。这里的 `MAX_ITERS`
计的是 SpeedTune 决策步，不是 dense physics/action steps；K=40 下 20000 decisions
大致对齐旧 K=20、40000 decisions 的执行动作预算。需要更长训练时显式覆盖：

```bash
MAX_ITERS=40000 bash scripts/run_speedtune_train_eval.sh
```

仍可选择其它 1–2 个后端：

```bash
BACKENDS="chunk_toppra" bash scripts/run_speedtune_train_eval.sh
```

默认运行流程：

1. preflight 检查路径、checkpoint、conda env、4 GPU 和端口；
2. 启动两个 RoboTwin server 并等待端口 ready；
3. 分别训练 fixed-time 和 whole-chunk 单-head DQN；
4. 每个 checkpoint 写 `speedtune_metadata.json`；
5. eval 恢复前强校验 backend、support、k-skip、atoms、speed grid 和 acc rule；
6. 两个后端使用相同 30 个 seed 和 `(seed, episode, decision_step)` pi0.5 noise key；
7. 只用双方共同 task-success 的 episode 计算主要配对速度比，同时报告 safe-success 配对结果。

## 时间与结果

主要物理指标：

```text
physics_time_s = dense_steps / 250
```

部署指标分别记录 pi0.5、DQN、env RPC 和 decision-loop 端到端墙钟时间。严格计时默认关闭视频。

产物位于：

```text
logs/speedtune_traineval_<timestamp>/
  server_<backend>.log
  train_<backend>.log
  speedtune_<backend>_<timestamp>/checkpoints/update_*/
  compare_eval/eval.log
  compare_eval/compare_summary.json
  compare_eval/compare_speedup.png
  compare_eval/compare_knob.png
```

## 退出与错误

Ctrl-C、SIGTERM、训练失败或 eval 失败都会触发统一清理：先对每个独立进程组发送 TERM，最多等待
5 秒，再对残留进程组发送 KILL。退出信息包含失败阶段、返回码、日志目录以及每个子进程的日志路径。

若上次进程是在宿主机级别被 `kill -9` 或机器重启中断，shell 无法执行 trap；重跑前检查：

```bash
ps -ef | grep -E 'run_robotwin_client|train_speedtune_async|eval_speedtune' | grep -v grep
nvidia-smi
```
