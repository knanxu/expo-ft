#!/usr/bin/env bash
# 同时运行三个动作执行方式（exec_backend）的 SpeedTune DQN 训练，全部上传同一 wandb project。
#
#   方式1 fixed_time         (RoboTwin streaming，论文式固定时长)
#   方式2 per_action_toppra  (RoboTwin per_action，逐 action TOPP)
#   方式3 chunk_toppra       (RoboTwin whole_chunk，整段 TOPPRA)
#
# 每个 backend 起一对进程：① client_robotwin RoboTwin env server（sim venv）；
# ② train_speedtune_async.py SpeedTune DQN 训练（expo-ft .venv，连各自 server，冻结 VLA + RainbowDQN）。
# 三者共享同一冻结 drift pi0.5 ckpt（微调后 action head），各自独立 GPU / port / wandb run。
#
# 用法：
#   export SPEEDTUNE_VLA_CKPT=/abs/path/to/drift_pi05/checkpoint   # 必填：微调后 drift pi0.5
#   export SPEEDTUNE_VLA_ASSETS=/abs/path/.../assets               # 选填：drift norm_stats 目录
#   export SPEEDTUNE_VLA_ASSET_ID=robotwin                         # 选填：asset id
#   export GPUS="0 1 2"                                            # 选填：三 run 各用一卡（默认 0 1 2）
#   bash scripts/run_speedtune_3backends.sh
#
# 停止：Ctrl-C（trap 会清理所有子进程）。日志在 logs/speedtune_3run_<stamp>/。

set -euo pipefail

# ===== 配置（环境变量可覆盖）=====
EXPO_ROOT="${EXPO_ROOT:-/home/xukainan/expo-ft}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/home/xukainan/RoboTwin}"
ROBOTWIN_ENV="${ROBOTWIN_ENV:-RoboTwin}"                 # RoboTwin sim 的 conda env 名
TASK_CONFIG="${TASK_CONFIG:-configs/task/robotwin_stack_blocks.py}"
WANDB_PROJECT="${WANDB_PROJECT:-expo-ft-speedtune}"
MAX_ITERS="${MAX_ITERS:-100000}"
SEED="${SEED:-42}"
SERVER_WAIT="${SERVER_WAIT:-30}"                         # 等 env server socket 就绪的秒数

: "${SPEEDTUNE_VLA_CKPT:?请先 export SPEEDTUNE_VLA_CKPT=<微调后 drift pi0.5 ckpt 绝对路径>}"
export SPEEDTUNE_VLA_CKPT
export SPEEDTUNE_VLA_ASSETS="${SPEEDTUNE_VLA_ASSETS:-}"
export SPEEDTUNE_VLA_ASSET_ID="${SPEEDTUNE_VLA_ASSET_ID:-}"

BACKENDS=(fixed_time per_action_toppra chunk_toppra)
PORTS=(8102 8103 8104)
read -ra GPUS <<< "${GPUS:-0 1 2}"

STAMP="$(date +%m%d_%H%M)"
LOGDIR="${EXPO_ROOT}/logs/speedtune_3run_${STAMP}"
mkdir -p "$LOGDIR"
echo "[*] 日志目录: $LOGDIR"
echo "[*] 冻结 VLA ckpt: $SPEEDTUNE_VLA_CKPT"

PIDS=()
cleanup() {
  echo ""
  echo "[cleanup] 终止所有子进程 ..."
  for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "$EXPO_ROOT"

# ---- 1) 启动三个 RoboTwin env server（sim venv；server 内部自行 chdir 到 RoboTwin）----
for i in "${!BACKENDS[@]}"; do
  be="${BACKENDS[$i]}"; port="${PORTS[$i]}"; gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  echo "[$be] 启动 env server  port=$port  gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" \
  conda run --no-capture-output -n "$ROBOTWIN_ENV" \
    python -m client_robotwin.run_robotwin_client \
      --config_task_path "$TASK_CONFIG" \
      --robotwin_root "$ROBOTWIN_ROOT" \
      --server_port "$port" \
      > "$LOGDIR/server_${be}.log" 2>&1 &
  PIDS+=($!)
done

echo "[*] 等待 env server 就绪 (${SERVER_WAIT}s) ..."
sleep "$SERVER_WAIT"

# ---- 2) 启动三个 SpeedTune DQN 训练（expo-ft .venv；连各自 server；wandb 上传）----
for i in "${!BACKENDS[@]}"; do
  be="${BACKENDS[$i]}"; port="${PORTS[$i]}"; gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  run_name="speedtune_${be}_${STAMP}"
  echo "[$be] 启动训练  gpu=$gpu  client_port=$port  wandb_run=$run_name"
  CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  WANDB_PROJECT="$WANDB_PROJECT" \
    "$EXPO_ROOT/.venv/bin/python" train_speedtune_async.py \
      --config configs/model/speedtune_dqn_config.py \
      --config.exec_backend "$be" \
      --config.max_iters "$MAX_ITERS" \
      --config_task "$TASK_CONFIG" \
      --client_host localhost --client_port "$port" \
      --seed "$SEED" \
      --project_name "$WANDB_PROJECT" --run_name "$run_name" \
      --output_dir "$LOGDIR" \
      > "$LOGDIR/train_${be}.log" 2>&1 &
  PIDS+=($!)
  sleep 5
done

echo ""
echo "[*] 三个 SpeedTune 训练已启动，wandb project = $WANDB_PROJECT"
echo "[*] 实时查看：tail -f $LOGDIR/train_per_action_toppra.log"
echo "[*] Ctrl-C 停止全部。"
wait
