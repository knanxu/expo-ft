#!/usr/bin/env bash
# 4 卡跑 2 个 backend 的 SpeedTune DQN 训练，server 与 train **分卡**避免光追 sapien 与 3B VLA 抢显存。
#
#   per_action_toppra:  server→GPU0, train→GPU1, port 8102
#   fixed_time:         server→GPU2, train→GPU3, port 8103
#
# 每个 backend 一对进程：① client_robotwin RoboTwin env server（sim venv，光追渲染独占一卡）；
# ② train_speedtune_async.py（expo-ft venv，3B VLA + RainbowDQN 独占一卡）。两个 wandb run 进同一 project。
#
# 用法（云端）：
#   export SPEEDTUNE_VLA_CKPT=/abs/path/to/drift_pi05/checkpoint     # 必填
#   export SPEEDTUNE_VLA_ASSETS=$SPEEDTUNE_VLA_CKPT/assets           # 选填（norm_stats）
#   export SPEEDTUNE_VLA_ASSET_ID=<assets 下子目录名>                # 选填
#   bash scripts/run_speedtune_2backends.sh
#
# 跑前务必清理残留（光追 sapien 退出不彻底会占着显存 → cannot create buffer）：
#   pkill -9 -f run_robotwin_client; pkill -9 -f train_speedtune_async; sleep 3; nvidia-smi
#
# 停止：Ctrl-C（trap 清理）。日志：logs/speedtune_2run_<stamp>/。

set -euo pipefail

# ===== 配置（环境变量可覆盖）=====
EXPO_ROOT="${EXPO_ROOT:-/home/chenlu/expo-ft}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/home/chenlu/RoboTwin}"
ROBOTWIN_ENV="${ROBOTWIN_ENV:-RoboTwin}"          # RoboTwin sim 的 conda env 名
PYTHON="${PYTHON:-uv run python}"                 # learner 端 python（云端 uv）
TASK_CONFIG="${TASK_CONFIG:-configs/task/robotwin_stack_blocks.py}"
WANDB_PROJECT="${WANDB_PROJECT:-expo-ft-speedtune}"
MAX_ITERS="${MAX_ITERS:-100000}"
SEED="${SEED:-42}"
SERVER_WAIT="${SERVER_WAIT:-45}"                  # 等 server 渲染自检/就绪秒数
TRAIN_MEM_FRAC="${TRAIN_MEM_FRAC:-0.85}"          # train VLA 单卡显存上限（独占一卡，留余量给碎片）
# fixed_time(streaming) 每个 action hold 的物理步：250/这个=等效控制Hz。
# 15≈16.7Hz=专家采集 save_freq 原速基线；增大→每 action 跟踪更久/更慢更平滑，减小→更快(<15 脱离真机)。
STREAM_HOLD_STEPS="${STREAM_HOLD_STEPS:-15}"

: "${SPEEDTUNE_VLA_CKPT:?请先 export SPEEDTUNE_VLA_CKPT=<微调后 drift pi0.5 ckpt 绝对路径>}"
export SPEEDTUNE_VLA_CKPT
export SPEEDTUNE_VLA_ASSETS="${SPEEDTUNE_VLA_ASSETS:-}"
export SPEEDTUNE_VLA_ASSET_ID="${SPEEDTUNE_VLA_ASSET_ID:-}"

# 2 backend；每个 server / train 各独占一卡（4 卡用满）。可用环境变量覆盖以换卡/换 backend。
BACKENDS=(${BACKENDS:-per_action_toppra fixed_time})
PORTS=(${PORTS:-8102 8103})
SERVER_GPUS=(${SERVER_GPUS:-0 2})                 # 每 backend 的 server（sapien 光追）卡
TRAIN_GPUS=(${TRAIN_GPUS:-1 3})                   # 每 backend 的 train（3B VLA + DQN）卡

STAMP="$(date +%m%d_%H%M)"
LOGDIR="${EXPO_ROOT}/logs/speedtune_2run_${STAMP}"
mkdir -p "$LOGDIR"
echo "[*] 日志目录: $LOGDIR"
echo "[*] 冻结 VLA ckpt: $SPEEDTUNE_VLA_CKPT"
echo "[*] 分卡: $(for i in "${!BACKENDS[@]}"; do printf '%s(server=GPU%s,train=GPU%s) ' "${BACKENDS[$i]}" "${SERVER_GPUS[$i]}" "${TRAIN_GPUS[$i]}"; done)"

PIDS=()
cleanup() {
  echo ""
  echo "[cleanup] 终止所有子进程 ..."
  for p in "${PIDS[@]:-}"; do kill -9 "$p" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "$EXPO_ROOT"

# ---- 1) 起 server（每个独占一卡，光追渲染）----
for i in "${!BACKENDS[@]}"; do
  be="${BACKENDS[$i]}"; port="${PORTS[$i]}"; sgpu="${SERVER_GPUS[$i]}"
  echo "[$be] 启动 env server  server_GPU=$sgpu  port=$port"
  CUDA_VISIBLE_DEVICES="$sgpu" \
  conda run --no-capture-output -n "$ROBOTWIN_ENV" \
    python -m client_robotwin.run_robotwin_client \
      --config_task_path "$TASK_CONFIG" --robotwin_root "$ROBOTWIN_ROOT" --server_port "$port" \
      > "$LOGDIR/server_${be}.log" 2>&1 &
  PIDS+=($!)
done

echo "[*] 等 server 渲染自检/就绪 (${SERVER_WAIT}s)；正常会在 server 日志打印 'Render Well' ..."
sleep "$SERVER_WAIT"
for i in "${!BACKENDS[@]}"; do
  be="${BACKENDS[$i]}"
  if grep -q "Render Well" "$LOGDIR/server_${be}.log" 2>/dev/null; then
    echo "[$be] server ✓ Render Well"
  elif grep -q "Render Error" "$LOGDIR/server_${be}.log" 2>/dev/null; then
    echo "[$be] server ✗ Render Error —— 渲染环境问题，看 $LOGDIR/server_${be}.log"
  else
    echo "[$be] server 渲染状态未知（可能还在初始化）；若 train 报 cannot create buffer 看 $LOGDIR/server_${be}.log"
  fi
done

# ---- 2) 起 train（每个独占一卡，与 server 分卡）----
for i in "${!BACKENDS[@]}"; do
  be="${BACKENDS[$i]}"; port="${PORTS[$i]}"; tgpu="${TRAIN_GPUS[$i]}"
  run_name="speedtune_${be}_${STAMP}"
  echo "[$be] 启动训练  train_GPU=$tgpu  client_port=$port  wandb_run=$run_name"
  CUDA_VISIBLE_DEVICES="$tgpu" XLA_PYTHON_CLIENT_MEM_FRACTION="$TRAIN_MEM_FRAC" \
  WANDB_PROJECT="$WANDB_PROJECT" \
    $PYTHON train_speedtune_async.py \
      --config configs/model/speedtune_dqn_config.py \
      --config.exec_backend "$be" \
      --config.max_iters "$MAX_ITERS" \
      --config.stream_hold_steps "$STREAM_HOLD_STEPS" \
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
echo "[*] 2 个 SpeedTune 训练已启动，wandb project = $WANDB_PROJECT"
echo "[*] 实时查看：tail -f $LOGDIR/train_per_action_toppra.log"
echo "[*] Ctrl-C 停止全部。"
wait
