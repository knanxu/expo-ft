#!/usr/bin/env bash
# SpeedTune 一体化：4 卡并行训练 per_action_toppra + fixed_time，训练完成后自动 eval 对比。
# 替代「run_speedtune_2backends.sh 训练 → 手填 CKPT → run_eval_compare.sh」两步手动衔接。
#
# 卡分配（4 卡；server 与 train 分卡避免光追 sapien 与 3B VLA 抢显存）：
#   per_action_toppra: server→GPU0:8102 | train→GPU1
#   fixed_time:        server→GPU2:8103 | train→GPU3
#   eval compare:      复用上面两个 server，compare→GPU1（train 退出后空出）
#
# 训练跑满 MAX_ITERS 自然退出 → final checkpoint 落盘（train finally: _stop+join 保证）→
# 自动找最新 update_N → 复用训练 server 跑 eval_speedtune_compare（不重起 server/不端口预检）。
#
# 用法（云端）：
#   export SPEEDTUNE_VLA_CKPT=/abs/path/.../params        # 必填
#   export SPEEDTUNE_VLA_ASSETS=/abs/path/.../assets      # 选填(norm_stats)
#   export SPEEDTUNE_VLA_ASSET_ID=trossen                 # 选填
#   bash scripts/run_speedtune_train_eval.sh
#   冒烟：MAX_ITERS=2000 N_EPISODES=3 bash scripts/run_speedtune_train_eval.sh
#
# 跑前清残留（光追退出不彻底会占显存 → cannot create buffer）：
#   pkill -9 -f run_robotwin_client; pkill -9 -f train_speedtune_async; pkill -9 -f eval_speedtune; sleep 3; nvidia-smi
# 停止：Ctrl-C（trap 清理）。

set -euo pipefail

# ===== 配置（环境变量可覆盖）=====
EXPO_ROOT="${EXPO_ROOT:-/home/chenlu/expo-ft}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/home/chenlu/RoboTwin}"
ROBOTWIN_ENV="${ROBOTWIN_ENV:-RoboTwin}"
PYTHON="${PYTHON:-uv run python}"
TASK_CONFIG="${TASK_CONFIG:-configs/task/robotwin_stack_blocks.py}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model/speedtune_dqn_config.py}"
WANDB_PROJECT="${WANDB_PROJECT:-expo-ft-speedtune}"
MAX_ITERS="${MAX_ITERS:-100000}"
SEED="${SEED:-42}"
SERVER_WAIT="${SERVER_WAIT:-45}"
TRAIN_MEM_FRAC="${TRAIN_MEM_FRAC:-0.85}"
COMPARE_MEM_FRAC="${COMPARE_MEM_FRAC:-0.85}"
STREAM_HOLD_STEPS="${STREAM_HOLD_STEPS:-15}"
N_EPISODES="${N_EPISODES:-5}"
MAX_DECISION_STEPS="${MAX_DECISION_STEPS:-400}"

# 执行层力矩底座（per-joint 单臂 τ_max, N·m；空串=不施加）。train+compare 进程都继承。
export SPEEDTUNE_FORCE_LIMIT="${SPEEDTUNE_FORCE_LIMIT:-30,40,30,15,10,10}"

: "${SPEEDTUNE_VLA_CKPT:?请先 export SPEEDTUNE_VLA_CKPT=<微调后 drift pi0.5 ckpt 绝对路径>}"
export SPEEDTUNE_VLA_CKPT
export SPEEDTUNE_VLA_ASSETS="${SPEEDTUNE_VLA_ASSETS:-}"
export SPEEDTUNE_VLA_ASSET_ID="${SPEEDTUNE_VLA_ASSET_ID:-}"

# 两 backend：名 / port / server 卡 / train 卡（与 eval 默认 port 对齐：per_action=8102, fixed_time=8103）。
BACKENDS=(per_action_toppra fixed_time)
PORTS=(8102 8103)
SERVER_GPUS=(0 2)
TRAIN_GPUS=(1 3)
COMPARE_GPU="${COMPARE_GPU:-1}"

STAMP="$(date +%m%d_%H%M)"
LOGDIR="${EXPO_ROOT}/logs/speedtune_traineval_${STAMP}"
OUTPUT_DIR="${LOGDIR}/compare_eval"
mkdir -p "$LOGDIR" "$OUTPUT_DIR"
echo "[*] 日志目录: $LOGDIR"
echo "[*] 冻结 VLA ckpt: $SPEEDTUNE_VLA_CKPT"
echo "[*] force_limit(τ_max): ${SPEEDTUNE_FORCE_LIMIT:-<空=不施加>}"

SERVER_PIDS=()
TRAIN_PIDS=()
cleanup() {
  echo ""
  echo "[cleanup] 终止所有子进程 ..."
  for p in "${TRAIN_PIDS[@]:-}" "${SERVER_PIDS[@]:-}"; do kill -9 "$p" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "$EXPO_ROOT"

# ---- 1) 起 2 个 server（每个独占一卡，光追渲染）----
for i in "${!BACKENDS[@]}"; do
  be="${BACKENDS[$i]}"; port="${PORTS[$i]}"; sgpu="${SERVER_GPUS[$i]}"
  echo "[$be] 启动 env server  server_GPU=$sgpu  port=$port"
  CUDA_VISIBLE_DEVICES="$sgpu" \
  conda run --no-capture-output -n "$ROBOTWIN_ENV" \
    python -m client_robotwin.run_robotwin_client \
      --config_task_path "$TASK_CONFIG" --robotwin_root "$ROBOTWIN_ROOT" --server_port "$port" \
      > "$LOGDIR/server_${be}.log" 2>&1 &
  SERVER_PIDS+=($!)
done

echo "[*] 等 server 渲染自检/就绪 (${SERVER_WAIT}s) ..."
sleep "$SERVER_WAIT"
for be in "${BACKENDS[@]}"; do
  if grep -q "address already in use" "$LOGDIR/server_${be}.log" 2>/dev/null; then
    echo "[ERROR] $be server 端口 bind 失败（address already in use）。先清理残留后重跑。" >&2
    exit 1
  fi
  if grep -q "Render Well" "$LOGDIR/server_${be}.log" 2>/dev/null; then
    echo "[$be] server ✓ Render Well"
  else
    echo "[$be] server 渲染状态未知（可能还在初始化）；若 train 报 cannot create buffer 看 server_${be}.log"
  fi
done

# ---- 2) 起 2 个 train（每个独占一卡，与 server 分卡）；只 wait train 进程 ----
for i in "${!BACKENDS[@]}"; do
  be="${BACKENDS[$i]}"; port="${PORTS[$i]}"; tgpu="${TRAIN_GPUS[$i]}"
  run_name="speedtune_${be}_${STAMP}"
  echo "[$be] 启动训练  train_GPU=$tgpu  client_port=$port  run=$run_name"
  CUDA_VISIBLE_DEVICES="$tgpu" XLA_PYTHON_CLIENT_MEM_FRACTION="$TRAIN_MEM_FRAC" \
  WANDB_PROJECT="$WANDB_PROJECT" \
    $PYTHON train_speedtune_async.py \
      --config "$MODEL_CONFIG" \
      --config.exec_backend "$be" \
      --config.max_iters "$MAX_ITERS" \
      --config.stream_hold_steps "$STREAM_HOLD_STEPS" \
      --config_task "$TASK_CONFIG" \
      --client_host localhost --client_port "$port" \
      --seed "$SEED" \
      --project_name "$WANDB_PROJECT" --run_name "$run_name" \
      --output_dir "$LOGDIR" \
      > "$LOGDIR/train_${be}.log" 2>&1 &
  TRAIN_PIDS+=($!)
  sleep 5
done

echo "[*] 训练中（MAX_ITERS=$MAX_ITERS）；只等 train 进程退出（server 常驻待 eval 复用）..."
echo "[*] 实时查看：tail -f $LOGDIR/train_per_action_toppra.log"
train_rc=0
for i in "${!TRAIN_PIDS[@]}"; do
  p="${TRAIN_PIDS[$i]}"; be="${BACKENDS[$i]}"
  if wait "$p"; then echo "[$be] 训练完成 ✓"; else echo "[$be] 训练异常退出 ✗（见 train_${be}.log）" >&2; train_rc=1; fi
done
if [ "$train_rc" -ne 0 ]; then
  echo "[ERROR] 有训练异常退出，跳过 eval。日志在 $LOGDIR" >&2
  exit 1
fi

# ---- 3) 自动发现各 backend 最新 checkpoint ----
CKPT_PER=$(ls -d "$LOGDIR/speedtune_per_action_toppra_${STAMP}/checkpoints/update_"* 2>/dev/null | sort -V | tail -1 || true)
CKPT_FIXED=$(ls -d "$LOGDIR/speedtune_fixed_time_${STAMP}/checkpoints/update_"* 2>/dev/null | sort -V | tail -1 || true)
if [ -z "$CKPT_PER" ] || [ -z "$CKPT_FIXED" ]; then
  echo "[ERROR] 找不到 checkpoint（per_action=$CKPT_PER fixed_time=$CKPT_FIXED）。看 train_*.log" >&2
  exit 1
fi
echo "[*] CKPT per_action_toppra = $CKPT_PER"
echo "[*] CKPT fixed_time        = $CKPT_FIXED"

# ---- 4) 起 compare（复用训练 server；A=fixed_time 基准 / B=per_action 被测）----
echo "[*] 启动对比 eval  compare_GPU=$COMPARE_GPU（复用 server，不重起/不端口预检）"
CUDA_VISIBLE_DEVICES="$COMPARE_GPU" XLA_PYTHON_CLIENT_MEM_FRACTION="$COMPARE_MEM_FRAC" \
  $PYTHON eval_speedtune_compare.py \
    --config "$MODEL_CONFIG" \
    --config_task "$TASK_CONFIG" \
    --config.stream_hold_steps "$STREAM_HOLD_STEPS" \
    --backend_a fixed_time        --ckpt_a "$CKPT_FIXED" --port_a 8103 \
    --backend_b per_action_toppra --ckpt_b "$CKPT_PER"   --port_b 8102 \
    --n_episodes "$N_EPISODES" --seed "$SEED" \
    --max_decision_steps "$MAX_DECISION_STEPS" \
    --client_host localhost \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/compare.log"

echo ""
echo "[*] 全部完成。训练日志: $LOGDIR ; 对比结果: $OUTPUT_DIR"
echo "    - compare_summary.json / compare_speedup.png / compare_knob.png"
echo "    - <backend>/videos/ + <backend>/episode*_speed.{json,png}"
