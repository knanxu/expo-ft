#!/usr/bin/env bash
# SpeedTune 双 backend 加速对比 eval：起两个 RoboTwin env server(不同 backend/port) + 一个 compare
# 进程(3B 冻结 VLA + 两个训练好的 DQN，各连一个 server)，对比 chunk_toppra vs fixed_time。
#
# 卡分配（3 卡；光追 sapien server 各独占一卡，与 compare 分卡避免抢显存）：
#   server A(BACKEND_A)→GPU0(port PORT_A) | server B(BACKEND_B)→GPU1(port PORT_B) | compare→GPU2
#
# 用法（云端；VLA 三路径已默认指向云端 drift ckpt，通常只需给两个 DQN checkpoint）：
#   CKPT_A=logs/.../fixed_time/checkpoints/update_<N> \
#   CKPT_B=logs/.../chunk_toppra/checkpoints/update_<N> \
#   bash scripts/run_eval_compare.sh
#   （VLA ckpt 换位置时：export SPEEDTUNE_VLA_ROOT=<根目录(params/ 与 assets/ 平级)> 覆盖，
#    或单独 export SPEEDTUNE_VLA_CKPT / SPEEDTUNE_VLA_ASSETS / SPEEDTUNE_VLA_ASSET_ID）
#
# 跑前清残留（光追退出不彻底会占显存 → cannot create buffer）：
#   pkill -9 -f run_robotwin_client; pkill -9 -f eval_speedtune_compare; sleep 3; nvidia-smi
# 停止：Ctrl-C（trap 清理）。结果在 $OUTPUT_DIR。

set -euo pipefail

# ===== 配置（环境变量可覆盖）=====
EXPO_ROOT="${EXPO_ROOT:-/home/chenlu/expo-ft}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/home/chenlu/RoboTwin}"
ROBOTWIN_ENV="${ROBOTWIN_ENV:-RoboTwin}"           # RoboTwin sim 的 conda env 名
PYTHON="${PYTHON:-uv run python}"                  # learner 端 python（云端 uv）
TASK_CONFIG="${TASK_CONFIG:-configs/task/robotwin_stack_blocks.py}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model/speedtune_dqn_config.py}"
N_EPISODES="${N_EPISODES:-30}"
SEED="${SEED:-0}"
MAX_DECISION_STEPS="${MAX_DECISION_STEPS:-400}"
# fixed_time(streaming) 每 action hold 的物理步：250/这个=等效控制Hz；务必与训练时一致（默认 15）。
STREAM_HOLD_STEPS="${STREAM_HOLD_STEPS:-15}"
FIXED_TIME_K_SKIP="${FIXED_TIME_K_SKIP:-10}"
CHUNK_TOPPRA_K_SKIP="${CHUNK_TOPPRA_K_SKIP:-40}"
SERVER_WAIT="${SERVER_WAIT:-45}"                   # 等 server 渲染自检/就绪秒数
COMPARE_MEM_FRAC="${COMPARE_MEM_FRAC:-0.85}"       # compare(3B VLA) 单卡显存上限

# 两 backend + 各自 DQN checkpoint（CKPT_A / CKPT_B 必填）。A=基准, B=被测。
BACKEND_A="${BACKEND_A:-fixed_time}"
BACKEND_B="${BACKEND_B:-chunk_toppra}"
PORT_A="${PORT_A:-8103}"
PORT_B="${PORT_B:-8102}"
: "${CKPT_A:?请 export CKPT_A=<backend_a 的 DQN checkpoint update_<N> 目录（含 q_net/）>}"
: "${CKPT_B:?请 export CKPT_B=<backend_b 的 DQN checkpoint update_<N> 目录（含 q_net/）>}"

# VLA（冻结 drift pi0.5, stack_blocks_two）：默认指向云端 ckpt 根目录（params/ 与 assets/ **平级**，
# norm_stats 在 assets/trossen/）；三个环境变量供 config 拼 weight_loader + AssetsConfig，可 env 覆盖。
# 坑：SPEEDTUNE_VLA_ASSETS 必须是 <根目录>/assets（不是 $SPEEDTUNE_VLA_CKPT/assets=params/assets）；asset_id=trossen。
_VLA_ROOT="${SPEEDTUNE_VLA_ROOT:-/home/chenlu/openpi/checkpoints/pi05_aloha_robotwin_drifting_stack_blocks_two/drifting_v1/29999}"
export SPEEDTUNE_VLA_CKPT="${SPEEDTUNE_VLA_CKPT:-${_VLA_ROOT}/params}"
export SPEEDTUNE_VLA_ASSETS="${SPEEDTUNE_VLA_ASSETS:-${_VLA_ROOT}/assets}"
export SPEEDTUNE_VLA_ASSET_ID="${SPEEDTUNE_VLA_ASSET_ID:-trossen}"

# 卡分配：两 server 各一卡（光追独占），compare 一卡。可用环境变量覆盖。
SERVER_GPUS=(${SERVER_GPUS:-0 1})
COMPARE_GPU="${COMPARE_GPU:-2}"

STAMP="$(date +%m%d_%H%M)"
OUTPUT_DIR="${OUTPUT_DIR:-${EXPO_ROOT}/logs/speedtune_compare_${STAMP}}"
mkdir -p "$OUTPUT_DIR"
echo "[*] 输出目录: $OUTPUT_DIR"
echo "[*] A=$BACKEND_A (port $PORT_A, GPU ${SERVER_GPUS[0]}, ckpt $CKPT_A)"
echo "[*] B=$BACKEND_B (port $PORT_B, GPU ${SERVER_GPUS[1]}, ckpt $CKPT_B)"
echo "[*] k_skip fixed_time=$FIXED_TIME_K_SKIP chunk_toppra=$CHUNK_TOPPRA_K_SKIP"
echo "[*] compare→GPU $COMPARE_GPU | 冻结 VLA: $SPEEDTUNE_VLA_CKPT"

PIDS=()
cleanup() {
  local rc="$?" p
  trap - EXIT INT TERM
  echo ""
  echo "[cleanup] 终止所有子进程 ..."
  for p in "${PIDS[@]:-}"; do kill -TERM -- "-$p" 2>/dev/null || true; done
  sleep 1
  for p in "${PIDS[@]:-}"; do kill -KILL -- "-$p" 2>/dev/null || true; done
  wait 2>/dev/null || true
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$EXPO_ROOT"

# ---- 0) 端口预检：被占用 = 有残留旧 server，必须先清理（否则会 silent 连到旧 server 跑旧代码、无视频）----
for _port in "$PORT_A" "$PORT_B"; do
  if (exec 3<>"/dev/tcp/127.0.0.1/${_port}") 2>/dev/null; then
    exec 3>&- 3<&- 2>/dev/null || true
    echo "[ERROR] 端口 ${_port} 已被占用——极可能是上次 RoboTwin server 残留进程。" >&2
    echo "        若继续会 silent 连到旧 server（跑旧代码、无 start_video、无 contact）。请先清理后重跑：" >&2
    echo "        pkill -9 -f run_robotwin_client; pkill -9 -f eval_speedtune; sleep 3; nvidia-smi" >&2
    exit 1
  fi
done

# ---- 1) 起两个 env server（每个独占一卡，光追渲染）----
BACKENDS=("$BACKEND_A" "$BACKEND_B")
PORTS=("$PORT_A" "$PORT_B")
for i in 0 1; do
  be="${BACKENDS[$i]}"; port="${PORTS[$i]}"; sgpu="${SERVER_GPUS[$i]}"
  echo "[$be] 启动 env server  server_GPU=$sgpu  port=$port"
  setsid env CUDA_VISIBLE_DEVICES="$sgpu" \
  conda run --no-capture-output -n "$ROBOTWIN_ENV" \
    python -m client_robotwin.run_robotwin_client \
      --config_task_path "$TASK_CONFIG" --robotwin_root "$ROBOTWIN_ROOT" --server_port "$port" \
      > "$OUTPUT_DIR/server_${be}.log" 2>&1 &
  PIDS+=($!)
done

echo "[*] 等 server 渲染自检/就绪 (${SERVER_WAIT}s)；正常会在 server 日志打印 'Render Well' ..."
sleep "$SERVER_WAIT"
for be in "$BACKEND_A" "$BACKEND_B"; do
  # bind 失败检测最优先：'Render Well' 会在 bind 之前打印, 不能只靠它判就绪（这正是上次 silent 连到旧 server 的坑）。
  if grep -q "address already in use" "$OUTPUT_DIR/server_${be}.log" 2>/dev/null; then
    echo "[ERROR] $be server 端口 bind 失败（address already in use，见 server_${be}.log）。" >&2
    echo "        新 server 没起来，继续会 silent 连到残留旧 server。中止。先清理后重跑：" >&2
    echo "        pkill -9 -f run_robotwin_client; pkill -9 -f eval_speedtune; sleep 3; nvidia-smi" >&2
    exit 1
  fi
  if grep -q "Render Well" "$OUTPUT_DIR/server_${be}.log" 2>/dev/null; then
    echo "[$be] server ✓ Render Well"
  elif grep -q "Render Error" "$OUTPUT_DIR/server_${be}.log" 2>/dev/null; then
    echo "[$be] server ✗ Render Error —— 渲染环境问题，看 $OUTPUT_DIR/server_${be}.log"
  else
    echo "[$be] server 渲染状态未知（可能还在初始化）；若 compare 报 cannot create buffer 看 $OUTPUT_DIR/server_${be}.log"
  fi
done

# ---- 2) 起 compare（一卡：3B 冻结 VLA + 两个 greedy DQN，分别连两个 server）----
echo "[*] 启动对比 eval  compare_GPU=$COMPARE_GPU"
CUDA_VISIBLE_DEVICES="$COMPARE_GPU" XLA_PYTHON_CLIENT_MEM_FRACTION="$COMPARE_MEM_FRAC" \
  $PYTHON eval_speedtune_compare.py \
    --config "$MODEL_CONFIG" \
    --config_task "$TASK_CONFIG" \
    --config.stream_hold_steps "$STREAM_HOLD_STEPS" \
    --config.fixed_time_k_skip "$FIXED_TIME_K_SKIP" \
    --config.chunk_toppra_k_skip "$CHUNK_TOPPRA_K_SKIP" \
    --backend_a "$BACKEND_A" --ckpt_a "$CKPT_A" --port_a "$PORT_A" \
    --backend_b "$BACKEND_B" --ckpt_b "$CKPT_B" --port_b "$PORT_B" \
    --n_episodes "$N_EPISODES" --seed "$SEED" \
    --max_decision_steps "$MAX_DECISION_STEPS" \
    --client_host localhost \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/compare.log"

echo ""
echo "[*] 对比完成。结果: $OUTPUT_DIR"
echo "    - compare_summary.json   （两 backend 平均执行步数 + 加速比）"
echo "    - compare_speedup.png    （per-episode 执行步数 bar + 激进度对比）"
echo "    - compare_knob.png       （各 backend 实际加速参数随决策步）"
echo "    - <backend>/videos/episode*.mp4 + <backend>/episode*_speed.{json,png}"
