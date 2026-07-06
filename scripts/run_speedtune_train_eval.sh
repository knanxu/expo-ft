#!/usr/bin/env bash
# Train one or two backend-specific SpeedTune DQNs, then evaluate them.
# Default comparison: fixed_time vs chunk_toppra.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_EXPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXPO_ROOT="${EXPO_ROOT:-$DEFAULT_EXPO_ROOT}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-$(dirname "$EXPO_ROOT")/RoboTwin}"
ROBOTWIN_ENV="${ROBOTWIN_ENV:-RoboTwin}"
PYTHON="${PYTHON:-uv run python}"
TASK_CONFIG="${TASK_CONFIG:-configs/task/robotwin_stack_blocks.py}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model/speedtune_dqn_config.py}"
WANDB_PROJECT="${WANDB_PROJECT:-expo-ft-speedtune}"
# MAX_ITERS counts SpeedTune decisions, not dense physics/action steps.
# With whole-chunk execution_steps=40, 20k decisions roughly matches the
# previous 40k-decision budget at execution_steps=20 in executed action budget.
MAX_ITERS="${MAX_ITERS:-20000}"
SEED="${SEED:-42}"
SERVER_WAIT="${SERVER_WAIT:-90}"
TRAIN_MEM_FRAC="${TRAIN_MEM_FRAC:-0.85}"
COMPARE_MEM_FRAC="${COMPARE_MEM_FRAC:-0.85}"
STREAM_HOLD_STEPS="${STREAM_HOLD_STEPS:-15}"
CHUNK_TOPPRA_K_SKIP="${CHUNK_TOPPRA_K_SKIP:-40}"
N_EPISODES="${N_EPISODES:-30}"
VIDEO_EPISODES="${VIDEO_EPISODES:-5}"
MAX_DECISION_STEPS="${MAX_DECISION_STEPS:-400}"
DRY_RUN="${DRY_RUN:-0}"
CLEANUP_SELF_TEST="${CLEANUP_SELF_TEST:-0}"
COMPARE_GPU="${COMPARE_GPU:-1}"

if [ -n "${TRAIN_MODE:-}" ]; then
  :
elif [ -t 0 ] && [ "$DRY_RUN" != "1" ]; then
  read -r -p "TRAIN_MODE [sync] (sync/async): " TRAIN_MODE
  TRAIN_MODE="${TRAIN_MODE:-sync}"
else
  TRAIN_MODE="sync"
fi
case "$TRAIN_MODE" in
  sync) TRAIN_ENTRY="train_speedtune_sync.py" ;;
  async) TRAIN_ENTRY="train_speedtune_async.py" ;;
  *) echo "[ERROR] TRAIN_MODE must be sync or async, got '$TRAIN_MODE'" >&2; exit 2 ;;
esac

export SPEEDTUNE_FORCE_LIMIT="${SPEEDTUNE_FORCE_LIMIT:-30,40,30,15,10,10}"
: "${SPEEDTUNE_VLA_CKPT:?export SPEEDTUNE_VLA_CKPT=<frozen pi0.5 checkpoint>}"
export SPEEDTUNE_VLA_CKPT
export SPEEDTUNE_VLA_ASSETS="${SPEEDTUNE_VLA_ASSETS:-}"
export SPEEDTUNE_VLA_ASSET_ID="${SPEEDTUNE_VLA_ASSET_ID:-}"

ALL_BACKENDS=(fixed_time per_action_toppra chunk_toppra)
PORTS=(8102 8103)
SERVER_GPUS=(0 2)
TRAIN_GPUS=(1 3)

if [ -n "${BACKENDS:-}" ]; then
  read -ra BACKENDS <<< "$BACKENDS"
elif [ -t 0 ] && [ "$DRY_RUN" != "1" ]; then
  echo "Backends: ${ALL_BACKENDS[*]} (choose one or two)"
  read -r -p "BACKENDS [fixed_time chunk_toppra]: " line
  if [ -n "$line" ]; then read -ra BACKENDS <<< "$line"; else BACKENDS=(fixed_time chunk_toppra); fi
else
  BACKENDS=(fixed_time chunk_toppra)
fi

if [ "${#BACKENDS[@]}" -lt 1 ] || [ "${#BACKENDS[@]}" -gt 2 ]; then
  echo "[ERROR] BACKENDS must contain one or two entries: ${BACKENDS[*]}" >&2
  exit 2
fi
for be in "${BACKENDS[@]}"; do
  case "$be" in
    fixed_time|per_action_toppra|chunk_toppra) ;;
    *) echo "[ERROR] invalid backend '$be'" >&2; exit 2 ;;
  esac
done

PORTS=("${PORTS[@]:0:${#BACKENDS[@]}}")
SERVER_GPUS=("${SERVER_GPUS[@]:0:${#BACKENDS[@]}}")
TRAIN_GPUS=("${TRAIN_GPUS[@]:0:${#BACKENDS[@]}}")
read -ra PYTHON_CMD <<< "$PYTHON"

require_path() {
  local path="$1" label="$2"
  [ -e "$path" ] || { echo "[ERROR] missing $label: $path" >&2; exit 2; }
}

require_path "$EXPO_ROOT" EXPO_ROOT
require_path "$ROBOTWIN_ROOT" ROBOTWIN_ROOT
require_path "$SPEEDTUNE_VLA_CKPT" SPEEDTUNE_VLA_CKPT
require_path "$EXPO_ROOT/$TASK_CONFIG" TASK_CONFIG
require_path "$EXPO_ROOT/$MODEL_CONFIG" MODEL_CONFIG
if [ -n "$SPEEDTUNE_VLA_ASSETS" ]; then require_path "$SPEEDTUNE_VLA_ASSETS" SPEEDTUNE_VLA_ASSETS; fi
command -v "${PYTHON_CMD[0]}" >/dev/null || { echo "[ERROR] missing command: ${PYTHON_CMD[0]}" >&2; exit 2; }
command -v conda >/dev/null || { echo "[ERROR] missing conda" >&2; exit 2; }
command -v setsid >/dev/null || { echo "[ERROR] missing setsid" >&2; exit 2; }
command -v git >/dev/null || { echo "[ERROR] missing git" >&2; exit 2; }

EXPO_COMMIT="$(git -C "$EXPO_ROOT" rev-parse HEAD)"
ROBOTWIN_COMMIT="$(git -C "$ROBOTWIN_ROOT" rev-parse HEAD)"

port_is_free() {
  python - "$1" <<'PY'
import socket, sys
s = socket.socket()
try:
    s.bind(("127.0.0.1", int(sys.argv[1])))
finally:
    s.close()
PY
}

port_is_listening() {
  python - "$1" <<'PY'
import socket, sys
s = socket.socket(); s.settimeout(.5)
try:
    ok = s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0
finally:
    s.close()
raise SystemExit(0 if ok else 1)
PY
}

if [ "$DRY_RUN" != "1" ] && [ "$CLEANUP_SELF_TEST" != "1" ]; then
  command -v nvidia-smi >/dev/null || { echo "[ERROR] nvidia-smi not found" >&2; exit 2; }
  gpu_count="$(nvidia-smi -L | wc -l)"
  [ "$gpu_count" -ge 4 ] || { echo "[ERROR] four GPUs required, found $gpu_count" >&2; exit 2; }
  conda env list | awk '{print $1}' | grep -Fxq "$ROBOTWIN_ENV" || {
    echo "[ERROR] conda env '$ROBOTWIN_ENV' not found" >&2; exit 2;
  }
  for port in "${PORTS[@]}"; do
    port_is_free "$port" || { echo "[ERROR] port $port is already in use" >&2; exit 2; }
  done
fi

STAMP="$(date +%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-$EXPO_ROOT/logs/speedtune_traineval_$STAMP}"
OUTPUT_DIR="$LOGDIR/compare_eval"

if [ "$DRY_RUN" = "1" ]; then
  echo "[DRY-RUN] backends: ${BACKENDS[*]}"
  echo "[DRY-RUN] max_iters: $MAX_ITERS"
  echo "[DRY-RUN] n_episodes: $N_EPISODES"
  echo "[DRY-RUN] video_episodes: $VIDEO_EPISODES"
  echo "[DRY-RUN] stream_hold_steps: $STREAM_HOLD_STEPS"
  echo "[DRY-RUN] chunk_toppra_k_skip: $CHUNK_TOPPRA_K_SKIP"
  echo "[DRY-RUN] max_decision_steps: $MAX_DECISION_STEPS"
  echo "[DRY-RUN] train_mode: $TRAIN_MODE"
  echo "[DRY-RUN] trainer: $TRAIN_ENTRY"
  echo "[DRY-RUN] expo_commit: $EXPO_COMMIT"
  echo "[DRY-RUN] robotwin_commit: $ROBOTWIN_COMMIT"
  for i in "${!BACKENDS[@]}"; do
    echo "[DRY-RUN] server backend=${BACKENDS[$i]} gpu=${SERVER_GPUS[$i]} port=${PORTS[$i]}"
    echo "[DRY-RUN] train backend=${BACKENDS[$i]} gpu=${TRAIN_GPUS[$i]} config.exec_backend=${BACKENDS[$i]}"
  done
  if [ "${#BACKENDS[@]}" -eq 2 ]; then
    echo "[DRY-RUN] eval backend_a=${BACKENDS[0]} port_a=${PORTS[0]} backend_b=${BACKENDS[1]} port_b=${PORTS[1]}"
  else
    echo "[DRY-RUN] eval backend=${BACKENDS[0]} port=${PORTS[0]}"
  fi
  exit 0
fi

mkdir -p "$LOGDIR" "$OUTPUT_DIR"
cd "$EXPO_ROOT"

declare -a ALL_PIDS=()
declare -a ALL_LABELS=()
declare -a ALL_LOGS=()
STAGE="startup"

cleanup() {
  local rc="$?" pid
  trap - EXIT INT TERM
  echo "[cleanup] stage=$STAGE rc=$rc logdir=$LOGDIR"
  for pid in "${ALL_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then kill -TERM -- "-$pid" 2>/dev/null || true; fi
  done
  for _ in 1 2 3 4 5; do
    local alive=0
    for pid in "${ALL_PIDS[@]:-}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
    [ "$alive" -eq 0 ] && break
    sleep 1
  done
  for pid in "${ALL_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then kill -KILL -- "-$pid" 2>/dev/null || true; fi
  done
  wait 2>/dev/null || true
  if [ "$rc" -ne 0 ]; then
    echo "[ERROR] failed during $STAGE. Logs:"
    for i in "${!ALL_LOGS[@]}"; do echo "  ${ALL_LABELS[$i]}: ${ALL_LOGS[$i]}"; done
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [ "$CLEANUP_SELF_TEST" = "1" ]; then
  STAGE="cleanup self-test"
  setsid sleep 60 &
  pid=$!
  ALL_PIDS+=("$pid"); ALL_LABELS+=("cleanup-self-test"); ALL_LOGS+=("$LOGDIR/cleanup-self-test.log")
  echo "cleanup_self_test_pid=$pid"
  kill -TERM "$$"
  sleep 2
  exit 1
fi

wait_server() {
  local pid="$1" port="$2" log="$3" backend="$4" elapsed=0
  while [ "$elapsed" -lt "$SERVER_WAIT" ]; do
    kill -0 "$pid" 2>/dev/null || { echo "[ERROR] $backend server exited"; tail -n 40 "$log"; return 1; }
    grep -Eq "Render Error|address already in use|Traceback" "$log" && {
      echo "[ERROR] $backend server initialization failed"; tail -n 40 "$log"; return 1;
    }
    if port_is_listening "$port"; then echo "[$backend] server ready on $port"; return 0; fi
    sleep 2; elapsed=$((elapsed + 2))
  done
  echo "[ERROR] $backend server not ready after ${SERVER_WAIT}s"; tail -n 40 "$log"; return 1
}

echo "[*] backends=${BACKENDS[*]} train_mode=$TRAIN_MODE episodes=$N_EPISODES videos=$VIDEO_EPISODES logdir=$LOGDIR"
echo "[*] expo_commit=$EXPO_COMMIT robotwin_commit=$ROBOTWIN_COMMIT"
STAGE="server startup"
SERVER_PIDS=()
for i in "${!BACKENDS[@]}"; do
  be="${BACKENDS[$i]}"; port="${PORTS[$i]}"; gpu="${SERVER_GPUS[$i]}"
  log="$LOGDIR/server_${be}.log"
  setsid env CUDA_VISIBLE_DEVICES="$gpu" conda run --no-capture-output -n "$ROBOTWIN_ENV" \
    python -m client_robotwin.run_robotwin_client \
      --config_task_path "$TASK_CONFIG" --robotwin_root "$ROBOTWIN_ROOT" --server_port "$port" \
      >"$log" 2>&1 &
  pid=$!; SERVER_PIDS+=("$pid"); ALL_PIDS+=("$pid"); ALL_LABELS+=("server:$be"); ALL_LOGS+=("$log")
done
for i in "${!BACKENDS[@]}"; do
  wait_server "${SERVER_PIDS[$i]}" "${PORTS[$i]}" "$LOGDIR/server_${BACKENDS[$i]}.log" "${BACKENDS[$i]}"
done

STAGE="training"
TRAIN_PIDS=()
for i in "${!BACKENDS[@]}"; do
  be="${BACKENDS[$i]}"; port="${PORTS[$i]}"; gpu="${TRAIN_GPUS[$i]}"
  run_name="speedtune_${TRAIN_MODE}_${be}_${STAMP}"; log="$LOGDIR/train_${be}.log"
  setsid env CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_MEM_FRACTION="$TRAIN_MEM_FRAC" \
    WANDB_PROJECT="$WANDB_PROJECT" "${PYTHON_CMD[@]}" "$TRAIN_ENTRY" \
      --config "$MODEL_CONFIG" --config.exec_backend "$be" --config.max_iters "$MAX_ITERS" \
      --config.stream_hold_steps "$STREAM_HOLD_STEPS" \
      --config.chunk_toppra_k_skip "$CHUNK_TOPPRA_K_SKIP" --config_task "$TASK_CONFIG" \
      --client_host localhost --client_port "$port" --seed "$SEED" \
      --project_name "$WANDB_PROJECT" --run_name "$run_name" --output_dir "$LOGDIR" \
      >"$log" 2>&1 &
  pid=$!; TRAIN_PIDS+=("$pid"); ALL_PIDS+=("$pid"); ALL_LABELS+=("train:$be"); ALL_LOGS+=("$log")
  echo "[$be] train pid=$pid log=$log"
done
train_failed=0
for i in "${!TRAIN_PIDS[@]}"; do
  if wait "${TRAIN_PIDS[$i]}"; then echo "[${BACKENDS[$i]}] training complete"; else train_failed=1; fi
done
[ "$train_failed" -eq 0 ] || { echo "[ERROR] one or more training processes failed"; exit 1; }

STAGE="checkpoint discovery"
CKPTS=()
for be in "${BACKENDS[@]}"; do
  ck="$(find "$LOGDIR/speedtune_${TRAIN_MODE}_${be}_${STAMP}/checkpoints" -maxdepth 1 -type d -name 'update_*' 2>/dev/null | sort -V | tail -1)"
  [ -n "$ck" ] || { echo "[ERROR] checkpoint not found for $be"; exit 1; }
  require_path "$ck/speedtune_metadata.json" "checkpoint metadata for $be"
  CKPTS+=("$ck"); echo "[$be] checkpoint=$ck"
done

STAGE="evaluation"
EVAL_LOG="$OUTPUT_DIR/eval.log"
if [ "${#BACKENDS[@]}" -eq 2 ]; then
  setsid env CUDA_VISIBLE_DEVICES="$COMPARE_GPU" XLA_PYTHON_CLIENT_MEM_FRACTION="$COMPARE_MEM_FRAC" \
    "${PYTHON_CMD[@]}" eval_speedtune_compare.py --config "$MODEL_CONFIG" --config_task "$TASK_CONFIG" \
      --config.stream_hold_steps "$STREAM_HOLD_STEPS" \
      --config.chunk_toppra_k_skip "$CHUNK_TOPPRA_K_SKIP" \
      --backend_a "${BACKENDS[0]}" --ckpt_a "${CKPTS[0]}" --port_a "${PORTS[0]}" \
      --backend_b "${BACKENDS[1]}" --ckpt_b "${CKPTS[1]}" --port_b "${PORTS[1]}" \
      --n_episodes "$N_EPISODES" --seed "$SEED" --max_decision_steps "$MAX_DECISION_STEPS" \
      --record_video --video_episodes "$VIDEO_EPISODES" \
      --client_host localhost --output_dir "$OUTPUT_DIR" >"$EVAL_LOG" 2>&1 &
else
  setsid env CUDA_VISIBLE_DEVICES="$COMPARE_GPU" XLA_PYTHON_CLIENT_MEM_FRACTION="$COMPARE_MEM_FRAC" \
    "${PYTHON_CMD[@]}" eval_speedtune.py --config "$MODEL_CONFIG" \
      --config.exec_backend "${BACKENDS[0]}" --config.stream_hold_steps "$STREAM_HOLD_STEPS" \
      --config.chunk_toppra_k_skip "$CHUNK_TOPPRA_K_SKIP" \
      --config_task "$TASK_CONFIG" --dqn_ckpt "${CKPTS[0]}" --client_port "${PORTS[0]}" \
      --n_episodes "$N_EPISODES" --seed "$SEED" --max_decision_steps "$MAX_DECISION_STEPS" \
      --record_video --video_episodes "$VIDEO_EPISODES" \
      --client_host localhost --output_dir "$OUTPUT_DIR" >"$EVAL_LOG" 2>&1 &
fi
pid=$!; ALL_PIDS+=("$pid"); ALL_LABELS+=("eval"); ALL_LOGS+=("$EVAL_LOG")
if ! wait "$pid"; then echo "[ERROR] eval failed"; tail -n 80 "$EVAL_LOG"; exit 1; fi

STAGE="complete"
echo "[*] complete: training=$LOGDIR eval=$OUTPUT_DIR"
