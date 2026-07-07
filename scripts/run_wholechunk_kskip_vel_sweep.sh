#!/usr/bin/env bash
# Whole-chunk TOPPRA vel_limit × k_skip sweep.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_EXPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXPO_ROOT="${EXPO_ROOT:-$DEFAULT_EXPO_ROOT}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-$(dirname "$EXPO_ROOT")/RoboTwin}"
ROBOTWIN_ENV="${ROBOTWIN_ENV:-RoboTwin}"
PYTHON="${PYTHON:-uv run python}"
TASK_CONFIG="${TASK_CONFIG:-configs/task/robotwin_stack_blocks.py}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model/speedtune_dqn_config.py}"
N_EPISODES="${N_EPISODES:-50}"
VEL_LIMITS="${VEL_LIMITS:-1,1.5,2,2.5,3,3.5,4}"
K_SKIPS="${K_SKIPS:-10,20,40,50}"
SEED="${SEED:-42}"
MAX_DECISION_STEPS="${MAX_DECISION_STEPS:-400}"
STREAM_HOLD_STEPS="${STREAM_HOLD_STEPS:-15}"
CHUNK_TOPPRA_PORT="${CHUNK_TOPPRA_PORT:-8102}"
SERVER_WAIT="${SERVER_WAIT:-90}"
EVAL_MEM_FRAC="${EVAL_MEM_FRAC:-0.85}"
SERVER_GPU="${SERVER_GPU:-0}"
EVAL_GPU="${EVAL_GPU:-1}"
DRY_RUN="${DRY_RUN:-0}"
CLEANUP_SELF_TEST="${CLEANUP_SELF_TEST:-0}"

export SPEEDTUNE_FORCE_LIMIT="${SPEEDTUNE_FORCE_LIMIT:-30,40,30,15,10,10}"
: "${SPEEDTUNE_VLA_CKPT:?export SPEEDTUNE_VLA_CKPT=<frozen pi0.5 checkpoint>}"
export SPEEDTUNE_VLA_CKPT
export SPEEDTUNE_VLA_ASSETS="${SPEEDTUNE_VLA_ASSETS:-}"
export SPEEDTUNE_VLA_ASSET_ID="${SPEEDTUNE_VLA_ASSET_ID:-}"

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
require_path "$EXPO_ROOT/eval_speedtune_chunk_kskip_sweep.py" sweep_entrypoint
if [ -n "$SPEEDTUNE_VLA_ASSETS" ]; then require_path "$SPEEDTUNE_VLA_ASSETS" SPEEDTUNE_VLA_ASSETS; fi
command -v "${PYTHON_CMD[0]}" >/dev/null || { echo "[ERROR] missing command: ${PYTHON_CMD[0]}" >&2; exit 2; }
command -v conda >/dev/null || { echo "[ERROR] missing conda" >&2; exit 2; }
command -v setsid >/dev/null || { echo "[ERROR] missing setsid" >&2; exit 2; }
command -v git >/dev/null || { echo "[ERROR] missing git" >&2; exit 2; }

EXPO_COMMIT="$(git -C "$EXPO_ROOT" rev-parse HEAD)"
ROBOTWIN_COMMIT="$(git -C "$ROBOTWIN_ROOT" rev-parse HEAD)"

IFS=',' read -ra VEL_VALUES <<< "$VEL_LIMITS"
IFS=',' read -ra K_VALUES <<< "$K_SKIPS"
[ "${#VEL_VALUES[@]}" -gt 0 ] || { echo "[ERROR] VEL_LIMITS cannot be empty" >&2; exit 2; }
[ "${#K_VALUES[@]}" -gt 0 ] || { echo "[ERROR] K_SKIPS cannot be empty" >&2; exit 2; }
TOTAL_CONFIGS=$((${#VEL_VALUES[@]} * ${#K_VALUES[@]}))
TOTAL_EPISODES=$((TOTAL_CONFIGS * N_EPISODES))

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
    listening = s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0
finally:
    s.close()
raise SystemExit(0 if listening else 1)
PY
}

if [ "$DRY_RUN" != "1" ] && [ "$CLEANUP_SELF_TEST" != "1" ]; then
  command -v nvidia-smi >/dev/null || { echo "[ERROR] nvidia-smi not found" >&2; exit 2; }
  gpu_count="$(nvidia-smi -L | wc -l)"
  [ "$gpu_count" -ge 2 ] || { echo "[ERROR] two GPUs required, found $gpu_count" >&2; exit 2; }
  conda env list | awk '{print $1}' | grep -Fxq "$ROBOTWIN_ENV" || {
    echo "[ERROR] conda env '$ROBOTWIN_ENV' not found" >&2; exit 2;
  }
  port_is_free "$CHUNK_TOPPRA_PORT" || { echo "[ERROR] port $CHUNK_TOPPRA_PORT is already in use" >&2; exit 2; }
fi

STAMP="$(date +%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-$EXPO_ROOT/logs/wholechunk_kskip_vel_sweep_$STAMP}"

if [ "$DRY_RUN" = "1" ]; then
  echo "[DRY-RUN] backend: chunk_toppra"
  echo "[DRY-RUN] n_episodes_per_config: $N_EPISODES"
  echo "[DRY-RUN] vel_limits: $VEL_LIMITS"
  echo "[DRY-RUN] k_skips: $K_SKIPS"
  echo "[DRY-RUN] total_configs: $TOTAL_CONFIGS"
  echo "[DRY-RUN] total_episodes: $TOTAL_EPISODES"
  echo "[DRY-RUN] expo_commit: $EXPO_COMMIT"
  echo "[DRY-RUN] robotwin_commit: $ROBOTWIN_COMMIT"
  echo "[DRY-RUN] server backend=chunk_toppra gpu=$SERVER_GPU port=$CHUNK_TOPPRA_PORT"
  echo "[DRY-RUN] eval entrypoint=eval_speedtune_chunk_kskip_sweep.py gpu=$EVAL_GPU"
  echo "[DRY-RUN] output: $LOGDIR"
  exit 0
fi

mkdir -p "$LOGDIR"
cd "$EXPO_ROOT"

declare -a ALL_PIDS=()
declare -a ALL_LABELS=()
declare -a ALL_LOGS=()
STAGE="startup"

cleanup() {
  local rc="$?" pid alive
  trap - EXIT INT TERM
  echo "[cleanup] stage=$STAGE rc=$rc logdir=$LOGDIR"
  for pid in "${ALL_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then kill -TERM -- "-$pid" 2>/dev/null || true; fi
  done
  for _ in 1 2 3 4 5; do
    alive=0
    for pid in "${ALL_PIDS[@]:-}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
    [ "$alive" -eq 0 ] && break
    sleep 1
  done
  for pid in "${ALL_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then kill -KILL -- "-$pid" 2>/dev/null || true; fi
  done
  wait 2>/dev/null || true
  if [ "$rc" -ne 0 ]; then
    echo "[ERROR] failed or interrupted during $STAGE. Logs:"
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
  local pid="$1" port="$2" log="$3" elapsed=0
  while [ "$elapsed" -lt "$SERVER_WAIT" ]; do
    kill -0 "$pid" 2>/dev/null || {
      echo "[ERROR] chunk_toppra server exited"; tail -n 60 "$log"; return 1;
    }
    grep -Eq "Render Error|address already in use|Traceback" "$log" && {
      echo "[ERROR] chunk_toppra server initialization failed"; tail -n 60 "$log"; return 1;
    }
    if port_is_listening "$port"; then echo "[chunk_toppra] server ready on $port"; return 0; fi
    sleep 2; elapsed=$((elapsed + 2))
  done
  echo "[ERROR] chunk_toppra server not ready after ${SERVER_WAIT}s"; tail -n 60 "$log"; return 1
}

echo "[*] whole-chunk TOPPRA vel_limit × k_skip sweep"
echo "[*] episodes/config=$N_EPISODES configs=$TOTAL_CONFIGS total=$TOTAL_EPISODES"
echo "[*] vel_limits=$VEL_LIMITS k_skips=$K_SKIPS"
echo "[*] expo_commit=$EXPO_COMMIT robotwin_commit=$ROBOTWIN_COMMIT logdir=$LOGDIR"

STAGE="server startup"
SERVER_LOG="$LOGDIR/server_chunk_toppra.log"
setsid env CUDA_VISIBLE_DEVICES="$SERVER_GPU" conda run --no-capture-output -n "$ROBOTWIN_ENV" \
  python -m client_robotwin.run_robotwin_client \
    --config_task_path "$TASK_CONFIG" --robotwin_root "$ROBOTWIN_ROOT" --server_port "$CHUNK_TOPPRA_PORT" \
    >"$SERVER_LOG" 2>&1 &
pid=$!; ALL_PIDS+=("$pid"); ALL_LABELS+=("server:chunk_toppra"); ALL_LOGS+=("$SERVER_LOG")
wait_server "$pid" "$CHUNK_TOPPRA_PORT" "$SERVER_LOG"

STAGE="whole-chunk k-skip sweep"
EVAL_LOG="$LOGDIR/eval.log"
setsid env CUDA_VISIBLE_DEVICES="$EVAL_GPU" XLA_PYTHON_CLIENT_MEM_FRACTION="$EVAL_MEM_FRAC" \
  "${PYTHON_CMD[@]}" eval_speedtune_chunk_kskip_sweep.py \
    --config "$MODEL_CONFIG" --config_task "$TASK_CONFIG" \
    --config.stream_hold_steps "$STREAM_HOLD_STEPS" \
    --chunk_toppra_port "$CHUNK_TOPPRA_PORT" \
    --n_episodes "$N_EPISODES" --vel_limits "$VEL_LIMITS" --k_skips "$K_SKIPS" \
    --seed "$SEED" --max_decision_steps "$MAX_DECISION_STEPS" --client_host localhost \
    --output_dir "$LOGDIR" >"$EVAL_LOG" 2>&1 &
pid=$!; ALL_PIDS+=("$pid"); ALL_LABELS+=("wholechunk-kskip-sweep"); ALL_LOGS+=("$EVAL_LOG")
if ! wait "$pid"; then
  echo "[ERROR] whole-chunk k-skip sweep failed"; tail -n 100 "$EVAL_LOG"; exit 1
fi

STAGE="complete"
echo "[*] whole-chunk k-skip sweep complete: $LOGDIR"
echo "    wholechunk_kskip_vel_summary.json / .csv / wholechunk_kskip_vel_report.md"
echo "    k*/vel_limit_*/episodes.json contains per-episode traces"
