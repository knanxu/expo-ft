#!/usr/bin/env bash
# TEMPORARY wrapper: train/eval both SpeedTune backends, then run the fixed-speed sweep.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPO_ROOT="${EXPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
TRAIN_EVAL_SCRIPT="$EXPO_ROOT/scripts/run_speedtune_train_eval.sh"
FIXED_SWEEP_SCRIPT="$EXPO_ROOT/scripts/run_speedtune_fixed_sweep.sh"

EVAL_N_EPISODES="${EVAL_N_EPISODES:-30}"
TRAIN_MAX_ITERS="${TRAIN_MAX_ITERS:-${MAX_ITERS:-20000}}"
TRAIN_MODE="${TRAIN_MODE:-sync}"
STREAM_HOLD_STEPS="${STREAM_HOLD_STEPS:-15}"
CHUNK_TOPPRA_K_SKIP="${CHUNK_TOPPRA_K_SKIP:-40}"
VIDEO_EPISODES="${VIDEO_EPISODES:-5}"
MAX_DECISION_STEPS="${MAX_DECISION_STEPS:-400}"
SWEEP_N_EPISODES="${SWEEP_N_EPISODES:-30}"
SWEEP_SPEEDS="${SWEEP_SPEEDS:-1,1.5,2,2.5,3,3.5,4}"
BACKENDS="fixed_time chunk_toppra"
DRY_RUN="${DRY_RUN:-0}"

STAMP="$(date +%m%d_%H%M%S)"
PIPELINE_ROOT="${PIPELINE_ROOT:-$EXPO_ROOT/logs/speedtune_full_pipeline_$STAMP}"
TRAIN_EVAL_DIR="$PIPELINE_ROOT/train_eval"
FIXED_SWEEP_DIR="$PIPELINE_ROOT/fixed_speed_sweep"

[ -x "$TRAIN_EVAL_SCRIPT" ] || {
  echo "[ERROR] missing executable: $TRAIN_EVAL_SCRIPT" >&2; exit 2;
}
[ -x "$FIXED_SWEEP_SCRIPT" ] || {
  echo "[ERROR] missing executable: $FIXED_SWEEP_SCRIPT" >&2; exit 2;
}

mkdir -p "$PIPELINE_ROOT"

CURRENT_PID=""
STAGE="startup"
cleanup() {
  local rc="$?"
  trap - EXIT INT TERM
  if [ -n "$CURRENT_PID" ] && kill -0 "$CURRENT_PID" 2>/dev/null; then
    kill -TERM -- "-$CURRENT_PID" 2>/dev/null || true
    wait "$CURRENT_PID" 2>/dev/null || true
  fi
  if [ "$rc" -ne 0 ]; then
    echo "[ERROR] temporary full pipeline stopped during: $STAGE" >&2
    echo "        output root: $PIPELINE_ROOT" >&2
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "[*] temporary SpeedTune full pipeline"
echo "[*] output root: $PIPELINE_ROOT"

STAGE="stage 1/2: train + trained-policy eval"
echo "[*] $STAGE"
setsid env \
  BACKENDS="$BACKENDS" \
  MAX_ITERS="$TRAIN_MAX_ITERS" \
  TRAIN_MODE="$TRAIN_MODE" \
  STREAM_HOLD_STEPS="$STREAM_HOLD_STEPS" \
  CHUNK_TOPPRA_K_SKIP="$CHUNK_TOPPRA_K_SKIP" \
  N_EPISODES="$EVAL_N_EPISODES" \
  VIDEO_EPISODES="$VIDEO_EPISODES" \
  MAX_DECISION_STEPS="$MAX_DECISION_STEPS" \
  LOGDIR="$TRAIN_EVAL_DIR" \
  DRY_RUN="$DRY_RUN" \
  bash "$TRAIN_EVAL_SCRIPT" &
CURRENT_PID=$!
wait "$CURRENT_PID"
CURRENT_PID=""

STAGE="stage 2/2: fixed raw-speed sweep"
echo "[*] $STAGE"
setsid env \
  N_EPISODES="$SWEEP_N_EPISODES" \
  SPEEDS="$SWEEP_SPEEDS" \
  CHUNK_TOPPRA_K_SKIP="$CHUNK_TOPPRA_K_SKIP" \
  LOGDIR="$FIXED_SWEEP_DIR" \
  DRY_RUN="$DRY_RUN" \
  bash "$FIXED_SWEEP_SCRIPT" &
CURRENT_PID=$!
wait "$CURRENT_PID"
CURRENT_PID=""

STAGE="complete"
echo "[*] temporary full pipeline complete"
echo "    train_eval_output: $TRAIN_EVAL_DIR"
echo "    fixed_sweep_output: $FIXED_SWEEP_DIR"
