#!/usr/bin/env bash
# Cloud helper: sweep paper SpeedTuning reward alpha for fixed_time and whole-chunk TOPPRA.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPO_ROOT="${EXPO_ROOT:-/home/chenlu/expo-ft}"
export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/home/chenlu/RoboTwin}"
export ROBOTWIN_ENV="${ROBOTWIN_ENV:-RoboTwin}"
export PYTHON="${PYTHON:-uv run python}"

CKPT_ROOT="${CKPT_ROOT:-/home/chenlu/openpi/checkpoints/pi05_aloha_robotwin_drifting_stack_blocks_two/drifting_v1/29999}"
export SPEEDTUNE_VLA_CKPT="${SPEEDTUNE_VLA_CKPT:-$CKPT_ROOT/params}"
export SPEEDTUNE_VLA_ASSETS="${SPEEDTUNE_VLA_ASSETS:-$CKPT_ROOT/assets}"
export SPEEDTUNE_VLA_ASSET_ID="${SPEEDTUNE_VLA_ASSET_ID:-trossen}"

ALPHAS="${ALPHAS:-1e-5 3e-5 1e-4 3e-4 1e-3}"
BACKENDS="${BACKENDS:-fixed_time chunk_toppra}"
TRAIN_MODE="${TRAIN_MODE:-sync}"
MAX_ITERS="${MAX_ITERS:-40000}"
N_EPISODES="${N_EPISODES:-30}"
VIDEO_EPISODES="${VIDEO_EPISODES:-5}"
CHUNK_TOPPRA_K_SKIP="${CHUNK_TOPPRA_K_SKIP:-40}"
STREAM_HOLD_STEPS="${STREAM_HOLD_STEPS:-15}"
EPSILON_DECAY_STEPS="${EPSILON_DECAY_STEPS:-8000}"
CURRICULUM_ENABLED="${CURRICULUM_ENABLED:-True}"
CURRICULUM_WINDOW_SIZE="${CURRICULUM_WINDOW_SIZE:-20}"
CURRICULUM_SUCCESS_THRESHOLD="${CURRICULUM_SUCCESS_THRESHOLD:-0.7}"
BASE_LOGDIR="${BASE_LOGDIR:-$EXPO_ROOT/logs/speedtune_paper_alpha_sweep_$(date +%m%d_%H%M%S)}"

echo "[*] paper alpha sweep"
echo "[*] expo_root: $EXPO_ROOT"
echo "[*] robotwin_root: $ROBOTWIN_ROOT"
echo "[*] alphas: $ALPHAS"
echo "[*] backends: $BACKENDS"
echo "[*] max_iters: $MAX_ITERS train_mode: $TRAIN_MODE"
echo "[*] curriculum: enabled=$CURRICULUM_ENABLED window=$CURRICULUM_WINDOW_SIZE threshold=$CURRICULUM_SUCCESS_THRESHOLD"
echo "[*] epsilon_decay_steps: $EPSILON_DECAY_STEPS"
echo "[*] base_logdir: $BASE_LOGDIR"

for alpha in $ALPHAS; do
  logdir="$BASE_LOGDIR/alpha_${alpha}"
  overrides="--config.reward_mode paper_speedtuning --config.reward_alpha $alpha --config.reward_beta 2.0 --config.curriculum_enabled $CURRICULUM_ENABLED --config.curriculum_window_size $CURRICULUM_WINDOW_SIZE --config.curriculum_success_threshold $CURRICULUM_SUCCESS_THRESHOLD --config.epsilon_decay_steps $EPSILON_DECAY_STEPS"

  echo "[*] alpha=$alpha logdir=$logdir"
  echo "[*] overrides: $overrides"

  BACKENDS="$BACKENDS" \
  TRAIN_MODE="$TRAIN_MODE" \
  MAX_ITERS="$MAX_ITERS" \
  N_EPISODES="$N_EPISODES" \
  VIDEO_EPISODES="$VIDEO_EPISODES" \
  CHUNK_TOPPRA_K_SKIP="$CHUNK_TOPPRA_K_SKIP" \
  STREAM_HOLD_STEPS="$STREAM_HOLD_STEPS" \
  LOGDIR="$logdir" \
  CONFIG_OVERRIDES="$overrides" \
  bash "$SCRIPT_DIR/run_speedtune_train_eval.sh"
done

echo "[*] paper alpha sweep complete: $BASE_LOGDIR"
