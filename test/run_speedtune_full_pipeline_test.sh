#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$ROOT/scripts/run_speedtune_full_pipeline.sh"

[ -x "$SCRIPT" ]
grep -q 'TEMPORARY' "$SCRIPT"
grep -q 'run_speedtune_train_eval.sh' "$SCRIPT"
grep -q 'run_speedtune_fixed_sweep.sh' "$SCRIPT"
grep -q 'EVAL_N_EPISODES="${EVAL_N_EPISODES:-30}"' "$SCRIPT"
grep -q 'TRAIN_MAX_ITERS="${TRAIN_MAX_ITERS:-${MAX_ITERS:-20000}}"' "$SCRIPT"
grep -q 'CHUNK_TOPPRA_K_SKIP="${CHUNK_TOPPRA_K_SKIP:-40}"' "$SCRIPT"
grep -q 'SWEEP_N_EPISODES="${SWEEP_N_EPISODES:-30}"' "$SCRIPT"
grep -q 'SWEEP_SPEEDS="${SWEEP_SPEEDS:-1,1.5,2,2.5,3,3.5,4}"' "$SCRIPT"
grep -q 'BACKENDS="fixed_time chunk_toppra"' "$SCRIPT"
grep -q 'MAX_ITERS="$TRAIN_MAX_ITERS"' "$SCRIPT"
grep -q 'CHUNK_TOPPRA_K_SKIP="$CHUNK_TOPPRA_K_SKIP"' "$SCRIPT"
grep -q 'kill -TERM -- "-\$CURRENT_PID"' "$SCRIPT"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
touch "$tmp/vla.ckpt"

output="$(DRY_RUN=1 PIPELINE_ROOT="$tmp/pipeline" \
  EXPO_ROOT="$ROOT" ROBOTWIN_ROOT="/home/xukainan/RoboTwin" \
  SPEEDTUNE_VLA_CKPT="$tmp/vla.ckpt" bash "$SCRIPT" 2>&1)"

grep -q 'stage 1/2: train + trained-policy eval' <<<"$output"
grep -q 'backends: fixed_time chunk_toppra' <<<"$output"
grep -q 'max_iters: 20000' <<<"$output"
grep -q 'chunk_toppra_k_skip: 40' <<<"$output"
grep -q 'stage 2/2: fixed raw-speed sweep' <<<"$output"
grep -q 'k_skip: fixed_time=10 chunk_toppra=40' <<<"$output"
grep -q 'n_episodes_per_config: 30' <<<"$output"
grep -q 'total_episodes: 420' <<<"$output"
grep -q "train_eval_output: $tmp/pipeline/train_eval" <<<"$output"
grep -q "fixed_sweep_output: $tmp/pipeline/fixed_speed_sweep" <<<"$output"

override="$(DRY_RUN=1 PIPELINE_ROOT="$tmp/pipeline-override" TRAIN_MAX_ITERS=1234 \
  EXPO_ROOT="$ROOT" ROBOTWIN_ROOT="/home/xukainan/RoboTwin" \
  SPEEDTUNE_VLA_CKPT="$tmp/vla.ckpt" bash "$SCRIPT" 2>&1)"
grep -q 'max_iters: 1234' <<<"$override"

echo "run_speedtune_full_pipeline dry-run test passed"
