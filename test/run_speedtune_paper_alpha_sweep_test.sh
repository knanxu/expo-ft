#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$ROOT/scripts/run_speedtune_paper_alpha_sweep.sh"

[ -x "$SCRIPT" ]
grep -q 'ALPHAS="${ALPHAS:-1e-5 3e-5 1e-4 3e-4 1e-3}"' "$SCRIPT"
grep -q 'BACKENDS="${BACKENDS:-fixed_time chunk_toppra}"' "$SCRIPT"
grep -q 'MAX_ITERS="${MAX_ITERS:-40000}"' "$SCRIPT"
grep -q 'EPSILON_DECAY_STEPS="${EPSILON_DECAY_STEPS:-8000}"' "$SCRIPT"
grep -q 'CURRICULUM_ENABLED="${CURRICULUM_ENABLED:-True}"' "$SCRIPT"
grep -q 'CURRICULUM_WINDOW_SIZE="${CURRICULUM_WINDOW_SIZE:-20}"' "$SCRIPT"
grep -q 'CURRICULUM_SUCCESS_THRESHOLD="${CURRICULUM_SUCCESS_THRESHOLD:-0.7}"' "$SCRIPT"
grep -q 'run_speedtune_train_eval.sh' "$SCRIPT"
grep -q 'paper_speedtuning' "$SCRIPT"
grep -q 'CONFIG_OVERRIDES=' "$SCRIPT"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
touch "$tmp/vla.ckpt"

output="$(DRY_RUN=1 ALPHAS='1e-5 1e-4' \
  EXPO_ROOT="$ROOT" ROBOTWIN_ROOT="/home/xukainan/RoboTwin" \
  SPEEDTUNE_VLA_CKPT="$tmp/vla.ckpt" \
  SPEEDTUNE_VLA_ASSETS="$tmp" SPEEDTUNE_VLA_ASSET_ID=trossen \
  bash "$SCRIPT" 2>&1)"

grep -q 'paper alpha sweep' <<<"$output"
grep -q 'alphas: 1e-5 1e-4' <<<"$output"
grep -q 'backends: fixed_time chunk_toppra' <<<"$output"
grep -q 'max_iters: 40000 train_mode: sync' <<<"$output"
grep -q 'curriculum: enabled=True window=20 threshold=0.7' <<<"$output"
grep -q 'epsilon_decay_steps: 8000' <<<"$output"
grep -q 'alpha=1e-5' <<<"$output"
grep -q 'alpha=1e-4' <<<"$output"
grep -q -- '--config.reward_mode paper_speedtuning' <<<"$output"
grep -q -- '--config.reward_alpha 1e-4' <<<"$output"
grep -q -- '--config.curriculum_enabled True' <<<"$output"
grep -q -- '--config.curriculum_window_size 20' <<<"$output"
grep -q -- '--config.curriculum_success_threshold 0.7' <<<"$output"
grep -q -- '--config.epsilon_decay_steps 8000' <<<"$output"

echo "run_speedtune_paper_alpha_sweep dry-run test passed"
