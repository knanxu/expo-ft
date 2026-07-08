#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$ROOT/scripts/run_speedtune_paper_alpha_sweep.sh"

[ -x "$SCRIPT" ]
grep -q 'ALPHAS="${ALPHAS:-1e-5 3e-5 1e-4 3e-4 1e-3}"' "$SCRIPT"
grep -q 'BACKENDS="${BACKENDS:-fixed_time chunk_toppra}"' "$SCRIPT"
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
grep -q 'alpha=1e-5' <<<"$output"
grep -q 'alpha=1e-4' <<<"$output"
grep -q -- '--config.reward_mode paper_speedtuning' <<<"$output"
grep -q -- '--config.reward_alpha 1e-4' <<<"$output"

echo "run_speedtune_paper_alpha_sweep dry-run test passed"
