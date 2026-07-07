#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$ROOT/scripts/run_wholechunk_kskip_vel_sweep.sh"

[ -x "$SCRIPT" ]
grep -q 'N_EPISODES="${N_EPISODES:-50}"' "$SCRIPT"
grep -q 'VEL_LIMITS="${VEL_LIMITS:-1,1.5,2,2.5,3,3.5,4}"' "$SCRIPT"
grep -q 'K_SKIPS="${K_SKIPS:-10,20,40,50}"' "$SCRIPT"
grep -q 'eval_speedtune_chunk_kskip_sweep.py' "$SCRIPT"
grep -q -- '--vel_limits "$VEL_LIMITS"' "$SCRIPT"
grep -q -- '--k_skips "$K_SKIPS"' "$SCRIPT"
grep -q 'kill -TERM -- "-\$pid"' "$SCRIPT"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
touch "$tmp/vla.ckpt"

output="$(DRY_RUN=1 EXPO_ROOT="$ROOT" ROBOTWIN_ROOT="/home/xukainan/RoboTwin" \
  SPEEDTUNE_VLA_CKPT="$tmp/vla.ckpt" bash "$SCRIPT" 2>&1)"
grep -q 'backend: chunk_toppra' <<<"$output"
grep -q 'n_episodes_per_config: 50' <<<"$output"
grep -q 'vel_limits: 1,1.5,2,2.5,3,3.5,4' <<<"$output"
grep -q 'k_skips: 10,20,40,50' <<<"$output"
grep -q 'total_configs: 28' <<<"$output"
grep -q 'total_episodes: 1400' <<<"$output"
grep -q 'server backend=chunk_toppra' <<<"$output"
grep -q 'eval entrypoint=eval_speedtune_chunk_kskip_sweep.py' <<<"$output"

override="$(DRY_RUN=1 EXPO_ROOT="$ROOT" ROBOTWIN_ROOT="/home/xukainan/RoboTwin" \
  SPEEDTUNE_VLA_CKPT="$tmp/vla.ckpt" VEL_LIMITS=1,2 K_SKIPS=20,40 N_EPISODES=3 \
  bash "$SCRIPT" 2>&1)"
grep -q 'n_episodes_per_config: 3' <<<"$override"
grep -q 'total_configs: 4' <<<"$override"
grep -q 'total_episodes: 12' <<<"$override"

echo "run_wholechunk_kskip_vel_sweep dry-run tests passed"
