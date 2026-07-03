#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$ROOT/scripts/run_speedtune_fixed_sweep.sh"

[ -x "$SCRIPT" ]
grep -q 'N_EPISODES="${N_EPISODES:-30}"' "$SCRIPT"
grep -q 'SPEEDS="${SPEEDS:-1,1.5,2,2.5,3,3.5,4}"' "$SCRIPT"
grep -q 'FIXED_TIME_K_SKIP="${FIXED_TIME_K_SKIP:-10}"' "$SCRIPT"
grep -q 'CHUNK_TOPPRA_K_SKIP="${CHUNK_TOPPRA_K_SKIP:-20}"' "$SCRIPT"
grep -q 'kill -TERM -- "-\$pid"' "$SCRIPT"
grep -q 'setsid env CUDA_VISIBLE_DEVICES' "$SCRIPT"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
touch "$tmp/vla.ckpt"

output="$(DRY_RUN=1 EXPO_ROOT="$ROOT" ROBOTWIN_ROOT="/home/xukainan/RoboTwin" \
  SPEEDTUNE_VLA_CKPT="$tmp/vla.ckpt" bash "$SCRIPT" 2>&1)"
grep -q 'backends: fixed_time chunk_toppra' <<<"$output"
grep -q 'n_episodes_per_config: 30' <<<"$output"
grep -q 'total_episodes: 420' <<<"$output"
grep -q 'speeds: 1,1.5,2,2.5,3,3.5,4' <<<"$output"
grep -q 'k_skip: fixed_time=10 chunk_toppra=20' <<<"$output"
grep -q 'expo_commit:' <<<"$output"
grep -q 'robotwin_commit:' <<<"$output"
grep -q 'server backend=fixed_time' <<<"$output"
grep -q 'server backend=chunk_toppra' <<<"$output"
grep -q 'eval entrypoint=eval_speedtune_fixed_sweep.py' <<<"$output"

set +e
cleanup_output="$(CLEANUP_SELF_TEST=1 LOGDIR="$tmp/cleanup-logs" \
  EXPO_ROOT="$ROOT" ROBOTWIN_ROOT="/home/xukainan/RoboTwin" \
  SPEEDTUNE_VLA_CKPT="$tmp/vla.ckpt" bash "$SCRIPT" 2>&1)"
cleanup_rc=$?
set -e
[ "$cleanup_rc" -eq 143 ]
cleanup_pid="$(sed -n 's/.*cleanup_self_test_pid=\([0-9][0-9]*\).*/\1/p' <<<"$cleanup_output")"
[ -n "$cleanup_pid" ]
if kill -0 "$cleanup_pid" 2>/dev/null; then
  echo "cleanup self-test left process $cleanup_pid running" >&2
  kill -9 "$cleanup_pid" 2>/dev/null || true
  exit 1
fi
grep -q 'stage=cleanup self-test' <<<"$cleanup_output"

echo "run_speedtune_fixed_sweep dry-run tests passed"
