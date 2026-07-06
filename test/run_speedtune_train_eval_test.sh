#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$ROOT/scripts/run_speedtune_train_eval.sh"

[ -x "$SCRIPT" ]
grep -q 'BACKENDS=(fixed_time chunk_toppra)' "$SCRIPT"
grep -q 'N_EPISODES="${N_EPISODES:-30}"' "$SCRIPT"
grep -q 'VIDEO_EPISODES="${VIDEO_EPISODES:-5}"' "$SCRIPT"
grep -q 'CHUNK_TOPPRA_K_SKIP="${CHUNK_TOPPRA_K_SKIP:-40}"' "$SCRIPT"
grep -q -- '--config.chunk_toppra_k_skip "$CHUNK_TOPPRA_K_SKIP"' "$SCRIPT"
grep -q -- '--record_video' "$SCRIPT"
grep -q 'DRY_RUN' "$SCRIPT"
grep -q 'CLEANUP_SELF_TEST' "$SCRIPT"
grep -q 'kill -TERM -- "-\$pid"' "$SCRIPT"
grep -qx 'cd "$EXPO_ROOT"' "$SCRIPT"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
touch "$tmp/vla.ckpt"

output="$({
  env -u BACKENDS DRY_RUN=1 \
    EXPO_ROOT="$ROOT" ROBOTWIN_ROOT="/home/xukainan/RoboTwin" \
    SPEEDTUNE_VLA_CKPT="$tmp/vla.ckpt" \
    bash "$SCRIPT"
} 2>&1)"
grep -q 'backends: fixed_time chunk_toppra' <<<"$output"
grep -q 'n_episodes: 30' <<<"$output"
grep -q 'chunk_toppra_k_skip: 40' <<<"$output"
grep -q 'train_mode: sync' <<<"$output"
grep -q 'trainer: train_speedtune_sync.py' <<<"$output"
grep -q 'video_episodes: 5' <<<"$output"
grep -q 'expo_commit:' <<<"$output"
grep -q 'robotwin_commit:' <<<"$output"
grep -q 'backend_a=fixed_time' <<<"$output"
grep -q 'backend_b=chunk_toppra' <<<"$output"

dynamic_root="$(env -u EXPO_ROOT -u ROBOTWIN_ROOT -u BACKENDS DRY_RUN=1 \
  SPEEDTUNE_VLA_CKPT="$tmp/vla.ckpt" bash "$SCRIPT" 2>&1)"
grep -q 'backends: fixed_time chunk_toppra' <<<"$dynamic_root"

override="$(DRY_RUN=1 BACKENDS='chunk_toppra' \
  EXPO_ROOT="$ROOT" ROBOTWIN_ROOT="/home/xukainan/RoboTwin" \
  SPEEDTUNE_VLA_CKPT="$tmp/vla.ckpt" bash "$SCRIPT" 2>&1)"
grep -q 'backends: chunk_toppra' <<<"$override"

kskip_override="$(DRY_RUN=1 BACKENDS='chunk_toppra' CHUNK_TOPPRA_K_SKIP=37 \
  EXPO_ROOT="$ROOT" ROBOTWIN_ROOT="/home/xukainan/RoboTwin" \
  SPEEDTUNE_VLA_CKPT="$tmp/vla.ckpt" bash "$SCRIPT" 2>&1)"
grep -q 'chunk_toppra_k_skip: 37' <<<"$kskip_override"

async_mode="$(DRY_RUN=1 TRAIN_MODE=async \
  EXPO_ROOT="$ROOT" ROBOTWIN_ROOT="/home/xukainan/RoboTwin" \
  SPEEDTUNE_VLA_CKPT="$tmp/vla.ckpt" bash "$SCRIPT" 2>&1)"
grep -q 'train_mode: async' <<<"$async_mode"
grep -q 'trainer: train_speedtune_async.py' <<<"$async_mode"

if DRY_RUN=1 TRAIN_MODE='bad_mode' EXPO_ROOT="$ROOT" \
  ROBOTWIN_ROOT="/home/xukainan/RoboTwin" SPEEDTUNE_VLA_CKPT="$tmp/vla.ckpt" \
  bash "$SCRIPT" >/dev/null 2>&1; then
  echo "invalid training mode unexpectedly succeeded" >&2
  exit 1
fi

if DRY_RUN=1 BACKENDS='bad_backend' EXPO_ROOT="$ROOT" \
  ROBOTWIN_ROOT="/home/xukainan/RoboTwin" SPEEDTUNE_VLA_CKPT="$tmp/vla.ckpt" \
  bash "$SCRIPT" >/dev/null 2>&1; then
  echo "invalid backend unexpectedly succeeded" >&2
  exit 1
fi

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

echo "run_speedtune_train_eval dry-run tests passed"
