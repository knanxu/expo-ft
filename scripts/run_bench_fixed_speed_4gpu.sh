#!/usr/bin/env bash
# 4 卡（RTX 5880 Ada 48GB）并行跑 bench_fixed_speed：每卡 1 个 (server + bench) pair，
# bench 用 --config_subset i/N 分摊固定加速配置（默认 25 个，both backends）。
#
# 关键：server(sapien 光追) 与 bench(3B 冻结 VLA, jax) **同卡**——48GB 够，但 bench 的
# jax 必须限显存(XLA_PYTHON_CLIENT_MEM_FRACTION=MEM_FRAC)给同卡 sapien 留空间，否则
# server 会 "cannot create buffer"。
#
# 卡分配（N_SHARDS=4）：
#   shard i  → server GPU i 端口 (BASE_PORT+i) | bench GPU i --config_subset i/4 --client_port (BASE_PORT+i)
#
# 用法（云端）：
#   export SPEEDTUNE_VLA_CKPT=/abs/.../params   # 或用下面默认 _VLA_ROOT
#   bash scripts/run_bench_fixed_speed_4gpu.sh
#   冒烟：N_EPISODES=3 bash scripts/run_bench_fixed_speed_4gpu.sh
#   纯速度系数(关力矩底座)：FORCE_LIMIT=none bash scripts/run_bench_fixed_speed_4gpu.sh
#
# 跑前清残留（光追退出不彻底会占显存 → cannot create buffer）：
#   pkill -9 -f run_robotwin_client; pkill -9 -f bench_fixed_speed; sleep 3; nvidia-smi
# 停止：Ctrl-C（trap 清理）。

set -euo pipefail

# ===== 配置（环境变量可覆盖）=====
EXPO_ROOT="${EXPO_ROOT:-/home/chenlu/expo-ft}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/home/chenlu/RoboTwin}"
ROBOTWIN_ENV="${ROBOTWIN_ENV:-RoboTwin}"           # RoboTwin sim 的 conda env 名
PYTHON="${PYTHON:-uv run python}"                  # learner 端 python（云端 uv）
TASK_CONFIG="${TASK_CONFIG:-configs/task/robotwin_stack_blocks.py}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model/speedtune_dqn_config.py}"
N_EPISODES="${N_EPISODES:-50}"
N_SHARDS="${N_SHARDS:-4}"                           # = 卡数 = 并行 bench 进程数
BACKENDS="${BACKENDS:-both}"                        # both | fixed_time | per_action
MAX_DECISION_STEPS="${MAX_DECISION_STEPS:-400}"
SEED="${SEED:-0}"
FORCE_LIMIT="${FORCE_LIMIT:-}"                      # 空=继承 config 力矩底座(30,40,...)；"none"=关掉(∞)
SERVER_WAIT="${SERVER_WAIT:-45}"                    # 等 server 渲染自检/就绪秒数
MEM_FRAC="${MEM_FRAC:-0.5}"                         # bench(jax) 单卡显存上限；留空间给同卡 sapien 光追
BASE_PORT="${BASE_PORT:-8102}"

# VLA（冻结 drift pi0.5, stack_blocks_two）：默认云端 ckpt 根目录（params/ 与 assets/ 平级，
# norm_stats 在 assets/trossen/）。坑：SPEEDTUNE_VLA_ASSETS 必须 <根>/assets，asset_id=trossen。
_VLA_ROOT="${SPEEDTUNE_VLA_ROOT:-/home/chenlu/openpi/checkpoints/pi05_aloha_robotwin_drifting_stack_blocks_two/drifting_v1/29999}"
export SPEEDTUNE_VLA_CKPT="${SPEEDTUNE_VLA_CKPT:-${_VLA_ROOT}/params}"
export SPEEDTUNE_VLA_ASSETS="${SPEEDTUNE_VLA_ASSETS:-${_VLA_ROOT}/assets}"
export SPEEDTUNE_VLA_ASSET_ID="${SPEEDTUNE_VLA_ASSET_ID:-trossen}"
# force_limit 力矩底座默认（config 也读它；bench --force_limit 不传时继承 config）。
export SPEEDTUNE_FORCE_LIMIT="${SPEEDTUNE_FORCE_LIMIT:-30,40,30,15,10,10}"

STAMP="$(date +%m%d_%H%M)"
LOGDIR="${LOGDIR:-${EXPO_ROOT}/logs/bench_fixed_speed_${STAMP}}"
mkdir -p "$LOGDIR"
echo "[*] 输出/日志: $LOGDIR"
echo "[*] 冻结 VLA: $SPEEDTUNE_VLA_CKPT"
echo "[*] N_SHARDS=$N_SHARDS N_EPISODES=$N_EPISODES backends=$BACKENDS MEM_FRAC=$MEM_FRAC force_limit=${FORCE_LIMIT:-<config默认>}"

SERVER_PIDS=(); BENCH_PIDS=()
cleanup() {
  echo ""
  echo "[cleanup] 终止所有子进程 ..."
  for p in "${BENCH_PIDS[@]:-}" "${SERVER_PIDS[@]:-}"; do kill -9 "$p" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM
cd "$EXPO_ROOT"

# ---- 0) 端口预检：被占用 = 残留旧 server，必须先清理（否则 silent 连到旧 server）----
for i in $(seq 0 $((N_SHARDS - 1))); do
  port=$((BASE_PORT + i))
  if (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null; then
    exec 3>&- 3<&- 2>/dev/null || true
    echo "[ERROR] 端口 ${port} 已被占用——极可能是残留 RoboTwin server。先清理后重跑：" >&2
    echo "        pkill -9 -f run_robotwin_client; pkill -9 -f bench_fixed_speed; sleep 3; nvidia-smi" >&2
    exit 1
  fi
done

# ---- 1) 每卡起 1 个 server（同卡 i，光追渲染，端口 BASE_PORT+i）----
for i in $(seq 0 $((N_SHARDS - 1))); do
  port=$((BASE_PORT + i))
  echo "[shard $i] 启动 server  GPU=$i  port=$port"
  CUDA_VISIBLE_DEVICES="$i" \
  conda run --no-capture-output -n "$ROBOTWIN_ENV" \
    python -m client_robotwin.run_robotwin_client \
      --config_task_path "$TASK_CONFIG" --robotwin_root "$ROBOTWIN_ROOT" --server_port "$port" \
      > "$LOGDIR/server_$i.log" 2>&1 &
  SERVER_PIDS+=($!)
done

echo "[*] 等 server 渲染自检/就绪 (${SERVER_WAIT}s)；正常会打印 'Render Well' ..."
sleep "$SERVER_WAIT"
for i in $(seq 0 $((N_SHARDS - 1))); do
  if grep -q "address already in use" "$LOGDIR/server_$i.log" 2>/dev/null; then
    echo "[ERROR] shard $i server 端口 bind 失败（见 server_$i.log）。清理后重跑。" >&2
    exit 1
  fi
  if grep -q "Render Well" "$LOGDIR/server_$i.log" 2>/dev/null; then
    echo "[shard $i] server ✓ Render Well"
  elif grep -q "Render Error" "$LOGDIR/server_$i.log" 2>/dev/null; then
    echo "[shard $i] server ✗ Render Error（看 server_$i.log）"
  else
    echo "[shard $i] server 状态未知（可能还在初始化）；若 bench 报 cannot create buffer 看 server_$i.log"
  fi
done

# ---- 2) 每卡起 1 个 bench（同卡 i，--config_subset i/N，--client_port BASE_PORT+i）----
FL_ARG=()
[ -n "$FORCE_LIMIT" ] && FL_ARG=(--force_limit "$FORCE_LIMIT")
for i in $(seq 0 $((N_SHARDS - 1))); do
  port=$((BASE_PORT + i))
  echo "[shard $i] 启动 bench  GPU=$i  subset=$i/$N_SHARDS  client_port=$port"
  CUDA_VISIBLE_DEVICES="$i" XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRAC" \
    $PYTHON scripts/bench_fixed_speed.py \
      --config "$MODEL_CONFIG" --config_task "$TASK_CONFIG" \
      --n_episodes "$N_EPISODES" --seed "$SEED" \
      --max_decision_steps "$MAX_DECISION_STEPS" \
      --backends "$BACKENDS" --config_subset "$i/$N_SHARDS" \
      "${FL_ARG[@]}" \
      --client_host localhost --client_port "$port" \
      --output_dir "$LOGDIR/shard_$i" \
      > "$LOGDIR/bench_$i.log" 2>&1 &
  BENCH_PIDS+=($!)
  sleep 5
done

echo "[*] $N_SHARDS 片 bench 跑中；实时看：tail -f $LOGDIR/bench_0.log"
rc=0
for i in "${!BENCH_PIDS[@]}"; do
  if wait "${BENCH_PIDS[$i]}"; then echo "[shard $i] 完成 ✓"; else echo "[shard $i] 异常退出 ✗（见 bench_$i.log）" >&2; rc=1; fi
done

# ---- 3) 合并各片 csv + 画完整图（纯 csv + matplotlib，用 CPU 不占 GPU）----
echo "[*] 合并 + 画完整图 → $LOGDIR/merged"
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
  $PYTHON scripts/bench_fixed_speed.py \
    --config "$MODEL_CONFIG" --config_task "$TASK_CONFIG" \
    --plot_from_csv "$LOGDIR/shard_*/bench_summary.csv" \
    --output_dir "$LOGDIR/merged" \
    > "$LOGDIR/merge.log" 2>&1 || echo "[WARN] 合并画图失败（看 merge.log）；可手动 cat shard_*/bench_summary.csv"

echo ""
echo "[*] 完成。各片: $LOGDIR/shard_*/bench_summary.csv ; 合并: $LOGDIR/merged/（bench_summary.csv + *.png）"
[ "$rc" -ne 0 ] && echo "[WARN] 有片异常退出，合并结果可能不全（看对应 bench_<i>.log）" >&2
exit "$rc"
