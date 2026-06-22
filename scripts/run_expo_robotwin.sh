#!/usr/bin/env bash
# EXPO-FT + drift pi0.5 + RoboTwin + **自动专家介入** 云端训练脚本。
#
# 结构：① RoboTwin env server（conda RoboTwin env，光追独占一卡）；② EXPO 训练
# （expo-ft venv，train_pi_robo_async 需 ≥2 卡：sample device[0] + update device[1:]）。
#
# 自动专家介入（替代人类在环，打破 rollout 0% 死锁）：rollout 用到 takeover_step_frac 比例 step
# 预算仍未 success → RoboTwin expert 从当前状态接管，录 play_once 专家带逐 step 回放
# （action_type="human"，复用 learner is_hil 通路）。参数见 configs/task/robotwin_stack_blocks.py
# 的 takeover_enable/takeover_step_frac/takeover_save_freq，可用 --config_task.xxx 覆盖。
#
# 用法（云端）：
#   export VLA_CKPT=/abs/drift_pi05/checkpoint              # 必填：微调后 drift pi0.5 ckpt
#   export VLA_ASSETS_DIR=$VLA_CKPT/assets                  # 必填：norm_stats assets 目录
#   export VLA_ASSET_ID=<assets 下子目录名,如 trossen>      # 必填：与 eval 同一份 norm_stats
#   export DATASET_PATH=/abs/robotwin/<task>/<cfg>/data     # 必填：offline demo（critic 暖启 + success-only BC）
#   bash scripts/run_expo_robotwin.sh
#
# 跑前清理残留（光追 sapien 退出不彻底会占显存）：
#   pkill -9 -f run_robotwin_client; pkill -9 -f train_pi_robo_async; sleep 3; nvidia-smi
#
# 对照实验（关掉接管，复现旧的纯 offline-demo 行为，rollout 应仍 0%）：
#   加 --config_task.takeover_enable=False
#
# 停止：Ctrl-C（trap 清理）。日志：logs/<run_name>/。

set -euo pipefail

# ===== 配置（环境变量可覆盖）=====
EXPO_ROOT="${EXPO_ROOT:-/home/chenlu/expo-ft}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/home/chenlu/RoboTwin}"
ROBOTWIN_ENV="${ROBOTWIN_ENV:-RoboTwin}"          # RoboTwin sim 的 conda env 名
PYTHON="${PYTHON:-uv run python}"                 # learner 端 python（云端 uv）
TASK_CONFIG="${TASK_CONFIG:-configs/task/robotwin_stack_blocks.py}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model/expo_ft_pi_drift_config.py}"
WANDB_PROJECT="${WANDB_PROJECT:-expo-ft-robotwin}"
RUN_NAME="${RUN_NAME:-expo_robotwin_takeover_$(date +%m%d_%H%M)}"
SERVER_PORT="${SERVER_PORT:-8102}"
SERVER_WAIT="${SERVER_WAIT:-45}"                  # 等 server 渲染自检/就绪秒数
REPLAN_STEPS="${REPLAN_STEPS:-8}"                 # EXPO 执行前多少步再 replan（drift 仍预测 H=50）
MAX_STEPS="${MAX_STEPS:-100000}"
SEED="${SEED:-42}"
SERVER_GPU="${SERVER_GPU:-0}"                     # server（sapien 光追）卡
TRAIN_GPUS="${TRAIN_GPUS:-1,2}"                   # train（3B VLA + critic）卡；需 ≥2（sample[0]+update[1:]）

: "${VLA_CKPT:?请 export VLA_CKPT=<微调后 drift pi0.5 ckpt 绝对路径>}"
: "${VLA_ASSETS_DIR:?请 export VLA_ASSETS_DIR=<norm_stats assets 目录，如 \$VLA_CKPT/assets>}"
: "${VLA_ASSET_ID:?请 export VLA_ASSET_ID=<assets 下子目录名，如 trossen>}"
: "${DATASET_PATH:?请 export DATASET_PATH=<RoboTwin offline demo data 目录>}"

LOGDIR="${EXPO_ROOT}/logs/${RUN_NAME}"
mkdir -p "$LOGDIR"
echo "[*] 日志目录: $LOGDIR"
echo "[*] VLA ckpt: $VLA_CKPT   norm_stats: $VLA_ASSETS_DIR/$VLA_ASSET_ID"
echo "[*] offline demo: $DATASET_PATH"
echo "[*] 分卡: server=GPU$SERVER_GPU  train=GPU$TRAIN_GPUS"

PIDS=()
cleanup() {
  echo ""
  echo "[cleanup] 终止所有子进程 ..."
  for p in "${PIDS[@]:-}"; do kill -9 "$p" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "$EXPO_ROOT"

# ---- 1) RoboTwin env server（conda RoboTwin env，光追独占 SERVER_GPU）----
echo "[server] 启动 env server  GPU=$SERVER_GPU  port=$SERVER_PORT"
CUDA_VISIBLE_DEVICES="$SERVER_GPU" \
conda run --no-capture-output -n "$ROBOTWIN_ENV" \
  python -m client_robotwin.run_robotwin_client \
    --config_task_path "$TASK_CONFIG" --robotwin_root "$ROBOTWIN_ROOT" --server_port "$SERVER_PORT" \
    > "$LOGDIR/server.log" 2>&1 &
PIDS+=($!)

echo "[*] 等 server 渲染自检/就绪 (${SERVER_WAIT}s)；正常会在 server 日志打印 'Render Well' ..."
sleep "$SERVER_WAIT"
if grep -q "Render Well" "$LOGDIR/server.log" 2>/dev/null; then
  echo "[server] ✓ Render Well"
elif grep -q "Render Error" "$LOGDIR/server.log" 2>/dev/null; then
  echo "[server] ✗ Render Error —— 看 $LOGDIR/server.log"
else
  echo "[server] 渲染状态未知（可能还在初始化）；若 train 报 cannot create buffer 看 $LOGDIR/server.log"
fi

# ---- 2) EXPO 训练（expo-ft venv，TRAIN_GPUS ≥2 卡）----
echo "[train] 启动训练  GPUs=$TRAIN_GPUS  run=$RUN_NAME"
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
WANDB_PROJECT="$WANDB_PROJECT" \
  $PYTHON train_pi_robo_async.py \
    --config "$MODEL_CONFIG" \
    --config.pi05_weight_loader_path "$VLA_CKPT" \
    --config.pi05_assets_dir "$VLA_ASSETS_DIR" \
    --config.pi05_asset_id "$VLA_ASSET_ID" \
    --config_task "$TASK_CONFIG" \
    --dataset_path "$DATASET_PATH" \
    --client_host localhost --client_port "$SERVER_PORT" \
    --replan_steps "$REPLAN_STEPS" \
    --max_steps "$MAX_STEPS" \
    --seed "$SEED" \
    --project_name "$WANDB_PROJECT" --run_name "$RUN_NAME" \
    --output_dir "${EXPO_ROOT}/logs" \
    > "$LOGDIR/train.log" 2>&1 &
PIDS+=($!)

echo ""
echo "[*] EXPO+RoboTwin+自动专家介入 训练已启动，wandb project = $WANDB_PROJECT"
echo "[*] 实时查看：tail -f $LOGDIR/train.log"
echo "[*] 验证接管生效：grep 'expert takeover started' $LOGDIR/server.log"
echo "[*] 关注 wandb：rollout 成功率脱离 0% / training/target_q_max 上升 / is_hil 比例。Ctrl-C 停止全部。"
wait
