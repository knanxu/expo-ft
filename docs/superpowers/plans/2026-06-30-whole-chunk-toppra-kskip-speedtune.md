# Whole chunk TOPPRA + k_skip 的 SpeedTune 训练 / eval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 whole chunk TOPPRA（`chunk_toppra`）执行后端支持 k_skip（整段对前 `k_skip` 帧做一次 TOPPRA、执行完重推 VLA），并参数化训练脚本与 eval，使维护人能在「最多 2 个并行」的云端训练含 chunk_toppra 的任意两个 backend。

**Architecture:** k_skip 在 RoboTwin `take_chunk_action` 内截断（reconstruct 后、TOPPRA 前，复用既有 `_apply_k_skip`，与 per_action 对齐）；expo-ft `robotwin_env.py` 把 `self._k_skip` 透传给 whole_chunk 分支（`_call_backend` 的 `inspect.signature` 过滤保证向后兼容）；SpeedTune 学习栈（DQN/buffer/network/训练入口/reward）零改动；训练脚本 `run_speedtune_train_eval.sh` 外露 `BACKENDS`、参数化 checkpoint 发现与 eval 调用。

**Tech Stack:** Python（JAX/flax 学习栈，numpy）、RoboTwin（sapien 仿真，TOPPRA 时间参数化）、pytest（云端跑）、bash（启动脚本）。

## Global Constraints

- **本地不可运行**：本地显存不足，**禁止**本地执行 `pytest` / `import jax|sapien|torch` / 训练 / rollout / eval。本地验证仅限：`python -c "import ast; ast.parse(...)"` 语法静检、`bash -n` 脚本语法检查、逻辑 review。真正的 pytest / 训练 / eval 全部**交云端**（维护人在 `/home/chenlu/expo-ft`、`/home/chenlu/RoboTwin` 跑）。
- **两个独立仓库**：代码改动跨 `/home/xukainan/expo-ft`（expo-ft）与 `/home/xukainan/RoboTwin`（RoboTwin 只读对照副本）。两者 git 独立，分别 commit。RoboTwin 改动由维护人同步到云端 `/home/chenlu/RoboTwin`。
- **commit 时机**：遵守仓库约定——**只在维护人明确要求时 commit**。本计划每个 Task 末尾的 commit 步骤是逻辑边界标记；执行时若维护人未要求 commit，则跳过实际 commit（保留改动待确认）。commit 前确认当前分支非默认分支。
- **零侵入**：不改 EXPO/BC 任何路径、其 config、`droid_env.py`、原 `train_pi_robo*.py`。不改 SpeedTune 学习栈（`speedtune_dqn.py`/`speedtune_buffer.py`/`rainbow_dqn.py`/`train_speedtune_async.py`）、`exec_backends.py`、`eval_speedtune.py`、`eval_speedtune_compare.py`。RoboTwin 新增参数 `max_actions: int = None`（默认 None = 旧行为，向后兼容）。
- **reward / 超参不变**：沿用 `exec_backends.py` 现成 success-gated `chunk_toppra` spec（3 head v/vel/acc）；`max_iters`/`epsilon_decay_steps`/`learning_starts` 共用现有 config（加 k_skip 后 chunk_toppra 决策粒度对齐 per_action）。
- **k_skip 语义**：reconstruct → 预算截断 → `_apply_k_skip(前 k_skip 帧)` → 整段 TOPPRA。窗口首尾 `sd=0`（轻微减速，已知代价，本次不消除）。

---

## File Structure

| 文件 | 仓库 | 职责 | 动作 |
|---|---|---|---|
| `envs/_base_task.py` `take_chunk_action`（:1787） | RoboTwin | whole_chunk 执行核：新增 `max_actions` 截断 | 修改 |
| `envs/whole_chunk_kskip_test.py` | RoboTwin | 单测：max_actions 截断生效 / None 不截断 | 新建 |
| `client_robotwin/envs/robotwin_env.py`（:421-424） | expo-ft | whole_chunk 分支透传 `max_actions=self._k_skip` | 修改 |
| `configs/model/speedtune_dqn_config.py`（:42-44） | expo-ft | `k_skip` 注释更新（三 backend 均生效） | 修改 |
| `scripts/run_speedtune_train_eval.sh` | expo-ft | 外露 `BACKENDS`（≤2）+ checkpoint 发现/eval 调用参数化 | 修改 |

---

## Task 1: RoboTwin `take_chunk_action` 支持 max_actions（whole chunk + k_skip）

**Files:**
- Modify: `/home/xukainan/RoboTwin/envs/_base_task.py`（`take_chunk_action` 签名 :1787-1788、docstring :1799-1809、`_apply_k_skip` 插入点 :1858 前）
- Test: `/home/xukainan/RoboTwin/envs/whole_chunk_kskip_test.py`（新建）

**Interfaces:**
- Consumes: 模块级 `_apply_k_skip(action_chunk, max_actions)`（`_base_task.py:37`，已存在）；`retime_chunk(**kwargs)`（kwarg `chunk_arm` 是截断后的 (N,12) 臂轨迹）。
- Produces: `take_chunk_action(self, action_chunk, vel_limit=5.0, acc_limit=10.0, v=1.0, video_save_freq=-1, max_actions=None)`。`max_actions=None`/`<=0` → 整段（旧行为）；`>0` 且 reconstruct 后帧数 > `max_actions` → 仅前 `max_actions` 帧进入 TOPPRA，返回 info 的 `take_action_cnt_delta` = 实际执行帧数 M。

- [ ] **Step 1: 写失败测试** — 新建 `/home/xukainan/RoboTwin/envs/whole_chunk_kskip_test.py`

```python
import numpy as np

from envs._base_task import Base_Task
import envs.robot.toppra_chunk_executor as tce


class FakeRobot:
    def get_left_arm_jointState(self):
        return [0.0] * 6 + [0.0]

    def get_right_arm_jointState(self):
        return [0.0] * 6 + [0.0]

    def get_left_arm_real_jointState(self):
        return [0.0] * 6 + [0.0]

    def get_right_arm_real_jointState(self):
        return [0.0] * 6 + [0.0]

    def get_left_gripper_val(self):
        return 0.0

    def get_right_gripper_val(self):
        return 0.0


def _fake_retime_capture(captured):
    # 让 retime 在入口处"截停"(status!=success → take_chunk_action 走 fallback 立即返回),
    # 免去 stub 250Hz 下发循环; 仍能捕获传入的 chunk_arm (= _apply_k_skip 之后的帧).
    def fake_retime_chunk(**kwargs):
        captured.update(kwargs)
        return {
            "status": "fallback",
            "fallback_reason": "stop after input capture",
            "duration": 0.0,
            "dense_arm_pos": None,
            "dense_arm_vel": None,
            "dense_gripper": None,
            "return_code": None,
        }
    return fake_retime_chunk


def _make_env():
    env = Base_Task()
    env.robot = FakeRobot()
    env.take_action_cnt = 0
    env.step_lim = 100
    env.eval_success = False
    return env


def test_whole_chunk_k_skip_truncates_chunk_before_toppra(monkeypatch):
    captured = {}
    monkeypatch.setattr(tce, "retime_chunk", _fake_retime_capture(captured))
    env = _make_env()

    # 5-frame chunk, max_actions=2 → 只前 2 帧进入整段 TOPPRA
    info = env.take_chunk_action(
        np.zeros((5, 14)), vel_limit=1.0, acc_limit=1.0, v=1.0, max_actions=2,
    )

    chunk_arm = np.asarray(captured["chunk_arm"], dtype=float)
    assert chunk_arm.shape[0] == 2, chunk_arm.shape
    assert info["take_action_cnt_delta"] == 2, info


def test_whole_chunk_none_max_actions_keeps_full_chunk(monkeypatch):
    captured = {}
    monkeypatch.setattr(tce, "retime_chunk", _fake_retime_capture(captured))
    env = _make_env()

    # max_actions=None → 整段保留 (旧行为)
    info = env.take_chunk_action(
        np.zeros((5, 14)), vel_limit=1.0, acc_limit=1.0, v=1.0, max_actions=None,
    )

    chunk_arm = np.asarray(captured["chunk_arm"], dtype=float)
    assert chunk_arm.shape[0] == 5, chunk_arm.shape
    assert info["take_action_cnt_delta"] == 5, info
```

- [ ] **Step 2: 本地语法静检测试文件**（本地不能跑 pytest）

Run: `python -c "import ast; ast.parse(open('/home/xukainan/RoboTwin/envs/whole_chunk_kskip_test.py').read()); print('OK')"`
Expected: 打印 `OK`（语法合法）。

- [ ] **Step 3: 改签名** — `_base_task.py:1787-1788`

old:
```python
    def take_chunk_action(self, action_chunk, vel_limit: float = 5.0, acc_limit: float = 10.0, v: float = 1.0,
                          video_save_freq: int = -1):
```
new:
```python
    def take_chunk_action(self, action_chunk, vel_limit: float = 5.0, acc_limit: float = 10.0, v: float = 1.0,
                          video_save_freq: int = -1, max_actions: int = None):
```

- [ ] **Step 4: 补 docstring** — `_base_task.py`，在 `video_save_freq` 说明（现 :1807-1809）之后、`Returns:`（现 :1811）之前插入

```python
            max_actions: k_skip (论文式 frame skip). >0 且 reconstruct 后帧数超过它时,
               只保留前 max_actions 帧再做整段 TOPPRA, 执行完即返回 (上层重推 VLA, 闭环).
               None/<=0 → 不截断, 整段执行 (旧行为, 向后兼容). 与 per_action 的 _apply_k_skip 一致.
```

- [ ] **Step 5: 插入 `_apply_k_skip`** — `_base_task.py`，在 `M = int(action_chunk.shape[0])`（现 :1859）之前插入（与 per_action :2023 对齐：预算截断后、M 前）

old:
```python
        # 压缩后 chunk 帧数 M: 代表策略实际下发的 action 数量, 也是本次调用消耗的 cnt 预算
        M = int(action_chunk.shape[0])
```
new:
```python
        # ---- k_skip (论文式 frame skip): 只取前 max_actions 帧再做整段 TOPPRA ----
        # 与 per_action (_apply_k_skip @ take_chunk_action_per_action) 一致: 在预算截断之后、
        # 拆臂/TOPPRA 之前截断; max_actions=None/<=0 → 整段执行 (旧行为).
        action_chunk = _apply_k_skip(action_chunk, max_actions)

        # 压缩后 chunk 帧数 M: 代表策略实际下发的 action 数量, 也是本次调用消耗的 cnt 预算
        M = int(action_chunk.shape[0])
```

- [ ] **Step 6: 本地语法静检实现文件**

Run: `python -c "import ast; ast.parse(open('/home/xukainan/RoboTwin/envs/_base_task.py').read()); print('OK')"`
Expected: 打印 `OK`。

- [ ] **Step 7: 云端跑测试**（交维护人在云端执行；本地跳过）

Run（云端）: `cd /home/chenlu/RoboTwin && conda run -n RoboTwin python -m pytest envs/whole_chunk_kskip_test.py -v`
Expected: 2 passed（`test_whole_chunk_k_skip_truncates_chunk_before_toppra`、`test_whole_chunk_none_max_actions_keeps_full_chunk`）。

- [ ] **Step 8: Commit**（仅维护人要求时；RoboTwin 仓库）

```bash
cd /home/xukainan/RoboTwin
git add envs/_base_task.py envs/whole_chunk_kskip_test.py
git commit -m "feat(whole_chunk): take_chunk_action 支持 max_actions (k_skip), reconstruct 后截断前 k 帧再整段 TOPPRA"
```

---

## Task 2: expo-ft 接线（whole_chunk 分支透传 k_skip + config 注释）

**Files:**
- Modify: `/home/xukainan/expo-ft/client_robotwin/envs/robotwin_env.py`（:421-424）
- Modify: `/home/xukainan/expo-ft/configs/model/speedtune_dqn_config.py`（:42-44）

**Interfaces:**
- Consumes: Task 1 的 `take_chunk_action(..., max_actions=None)`；`self._k_skip`（robotwin_env `__init__` 已设，:67-68）；`_call_backend`（:372-384，`inspect.signature` 过滤 kwargs → 向后兼容）。
- Produces: whole_chunk 分支调用 `take_chunk_action(..., max_actions=self._k_skip)`。无新函数签名对外。

- [ ] **Step 1: whole_chunk 分支传 max_actions** — `robotwin_env.py:421-424`

old:
```python
        else:  # whole_chunk：整段 TOPPRA，k_skip 不适用
            info = self._call_backend(self.env.take_chunk_action, chunk,
                                      vel_limit=vel_limit, acc_limit=acc_limit, v=v,
                                      video_save_freq=vsf)
```
new:
```python
        else:  # whole_chunk：整段 TOPPRA；k_skip 取前 k 帧再整段重参数化（_call_backend 按签名过滤，向后兼容）
            info = self._call_backend(self.env.take_chunk_action, chunk,
                                      vel_limit=vel_limit, acc_limit=acc_limit, v=v,
                                      max_actions=self._k_skip, video_save_freq=vsf)
```

- [ ] **Step 2: 更新 config k_skip 注释** — `speedtune_dqn_config.py:42-44`

old:
```python
    # 论文式 frame skip：streaming/per_action 每决策只执行 reconstruct(v) 后前 k_skip 个 action 即重推
    # VLA（闭环），缩短 MDP horizon。whole_chunk（整段 TOPPRA）忽略此项。None/0=整段。
    config.k_skip = 10
```
new:
```python
    # 论文式 frame skip：每决策只执行 reconstruct(v) 后前 k_skip 个 action 即重推 VLA（闭环），
    # 缩短 MDP horizon。三种执行方式均生效：streaming/per_action 逐帧；whole_chunk（整段 TOPPRA）
    # 取前 k_skip 帧做一次整段 TOPPRA。None/0=整段执行。
    config.k_skip = 10
```

- [ ] **Step 3: 本地语法静检两个文件**

Run:
```bash
python -c "import ast; ast.parse(open('/home/xukainan/expo-ft/client_robotwin/envs/robotwin_env.py').read()); ast.parse(open('/home/xukainan/expo-ft/configs/model/speedtune_dqn_config.py').read()); print('OK')"
```
Expected: 打印 `OK`。

- [ ] **Step 4: 逻辑 review**（人工 / reviewer）

确认：① whole_chunk 分支现传 `max_actions=self._k_skip`；② `_call_backend` 对不支持 `max_actions` 的 RoboTwin 版本会过滤该 kwarg（不报错，降级整段执行）；③ config 注释不再声称 whole_chunk 忽略 k_skip；④ 未改 streaming/per_action 分支与其它逻辑。

- [ ] **Step 5: Commit**（仅维护人要求时；expo-ft 仓库）

```bash
cd /home/xukainan/expo-ft
git add client_robotwin/envs/robotwin_env.py configs/model/speedtune_dqn_config.py
git commit -m "feat(speedtune): whole_chunk 分支透传 k_skip 到 take_chunk_action(max_actions); 更新 k_skip 注释"
```

---

## Task 3: 训练脚本参数化（外露 BACKENDS + checkpoint/eval 参数化）

**Files:**
- Modify: `/home/xukainan/expo-ft/scripts/run_speedtune_train_eval.sh`（:51-56 资源定义、:138-146 checkpoint 发现、:148-161 eval 调用）

**Interfaces:**
- Consumes: `eval_speedtune_compare.py`（flags `--backend_a/--ckpt_a/--port_a` `--backend_b/--ckpt_b/--port_b`，任意 2 backend）；`eval_speedtune.py`（flags `--config.exec_backend`、`--dqn_ckpt`、`--client_port`，单 backend）。
- Produces: 环境变量 `BACKENDS="A B"`（1~2 个，空格分隔）控制训练哪些 backend；非交互未传时默认 `per_action_toppra chunk_toppra`；交互终端询问。

- [ ] **Step 1: 资源定义 + 外露 BACKENDS + 校验 + 切片** — 替换 `run_speedtune_train_eval.sh:51-56`

old:
```bash
# 两 backend：名 / port / server 卡 / train 卡（与 eval 默认 port 对齐：per_action=8102, fixed_time=8103）。
BACKENDS=(per_action_toppra fixed_time)
PORTS=(8102 8103)
SERVER_GPUS=(0 2)
TRAIN_GPUS=(1 3)
COMPARE_GPU="${COMPARE_GPU:-1}"
```
new:
```bash
# 资源池（server 与 train 分卡避免光追 sapien 与 3B VLA 抢显存；与 eval 默认 port 对齐 8102/8103）。
ALL_BACKENDS=(fixed_time per_action_toppra chunk_toppra)
PORTS=(8102 8103)
SERVER_GPUS=(0 2)
TRAIN_GPUS=(1 3)
COMPARE_GPU="${COMPARE_GPU:-1}"

# 外露 BACKENDS：训练前传 BACKENDS="A B"（≤2 个，云端只支持 2 个并行）。
#   已设          → 用之；
#   未设 + 交互终端 → 列出三选项询问；
#   未设 + 非交互   → 默认 per_action_toppra chunk_toppra（并打印提示，nohup 后台不卡）。
if [ -n "${BACKENDS:-}" ]; then
  read -ra BACKENDS <<< "$BACKENDS"
elif [ -t 0 ]; then
  echo "可选执行后端：${ALL_BACKENDS[*]}（云端只支持同时训练 2 个）"
  read -r -p "请输入要训练的 backend（空格分隔，最多 2 个，回车用默认 per_action_toppra chunk_toppra）：" _line
  if [ -n "$_line" ]; then read -ra BACKENDS <<< "$_line"; else BACKENDS=(per_action_toppra chunk_toppra); fi
else
  BACKENDS=(per_action_toppra chunk_toppra)
  echo "[*] 未设 BACKENDS 且非交互终端 → 默认: ${BACKENDS[*]}"
fi

# 校验数量与合法性
if [ "${#BACKENDS[@]}" -lt 1 ] || [ "${#BACKENDS[@]}" -gt 2 ]; then
  echo "[ERROR] BACKENDS 必须是 1~2 个（云端只支持 2 个并行），当前(${#BACKENDS[@]}): ${BACKENDS[*]}" >&2
  exit 1
fi
for be in "${BACKENDS[@]}"; do
  case "$be" in
    fixed_time|per_action_toppra|chunk_toppra) ;;
    *) echo "[ERROR] 未知 backend: '$be'（合法: ${ALL_BACKENDS[*]}）" >&2; exit 1 ;;
  esac
done

# 按 backend 数切片资源池
PORTS=("${PORTS[@]:0:${#BACKENDS[@]}}")
SERVER_GPUS=("${SERVER_GPUS[@]:0:${#BACKENDS[@]}}")
TRAIN_GPUS=("${TRAIN_GPUS[@]:0:${#BACKENDS[@]}}")
echo "[*] 训练 backend: ${BACKENDS[*]} | ports: ${PORTS[*]} | server_gpus: ${SERVER_GPUS[*]} | train_gpus: ${TRAIN_GPUS[*]}"
```

- [ ] **Step 2: checkpoint 发现循环化** — 替换 `run_speedtune_train_eval.sh:138-146`

old:
```bash
# ---- 3) 自动发现各 backend 最新 checkpoint ----
CKPT_PER=$(ls -d "$LOGDIR/speedtune_per_action_toppra_${STAMP}/checkpoints/update_"* 2>/dev/null | sort -V | tail -1 || true)
CKPT_FIXED=$(ls -d "$LOGDIR/speedtune_fixed_time_${STAMP}/checkpoints/update_"* 2>/dev/null | sort -V | tail -1 || true)
if [ -z "$CKPT_PER" ] || [ -z "$CKPT_FIXED" ]; then
  echo "[ERROR] 找不到 checkpoint（per_action=$CKPT_PER fixed_time=$CKPT_FIXED）。看 train_*.log" >&2
  exit 1
fi
echo "[*] CKPT per_action_toppra = $CKPT_PER"
echo "[*] CKPT fixed_time        = $CKPT_FIXED"
```
new:
```bash
# ---- 3) 自动发现各 backend 最新 checkpoint（循环，与 BACKENDS 对齐）----
CKPTS=()
for be in "${BACKENDS[@]}"; do
  ck=$(ls -d "$LOGDIR/speedtune_${be}_${STAMP}/checkpoints/update_"* 2>/dev/null | sort -V | tail -1 || true)
  if [ -z "$ck" ]; then
    echo "[ERROR] 找不到 $be 的 checkpoint。看 $LOGDIR/train_${be}.log" >&2
    exit 1
  fi
  echo "[*] CKPT $be = $ck"
  CKPTS+=("$ck")
done
```

- [ ] **Step 3: eval 调用参数化** — 替换 `run_speedtune_train_eval.sh:148-161`

old:
```bash
# ---- 4) 起 compare（复用训练 server；A=fixed_time 基准 / B=per_action 被测）----
echo "[*] 启动对比 eval  compare_GPU=$COMPARE_GPU（复用 server，不重起/不端口预检）"
CUDA_VISIBLE_DEVICES="$COMPARE_GPU" XLA_PYTHON_CLIENT_MEM_FRACTION="$COMPARE_MEM_FRAC" \
  $PYTHON eval_speedtune_compare.py \
    --config "$MODEL_CONFIG" \
    --config_task "$TASK_CONFIG" \
    --config.stream_hold_steps "$STREAM_HOLD_STEPS" \
    --backend_a fixed_time        --ckpt_a "$CKPT_FIXED" --port_a 8103 \
    --backend_b per_action_toppra --ckpt_b "$CKPT_PER"   --port_b 8102 \
    --n_episodes "$N_EPISODES" --seed "$SEED" \
    --max_decision_steps "$MAX_DECISION_STEPS" \
    --client_host localhost \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/compare.log"
```
new:
```bash
# ---- 4) eval（复用训练 server，不重起/不端口预检）----
if [ "${#BACKENDS[@]}" -eq 2 ]; then
  echo "[*] 启动 2 路对比 eval  compare_GPU=$COMPARE_GPU（A=${BACKENDS[0]} / B=${BACKENDS[1]}，复用 server）"
  CUDA_VISIBLE_DEVICES="$COMPARE_GPU" XLA_PYTHON_CLIENT_MEM_FRACTION="$COMPARE_MEM_FRAC" \
    $PYTHON eval_speedtune_compare.py \
      --config "$MODEL_CONFIG" \
      --config_task "$TASK_CONFIG" \
      --config.stream_hold_steps "$STREAM_HOLD_STEPS" \
      --backend_a "${BACKENDS[0]}" --ckpt_a "${CKPTS[0]}" --port_a "${PORTS[0]}" \
      --backend_b "${BACKENDS[1]}" --ckpt_b "${CKPTS[1]}" --port_b "${PORTS[1]}" \
      --n_episodes "$N_EPISODES" --seed "$SEED" \
      --max_decision_steps "$MAX_DECISION_STEPS" \
      --client_host localhost \
      --output_dir "$OUTPUT_DIR" \
      2>&1 | tee "$OUTPUT_DIR/compare.log"
else
  be="${BACKENDS[0]}"
  echo "[*] 单 backend eval（$be）  eval_GPU=$COMPARE_GPU（复用 server）"
  CUDA_VISIBLE_DEVICES="$COMPARE_GPU" XLA_PYTHON_CLIENT_MEM_FRACTION="$COMPARE_MEM_FRAC" \
    $PYTHON eval_speedtune.py \
      --config "$MODEL_CONFIG" \
      --config.exec_backend "$be" \
      --config.stream_hold_steps "$STREAM_HOLD_STEPS" \
      --config_task "$TASK_CONFIG" \
      --dqn_ckpt "${CKPTS[0]}" \
      --n_episodes "$N_EPISODES" --seed "$SEED" \
      --max_decision_steps "$MAX_DECISION_STEPS" \
      --client_host localhost --client_port "${PORTS[0]}" \
      --output_dir "$OUTPUT_DIR" \
      2>&1 | tee "$OUTPUT_DIR/eval_${be}.log"
fi
```

- [ ] **Step 4: 修正硬编码提示**（tail 日志名 + 头注释）

(a) 修 `tail` 提示硬编码 `per_action_toppra`（现 :127；若训练组合不含它会指向不存在的日志）：

old:
```bash
echo "[*] 实时查看：tail -f $LOGDIR/train_per_action_toppra.log"
```
new:
```bash
echo "[*] 实时查看：tail -f $LOGDIR/train_${BACKENDS[0]}.log"
```

(b) 用法注释（:13-18）在 `#   冒烟：MAX_ITERS=2000 ...`（:18）之后插入：
```bash
#   指定 backend（≤2，含 chunk_toppra）：BACKENDS="per_action_toppra chunk_toppra" bash scripts/run_speedtune_train_eval.sh
#   未传 BACKENDS：交互终端会询问；nohup 后台用默认 per_action_toppra chunk_toppra。
```

- [ ] **Step 5: 本地脚本语法检查**

Run: `bash -n /home/xukainan/expo-ft/scripts/run_speedtune_train_eval.sh && echo "OK"`
Expected: 打印 `OK`（无语法错误）。

- [ ] **Step 6: 逻辑 review**（人工 / reviewer）

确认：① `BACKENDS` 三种来源（已设 / 交互 / 非交互默认）都产出 1~2 个合法 backend；② >2 个 或 非法名 报错退出；③ server/train 循环（:79-132，未改）按切片后的 `BACKENDS`/`PORTS`/`*_GPUS` 正确起；④ checkpoint 发现按 `BACKENDS` 找 `speedtune_${be}_${STAMP}`；⑤ 2 backend 走 compare、1 backend 走单 eval，port/ckpt 与 `BACKENDS` 顺序一致；⑥ 默认 `per_action_toppra chunk_toppra` 不破坏原有冒烟流程。

- [ ] **Step 7: 云端冒烟**（交维护人；本地跳过）

Run（云端）:
```bash
export SPEEDTUNE_VLA_CKPT=/home/chenlu/openpi/checkpoints/pi05_aloha_robotwin_drifting_stack_blocks_two/drifting_v1/29999/params
export SPEEDTUNE_VLA_ASSETS=/home/chenlu/openpi/checkpoints/pi05_aloha_robotwin_drifting_stack_blocks_two/drifting_v1/29999/assets
export SPEEDTUNE_VLA_ASSET_ID=trossen
BACKENDS="per_action_toppra chunk_toppra" MAX_ITERS=2000 N_EPISODES=3 bash scripts/run_speedtune_train_eval.sh
```
Expected: 两 backend（含 chunk_toppra）并行训练 2000 步 → 各落 checkpoint → 自动 2 路对比 → `$OUTPUT_DIR/compare_summary.json` + `compare_speedup.png`。检查 chunk_toppra 决策步 `dense_steps` 反映 k_skip 帧执行、`exec_status` 非长期 `topp_fallback`。

- [ ] **Step 8: Commit**（仅维护人要求时；expo-ft 仓库）

```bash
cd /home/xukainan/expo-ft
git add scripts/run_speedtune_train_eval.sh
git commit -m "feat(speedtune): run_speedtune_train_eval 外露 BACKENDS(≤2) + checkpoint/eval 调用参数化, 支持 chunk_toppra"
```

---

## 验证总览（云端）

1. 本地：全部改动文件 `ast.parse` / `bash -n` 静检通过。
2. 云端 RoboTwin：`pytest envs/whole_chunk_kskip_test.py -v` → 2 passed。
3. 云端 expo-ft：`BACKENDS="per_action_toppra chunk_toppra" bash scripts/run_speedtune_train_eval.sh`（冒烟参数）→ 训练 + 2 路对比产出。
4. 解读：chunk_toppra 加 k_skip 后决策步数与 per_action 同量级；`dense_steps` 跨 backend 可比执行时长。
