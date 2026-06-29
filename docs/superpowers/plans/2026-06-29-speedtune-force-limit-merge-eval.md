# SpeedTune force_limit 接通 + train/eval 合并脚本 + 图表英文化 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 SpeedTune 执行层力矩底座 `force_limit=τ_max` 全链路接通，新增 train+eval 一体化脚本，修复 eval 图表中文渲染（豆腐块）→ 英文。

**Architecture:** force_limit 走环境变量 `SPEEDTUNE_FORCE_LIMIT` → `speedtune_dqn_config.force_limit`(str) → 新纯函数 `parse_force_limit` → `env_creation_request` → `run_robotwin_client` 白名单透传 → 已实现的 `RoboTwinEnv._apply_force_limit`。合并脚本先 4 卡并行训练 per_action+fixed_time，等 train 进程退出后自动找 final checkpoint、复用训练 server 跑 eval 对比。图表把所有渲染到图上的中文/非 ASCII 改英文。

**Tech Stack:** Python（numpy / jax / ml_collections / matplotlib）、bash、orbax checkpoint、RoboTwin sapien（云端）。

## Global Constraints

- **零侵入 EXPO/BC**：不改 `train_pi_robo_async.py` 的 `train_env_creation_request`（`:153`，不传 force_limit）；不改 `robot.py`；force_limit 默认仅在 SpeedTune config，`expo_ft_pi_*` config 不含。
- **本地只做静态检查**（CLAUDE.md）：`python -m py_compile <f>` 语法、`bash -n` 脚本语法、逻辑 review；`import jax/sapien/torch` 与训练/仿真/eval **不本地跑**，交云端。例外：`parse_force_limit` 单测纯 numpy（不依赖 jax/sapien），本地或云端均可跑。
- **commit 需维护人批准**（CLAUDE.md）：commit 步骤照写（含 `Co-Authored-By` trailer），执行时由维护人确认。
- **本批全在 expo-ft repo**（RoboTwin repo 本批不动）。
- **grid 不动**（维护人已改）：`per_action_toppra` head_sizes=`(5,3,5)`、`fixed_time`=`(5,)`。本批不改 head_sizes。
- **force_limit 语义**：per-joint 单臂 arm dof（6，ARX5）；`None`/空串 = 不施加（∞）= EXPO/BC 旧行为。默认 `30,40,30,15,10,10`（arx5-sdk `config.h`）。

---

## Task 1: `parse_force_limit` 纯函数 + 单测

**Files:**
- Modify: `expo_ft/speedtune/exec_backends.py`（加 `parse_force_limit`、`import Optional`、清 `:38` docstring 过时举例）
- Test: `test/exec_backends_test.py`（顶部 import + 3 个测试）

**Interfaces:**
- Produces: `parse_force_limit(spec) -> Optional[List[float]]`
  - `"30,40,30,15,10,10"` → `[30.0, 40.0, 30.0, 15.0, 10.0, 10.0]`
  - `""` / `"  "` / `None` → `None`
  - `[30, 40]` / `(30.0, 40.0)` → `[30.0, 40.0]`
  - `"30, 40 ,30"`（含空格） → `[30.0, 40.0, 30.0]`

- [ ] **Step 1: 写失败测试**

`test/exec_backends_test.py` 顶部 import（`:6`）改为：

```python
from expo_ft.speedtune.exec_backends import build_backend, list_backends, parse_force_limit
```

文件末尾追加：

```python
def test_parse_force_limit_str():
    assert parse_force_limit("30,40,30,15,10,10") == [30.0, 40.0, 30.0, 15.0, 10.0, 10.0]


def test_parse_force_limit_empty_and_none():
    assert parse_force_limit("") is None
    assert parse_force_limit("   ") is None
    assert parse_force_limit(None) is None


def test_parse_force_limit_list_tuple_and_spaces():
    assert parse_force_limit([30, 40]) == [30.0, 40.0]
    assert parse_force_limit((30.0, 40.0)) == [30.0, 40.0]
    assert parse_force_limit("30, 40 ,30") == [30.0, 40.0, 30.0]
```

- [ ] **Step 2: 跑测试确认失败**

Run（本地纯 numpy 可跑，不依赖 jax/sapien；亦可云端）：`python -m pytest test/exec_backends_test.py -k parse_force_limit -v`
Expected: FAIL（`ImportError: cannot import name 'parse_force_limit'`）

- [ ] **Step 3: 实现 `parse_force_limit`**

`exec_backends.py` 顶部 import（`from typing import Dict, List, Sequence, Tuple`）加 `Optional`：

```python
from typing import Dict, List, Optional, Sequence, Tuple
```

文件末尾（`build_backend` 之后）追加：

```python
def parse_force_limit(spec) -> Optional[List[float]]:
    """force_limit 配置 → per-joint 单臂力矩上限 list（空/None → None = 不施加 = ∞）。

    接受逗号分隔字符串 "30,40,30,15,10,10" / list / tuple；空串、纯空白、None → None。
    长度应 = 单臂 arm dof（6, ARX5）；不强制校验——由 RoboTwinEnv._apply_force_limit 的 zip
    自然处理（多余截断、不足只设前 N 个关节）。SpeedTune 执行层力矩底座用，EXPO/BC 不传 → None。
    """
    if spec is None:
        return None
    if isinstance(spec, (list, tuple)):
        vals = [float(x) for x in spec]
        return vals or None
    s = str(spec).strip()
    if not s:
        return None
    return [float(x) for x in s.split(",") if x.strip()]
```

- [ ] **Step 4: 清 `:38` docstring 过时举例**

`SpeedVar` docstring 里的：

```python
      name:      变量名（也是 ``speed_params`` 的 key），如 "compress"/"vel_scale"/"acc_scale"。
```

改为：

```python
      name:      变量名（也是 ``speed_params`` 的 key），如 "v"/"vel_limit"/"acc_limit"。
```

- [ ] **Step 5: 本地语法检查**

Run: `python -m py_compile expo_ft/speedtune/exec_backends.py test/exec_backends_test.py && echo OK`
Expected: `OK`

- [ ] **Step 6: 跑测试确认通过**

Run（本地纯 numpy 可跑；亦可云端）：`python -m pytest test/exec_backends_test.py -v`
Expected: PASS（含原有 `test_head_sizes` / `test_decode_lengths` 等 + 3 个新 parse 测试）

- [ ] **Step 7: Commit（维护人确认后）**

```bash
git add expo_ft/speedtune/exec_backends.py test/exec_backends_test.py
git commit -m "$(cat <<'EOF'
feat(speedtune): force_limit 配置解析 parse_force_limit + 单测

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: force_limit 配置源 + 全链路透传

> 本 task 为配置/胶水透传，**无合理单元测试**（依赖 jax/sapien 的 env_client + server）；force_limit 解析逻辑已在 Task 1 单测覆盖。本 task 用 `py_compile` 静态验证，运行时 force_limit 生效验证见 Task 4 端到端冒烟。

**Files:**
- Modify: `configs/model/speedtune_dqn_config.py`（`:32` 后加 `config.force_limit`）
- Modify: `train_speedtune_async.py`（`:40` import + `:192-200` request）
- Modify: `eval_speedtune_compare.py`（顶部 import + `_make_env` request）
- Modify: `eval_speedtune.py`（顶部 import + `:339-349` request）
- Modify: `client_robotwin/run_robotwin_client.py`（`:76` 白名单）
- Modify: `client_robotwin/envs/robotwin_env.py`（`:369` docstring 清理）

**Interfaces:**
- Consumes: `parse_force_limit`（Task 1）
- Produces: 三处 `env_creation_request` 带 `"force_limit"`；server 端 `env_kwargs["force_limit"]` 透传给 `RoboTwinEnv`

- [ ] **Step 1: config 加 force_limit**

`configs/model/speedtune_dqn_config.py`，在 `config.pi05_asset_id = ...`（`:32`）一行**之后**插入：

```python

    # --- SpeedTune 执行层力矩底座（真机 ARX5 τ_max）。per-joint 单臂(6), N·m；空串=不施加(∞)=旧行为。
    #     仅 SpeedTune 读此字段；EXPO/BC config 不含 → 零侵入。learner 端 parse_force_limit 解析后经
    #     env_creation_request 透传，env 每次 reset 后施加（见 robotwin_env._apply_force_limit）。---
    config.force_limit = os.environ.get("SPEEDTUNE_FORCE_LIMIT", "30,40,30,15,10,10")
```

- [ ] **Step 2: train 端 import + 透传**

`train_speedtune_async.py:40` 改为：

```python
from expo_ft.speedtune.exec_backends import build_backend, parse_force_limit
```

`:192-200` 的 `EnvClientWrapper` 调用，在 `env_creation_request` 字典里 `"stream_hold_steps"` 一行后加 `force_limit`：

```python
    env = EnvClientWrapper(
        env_creation_request={"example_action": example_action, "env_usage": "train",
                              "video_dir": os.path.join(log_dir, "train_videos"),
                              # SpeedTune：env 创建即设执行后端（RoboTwin take_chunk_action_backend 用）。
                              "exec_backend": str(config.exec_backend),
                              "k_skip": config.get("k_skip", None),
                              "stream_hold_steps": int(config.get("stream_hold_steps", 15)),
                              "force_limit": parse_force_limit(config.get("force_limit", ""))},
        host=FLAGS.client_host, port=FLAGS.client_port,
    )
```

- [ ] **Step 3: eval_compare 端 import + 透传**

`eval_speedtune_compare.py` 顶部 import 区（`from expo_ft.env.env_client import EnvClientWrapper` 一行后，`:36` 附近）加：

```python
from expo_ft.speedtune.exec_backends import parse_force_limit
```

`_make_env`（`:70-80`）的 `env_creation_request` 里 `"eval_video_save_freq": 25,` 一行后加：

```python
            "eval_video_save_freq": 25,
            "force_limit": parse_force_limit(config.get("force_limit", "")),
```

- [ ] **Step 4: eval_speedtune 端 import + 透传**

`eval_speedtune.py:43` 改为：

```python
from expo_ft.speedtune.exec_backends import build_backend, parse_force_limit
```

`:339-349` 的 `env_creation_request` 里 `"eval_video_save_freq": 25,` 一行后加：

```python
            "eval_video_save_freq": 25,
            "force_limit": parse_force_limit(config.get("force_limit", "")),
```

- [ ] **Step 5: server 端白名单透传**

`client_robotwin/run_robotwin_client.py:76` 改为：

```python
                    for _sk in ("exec_backend", "k_skip", "stream_hold_steps", "force_limit"):
```

- [ ] **Step 6: robotwin_env docstring 清理**

`client_robotwin/envs/robotwin_env.py:369`，把 docstring 里：

```python
        内部完成，本适配器**只透传** v/vel_scale/acc_scale（不再自行压缩）。
```

改为：

```python
        内部完成，本适配器**只透传** v/vel_limit/acc_limit（不再自行压缩）。
```

- [ ] **Step 7: 本地语法检查**

Run:
```bash
python -m py_compile \
  configs/model/speedtune_dqn_config.py \
  train_speedtune_async.py \
  eval_speedtune_compare.py \
  eval_speedtune.py \
  client_robotwin/run_robotwin_client.py \
  client_robotwin/envs/robotwin_env.py && echo OK
```
Expected: `OK`

- [ ] **Step 8: Commit（维护人确认后）**

```bash
git add configs/model/speedtune_dqn_config.py train_speedtune_async.py \
        eval_speedtune_compare.py eval_speedtune.py \
        client_robotwin/run_robotwin_client.py client_robotwin/envs/robotwin_env.py
git commit -m "$(cat <<'EOF'
feat(speedtune): force_limit 力矩底座全链路接通(config→request→server白名单)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: eval 图表中文 → 英文（+ 残留注释清理）

> 仅改**会渲染到图片上**的字符串；代码注释/docstring/日志里的中文（项目要求）保留。无单元测试（matplotlib 渲染），`py_compile` 静态 + Task 4 云端 eval 看图验证。

**Files:**
- Modify: `eval_speedtune_compare.py`（`_plot_compare` 7 处文本 + bar 标记 ✓✗ + `:92`/`:196` 注释）
- Modify: `eval_speedtune.py`（`_save_and_plot` 2 处 label + `:5`/`:274` 注释）

- [ ] **Step 1: compare 图表文本英文化（图1 部分）**

`eval_speedtune_compare.py:172`：
```python
    axes[0].set_title("per-episode 执行步数（越低越快; ✓=成功 ✗=失败）")
```
→
```python
    axes[0].set_title("Per-episode exec steps (lower=faster; S=success F=fail)")
```

`:184`：
```python
    axes[1].set_ylabel("aggressiveness (0慢~1快)")
```
→
```python
    axes[1].set_ylabel("aggressiveness (0=slow ~ 1=fast)")
```

`:186`：
```python
    axes[1].set_title("加速激进度（归一化）随决策步")
```
→
```python
    axes[1].set_title("Normalized aggressiveness vs decision step")
```

`:191`：
```python
    fig.suptitle(f"SpeedTune backend 对比: {na} vs {nb}{sp_txt}")
```
→
```python
    fig.suptitle(f"SpeedTune backend comparison: {na} vs {nb}{sp_txt}")
```

- [ ] **Step 2: compare bar 标记 ✓/✗ → S/F（避开非 ASCII 字体问题）**

`:160-167` 两个 `mk, col = ...` 行（共 2 处，内容相同）：
```python
        mk, col = ("✓", "green") if e["success"] else ("✗", "red")
```
两处都改为：
```python
        mk, col = ("S", "green") if e["success"] else ("F", "red")
```
（用 `replace_all` 或逐处改；该行在文件中出现 2 次，均改。）

- [ ] **Step 3: compare 图表文本英文化（图2 部分）**

`:200`：
```python
            ax.set_title(f"{name}: 无 episode")
```
→
```python
            ax.set_title(f"{name}: no episode")
```

`:209`：
```python
        ax.set_title(f"{name} ep{rep['ep']}{_tag}: 加速参数 + 抓取/接触")
```
→
```python
        ax.set_title(f"{name} ep{rep['ep']}{_tag}: speed params + grasp/contact")
```

`:211`：
```python
    fig2.suptitle("各 backend 实际下发的加速控制参数随决策步")
```
→
```python
    fig2.suptitle("Speed-control params issued per backend, vs decision step")
```

- [ ] **Step 4: compare 残留注释清理（不影响图，顺手）**

`:92`：
```python
    """从代表 episode 的首个 rec 反推该 backend 实际下发的加速参数 keys（v/vel_scale/acc_scale）。"""
```
→
```python
    """从代表 episode 的首个 rec 反推该 backend 实际下发的加速参数 keys（v/vel_limit/acc_limit）。"""
```

`:196`：
```python
    # ---- 图2 compare_knob.png：各 backend 实际加速参数 (v/vel_scale/acc_scale) 随决策步 ----
```
→
```python
    # ---- 图2 compare_knob.png：各 backend 实际加速参数 (v/vel_limit/acc_limit) 随决策步 ----
```

- [ ] **Step 5: eval_speedtune 图表 label 英文化**

`eval_speedtune.py:202`：
```python
    ax.plot(t, aggr, "-o", color="tab:blue", label="aggressiveness (0慢~1快)")
```
→
```python
    ax.plot(t, aggr, "-o", color="tab:blue", label="aggressiveness (0=slow ~ 1=fast)")
```

`:203`：
```python
    ax.plot(t, lg, "--", color="tab:green", alpha=0.6, label="left gripper (0闭~1开)")
```
→
```python
    ax.plot(t, lg, "--", color="tab:green", alpha=0.6, label="left gripper (0=closed ~ 1=open)")
```

- [ ] **Step 6: eval_speedtune 残留注释清理**

`:5`（模块 docstring）：
```python
(v/vel_scale/acc_scale + 归一化激进度) 与接触信号(夹爪值 + sapien 物理接触) + 累计仿真时间；
```
→
```python
(v/vel_limit/acc_limit + 归一化激进度) 与接触信号(夹爪值 + sapien 物理接触) + 累计仿真时间；
```

`:274`：
```python
            rec.update({k: float(v) for k, v in speed_params.items()})  # v / vel_scale / acc_scale
```
→
```python
            rec.update({k: float(v) for k, v in speed_params.items()})  # v / vel_limit / acc_limit
```

- [ ] **Step 7: 本地语法检查**

Run: `python -m py_compile eval_speedtune_compare.py eval_speedtune.py && echo OK`
Expected: `OK`

- [ ] **Step 8: 确认无残留图上中文**

Run: `grep -nP "set_title|set_xlabel|set_ylabel|suptitle|label=" eval_speedtune_compare.py eval_speedtune.py | grep -P "[\x{4e00}-\x{9fff}]" || echo "图上文本无中文 OK"`
Expected: `图上文本无中文 OK`

- [ ] **Step 9: Commit（维护人确认后）**

```bash
git add eval_speedtune_compare.py eval_speedtune.py
git commit -m "$(cat <<'EOF'
fix(speedtune-eval): 图表中文/✓✗ 改英文(matplotlib 无 CJK 字形→豆腐块)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: train+eval 一体化脚本 `scripts/run_speedtune_train_eval.sh`（新建）

**Files:**
- Create: `scripts/run_speedtune_train_eval.sh`

**Interfaces:**
- Consumes: `train_speedtune_async.py`（final checkpoint 落 `<output_dir>/<run_name>/checkpoints/update_<N>`）、`eval_speedtune_compare.py`、`SPEEDTUNE_FORCE_LIMIT`（Task 2）、`client_robotwin.run_robotwin_client`
- Produces: 一条命令完成「4 卡训练 per_action+fixed_time → 自动 eval 对比」，产物在 `$LOGDIR`（训练）+ `$LOGDIR/compare_eval`（对比图表 json/png）

- [ ] **Step 1: 写脚本**

写入 `scripts/run_speedtune_train_eval.sh`：

```bash
#!/usr/bin/env bash
# SpeedTune 一体化：4 卡并行训练 per_action_toppra + fixed_time，训练完成后自动 eval 对比。
# 替代「run_speedtune_2backends.sh 训练 → 手填 CKPT → run_eval_compare.sh」两步手动衔接。
#
# 卡分配（4 卡；server 与 train 分卡避免光追 sapien 与 3B VLA 抢显存）：
#   per_action_toppra: server→GPU0:8102 | train→GPU1
#   fixed_time:        server→GPU2:8103 | train→GPU3
#   eval compare:      复用上面两个 server，compare→GPU1（train 退出后空出）
#
# 训练跑满 MAX_ITERS 自然退出 → final checkpoint 落盘（train finally: _stop+join 保证）→
# 自动找最新 update_N → 复用训练 server 跑 eval_speedtune_compare（不重起 server/不端口预检）。
#
# 用法（云端）：
#   export SPEEDTUNE_VLA_CKPT=/abs/path/.../params        # 必填
#   export SPEEDTUNE_VLA_ASSETS=/abs/path/.../assets      # 选填(norm_stats)
#   export SPEEDTUNE_VLA_ASSET_ID=trossen                 # 选填
#   bash scripts/run_speedtune_train_eval.sh
#   冒烟：MAX_ITERS=2000 N_EPISODES=3 bash scripts/run_speedtune_train_eval.sh
#
# 跑前清残留（光追退出不彻底会占显存 → cannot create buffer）：
#   pkill -9 -f run_robotwin_client; pkill -9 -f train_speedtune_async; pkill -9 -f eval_speedtune; sleep 3; nvidia-smi
# 停止：Ctrl-C（trap 清理）。

set -euo pipefail

# ===== 配置（环境变量可覆盖）=====
EXPO_ROOT="${EXPO_ROOT:-/home/chenlu/expo-ft}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/home/chenlu/RoboTwin}"
ROBOTWIN_ENV="${ROBOTWIN_ENV:-RoboTwin}"
PYTHON="${PYTHON:-uv run python}"
TASK_CONFIG="${TASK_CONFIG:-configs/task/robotwin_stack_blocks.py}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model/speedtune_dqn_config.py}"
WANDB_PROJECT="${WANDB_PROJECT:-expo-ft-speedtune}"
MAX_ITERS="${MAX_ITERS:-100000}"
SEED="${SEED:-42}"
SERVER_WAIT="${SERVER_WAIT:-45}"
TRAIN_MEM_FRAC="${TRAIN_MEM_FRAC:-0.85}"
COMPARE_MEM_FRAC="${COMPARE_MEM_FRAC:-0.85}"
STREAM_HOLD_STEPS="${STREAM_HOLD_STEPS:-15}"
N_EPISODES="${N_EPISODES:-5}"
MAX_DECISION_STEPS="${MAX_DECISION_STEPS:-400}"

# 执行层力矩底座（per-joint 单臂 τ_max, N·m；空串=不施加）。train+compare 进程都继承。
export SPEEDTUNE_FORCE_LIMIT="${SPEEDTUNE_FORCE_LIMIT:-30,40,30,15,10,10}"

: "${SPEEDTUNE_VLA_CKPT:?请先 export SPEEDTUNE_VLA_CKPT=<微调后 drift pi0.5 ckpt 绝对路径>}"
export SPEEDTUNE_VLA_CKPT
export SPEEDTUNE_VLA_ASSETS="${SPEEDTUNE_VLA_ASSETS:-}"
export SPEEDTUNE_VLA_ASSET_ID="${SPEEDTUNE_VLA_ASSET_ID:-}"

# 两 backend：名 / port / server 卡 / train 卡（与 eval 默认 port 对齐：per_action=8102, fixed_time=8103）。
BACKENDS=(per_action_toppra fixed_time)
PORTS=(8102 8103)
SERVER_GPUS=(0 2)
TRAIN_GPUS=(1 3)
COMPARE_GPU="${COMPARE_GPU:-1}"

STAMP="$(date +%m%d_%H%M)"
LOGDIR="${EXPO_ROOT}/logs/speedtune_traineval_${STAMP}"
OUTPUT_DIR="${LOGDIR}/compare_eval"
mkdir -p "$LOGDIR" "$OUTPUT_DIR"
echo "[*] 日志目录: $LOGDIR"
echo "[*] 冻结 VLA ckpt: $SPEEDTUNE_VLA_CKPT"
echo "[*] force_limit(τ_max): ${SPEEDTUNE_FORCE_LIMIT:-<空=不施加>}"

SERVER_PIDS=()
TRAIN_PIDS=()
cleanup() {
  echo ""
  echo "[cleanup] 终止所有子进程 ..."
  for p in "${TRAIN_PIDS[@]:-}" "${SERVER_PIDS[@]:-}"; do kill -9 "$p" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "$EXPO_ROOT"

# ---- 1) 起 2 个 server（每个独占一卡，光追渲染）----
for i in "${!BACKENDS[@]}"; do
  be="${BACKENDS[$i]}"; port="${PORTS[$i]}"; sgpu="${SERVER_GPUS[$i]}"
  echo "[$be] 启动 env server  server_GPU=$sgpu  port=$port"
  CUDA_VISIBLE_DEVICES="$sgpu" \
  conda run --no-capture-output -n "$ROBOTWIN_ENV" \
    python -m client_robotwin.run_robotwin_client \
      --config_task_path "$TASK_CONFIG" --robotwin_root "$ROBOTWIN_ROOT" --server_port "$port" \
      > "$LOGDIR/server_${be}.log" 2>&1 &
  SERVER_PIDS+=($!)
done

echo "[*] 等 server 渲染自检/就绪 (${SERVER_WAIT}s) ..."
sleep "$SERVER_WAIT"
for be in "${BACKENDS[@]}"; do
  if grep -q "address already in use" "$LOGDIR/server_${be}.log" 2>/dev/null; then
    echo "[ERROR] $be server 端口 bind 失败（address already in use）。先清理残留后重跑。" >&2
    exit 1
  fi
  if grep -q "Render Well" "$LOGDIR/server_${be}.log" 2>/dev/null; then
    echo "[$be] server ✓ Render Well"
  else
    echo "[$be] server 渲染状态未知（可能还在初始化）；若 train 报 cannot create buffer 看 server_${be}.log"
  fi
done

# ---- 2) 起 2 个 train（每个独占一卡，与 server 分卡）；只 wait train 进程 ----
for i in "${!BACKENDS[@]}"; do
  be="${BACKENDS[$i]}"; port="${PORTS[$i]}"; tgpu="${TRAIN_GPUS[$i]}"
  run_name="speedtune_${be}_${STAMP}"
  echo "[$be] 启动训练  train_GPU=$tgpu  client_port=$port  run=$run_name"
  CUDA_VISIBLE_DEVICES="$tgpu" XLA_PYTHON_CLIENT_MEM_FRACTION="$TRAIN_MEM_FRAC" \
  WANDB_PROJECT="$WANDB_PROJECT" \
    $PYTHON train_speedtune_async.py \
      --config "$MODEL_CONFIG" \
      --config.exec_backend "$be" \
      --config.max_iters "$MAX_ITERS" \
      --config.stream_hold_steps "$STREAM_HOLD_STEPS" \
      --config_task "$TASK_CONFIG" \
      --client_host localhost --client_port "$port" \
      --seed "$SEED" \
      --project_name "$WANDB_PROJECT" --run_name "$run_name" \
      --output_dir "$LOGDIR" \
      > "$LOGDIR/train_${be}.log" 2>&1 &
  TRAIN_PIDS+=($!)
  sleep 5
done

echo "[*] 训练中（MAX_ITERS=$MAX_ITERS）；只等 train 进程退出（server 常驻待 eval 复用）..."
echo "[*] 实时查看：tail -f $LOGDIR/train_per_action_toppra.log"
train_rc=0
for i in "${!TRAIN_PIDS[@]}"; do
  p="${TRAIN_PIDS[$i]}"; be="${BACKENDS[$i]}"
  if wait "$p"; then echo "[$be] 训练完成 ✓"; else echo "[$be] 训练异常退出 ✗（见 train_${be}.log）" >&2; train_rc=1; fi
done
if [ "$train_rc" -ne 0 ]; then
  echo "[ERROR] 有训练异常退出，跳过 eval。日志在 $LOGDIR" >&2
  exit 1
fi

# ---- 3) 自动发现各 backend 最新 checkpoint ----
CKPT_PER=$(ls -d "$LOGDIR/speedtune_per_action_toppra_${STAMP}/checkpoints/update_"* 2>/dev/null | sort -V | tail -1 || true)
CKPT_FIXED=$(ls -d "$LOGDIR/speedtune_fixed_time_${STAMP}/checkpoints/update_"* 2>/dev/null | sort -V | tail -1 || true)
if [ -z "$CKPT_PER" ] || [ -z "$CKPT_FIXED" ]; then
  echo "[ERROR] 找不到 checkpoint（per_action=$CKPT_PER fixed_time=$CKPT_FIXED）。看 train_*.log" >&2
  exit 1
fi
echo "[*] CKPT per_action_toppra = $CKPT_PER"
echo "[*] CKPT fixed_time        = $CKPT_FIXED"

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

echo ""
echo "[*] 全部完成。训练日志: $LOGDIR ; 对比结果: $OUTPUT_DIR"
echo "    - compare_summary.json / compare_speedup.png / compare_knob.png"
echo "    - <backend>/videos/ + <backend>/episode*_speed.{json,png}"
```

- [ ] **Step 2: 脚本语法检查 + 可执行**

Run: `bash -n scripts/run_speedtune_train_eval.sh && echo OK`
Expected: `OK`

Run: `chmod +x scripts/run_speedtune_train_eval.sh`

- [ ] **Step 3: Run（云端）端到端冒烟**

Run（云端，维护人执行；先小预算）:
```bash
cd /home/chenlu/expo-ft
export SPEEDTUNE_VLA_CKPT=/home/chenlu/openpi/checkpoints/pi05_aloha_robotwin_drifting_stack_blocks_two/drifting_v1/29999/params
export SPEEDTUNE_VLA_ASSETS=/home/chenlu/openpi/checkpoints/pi05_aloha_robotwin_drifting_stack_blocks_two/drifting_v1/29999/assets
export SPEEDTUNE_VLA_ASSET_ID=trossen
MAX_ITERS=2000 N_EPISODES=3 bash scripts/run_speedtune_train_eval.sh
```
Expected:
- 两 server `Render Well`；两 train 跑满 2000 步后 `训练完成 ✓`；
- final checkpoint 在 `$LOGDIR/speedtune_*_<stamp>/checkpoints/update_*`；
- compare 正常产出 `compare_eval/compare_summary.json` + `compare_speedup.png` + `compare_knob.png`，**图全英文无方框**；
- server log 无 `apply force_limit failed`（force_limit 施加成功）。

- [ ] **Step 4: Commit（维护人确认后）**

```bash
git add scripts/run_speedtune_train_eval.sh
git commit -m "$(cat <<'EOF'
feat(speedtune): train+eval 一体化脚本(4卡训练→自动找ckpt→复用server eval对比)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## 集成验证（全部 task 后，云端）

- [ ] 正式跑（满预算）：`bash scripts/run_speedtune_train_eval.sh`（默认 `MAX_ITERS=100000`、`N_EPISODES=5`），确认：
  1. 两 backend 训练→eval 全自动衔接；
  2. force_limit 生效：per_action 力矩**不饱和**、fixed_time 激进时**饱和**（design 2026-06-26 §3）；
  3. 图表全英文、无豆腐块；
  4. `compare_summary.json` 给出 per_action vs fixed_time 加速比。
- [ ] 零侵入回归：EXPO/BC（`train_pi_robo_async.py`）仍正常、`RoboTwinEnv` 不传 force_limit 时默认 None 不施加。

---

## Self-Review

**Spec coverage（对照 spec §2/§3/§4）：**
- §2 A1 config.force_limit → Task 2 Step 1 ✓
- §2 A2 parse_force_limit + 单测 → Task 1 ✓
- §2 A3/A4/A5 三处 env_creation_request 透传 → Task 2 Step 2/3/4 ✓
- §2 A6 run_robotwin_client 白名单 → Task 2 Step 5 ✓
- §2 A7 robotwin_env docstring（_apply_force_limit 不改）→ Task 2 Step 6 ✓
- §3 合并脚本（4 卡训练→checkpoint 发现→复用 server eval）→ Task 4 ✓
- §4 图表英文化（compare 7 处 + bar + eval 2 处 + 注释）→ Task 3 ✓
- §5 零侵入（train_pi_robo 不动、robot.py 不动）→ Global Constraints + 集成验证 ✓
- §6 测试（parse 单测 + py_compile + bash -n + 云端集成）→ 各 task Step ✓

**Placeholder scan:** 无 TBD/TODO；每步给确切 old→new 代码或确切命令 + Expected。

**Type consistency:** `parse_force_limit(spec)->Optional[List[float]]` 在 Task 1 定义、Task 2 三处调用一致；`config.force_limit`(str) Task 2 Step 1 产、Step 2/3/4 用 `config.get("force_limit","")` 消，一致；`env_creation_request["force_limit"]` Task 2 产、`run_robotwin_client` 白名单（Step 5）消，一致；checkpoint 路径 `<output_dir>/<run_name>/checkpoints/update_<N>` 与 Task 4 的 `ls .../speedtune_<be>_<stamp>/checkpoints/update_*` 一致。
