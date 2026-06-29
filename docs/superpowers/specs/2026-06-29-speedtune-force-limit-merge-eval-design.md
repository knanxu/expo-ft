# SpeedTune force_limit 接通 + train/eval 合并脚本 + 图表英文化

> 日期：2026-06-29 ｜ 维护人：polar823 ｜ 分支：dbpo-robotwin
> 范围：在 2026-06-26「per_action 绝对值制 + 力矩底座」设计之上，**补完执行层 force_limit 全链路接通**、
> 新增 **train+eval 一体化脚本**、修复 **eval 图表中文渲染（豆腐块）→ 英文**。
> 运行环境：训练/rollout/eval 均在云端；本地仅静态检查（不跑 pytest/仿真/sapien）。**零侵入** EXPO/BC。

---

## 1. 背景与现状

2026-06-26 阶段 1（绝对值制 + 力矩底座）已大部分落地，本地 bench 验证后又做了若干 debug 修订（方法B 非零边界、sd_reachable 钳、段内开环前馈）。当前接口已三方自洽：

- `exec_backends.py` 绝对值制（`vel_limit/acc_limit`）、`decode` key 已切换；
- `robotwin_env.step_chunk` 读新 key 透传 `take_chunk_action_per_action(vel_limit, acc_limit, v)`；
- RoboTwin 侧 `take_chunk_action_*` 已绝对值制（git `39b59ee`/`d73efab`）。

**grid 维护人已自行调整**（本设计不动 grid）：

| backend | head_sizes（变量档数） | 重训? |
|---|---|---|
| `fixed_time` | `(5,)`（v 5 档） | head 未变，可选复用 |
| `per_action_toppra` | `(5, 3, 5)`（v=5, vel_limit=`(1,2,3)`=3, acc_limit=`(1,3,5,7,9)`=5） | **必须重训** |
| `chunk_toppra` | `(5, 3, 5)` | 必须重训 |

> 本批改动**不改 head_sizes**，但因 grid 已变、per_action checkpoint 作废，本次训练本就是全新 run。

**三个待补的缺口：**

1. **force_limit 链路断开**：`robotwin_env._apply_force_limit` 已实现（默认 `None` 不施加），但 ① task config 无字段；② `train_speedtune_async.py` / `eval_speedtune_compare.py` / `eval_speedtune.py` 的 `env_creation_request` 没传 force_limit；③ `run_robotwin_client.py` 透传白名单 `(exec_backend, k_skip, stream_hold_steps)` 没 force_limit。bench 脚本里是手动 monkey-patch 加的，正式链路缺。
2. **train 与 eval 分两脚本**：`run_speedtune_2backends.sh`（训练 2 backend）与 `run_eval_compare.sh`（eval 对比）需手动衔接（手填 CKPT_A/CKPT_B）。
3. **eval 图表中文渲染豆腐块**：matplotlib 默认字体（DejaVu Sans）无 CJK 字形，标题/标签中文显示为方框。

---

## 2. 设计 A：force_limit 力矩底座全链路接通

**配置源选型**：用环境变量 `SPEEDTUNE_FORCE_LIMIT`（对齐现有 `pi05_*` 路径风格 + 脚本好传）。**不**放共用 `configs/task/robotwin_stack_blocks.py`——那会被 EXPO/BC 一起吃到，破坏零侵入（design 2026-06-26 §7）。

**数据流：**

```
脚本 export SPEEDTUNE_FORCE_LIMIT="30,40,30,15,10,10"
  → speedtune_dqn_config.force_limit (str)
  → learner 端 parse_force_limit() → list[float] | None
  → env_creation_request["force_limit"]
  → msgpack → run_robotwin_client 白名单透传
  → env_kwargs["force_limit"] → RoboTwinEnv(**kwargs)
  → 每次 reset() 后 _apply_force_limit() 对左右臂各 6 个 arm 关节 set_drive_property(force_limit=τ_max)
```

**改动清单（7 处）：**

| # | 文件 | 改动 |
|---|---|---|
| A1 | `configs/model/speedtune_dqn_config.py`（`:32` 后） | 新增 `config.force_limit = os.environ.get("SPEEDTUNE_FORCE_LIMIT", "30,40,30,15,10,10")`。per-joint 单臂 τ_max(N·m)，ARX5；空串=不施加(∞)=旧行为。注释说明。 |
| A2 | `expo_ft/speedtune/exec_backends.py` | 新增纯函数 `parse_force_limit(spec) -> list[float] | None`：逗号串/list/tuple → `[float...]`，空/None → `None`。**纯 numpy，本地可单测**。 |
| A3 | `train_speedtune_async.py`（`:192-200`） | import `parse_force_limit`；`env_creation_request` 加 `"force_limit": parse_force_limit(config.get("force_limit", ""))`。 |
| A4 | `eval_speedtune_compare.py`（`_make_env`, `:70-78`） | 同 A3：`env_creation_request` 加 `"force_limit": parse_force_limit(config.get("force_limit", ""))`（`_make_env` 已有 `config` 参数）。 |
| A5 | `eval_speedtune.py`（`:339-345`） | 同 A3（单 backend eval 入口，保持一致）。 |
| A6 | `client_robotwin/run_robotwin_client.py`（`:76`） | 透传白名单 `("exec_backend", "k_skip", "stream_hold_steps")` → 加 `"force_limit"`。已有 `if request.get(_sk) is not None` 守护，`None` 自动跳过。 |
| A7 | `client_robotwin/envs/robotwin_env.py` | `_apply_force_limit` **已实现不改**；仅清 `:369` docstring 过时 `vel_scale/acc_scale`。 |

**`parse_force_limit` 语义**（写进单测）：

| 输入 | 输出 |
|---|---|
| `"30,40,30,15,10,10"` | `[30.0, 40.0, 30.0, 15.0, 10.0, 10.0]` |
| `""` / `"  "` / `None` | `None`（=不施加） |
| `[30, 40]` / `(30.0, 40.0)` | `[30.0, 40.0]` |
| `"30, 40 ,30"`（含空格） | `[30.0, 40.0, 30.0]` |

`None` 经 `env_creation_request` 传到 `run_robotwin_client` 被 `is not None` 跳过 → `RoboTwinEnv` 默认 `None` → 不施加。

---

## 3. 设计 B：train+eval 一体化脚本 `scripts/run_speedtune_train_eval.sh`（新建）

新建自包含脚本；原 `run_speedtune_2backends.sh` / `run_eval_compare.sh` **保留不动**（仍可单独用）。

**checkpoint 可靠性（已核实）**：`train_speedtune_async.py` 主循环 `finally`（`:341-345`）执行 `_stop.set()` + `_thread.join(timeout=120)`，update 线程退出前必跑 `:263` 的 `_save_checkpoint` → 训练跑满 `MAX_ITERS` 自然退出时，final checkpoint 可靠落盘到 `<log_dir>/checkpoints/update_<N>`（与 `checkpoint_model` flag 无关）。`log_dir = output_dir/run_name`。

**脚本结构：**

```bash
Phase 0 配置
  export SPEEDTUNE_FORCE_LIMIT="${SPEEDTUNE_FORCE_LIMIT:-30,40,30,15,10,10}"
  STAMP / LOGDIR / 卡 / port / MAX_ITERS / SERVER_WAIT / 跑前清残留提示

Phase 1 训练（4 卡；server 与 train 分卡避免抢显存）
  server: per_action_toppra→GPU0:8102 | fixed_time→GPU2:8103   （conda run -n RoboTwin，光追独占）
  train : per_action_toppra→GPU1      | fixed_time→GPU3         （uv run，3B VLA+DQN 独占）
  run_name=speedtune_<be>_<STAMP>，--output_dir $LOGDIR
  等 server "Render Well"（含 "address already in use" bind 失败检测 → abort，沿用 9713d8a 逻辑）
  记录两个 TRAIN PID → `wait $TPID1 $TPID2`（**只等 train，不等 server**）
  捕获退出码：任一 train 非 0 → 报错、跳过 eval、cleanup 退出

Phase 2 eval（复用训练的 2 个 server，省重起/渲染自检）
  自动发现 checkpoint（取最新 update_N）：
    CKPT_B=$(ls -d $LOGDIR/speedtune_per_action_toppra_$STAMP/checkpoints/update_* | sort -V | tail -1)
    CKPT_A=$(ls -d $LOGDIR/speedtune_fixed_time_$STAMP/checkpoints/update_*        | sort -V | tail -1)
  缺 checkpoint → 报错退出
  起 1 个 compare（GPU1，train 退出后空出）：直接调 eval_speedtune_compare.py
    --backend_a fixed_time --ckpt_a $CKPT_A --port_a 8103
    --backend_b per_action_toppra --ckpt_b $CKPT_B --port_b 8102
    --config.force_limit 经 SPEEDTUNE_FORCE_LIMIT 环境变量自动生效
  **不做** run_eval_compare.sh 的端口预检（server 是本脚本自己起的，端口被占是预期）

Phase 3 cleanup：trap EXIT/INT/TERM 杀全部 server + 子进程
```

- **port/backend 与 eval 默认对齐**（per_action=8102, fixed_time=8103）→ server 直接复用。
- 关键环境变量可覆盖：`MAX_ITERS`、`SEED`、`STREAM_HOLD_STEPS`、`SPEEDTUNE_FORCE_LIMIT`、`N_EPISODES`、`MAX_DECISION_STEPS`、卡/port、`SPEEDTUNE_VLA_*`。

---

## 4. 设计 C：eval 图表中文 → 英文（+ 非 ASCII 标记 ASCII 化）

仅改**会渲染到图片上**的字符串；代码注释/docstring/日志里的中文保留（项目中文要求）。

**`eval_speedtune_compare.py`（`_plot_compare`）：**

| 行 | 现 | 改为 |
|---|---|---|
| `:172` | `per-episode 执行步数（越低越快; ✓=成功 ✗=失败）` | `Per-episode exec steps (lower=faster; S=success F=fail)` |
| `:184` | `aggressiveness (0慢~1快)` | `aggressiveness (0=slow ~ 1=fast)` |
| `:186` | `加速激进度（归一化）随决策步` | `Normalized aggressiveness vs decision step` |
| `:191` | `SpeedTune backend 对比: {na} vs {nb}{sp_txt}` | `SpeedTune backend comparison: {na} vs {nb}{sp_txt}` |
| `:200` | `{name}: 无 episode` | `{name}: no episode` |
| `:209` | `{name} ep{ep}{_tag}: 加速参数 + 抓取/接触` | `{name} ep{ep}{_tag}: speed params + grasp/contact` |
| `:211` | `各 backend 实际下发的加速控制参数随决策步` | `Speed-control params issued per backend, vs decision step` |
| `:162-167` | bar 顶标记 `✓` / `✗` | `S`(绿) / `F`(红) — 避开 ✓✗ 字体不确定性 |

**`eval_speedtune.py`（`_save_and_plot`）：**

| 行 | 现 | 改为 |
|---|---|---|
| `:202` | `aggressiveness (0慢~1快)` | `aggressiveness (0=slow ~ 1=fast)` |
| `:203` | `left gripper (0闭~1开)` | `left gripper (0=closed ~ 1=open)` |

（`:168/:175/:204/:209` 已英文，不动。）

---

## 5. 零侵入边界

- **EXPO/BC**：`train_pi_robo_async.py:153` 的 `train_env_creation_request` **不碰 force_limit** → `RoboTwinEnv` 默认 `None` → 不施加，行为不变。
- **force_limit 默认值仅在 SpeedTune config**（`speedtune_dqn_config.py`），EXPO/BC config（`expo_ft_pi_*`）不含。
- **不改 `robot.py`**；`_apply_force_limit` 在 env 层覆盖 `set_drive_property`，保留 robot 已设 stiffness/damping。
- 原 `run_speedtune_2backends.sh` / `run_eval_compare.sh` 保留不动。

---

## 6. 测试

- **本地（静态）**：`ast.parse` 全部改动 `.py`；`bash -n scripts/run_speedtune_train_eval.sh`。
- **`test/exec_backends_test.py` 加 `parse_force_limit` 单测**（纯逻辑，不依赖 sapien，本地/云端均可）：覆盖 §2 表的 6 种输入。
- **图表英文化**：无法单测（matplotlib 渲染），云端 eval 后人工查图无豆腐块。
- **force_limit 链路 / 合并脚本**：依赖 sapien，**云端集成验证**（§7）。

---

## 7. 云端验证（维护人执行）

1. `bash scripts/run_speedtune_train_eval.sh`（设 `MAX_ITERS` 小值先冒烟，如 2000）端到端：
   - Phase 1 两 backend 训练正常退出、final checkpoint 落盘；
   - Phase 2 自动发现 checkpoint、compare 正常产出。
2. force_limit 生效：server log 无 `set_drive_property` 异常；per_action 力矩**不饱和**、fixed_time 激进时**饱和**（design 2026-06-26 §3）。
3. 产物：`compare_summary.json` + `compare_speedup.png` + `compare_knob.png` + 各 backend `videos/` + `episode*_speed.png`，**图全英文、无方框**。

---

## 8. 输入参数（维护人提供，已有默认）

| 参数 | 默认 | 来源 |
|---|---|---|
| `SPEEDTUNE_FORCE_LIMIT`（per-joint 单臂 τ_max, N·m） | `30,40,30,15,10,10` | ARX5 arx5-sdk `config.h`（bench 已用） |
| `MAX_ITERS`（总决策步） | `100000`（冒烟建议先 2000） | — |
| `SPEEDTUNE_VLA_CKPT/ASSETS/ASSET_ID` | 云端 drift ckpt 根目录 | 见 CLAUDE.md |

---

## 9. 实现顺序（供 plan 拆 task）

1. A2 `parse_force_limit` + 单测（TDD：先测后实现）。
2. A1 config.force_limit。
3. A3/A4/A5 三处 env_creation_request 透传 + import。
4. A6 run_robotwin_client 白名单。
5. A7 + §4 注释/docstring 清理。
6. 设计 C 图表英文化（compare + eval 两文件）。
7. 设计 B 合并脚本新建 + `bash -n`。
8. 全量 `ast.parse` + 静态 review；commit（**维护人确认后**，分 expo-ft 一处提交）。

> commit 项目约定需维护人批准（CLAUDE.md）；本批改动全在 expo-ft repo（RoboTwin repo 本批不动）。
