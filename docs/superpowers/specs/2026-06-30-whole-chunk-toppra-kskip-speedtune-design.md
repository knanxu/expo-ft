# Whole chunk TOPPRA + k_skip 的 SpeedTune 训练 / eval — 设计文档

- 日期：2026-06-30
- 维护人：polar823
- 分支：dbpo-robotwin
- 状态：设计已与维护人对齐，待写实现计划（writing-plans）

## 1. 背景与动机

SpeedTune 加速模块（冻结 VLA 之上的 branching Rainbow-DQN，学速度档）当前支持三种动作执行方式（exec_backend），由 `--config.exec_backend` 字符串切换，学习栈本身 backend 无关：

| backend 名 | RoboTwin 方法 | DQN heads | 执行语义 | k_skip |
|---|---|---|---|---|
| `fixed_time` | `take_chunk_action_streaming` | `(5,)` v | 线性插值 + hold，无 TOPP | 生效 |
| `per_action_toppra` | `take_chunk_action_per_action` | `(5,3,5)` v/vel/acc | 逐 action 两点零速 TOPP（段间 stop-and-go） | 生效 |
| `chunk_toppra`（whole chunk TOPPRA） | `take_chunk_action` | `(5,3,5)` v/vel/acc | 整段一次 TOPPRA，段间速度连续 | **当前不生效** |

`chunk_toppra` 在 `exec_backends.py`（spec）和 `robotwin_env.py`（`chunk_toppra→whole_chunk→take_chunk_action` 分发）两个接缝都已接好，单 backend 训练/eval 今天就能跑。**但 whole chunk TOPPRA 路径当前完全没有 k_skip 接线**：

- RoboTwin `take_chunk_action`（`envs/_base_task.py:1787`）签名无 `max_actions`，内部不调 `_apply_k_skip`（后者只在 per_action `:2023`、streaming `:2199` 被调用）。
- expo-ft `robotwin_env.py` whole_chunk 分支（`:421-424`）不传 `max_actions`，注释为「k_skip 不适用」。
- `config.k_skip` 注释（`:42-44`）写「whole_chunk 忽略此项」。

维护人已用 codex 在云端实现 k_skip 改进（本地 `/home/xukainan/RoboTwin` 是只读对照副本，未同步），目标是把 whole chunk TOPPRA 也接入 k_skip，并补上训练脚本与 eval。

## 2. 目标 / 非目标

**目标**

1. 让 whole chunk TOPPRA（`chunk_toppra`）支持 k_skip：整段对前 `k_skip` 帧做一次 TOPPRA，执行完即重推 VLA（receding-horizon 闭环）。
2. 训练脚本支持训练 `chunk_toppra`，且**外露 `BACKENDS` 参数**让维护人在训练前指定要训的两个 backend（云端只支持 2 个并行）。
3. eval 能对训练的 backend 做对比（含 chunk_toppra）。

**非目标 / YAGNI**

- 不消除 k_skip 窗口首尾零速（`sd_start=sd_end=0`）带来的轻微减速；相邻窗口速度衔接是后续优化。
- 不改 reward 设计（沿用 success-gated）。
- 不改 EXPO/BC 任何路径；不改 DQN/buffer/network/`train_speedtune_async.py`。
- 不把 `eval_speedtune_compare.py` 扩成 3 路（最多 2 个并行 → 永远 2 路对比足够）。

## 3. 关键语义：whole chunk TOPPRA + k_skip

维护人确认的语义（「chunk 重构生成新 chunk，只取前 k_skip 个 actions 传入后续的样条插值和 TOPPRA 过程」），对应 `take_chunk_action` 执行流，截断点插在 **reconstruct 之后、TOPPRA 之前**：

```
take_chunk_action(chunk, vel_limit, acc_limit, v, max_actions):
  1. reconstruct_chunk(chunk, v)        # chunk 重构（v 压缩/插帧）        (现 _base_task.py:1841-1847)
  2. _apply_k_skip(chunk, max_actions)  # 只取前 k_skip 帧                 【新增 1 行】
  3. remaining_budget 截断 + M = len(chunk)                              (现 :1849-1859)
  4. 拆双臂 → retime_chunk(SplineInterpolator 样条 + TOPPRA)             (现 :1861+)
  5. 250Hz 密集下发 → 返回 → 上层重推 VLA（闭环）
```

与 per_action/streaming 的 `_apply_k_skip` 用法一致；区别在 whole_chunk 对这 `k_skip` 帧做**一次整段** TOPPRA（窗口内速度连续），而非逐帧两点零速。`k_skip` 窗口首尾仍 `sd=0`（轻微减速），但远轻于 per_action 的逐帧 stop-and-go。

## 4. 改动清单

### 4.1 RoboTwin 侧（本地镜像实现 `/home/xukainan/RoboTwin`，维护人再同步云端）

| 文件 | 改动 |
|---|---|
| `envs/_base_task.py` `take_chunk_action`（:1787） | ① 签名加 `max_actions: int = None`；② 在 `reconstruct_chunk` 之后、`remaining_budget` 截断之前插入 `action_chunk = _apply_k_skip(action_chunk, max_actions)`；③ docstring 补 `max_actions` 说明 |
| `envs/_base_task.py` `take_chunk_action_backend`（:2274） | **默认不改**（expo-ft 走 `step_chunk`→`take_chunk_action`，不经此入口）。仅当维护人要 RoboTwin 自用 eval 也支持 whole_chunk+k_skip 时再加 `max_actions=None` 透传 |
| `envs/whole_chunk_kskip_test.py`（新增） | 仿 `whole_chunk_start_state_test.py` / `per_action_zero_boundary_test.py`，断言 `max_actions` 截断后只执行前 k 帧；本地只 `ast.parse` 静检，不执行 |

注：本地 RoboTwin 是只读对照副本。此处镜像实现以统一签名（参数名 `max_actions`）、供本地静检与云端对齐；最终以维护人云端 `/home/chenlu/RoboTwin` 为准。`_call_backend`（见 4.2）的签名过滤保证两侧即使短暂不一致也不报错。

### 4.2 expo-ft 侧

| 文件 | 改动 |
|---|---|
| `client_robotwin/envs/robotwin_env.py`（:421-424） | whole_chunk 分支改为传 `max_actions=self._k_skip`；更新注释（whole_chunk 现支持 k_skip）。`_call_backend`（:372-384）已用 `inspect.signature` 过滤 kwargs → 向后兼容：云端 RoboTwin 支持 `max_actions` 即生效，不支持则被过滤、降级整段执行，不报错 |
| `configs/model/speedtune_dqn_config.py`（:42-44） | 更新 `k_skip` 注释：去掉「whole_chunk 忽略此项」，说明三种执行方式均生效（whole_chunk 取前 k_skip 帧整段 TOPPRA） |

**零改动（确认）**：`expo_ft/speedtune/exec_backends.py`（`chunk_toppra` spec 已存在，3 head v/vel/acc，success-gated reward）、`speedtune_dqn.py`、`speedtune_buffer.py`、`rainbow_dqn.py`、`train_speedtune_async.py`、`eval_speedtune.py`、`eval_speedtune_compare.py`。

### 4.3 训练脚本：参数化 `scripts/run_speedtune_train_eval.sh`

现状：server/train/wait 循环已数组驱动（L79-132），但 `BACKENDS`/`PORTS`/`SERVER_GPUS`/`TRAIN_GPUS` 硬编码（L52-55），checkpoint 发现（L139-146）和 eval compare 调用（L155-156）硬编码 `per_action`/`fixed_time`。

参数化改动：

1. **`BACKENDS` 外露**：`BACKENDS="${BACKENDS:-per_action_toppra chunk_toppra}"` 经环境变量传入，`read -ra` 拆成数组。默认值改为 `per_action_toppra chunk_toppra`（主线变为训练对照 chunk_toppra）。
2. **校验**：`${#BACKENDS[@]}` 必须 ∈ {1,2}；>2 报错退出并提示「云端只支持 2 个并行」。
3. **交互 fallback**：未设 `BACKENDS` 且 stdin 是 TTY（`[ -t 0 ]`）→ 列出 `fixed_time / per_action_toppra / chunk_toppra`，`read` 让维护人选 2 个；非交互（nohup 后台）→ 用默认值并打印提示，不卡住。
4. **`PORTS`/`SERVER_GPUS`/`TRAIN_GPUS` 切片**：按 backend 数取前 N 个（`PORTS=(8102 8103)` 等基础数组切片）。
5. **checkpoint 发现循环化**（替换 L139-146）：对每个 `be` in `BACKENDS` 做 `ls -d "$LOGDIR/speedtune_${be}_${STAMP}/checkpoints/update_"* | sort -V | tail -1`，存入数组。
6. **eval 调用参数化**（替换 L148-161）：
   - 2 个 backend：`eval_speedtune_compare.py` `--backend_a ${BACKENDS[0]} --ckpt_a <ckpt0> --port_a ${PORTS[0]} --backend_b ${BACKENDS[1]} --ckpt_b <ckpt1> --port_b ${PORTS[1]}`。
   - 1 个 backend：跳过 compare，改用单 backend `eval_speedtune.py --config.exec_backend ${BACKENDS[0]} --dqn_ckpt <ckpt0> --client_port ${PORTS[0]}`。

### 4.4 eval

- `eval_speedtune_compare.py`：**不改**（本就支持任意 2 backend 组合）。
- `eval_speedtune.py`：**不改**（已支持 `--config.exec_backend chunk_toppra`）。
- `run_backend_episodes` 的 `dense_steps` 跨 backend 可比逻辑不变。加 k_skip 后 chunk_toppra 决策粒度变为 k_skip 帧，决策步数与 per_action 同量级。

## 5. reward 与超参

- **reward**：沿用 `exec_backends.py` 现成 success-gated `chunk_toppra` spec（`r_task + 1[success]·Σ αᵢvᵢ^βᵢ`，3 head v/vel/acc，与 per_action 同 grid、同 α=0.5/β=1.0）。零改动、跨 backend 可比、防 reward-hacking。
- **超参不特殊调**：加 k_skip 后 chunk_toppra 决策粒度从整段 50 帧变为 k_skip(默认 10) 帧，与 per_action 同量级，`max_iters`/`epsilon_decay_steps`/`learning_starts`（按决策步计）直接共用现有 config。

## 6. 零侵入核对

- EXPO/BC 路径、其 config、`droid_env.py`、原 `train_pi_robo*.py` 零触碰。
- SpeedTune 学习栈（DQN/buffer/network/`train_speedtune_async.py`）零改动。
- 仅动：SpeedTune env 分发（robotwin_env whole_chunk 分支）、config 注释、SpeedTune 脚本、RoboTwin 后端内新增**可选参数**（`max_actions` 默认 None = 旧行为，向后兼容）。

## 7. 测试

- `test/exec_backends_test.py`：已断言 `chunk_toppra` 构建 3 head，不改。
- 新增 RoboTwin `whole_chunk_kskip_test.py`：断言 `max_actions=k` 时只执行前 k 帧（dense_steps / take_action_cnt_delta 对应 k 帧）。本地只 `ast.parse` 静检，不执行。

## 8. 验证（全部云端）

1. 本地：`ast.parse` 静检所有改动文件语法。
2. 云端：`BACKENDS="per_action_toppra chunk_toppra" bash scripts/run_speedtune_train_eval.sh`（冒烟 `MAX_ITERS=2000 N_EPISODES=3`）→ 两 backend 并行训练 → 自动 2 路对比。
3. 检查 chunk_toppra 决策步的 `dense_steps`（应反映 k_skip 帧执行）、`exec_status`、对比 `compare_summary.json` 的成功率 / 速度。

## 9. 风险 / 已知差距

- 本地无法运行（显存/依赖），所有训练/eval/测试在云端验证。
- 云端 RoboTwin 的 codex 改动若参数名不是 `max_actions`，则 expo-ft 侧传的 `max_actions` 会被 `_call_backend` 过滤、k_skip 不生效（降级整段执行，不报错）。需以本设计统一参数名为 `max_actions`，或维护人核对云端签名后告知。
- k_skip 窗口首尾零速的轻微减速是已知代价（YAGNI 不在本次消除）。

## 10. 决策记录

- **k_skip 语义**：整段 TOPPRA + 执行前 k_skip 帧重推（reconstruct 后、TOPPRA 前截断）。
- **RoboTwin 改动**：本地镜像实现，统一参数名 `max_actions`，维护人同步云端。
- **reward**：沿用现有 success-gated。
- **脚本形态**：参数化主线 `run_speedtune_train_eval.sh`，外露 `BACKENDS`（最多 2 个）+ 交互 fallback。
- **eval**：保持 2 路 compare（不扩 3 路）。
