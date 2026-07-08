# Paper SpeedTuning Reward Design

## 背景

当前 SpeedTune 训练脚本使用 episode-end relabel：episode 成功且 fixed-time 没有速度违规时，整条 episode 的每个决策 transition 才获得 success-gated raw-speed reward；失败 transition reward 为 0。这是为了避免“无脑最大速度”的 reward hacking。

这次要新增的是复现 `speedtuning_icra.pdf` 的 baseline reward，而不是替换现有 reward。论文的可复现核心是：

```text
r_ST = alpha * r_speed(v_t) + r_task(s_t, a_t)
r_speed(v) = v^beta
```

论文图中使用 `beta=2`，没有在正文中给出固定 `alpha` 默认值。因此实现必须保留 `alpha` 为 config 参数，便于扫参。

## 需求

1. 新增 `config.reward_mode = "paper_speedtuning"`，默认保持现有 `"success_gated"`。
2. `"success_gated"` 走现有逻辑，不改变旧实验含义。
3. `"paper_speedtuning"` 复现论文 reward：速度项不 success-gated，失败时也保留 `alpha * v^beta`，不额外加入失败速度惩罚。
4. 任务 reward 只用环境本身返回的 `r_task`。通常只有 terminal transition 有 `r_task=1`，不把成功 reward relabel 到全 episode。
5. 当前训练只存每个 DQN 决策的一条 transition；SpeedTuning 论文的 action、reward 和 discount 都是在 chunk 这一层定义，不展开 chunk 内组成动作。
6. ReplayBuffer 可保留 per-transition discount 能力，但 paper baseline 写入 replay 时使用 chunk-level `gamma`，不是 `gamma^execution_steps`。

## Reward 和 Discount 语义

对一个 chunk transition，速度 reward 输入为 `v_list`，折扣为 `gamma`。论文 baseline 不把 chunk 内组成动作展开为多个 DQN transition。

```text
reward_chunk = alpha * sum_i(v_i^beta) + r_task
```

任务项只来自当前 chunk 的环境返回 `r_task`；通常只有 terminal chunk 为 1，不把成功 reward relabel 到全 episode。写入 replay 的 transition discount 为：

```text
transition_discount = gamma
```

若 transition `done=True`，ReplayBuffer 在 n-step 聚合时仍把 bootstrap discount 截断为 0。

## Alpha 选择

论文未给固定 `alpha`，这里不写死。由于 paper mode 会给失败 episode 速度 reward，`alpha` 决定“追速度”和“等 terminal success”的比例。

在 `gamma=0.99`、max raw speed `v=4`、`beta=2` 时，单个 chunk 的最大速度项为 16。C51 support 按最多 chunk 决策数估算，例如 fixed-time `k_skip=10`、800 个高层 action step 约为 80 个 chunk：

```text
16 * (1 - 0.99^80) / (1 - 0.99) ~= 884.0
```

因此：

```text
alpha=1e-5  -> fixed-time max speed return ~= 0.0088
alpha=1e-4  -> fixed-time max speed return ~= 0.0884
alpha=3e-4  -> fixed-time max speed return ~= 0.265
alpha=1e-3  -> fixed-time max speed return ~= 0.884
```

为了复现 baseline 且避免 C51 support 太粗，paper mode 的 support 应随 `alpha` 和 `beta` 动态缩放，而不是沿用 success-gated 的 `[0,900]` / `[0,550]`。

推荐实验先扫：

```text
alpha in {1e-5, 3e-5, 1e-4, 3e-4, 1e-3}
beta = 2
epsilon_decay_steps in {1000, 2000, 4000}
```

如果目标是更高实机样本效率，另一个方向是保留默认 success-gated reward，同时缩短 epsilon schedule、提高 warmup 后每 episode update 数、做离线 warm-start 或用仿真预训练 Q 网络。paper baseline 本身不保证实机友好，因为失败高速也有正 reward。

## 代码结构

修改点保持在 SpeedTune 边界内：

- `configs/model/speedtune_dqn_config.py`：新增 `reward_mode`、paper support 估算所需 horizon 参数，并把 epsilon decay 默认调小。
- `expo_ft/speedtune/runtime_config.py`：根据 reward mode 返回 backend support；paper mode 支持动态 C51 support。
- `expo_ft/speedtune/training_runtime.py`：新增 paper reward helper，并在 `flush_pending_episode` 内按 mode 选择 reward；paper 和 success-gated 都按 chunk 写入 `gamma`。
- `expo_ft/data/speedtune_buffer.py`：`insert` 增加可选 `discount`，默认仍为 `self.gamma`，旧调用不受影响。
- `train_speedtune_async.py` 和 `train_speedtune_sync.py`：pending transition 保存 `execution_steps`，flush 时传入 reward mode / alpha / beta / gamma。
- `test/*speedtune*`：用 TDD 增加 variable discount、paper reward、配置和训练脚本接线测试。

## 测试策略

1. ReplayBuffer 单元测试覆盖 variable discount n-step 聚合：
   - reward 使用每条 transition 的折扣累计；
   - terminal transition bootstrap discount 为 0。
2. Runtime 单元测试覆盖 paper mode：
   - `execution_steps=3, gamma=0.5, alpha=0.1, beta=2, v=4` 时 reward 仍为 chunk-level `0.1*16 + r_task`；
   - 失败 episode 仍保留速度 reward；
   - `discount` 参数写入为 `gamma`。
3. Config 测试覆盖默认 mode 不变、paper mode support 随 alpha 缩放。
4. 训练脚本源码测试覆盖 pending transition 包含 `execution_steps`，flush 调用传入 reward config。

## 自检

- 没有改动现有 success-gated reward 的默认语义。
- paper mode 不加失败速度惩罚，符合 baseline 复现要求。
- 任务 reward 不 episode-wide relabel，符合“只给最后一个 transition”的要求。
- 折扣使用 chunk-level `gamma`，不是 `gamma^execution_steps`、`gamma^n_exec_steps` 或物理仿真 dense step。
