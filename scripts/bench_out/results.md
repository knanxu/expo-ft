# 三后端执行对比 (stack_blocks_two)

> ⚠️ **本文件由 `scripts/bench_exec_backends.py` 在本地重跑时自动覆盖生成**（纯专家数据回放，不碰 VLA/jax/torch，本地可跑）。下方「上一轮」为**旧 sweep 的真实结果**
> （缺 vel/acc=1.0 下端、粒度粗、无零边界 A/B）。脚本已扩充，云端重跑后会得到：
>
> 1. **per_action_zero 组**（段间边界速度强制为 0，还原 RoboTwin 原版逐 action 两点零速 stop-and-go）+ 文末
>    **A/B 对照表**（zero/B 步数比）→ 回答补充测试①「零边界 TOPPRA 完成任务需多少物理步」。
>    *与 method B 唯一变量 = 边界速度*（同一 retime_chunk 执行核、同绝对 vel/acc、同开环前馈），步数口径诚实（250Hz 逐步采样）。
> 2. **vel/acc sweep 下探到 1.0 且加密**：`v∈{1,2,3}`、`vel∈{1,2,3,4,5}`、`acc∈{1,2,3,4,6,8}`（各只动一个变量，
>    其余固定 baseline=v1/vel5/acc8）→ 回答补充测试②（旧 sweep 缺 1.0、看不出 vel_limit 何时真 binding）。
>
> **配置量**：fixed_time 7 + per_action 12 + per_action_zero 12 + whole_chunk 12 = **43 config × 50 ep ≈ 2150 episode**（本地回放，偏大耗时）。
> 嫌慢可：①注释脚本里 `per_action_zero` 那行（A/B 表自动跳过）；②`--episodes 20` 先粗测；③`--quick` 只跑 8 config 验证流程。
>
> **本地重跑**：
> ```bash
> cd /home/xukainan/RoboTwin
> PATH=$CONDA/bin:$PATH PYTHONPATH=. python /home/xukainan/expo-ft/scripts/bench_exec_backends.py \
>     --task stack_blocks_two --rollout demo_clean --task_config demo_clean --episodes 50
> ```

---

## 上一轮 (旧 sweep, 真实数据 — 待本地重跑覆盖)

| config | success_rate | dense_steps(成功,÷250=s) | dense_steps(全部) | n_success |
|---|---|---|---|---|
| fixed_time_v1.0 | 98% | 4368 | 4370 | 49/50 |
| fixed_time_v1.5 | 98% | 2891 | 2892 | 49/50 |
| fixed_time_v2.0 | 98% | 2188 | 2189 | 49/50 |
| fixed_time_v2.5 | 98% | 1754 | 1755 | 49/50 |
| fixed_time_v3.0 | 98% | 1488 | 1488 | 49/50 |
| fixed_time_v3.5 | 100% | 1310 | 1310 | 50/50 |
| fixed_time_v4.0 | 90% | 1141 | 1146 | 45/50 |
| per_action_v1.0_vel5_acc8 | 86% | 1750 | 1742 | 43/50 |
| per_action_v2.0_vel5_acc8 | 82% | 1480 | 1471 | 41/50 |
| per_action_v3.0_vel5_acc8 | 80% | 1338 | 1324 | 40/50 |
| per_action_v1.0_vel2_acc8 | 88% | 1856 | 1845 | 44/50 |
| per_action_v1.0_vel3.5_acc8 | 88% | 1735 | 1728 | 44/50 |
| per_action_v1.0_vel5_acc2 | 94% | 2975 | 2968 | 47/50 |
| per_action_v1.0_vel5_acc5 | 92% | 2075 | 2067 | 46/50 |
| whole_chunk_v1.0_vel5_acc8 | 92% | 1822 | 1817 | 46/50 |
| whole_chunk_v2.0_vel5_acc8 | 90% | 1825 | 1816 | 45/50 |
| whole_chunk_v3.0_vel5_acc8 | 86% | 1814 | 1800 | 43/50 |
| whole_chunk_v1.0_vel2_acc8 | 92% | 1962 | 1952 | 46/50 |
| whole_chunk_v1.0_vel3.5_acc8 | 92% | 1826 | 1820 | 46/50 |
| whole_chunk_v1.0_vel5_acc2 | 96% | 3622 | 3622 | 48/50 |
| whole_chunk_v1.0_vel5_acc5 | 94% | 2298 | 2294 | 47/50 |

> 旧数据要点（已分析）：① fixed_time 加 v 几乎免费提速、成功率到 v3.5 不掉（streaming 无 vel/acc 约束 + 理想化跟踪）；
> ② `acc_limit` 对时间影响大（acc8→acc2：1822→3622），`vel_limit` 几乎不 binding（短动作加速度受限，峰值到不了 5 rad/s）；
> ③ whole_chunk 的 `v` 是死旋钮（1822/1825/1814 不变）。新 sweep 会把 vel/acc 压到 1.0 验证下端拐点。
