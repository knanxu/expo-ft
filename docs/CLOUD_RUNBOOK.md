# 云端跑通 DBPO×RoboTwin 复现测试 — 分步 Runbook

> 目标：在云端（已有 openpi + 训好的 `stack_blocks_two` drift checkpoint + RoboTwin，但无 expo-ft）
> 从零把 DBPO on-policy RL（方案 A async）在 RoboTwin stack-two-blocks 上跑起来。
> 每步带**验证 gate**，过了再下一步。`<...>` 是你要填的云端路径。

架构回顾（两进程，分离①）：
```
[RoboTwin venv] client_robotwin.run_robotwin_client  ──websocket(8102)──▶  [expo-ft venv] train_pi_robo_dbpo_async
   RoboTwin sim, take_action 方式2                                          actor线程 rollout + learner线程 update
```
**两个独立 venv**（依赖隔离）：① expo-ft learner venv（JAX + 合并版 openpi drift）；② RoboTwin client venv（sapien/curobo/robotwin + openpi-client + websockets）。

---

## 涉及的 3 个 GitHub 仓库

| 仓库 | 内容 | 你的动作 |
|---|---|---|
| **expo-ft**（fork `pd-perry/expo-ft`）| DBPO 代码（新增件 + 改动）| fork → push 你的分支 |
| **openpi 合并版**（fork `pd-perry/openpi`，分支 `expo_ft_drift`）| pd-perry infra + drift 合并（内嵌在 expo-ft，**被 .gitignore 忽略，不随 expo-ft 走**）| fork → push `expo_ft_drift` |
| **RoboTwin**（`knanxu/RoboTwin`，已存在）| 你改的含三种执行方式的 RoboTwin | 确认最新改动已 push |

⚠️ learner **必须**用合并版 openpi（pd-perry `infer(is_batch,for_training)` + drift）。**不能**用你云端单独那个 openpi（缺 pd-perry infer 改动 → `pi05.py` 崩）。checkpoint 是**数据**（不入 git），留在云端磁盘，config 指过去即可。

---

## 步骤 0 — 本地：push 到 GitHub（3 仓库）

**0a. push 合并版 openpi**（在内嵌 openpi 目录，当前分支 `expo_ft_drift` = 合并提交 e500a21）：
```bash
cd /home/xukainan/expo-ft/expo_ft/agents/vla/openpi
# 先在 GitHub 网页 fork pd-perry/openpi → knanxu/openpi
git remote add myopenpi https://github.com/knanxu/openpi.git
git push myopenpi expo_ft_drift          # 含 drift 合并的全部提交
```
**0b. push expo-ft DBPO work**（主仓；内嵌 openpi 被 .gitignore 忽略，不会误传）：
```bash
cd /home/xukainan/expo-ft
# 先在 GitHub 网页 fork pd-perry/expo-ft → knanxu/expo-ft
git remote add myexpo https://github.com/knanxu/expo-ft.git
git checkout -b dbpo-robotwin
git add expo_ft/agents/alg/dbpo*.py expo_ft/data/rollout_buffer.py expo_ft/networks/dbpo_heads.py \
        expo_ft/agents/alg/__init__.py configs/model/dbpo_pi_config.py configs/task/robotwin_stack_blocks.py \
        client_robotwin/ train_pi_robo_dbpo_async.py CLAUDE.md docs/
git commit -m "DBPO×BPO on-policy RL on RoboTwin (drift pi0.5)"
git push myexpo dbpo-robotwin
```
**0c. 确认 RoboTwin 已 push**：`cd /home/xukainan/RoboTwin && git status`，把含三种执行方式的改动 commit+push 到 `knanxu/RoboTwin`。

---

## 步骤 1 — 云端：clone 三仓库并组装目录

```bash
cd <CLOUD>
git clone https://github.com/knanxu/expo-ft.git && cd expo-ft && git checkout dbpo-robotwin && cd ..
# 内嵌 openpi（被忽略，单独 clone 进固定路径，分支 expo_ft_drift）：
git clone -b expo_ft_drift https://github.com/knanxu/openpi.git expo-ft/expo_ft/agents/vla/openpi
git clone https://github.com/knanxu/RoboTwin.git          # 若云端已有可跳过
```
**验证**：
```bash
cd <CLOUD>/expo-ft
git -C expo_ft/agents/vla/openpi log --oneline -1        # 见 e500a21 Merge ... drift
ls client_robotwin/run_robotwin_client.py configs/task/robotwin_stack_blocks.py train_pi_robo_dbpo_async.py
```

---

## 步骤 2a — learner venv（JAX + 合并 openpi，uv）

```bash
cd <CLOUD>/expo-ft
uv sync                      # editable 装内嵌 openpi(drift) + openpi-client；py>=3.11
# GPU JAX：按云端 CUDA 版补（uv.lock 若是 cpu jax）：uv pip install -U "jax[cuda12]"
```
**验证 gate 2a**：
```bash
JAX_PLATFORMS=cpu uv run python -m expo_ft.agents.alg.dbpo_dryrun_test     # 7 passed
JAX_PLATFORMS=cpu uv run python -m expo_ft.agents.alg.dbpo_pi05_test       # 5 passed
uv run python -c "import openpi.models.pi0_config as c; print('drift?', c.Pi0Config().use_drifting_loss)"
uv run python -c "import openpi.training.config as C; print(C.get_config('pi05_aloha_robotwin_drifting_stack_blocks_two').name)"
```

---

## 步骤 2b — RoboTwin client venv（conda py3.10 + sapien/curobo + 补 client 依赖）

RoboTwin 2.0 标准安装（py3.10，独立于 learner 的 py3.11 venv）：
```bash
conda create -n RoboTwin python=3.10 -y && conda activate RoboTwin
pip install torch==2.4.1 torchvision      # 按云端 CUDA 选 wheel
cd <CLOUD>/RoboTwin
bash script/_install.sh                    # requirements.txt + pytorch3d + sapien/mplib 补丁 + curobo v0.7.8
# ⚠️ 下载 RoboTwin assets（必须，否则 env 起不来）：见 RoboTwin INSTALLATION.md / huggingface
#    （embodiment=aloha-agilex 的 urdf/mesh 等）
# 补 client 侧依赖（server 用）：
pip install "websockets~=13.1" tyro msgpack
pip install -e <CLOUD>/expo-ft/expo_ft/agents/vla/openpi/packages/openpi-client
```
**验证 gate 2b（RoboTwin + 适配器 import OK）**：
```bash
# RoboTwin 在 import 期就用**相对路径**读 ./assets/objects/objaverse/list.json（任何 import envs.* 都触发，
# 见排查表），故 gate 必须从 RoboTwin 根跑、expo-ft 放 PYTHONPATH。先确认该索引存在（不入 git，须由 assets 下载提供）：
ls <CLOUD>/RoboTwin/assets/objects/objaverse/list.json
cd <CLOUD>/RoboTwin
PYTHONPATH=<CLOUD>/expo-ft python -c "
import envs.stack_blocks_two as m; print('robotwin task OK:', hasattr(m,'stack_blocks_two'))
from client_robotwin.envs.robotwin_env import RoboTwinEnv; print('adapter import OK')
"
```
（server 端不用手动 cd：`run_robotwin_client.py` 启动时已 `os.chdir(robotwin_root)` 并把 expo-ft 根钉进 sys.path。）
（`envs.stack_blocks_two`：RoboTwin 里 stack-two-blocks 的任务模块，类名应=文件名；不一致就改 `configs/task/robotwin_stack_blocks.py` 的 `task_name`。）

---

## 步骤 3 — 填配置（checkpoint + 任务 + 场景）

编辑 `<CLOUD>/expo-ft/configs/model/dbpo_pi_config.py`：
- `config.pi05_weight_loader_path = "<你训好的 stack_blocks_two drift checkpoint 目录>"`（orbax params dir）。
- `config.pi05_assets_dir / pi05_asset_id`：若 norm_stats 在你 checkpoint 的 assets 里，指过去（否则用配置默认 trossen assets）。
- 其余已对齐 DBPO（pi05_config_name=pi05_aloha_robotwin_drifting_stack_blocks_two / normalize_dims / clip_eps=0.02 / logσ∈[log0.03,log0.10] / ent=0.01 / anchor / replan_steps=25 / n_real_dims=14 / 稳定器）。

编辑 `<CLOUD>/expo-ft/configs/task/robotwin_stack_blocks.py`：
- `config.task_name = "stack_blocks_two"`（与 RoboTwin envs/ 文件名一致）。
- `config.control_hz`：按你 RoboTwin 设置。
- `config.robotwin_task_config`：补 RoboTwin 场景配置（**必须含 rgb(head/left/right_camera)+qpos**）。建议直接读你的 RoboTwin yaml：
  `/home/xukainan/RoboTwin/task_config/{demo_clean.yml + _camera_config.yml + _embodiment_config.yml}` 合成的 dict
  （camera 名、render_freq、domain_randomization、embodiment=aloha-agilex）。

**验证 gate 3（checkpoint 能被合并 openpi 加载）**：
```bash
cd <CLOUD>/expo-ft   # learner venv
uv run python -c "
from configs.model.dbpo_pi_config import get_config as gm
from configs.task.robotwin_stack_blocks import get_config as gt
import jax, openpi.training.sharding as S
cfg, ct = gm(), gt()
from expo_ft.agents.vla.pi05 import build_pi05
mesh=S.make_mesh(1); ds=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(S.DATA_AXIS)); rs=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
actor, ts, *_ = build_pi05(cfg, 0, mesh, ds, rs, False, ct.language_instruction)
print('checkpoint loaded; action_horizon=', actor.model_config.action_horizon, 'action_dim=', actor.model_config.action_dim)
"
```
报 shape/key mismatch → checkpoint 与 config 的 pi0.5 结构不一致（核对 paligemma/action_expert variant、action_dim/horizon、是否 LoRA）。

---

## 步骤 4 — ⭐ 前置硬 gate：greedy baseline（确认 checkpoint 本身能做任务）

RL 在 BC/DBP 流形上 bootstrap，**确定性策略做不动则 RL 无正信号**。先用你 RoboTwin 现有的 eval 工具评这份 checkpoint（最快）：
```bash
conda activate RoboTwin; cd <CLOUD>/RoboTwin
# 用 RoboTwin 自带 eval（pi05 deploy），pi0_step=25（=我们的 H_e），确认 stack_blocks_two 成功率 >0
bash script/run_eval_policy.sh stack_blocks_two 0   # 按你的 eval 入口/参数调
```
**gate**：success_rate 明显 >0（如 ≥0.3）。为 0 → 先回去把 DBP checkpoint 训好，RL 暂不要上。

---

## 步骤 5 — 启 RoboTwin env-server

```bash
conda activate RoboTwin; cd <CLOUD>/expo-ft
python -m client_robotwin.run_robotwin_client \
    --config_task_path configs/task/robotwin_stack_blocks.py \
    --robotwin_root <CLOUD>/RoboTwin \
    --server_host 0.0.0.0 --server_port 8102
```
**验证 gate 5（协议连通）**：另开一个 learner venv 终端：
```bash
cd <CLOUD>/expo-ft
uv run python -c "
from expo_ft.env.env_client import EnvClientWrapper
import numpy as np
env=EnvClientWrapper({'example_action':np.zeros((1,50,14)),'env_usage':'train','video_dir':''},host='localhost',port=8102)
o=env.reset(); print('obs keys:', list(o.keys()))
ra,t=env.step(np.zeros(14).tolist()); print('exec shape:', np.asarray(ra).shape, 'type:', t)
print('info:', env.get_info_for_step())
"
```
应见 `observation.images.cam_high/...`、`observation.state`、executed_action 形状 (14,)、(done,success,reward,mask)。

---

## 步骤 6 — 启 learner（DBPO async 训练）

```bash
cd <CLOUD>/expo-ft   # learner venv（GPU）
uv run python train_pi_robo_dbpo_async.py \
    --config configs/model/dbpo_pi_config.py \
    --config_task configs/task/robotwin_stack_blocks.py \
    --client_host localhost --client_port 8102 \
    --dataset_path "<可空或一条 demo 用于 example_action 形状>" \
    --run_name dbpo_stack_blocks_smoke --max_iters 50 \
    --fsdp_devices 1
```
（`--dataset_path` 仅用于取 `example_action` 形状建 env；若没有可用 demo，可临时改脚本用 `np.zeros((1,50,14))` 占位。）

---

## 步骤 7 — 监控（首跑重点验这些）

wandb / 日志看：
- **`ppo/ratio_mean` 首次更新 ≈ 1**（z 复用正确性命门）；`ppo/approx_kl` 起步 ≈0。
- `rollout/success_rate` 是否 >0、是否随迭代上升。
- `logstd/mean` 在 [log0.03, log0.10] 带内演化；`loss/anchor` 有限且不爆；`loss/value` 下降。
- `training/kl_early_stop` 偶发正常（高维 ratio 触发早停）；`ppo/ratio_max` 不应频繁顶到极端。

---

## 常见断点排查
| 现象 | 多半原因 / 处理 |
|---|---|
| 步骤1 import openpi 无 `use_drifting_loss` | 装成了非合并版 openpi；确认 editable 指向 `expo_ft/agents/vla/openpi`@`expo_ft_drift` |
| 步骤3 checkpoint key/shape mismatch | config 的 pi0.5 结构与 checkpoint 不符（variant/action_dim/horizon/LoRA） |
| 步骤5 `flat_item['action']` KeyError | repack 需 `action`——SEAM-2 preprocess 已补 dummy；若仍报，确认走的是 `train_pi_robo_dbpo_async` 的 preprocess（非 RoboTwin deploy 路径） |
| obs transform 报相机/键错 | `robotwin_task_config` 未启用对应相机 / RoboTwin get_obs 相机名≠head/left/right_camera |
| `assets/objects/objaverse/list.json` FileNotFoundError（import envs 或起 server 时） | ① **cwd 不对**：RoboTwin import 期用相对路径读 assets——gate 2b 从 RoboTwin 根跑；server 端 `run_robotwin_client` 已 `os.chdir(robotwin_root)` 兜底。② **文件真缺**：该索引不入 git，须由 RoboTwin assets 下载提供（仅 import 需 list.json 这 22KB 索引；stack_blocks_two 运行期不加载 objaverse mesh） |
| ratio 首更新 ≠1（>1.01） | matmul 精度（脚本已设 highest）/ z 未正确复用 / logp_old 未在采集时存 |
| success 一直 0 | 回步骤4：DBP baseline 本就做不动；或 H_e/control_hz 与训练不一致致分布漂移 |

---

## 备注（非阻塞，按需）
- DBPO 采样侧 `±3σ 截断 / min_sampling_std / logprob_min`（config 旋钮已在）尚未接进 `sample_actions`，首跑可不接；若 ratio 尾部不稳再补。
- `--dataset_path` 的 example_action 仅形状用途；后续可去掉该依赖。
