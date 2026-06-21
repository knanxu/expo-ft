# 云端跑通 EXPO-FT / DBPO × RoboTwin 复现测试 — 分步 Runbook

> 目标：在云端（已有 openpi + 训好的 `stack_blocks_two` drift checkpoint + RoboTwin，但无 expo-ft）
> 把 RoboTwin stack-two-blocks 上的在线 RL 跑起来。每步带**验证 gate**，过了再下一步。`<...>` 是你要填的云端路径。
>
> **两条 track（步骤 0–5 完全共用，仅步骤 3/6/7 分叉）：**
> - **track A = EXPO-FT 原算法（`EXPOLearner`，当前第一优先级，2026-06-21 起）** —— off-policy actor-critic，
>   走**现成** `train_pi_robo_async.py`，吃 **RoboTwin demo 数据集**（`process_robotwin_dataset`）。本 track 是
>   风险最低、最快闭环的路径，先跑它（见步骤 3-EXPO / 6-EXPO / 7-EXPO）。
> - **track B = DBPO on-policy RL（`DBPOLearner`）** —— 走独立 `train_pi_robo_dbpo_async.py`，无需离线数据集
>   （可用占位 example_action）。EXPO-FT 跑通后再回到它（见步骤 3-DBPO / 6-DBPO / 7-DBPO）。

架构回顾（两进程，分离①；两 track 共用同一个 RoboTwin env-server）：
```
[RoboTwin venv] client_robotwin.run_robotwin_client ─websocket(8102)─▶ [expo-ft venv] train_pi_robo_async        (track A, EXPO-FT)
   RoboTwin sim, take_action 方式2                                    └────────────────  train_pi_robo_dbpo_async (track B, DBPO)
                                                                       actor线程 rollout + learner线程 update
```
**两个独立 venv**（依赖隔离）：① expo-ft learner venv（JAX + 合并版 openpi drift）；② RoboTwin client venv（sapien/curobo/robotwin + openpi-client + websockets）。
**⚠️ learner 入口（两 track 都用 async 双线程）要求 ≥2 GPU**：device[0] 采样、device[1:] 更新（`train_pi_robo_async.py` 启动即断言 `num_gpus≥2`）。单卡跑不起来。

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
# track A（EXPO-FT）新增/改动件（与 DBPO 同分支，单独一条 commit 即可）：
git add configs/model/expo_ft_pi_drift_config.py expo_ft/env/robotwin_utils.py \
        train_pi_robo_async.py expo_ft/data/replay_buffer.py expo_ft/agents/vla/pi05.py \
        configs/task/robotwin_stack_blocks.py
git commit -m "EXPO-FT on drift pi0.5 + RoboTwin (Phase 4-EXPO)"
git push myexpo dbpo-robotwin
```
> **云端已搭好的情况（本次场景）**：不用重跑 0/1，只需在云端 `cd <CLOUD>/expo-ft && git pull` 拉到上面两条
> commit（含 EXPO-FT 的新 config / loader / dispatch + `replay_buffer.py`/`pi05.py` 的 `action` 键修复），
> 内嵌 openpi 不受影响（被 .gitignore 忽略，无 EXPO 改动）。
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
JAX_PLATFORMS=cpu uv run python -m expo_ft.agents.alg.dbpo_dryrun_test     # 7 passed（DBPO 回归）
JAX_PLATFORMS=cpu uv run python -m expo_ft.agents.alg.dbpo_pi05_test       # 5 passed（DBPO 回归；共享 pi05.py/replay_buffer.py，验 action 修复无回归）
uv run python -c "import openpi.models.pi0_config as c; print('drift?', c.Pi0Config().use_drifting_loss)"
uv run python -c "import openpi.training.config as C; print(C.get_config('pi05_aloha_robotwin_drifting_stack_blocks_two').name)"
# track A（EXPO-FT）专属：model config + robotwin loader import（learner 侧无 RoboTwin 依赖，loader 仅需 cv2/h5py）
uv run python -c "from configs.model.expo_ft_pi_drift_config import get_config as g; c=g(); print('EXPO cfg:', c.model_cls, c.pi05_config_name)"
uv run python -c "from expo_ft.env.robotwin_utils import process_robotwin_dataset; print('robotwin loader import OK')"
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
# 补 client 侧依赖（server 用）：ml_collections 是 configs/task/*.py 共用 config 必需（server load_task_config 要 import）。
pip install "websockets~=13.1" tyro msgpack ml_collections
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

### 3-EXPO（track A，当前优先）— 编辑 `<CLOUD>/expo-ft/configs/model/expo_ft_pi_drift_config.py`
🔴 **两项必填**，否则 learner 起不来：
- `config.pi05_weight_loader_path = "<你训好的 stack_blocks_two drift checkpoint 目录>"`（orbax params dir）。
  留空则 actor 从 base flow 权重初始化、drift 单步生成无效（greedy baseline 必然 0）。
- `config.pi05_assets_dir` + `config.pi05_asset_id`：指向你 **DBP 训练算出的 norm_stats**（`assets_dir/asset_id/norm_stats.json`），
  即 **eval 用的同一份**——RoboTwin eval `create_trained_policy` 从 `<ckpt>/assets/<asset_id>` 加载（`asset_id`=`<ckpt>/assets/` 下的子目录名，如 `trossen`）。
  ⚠️ **不能留空**：留空**不会报错**，而是**静默**落到 openpi 默认远程 `gs://.../pi05_base/assets/trossen`（通用 base-aloha 统计，**非**你 RoboTwin 训练统计）→ state 归一化 + action 反归一化全错 → rollout 0%（这是个隐蔽坑）。
  填法：`pi05_assets_dir="<ckpt>/assets"`、`pi05_asset_id="<子目录名>"`（与步骤4 eval 的 `<ckpt>/assets/` 同源）。
- 其余已对齐 EXPO（`model_cls=EXPOLearner` / `pi05_config_name=pi05_aloha_robotwin_drifting_stack_blocks_two` /
  `residual_action_xyzg=False` / `freeze_pi05_encoder=True` / `actor_success_only=True` / `num_qs=10` / `N=8`）。
  norm_stats 通常就在 `pi05_weight_loader_path` 同级的 `assets/` 里——优先指那里，与 greedy baseline 同源。

### 3-DBPO（track B）— 编辑 `<CLOUD>/expo-ft/configs/model/dbpo_pi_config.py`
- `config.pi05_weight_loader_path = "<你训好的 stack_blocks_two drift checkpoint 目录>"`（orbax params dir）。
- `config.pi05_assets_dir / pi05_asset_id`：若 norm_stats 在你 checkpoint 的 assets 里，指过去（否则用配置默认 trossen assets）。
- 其余已对齐 DBPO（pi05_config_name=pi05_aloha_robotwin_drifting_stack_blocks_two / normalize_dims / clip_eps=0.02 / logσ∈[log0.03,log0.10] / ent=0.01 / anchor / replan_steps=25 / n_real_dims=14 / 稳定器）。

编辑 `<CLOUD>/expo-ft/configs/task/robotwin_stack_blocks.py`：
- `config.task_name = "stack_blocks_two"`（与 RoboTwin envs/ 文件名一致）。
- `config.control_hz`：按你 RoboTwin 设置。
- `config.robotwin_task_config`：**新 config 加载模式**——只填要加载的 task_config yaml **名** + 少量覆盖项，
  **不再手写大段嵌套 dict**。camera / embodiment / data_type / domain_randomization 由适配器
  `client_robotwin/envs/robotwin_env.py::_resolve_setup_kwargs` **按 RoboTwin `script/eval_policy.py` 的方式从该 yaml
  二次解析**（加载 `task_config/{名}.yml` + `_camera_config.yml` + `_embodiment_config.yml`，并填 camera h/w、
  left/right embodiment config、`eval_mode` 等），与 gate 4 的 eval client **完全同源**。默认即：
  ```python
  config.robotwin_task_config = ml_collections.ConfigDict({
      "task_config": "demo_clean",   # 加载 <robotwin_root>/task_config/demo_clean.yml
      "eval_mode": True,             # 对齐 RoboTwin eval（unseen 纹理），与步骤4 baseline 一致；换 seen 设 False
      "render_freq": 0,              # headless 无屏渲染
  })
  ```
  （demo_clean = 干净桌面、无杂物/无背景随机，data_type 含 rgb(head+wrist)+qpos，适合 RL bootstrap；
  用户覆盖项盖在 yaml 之上。相对路径假定 cwd=RoboTwin 根，`run_robotwin_client` 启动时已 `os.chdir(robotwin_root)`。）

**验证 gate 3（checkpoint 能被合并 openpi 加载）**：
```bash
cd <CLOUD>/expo-ft   # learner venv
uv run python -c "
from configs.model.expo_ft_pi_drift_config import get_config as gm   # track A（EXPO）；track B 换成 configs.model.dbpo_pi_config
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

RL 在 BC/DBP 流形上 bootstrap，**确定性策略做不动则 RL 无正信号**。用 RoboTwin 自带的 **server-client eval**
评这份 checkpoint：**server 端**（`script/policy_model_server.py`，进程内加载 pi0.5 模型，裸 socket 暴露）+ **client 端**
（`script/eval_policy_client.py`，在 RoboTwin venv 跑 sim，`ModelClient.call` 远程取动作），**两个终端**。
**不要**用 `eval_policy.py`——它单进程、在 RoboTwin venv 里直接加载 jax 模型（无 server-client 分离）。

> ⚠️ **前置（pi05 需 client 兼容 eval）**：`eval_policy_client.py` 要求 policy 的 `eval()` **只用 `model.call(...)`**
> 访问模型（参考 `policy/DP/deploy_policy_double_env.py`：`model.call(func_name='get_action', obs=obs)`）。
> `policy/pi05/deploy_policy.py` 的 `eval()` 直接访问 `model.observation_window/set_language/get_action` —— **非 client 兼容**，
> 直接跑 client 会 `AttributeError`。若你还没补，先在 `policy/pi05/` 下新增 `model.call`-based 的 eval（仿 DP 的 double_env）。
>
> ⚠️ **checkpoint 路径约定**：RoboTwin `pi05` 从 `policy/pi05/checkpoints/{train_config_name}/{model_name}/{checkpoint_id}/`
> 加载，并用 **vendored openpi** 的 `get_config(train_config_name)`。drift checkpoint 含 drift-head 多余参数 + drift config
> 名（`pi05_aloha_robotwin_drifting_stack_blocks_two`）须该 openpi 能解析；若报 key/shape/config mismatch，说明 server 端
> 要用**合并版 openpi(drift)** 加载（或把 server 指向 expo-ft 的 drift openpi）。greedy baseline 只用 **mean action**，
> 不需要 drift 采样头。把你训好的 checkpoint 软链/拷到上面那个路径。

**终端 A（server，能加载 pi0.5 的 env）**：
```bash
conda activate <能加载 pi0.5 的 env>; cd <CLOUD>/RoboTwin
XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=0 \
python script/policy_model_server.py \
    --port 9000 \
    --config policy/pi05/deploy_policy.yml \
    --overrides \
    --task_name stack_blocks_two \
    --task_config demo_clean \
    --train_config_name pi05_aloha_robotwin_drifting_stack_blocks_two \
    --model_name   <你的训练 run/exp 名> \
    --ckpt_setting <你的训练 run/exp 名> \
    --checkpoint_id <如 30000> \
    --policy_name pi05 \
    --seed 0
```

**终端 B（client，RoboTwin sapien venv）**：
```bash
conda activate RoboTwin; cd <CLOUD>/RoboTwin
python script/eval_policy_client.py \
    --port 9000 \
    --config policy/pi05/deploy_policy.yml \
    --exec_backend whole_chunk \
    --overrides \
    --task_name stack_blocks_two \
    --task_config demo_clean \
    --train_config_name pi05_aloha_robotwin_drifting_stack_blocks_two \
    --model_name   <你的训练 run/exp 名> \
    --ckpt_setting <你的训练 run/exp 名> \
    --checkpoint_id <如 30000> \
    --seed 0 \
    --policy_name pi05
```
（`--exec_backend whole_chunk` = 整段 TOPPRA，对齐我们 DBPO 的 H_e=50 整段执行；想逐 action 用 `per_action`。
`pi0_step` 等在 `policy/pi05/deploy_policy.yml` 调。成功率写到
`eval_result/stack_blocks_two/pi05/demo_clean/<时间戳>/_result.txt`，eval 视频在同目录。）

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

## 步骤 6-EXPO（track A，当前优先）— 启 learner（EXPO-FT async 训练）

⚠️ **必须 ≥2 GPU**（`train_pi_robo_async.py:82` 启动即断言）：device[0] 采样、device[1:] 更新。
`--fsdp_devices` 必须整除「更新设备数 = 总 GPU 数 − 1」（2 卡→`--fsdp_devices 1`；4 卡→1 或 3）。

```bash
conda deactivate 2>/dev/null; cd <CLOUD>/expo-ft   # learner venv（GPU）
CUDA_VISIBLE_DEVICES=0,1,2,3 \
uv run python train_pi_robo_async.py \
    --config configs/model/expo_ft_pi_drift_config.py \
    --config_task configs/task/robotwin_stack_blocks.py \
    --client_host localhost --client_port 8102 \
    --dataset_path "<RoboTwin demo 目录，含 episode*.hdf5，如 <CLOUD>/RoboTwin/data/stack_blocks_two/demo_clean/data>" \
    --run_name expo_stack_blocks_smoke \
    --replan_steps 8 \
    --num_data 20 \
    --batch_size 64 --utd_ratio 20 \
    --max_steps 2000 \
    --overwrite \
    --fsdp_devices 1   # 4 卡 RTX 5880：device0 采样 + device1-3 更新（3 路数据并行）；显存紧可改 3（模型分片）
```
关键 flag（与 DBPO 不同点）：
- **`--dataset_path` 必填且必须是真实 RoboTwin demo 目录**（递归找 `episode{N}.hdf5`）。`process_robotwin_dataset`
  读 `joint_action/vector`(T,14) + `observation/{head,left,right}_camera/rgb`（逐帧 JPEG）。EXPO 用这些 demo 暖启
  critic + success-only actor 更新（`offline_ratio=0` 默认 → demo 灌进**在线** buffer 并标 is_success/is_hil）。
  ⚠️ demo 必须**三相机 rgb 齐全**（head+left+right），否则 loader `_CAM_MAP` 取相机时 KeyError；用 `collect_data.py`
  采的原始 demo（**不是** LeRobot 转换后的目录）。`--num_data N` 限制载入 episode 数（首跑取 ~20 即可，0=全部）。
- **`--max_steps`**（不是 DBPO 的 `--max_iters`）：env-step 总数（per-action 计）。冒烟 2000 足够看到首批 update。
- **`--replan_steps 8`**：EXPO 执行前 8 个 action 再 replan（drift 仍预测整段 H=50）。Q/residual 维度 = 8×14=112（可控）。
- `--offline_ratio 0.0`（默认）：demo 进在线 buffer；`--checkpoint_model` + `--checkpoint_interval N` 按需存盘。
- **`--overwrite`（重跑/首跑都建议加）**：脚本在 checkpoint 守卫前已 `mkdir logs/<run_name>/checkpoints`，
  故该目录一存在（含首跑）就 `raise FileExistsError`。`--overwrite` 清空重来；续训用 `--resume`。

**验证 gate 6-EXPO（数据管线 + 首批 update 不崩）**：日志依次应见
1. `Found <N> RoboTwin episodes; using <k>`（loader 找到 demo）；**不报** `flat_item['action']` KeyError
   （该 repack 接缝已在 `replay_buffer.py`/`pi05.py` 修；若仍报，确认云端已 `git pull` 到 action 修复那条 commit）。
2. `Created training environment ...` + 首次 `sample_actions` 编译完成（pi0.5 编译慢，首步可能数分钟）。
3. `Replay buffer ready (...), starting update thread.` → wandb `training/num_updates` 持续增长。
4. **无 NaN**：`training/*` loss 有限；`rollout/success_rate` 出现（>0 最好，0 也先看是否 episode 正常 done）。
   （`rollout/success_rate` 恒 0 → 回步骤 4 确认 greedy baseline 本身 >0，否则 RL 无正信号。）

> **诊断 `--actor_only_base_actions`**：EXPO rollout 默认不是 greedy——它从 N 个 base 候选 + residual 编辑候选里用
> critic `argmax` 选一个，训练初期 critic 随机 → 选择差 → 早期成功率低/0 属正常冷启动。加 `--actor_only_base_actions`
> 让 rollout **只用 base drift 动作**（不编辑、不 Q 选择），用来判别：base-only 能复现 ~50% → 配置正确、是冷启动（继续训会好）；
> base-only 仍 0% → 是权重/norm_stats/obs/动作问题，回步骤 3-EXPO 查。默认 False，不影响正常训练。

---

## 步骤 6-DBPO（track B）— 启 learner（DBPO async 训练）

```bash
cd <CLOUD>/expo-ft   # learner venv（GPU，同样 ≥2 GPU）
CUDA_VISIBLE_DEVICES=0,1 \
uv run python train_pi_robo_dbpo_async.py \
    --config configs/model/dbpo_pi_config.py \
    --config_task configs/task/robotwin_stack_blocks.py \
    --client_host localhost --client_port 8102 \
    --dataset_path "<可空或一条 demo 用于 example_action 形状>" \
    --run_name dbpo_stack_blocks_smoke --max_iters 50 \
    --overwrite \
    --fsdp_devices 1
```
（DBPO 的 `--dataset_path` 仅用于取 `example_action` 形状建 env；若没有可用 demo，可临时改脚本用 `np.zeros((1,50,14))` 占位。）

---

## 步骤 7-EXPO（track A）— 监控（首跑重点验这些）

EXPO 是 off-policy actor-critic（SAC 风格 + residual editing），`update_info` 键在 wandb **`training/`** 前缀下：
- **不崩 + 不 NaN 优先**：`training/critic_loss` 有限且总体下降；`training/residual_q`（edit 策略的 Q）有限、不爆正/负无穷。
- `training/residual_actor_loss`、`training/entropy`、`training/temperature` 演化平稳（temperature 不塌到 0 或炸大）。
- `training/num_updates` 持续增长；`training/update_paused` 长期为 0（>0 说明长时间无 episode 完成 → 看 env-server）。
- **`rollout/success_rate`** 是否 >0、是否随训练上升（核心目标）。恒 0 → 回步骤 4 确认 greedy baseline >0。
- 首批 update 前 buffer 要先暖（≥`batch_size` 条且有 success 标注的 demo）；`actor_success_only=True` 下若 success 池空，
  actor 批暂为 None（正常，待 rollout 出 success 或 demo 已标 is_success）。

## 步骤 7-DBPO（track B）— 监控（首跑重点验这些）

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
| 步骤6 `FileExistsError: Checkpoint directory ... already exists` | 脚本守卫前已 mkdir `logs/<run_name>/checkpoints`，故首跑/重跑都会撞。加 `--overwrite`（清空重来）或 `--resume`（续训） |
| 步骤6 `flat_item['action']` KeyError | aloha/robotwin repack 读单数 `action`，而 EXPO 数据管线给的是复数 `actions`。**已修**：`replay_buffer.py::insert`（离线/在线 transition）与 `pi05.py::process_raw_inputs`（在线采样）各补一个 `action` 别名（DROID repack 读 `actions`、忽略此键，行为不变）。若仍报 → 云端未 `git pull` 到该 commit |
| 步骤6-EXPO `At least 2 GPUs required` | async 双线程要 ≥2 GPU；`CUDA_VISIBLE_DEVICES` 至少暴露 2 张，且 `--fsdp_devices` 整除「GPU 数−1」 |
| 步骤6-EXPO loader `No RoboTwin episode*.hdf5 found` | `--dataset_path` 没指到含 `episode{N}.hdf5` 的目录（指 `collect_data.py` 原始 demo 目录，非 LeRobot 转换目录） |
| 步骤6-EXPO loader KeyError `left_camera`/`right_camera` | demo 采集时未存三相机 rgb；`process_robotwin_dataset._CAM_MAP` 要 head+left+right。换三相机齐全的 demo（与 demo_clean 同 data_type） |
| 步骤6-EXPO `Normalization stats not found ... raise ValueError` | EXPO replay buffer 强制要 norm_stats；`expo_ft_pi_drift_config.py` 的 `pi05_assets_dir/pi05_asset_id` 必填，指 DBP norm_stats（见步骤 3-EXPO） |
| 步骤6-EXPO rollout 成功率 0（但步骤4 greedy baseline 正常） | 三个独立元凶，逐一排查（同 ckpt eval 50-60% 却 rollout 0）：**①norm_stats 用错**——`expo_ft_pi_drift_config.py` 的 `pi05_assets_dir/pi05_asset_id` 留空会用远程 base `pi05_base/assets/trossen`（通用统计），必须指 checkpoint 自带的那份（= eval `create_trained_policy` 用的 `<ckpt>/assets/<asset_id>`，见步骤 3-EXPO）。**②delta 动作未加当前位姿**——aloha config `use_delta_joint_actions=True`，策略输出 delta，须 `+当前 state` 才是绝对 qpos；旧 `process_transformed_outputs` 用 dummy zeros → 跳到平均位姿。**已修**（pi05.py 传归一化 state），云端需 `git pull`。**③language instruction 不符**——已按 RLinf 修；确认 `config_task.instruction_type` 与 gate-4 `--instruction_type` 一致，env 启动应见 `cached episode_info for instructions: {...}`。自检：dump `raw_actions[0]` 应在当前 qpos 附近的小幅运动，而非平均位姿 |
| obs transform 报相机/键错 | `robotwin_task_config` 未启用对应相机 / RoboTwin get_obs 相机名≠head/left/right_camera |
| `assets/objects/objaverse/list.json` FileNotFoundError（import envs 或起 server 时） | ① **cwd 不对**：RoboTwin import 期用相对路径读 assets——gate 2b 从 RoboTwin 根跑；server 端 `run_robotwin_client` 已 `os.chdir(robotwin_root)` 兜底。② **文件真缺**：该索引不入 git，须由 RoboTwin assets 下载提供（仅 import 需 list.json 这 22KB 索引；stack_blocks_two 运行期不加载 objaverse mesh） |
| ratio 首更新 ≠1（>1.01） | matmul 精度（脚本已设 highest）/ z 未正确复用 / logp_old 未在采集时存 |
| success 一直 0 | 回步骤4：DBP baseline 本就做不动；或 H_e/control_hz 与训练不一致致分布漂移 |

---

## 备注（非阻塞，按需）
- DBPO 采样侧 `±3σ 截断 / min_sampling_std / logprob_min`（config 旋钮已在）尚未接进 `sample_actions`，首跑可不接；若 ratio 尾部不稳再补。
- `--dataset_path` 的 example_action 仅形状用途；后续可去掉该依赖。
