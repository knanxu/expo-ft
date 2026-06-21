"""RoboTwin offline-demo loader for EXPO-FT (EXPOLearner/BCLearner) online RL.

镜像 `expo_ft/env/droid_utils.py::process_droid_dataset` 的**返回契约**（per-transition
dict 列表 `{observations, actions, rewards, masks, dones}`），但读 RoboTwin 原始 demo hdf5
（`envs/utils/pkl2hdf5.py` 写出的格式），且 `observations` 用与
`client_robotwin/envs/robotwin_env.py::RoboTwinEnv.get_observation` **完全相同的扁平键** ——
这样在线 rollout obs 与离线 demo obs 流经**同一套** openpi `pi05_aloha_robotwin_*` transforms /
replay buffer，分布严格一致。

**零侵入**：新文件，**不改** `droid_utils.py`。learner 按 `config_task.dataset_loader` 分发到这里。

RoboTwin hdf5 结构（核对自真实 episode + `pkl2hdf5.py` / `policy/ACT/process_data.py`）：
  - `joint_action/vector`: (T, 14) float —— [左臂6, 左夹爪1, 右臂6, 右夹爪1]，**正是 aloha state/action 布局**
    （openpi `aloha_policy._decode_aloha` 的 [6,1,6,1]）。直接作每帧 state 与 action。
  - `observation/{head,left,right}_camera/rgb`: (T,) `S{len}` —— **逐帧 JPEG 编码字节**，需 `cv2.imdecode`。
    解码后 HWC uint8 → 转 **CHW**（AlohaInputs 期望 [C,H,W]，见 `aloha_policy.py:171` `c h w -> h w c`）。
    与 RoboTwinEnv `_chw` 一致：CHW 是两侧共同约定，若改 HWC 须两处同步。

奖励/终止：RoboTwin `data/` 下的 demo 均为成功专家轨迹 → 稀疏终局（末帧 reward=1, done=1），同 droid loader。
这些 demo 经 `PiReplayBuffer.insert_dataset` 标 is_hil/is_success=True，进入 EXPO 的 success-only actor 更新与 critic 暖启。
"""

import json
import os

import cv2
import h5py
import numpy as np
from tqdm import tqdm


def _discover_episode_files(datapath):
    """Return sorted RoboTwin `episode{N}.hdf5` paths under `datapath` (recursive)."""
    files = []
    for root, _dirs, fnames in os.walk(datapath):
        for fn in fnames:
            stem, ext = os.path.splitext(fn)
            if ext == ".hdf5" and stem.startswith("episode") and stem[len("episode"):].isdigit():
                files.append(os.path.join(root, fn))
    files.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0][len("episode"):]))
    return files


def _decode_rgb_chw(raw_bytes):
    """JPEG bytes (S{len}, 可能 \\0 尾填充) → CHW uint8（轴序对齐 RoboTwinEnv._chw / AlohaInputs）。"""
    buf = np.frombuffer(raw_bytes.rstrip(b"\0"), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # HWC；cv2 encode/decode 通道顺序自洽 → 还原原始 RGB
    if img is None:
        raise ValueError("cv2.imdecode failed on a RoboTwin rgb frame (corrupt JPEG bytes?)")
    return np.ascontiguousarray(np.transpose(img, (2, 0, 1)).astype(np.uint8))  # HWC -> CHW


# RoboTwin 相机名 → RoboTwinEnv/openpi 扁平键（head=主图，left/right=腕部）。
_CAM_MAP = {
    "head_camera": "observation.images.cam_high",
    "left_camera": "observation.images.cam_left_wrist",
    "right_camera": "observation.images.cam_right_wrist",
}


def _episode_instruction(ep_file, instruction_type, fallback):
    """读该 episode 的真实指令（RoboTwin 把每集指令存在 data/ 的 sibling
    `instructions/episode{N}.json`，含 {"seen":[...], "unseen":[...]}），随机选一条 instruction_type 模板。
    与在线 rollout（RoboTwinEnv._create_instruction）/ DBP 训练同分布——而非固定的 task language_instruction，
    否则离线 demo 的 prompt 与策略期望不符（详见 docs/DBPO_DEV.md 的 instruction 接缝）。缺失则回退 fallback。
    """
    stem = os.path.splitext(os.path.basename(ep_file))[0]                # episode{N}
    # ep_file = <config>/data/episode{N}.hdf5 → <config>/instructions/episode{N}.json
    jpath = os.path.join(os.path.dirname(os.path.dirname(ep_file)), "instructions", f"{stem}.json")
    try:
        with open(jpath, "r", encoding="utf-8") as jf:
            d = json.load(jf)
        pool = d.get(instruction_type) or d.get("seen") or []
        if len(pool) > 0:
            return str(np.random.choice(pool))
    except Exception:
        pass
    return fallback


def process_robotwin_dataset(datapath, task_config, episode_indices=None, num_data=None):
    """Load RoboTwin demos as EXPO-FT transitions (drop-in for process_droid_dataset).

    Args:
      datapath: dir containing RoboTwin `episode{N}.hdf5` (searched recursively).
      task_config: ml_collections config; reads `language_instruction` (prompt).
      episode_indices / num_data: subset selection, 同 process_droid_dataset。
    """
    ep_files = _discover_episode_files(datapath)
    if not ep_files:
        raise ValueError(
            f"No RoboTwin episode*.hdf5 found under '{datapath}'. "
            "Point --dataset_path at a RoboTwin demo dir (e.g. "
            "<RoboTwin>/data/<task>/<config>/data)."
        )
    if episode_indices is not None:
        ep_files = [ep_files[i] for i in episode_indices if 0 <= i < len(ep_files)]
    elif num_data is not None and num_data > 0:
        ep_files = ep_files[:num_data]

    prompt = getattr(task_config, "language_instruction", "")
    instruction_type = getattr(task_config, "instruction_type", "seen")
    print(f"Found {len(_discover_episode_files(datapath))} RoboTwin episodes; using {len(ep_files)}")

    data = []
    for ep in tqdm(ep_files):
        ep_prompt = _episode_instruction(ep, instruction_type, prompt)
        with h5py.File(ep, "r") as f:
            vector = np.asarray(f["joint_action"]["vector"], dtype=np.float32)  # (T, 14)
            T = len(vector)
            if T < 2:
                continue  # 至少要 1 个 transition（obs[t]→action=state[t+1]）

            # 动作时序对齐 DBP 训练管线（policy/pi05/scripts/process_data.py:103-124）：
            #   observation.state[t] = vector[t]，**action[t] = vector[t+1]**（下一帧绝对 qpos 目标，
            #   非 delta），共 T-1 行（丢末帧）。这与在线 rollout 一致：策略输出"下一目标 qpos"，
            #   RoboTwinEnv.step 经 take_action(qpos) 下发绝对目标。错一帧会让离线 demo 与在线动作约定不符。
            n = T - 1
            cams = {}
            for cam_name, flat_key in _CAM_MAP.items():
                rgb = f["observation"][cam_name]["rgb"]
                cams[flat_key] = [_decode_rgb_chw(rgb[t]) for t in range(n)]  # 帧 0..T-2

            # sparse terminal reward: 成功专家 demo → 末个 transition reward=1, done=1。
            ep_dones = np.zeros((n,), dtype=np.float32)
            ep_dones[-1] = 1.0
            ep_rewards = np.zeros((n,), dtype=np.float32)
            ep_rewards[-1] = 1.0

            for t in range(n):
                obs = {flat_key: cams[flat_key][t] for flat_key in _CAM_MAP.values()}
                obs["observation.state"] = vector[t]
                obs["prompt"] = ep_prompt
                data.append({
                    "observations": obs,
                    "actions": vector[t + 1],      # 下一帧绝对 qpos 目标（对齐 DBP action[t]=state[t+1]）
                    "rewards": ep_rewards[t],
                    "masks": 1.0 - ep_dones[t],
                    "dones": ep_dones[t],
                })

    return data
