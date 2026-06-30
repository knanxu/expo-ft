"""`expo_ft.env.robotwin_utils` 单元测试（纯 CPU：cv2/h5py/numpy，**不依赖** jax/torch/sapien）。

锁住 RoboTwin demo loader 的几个正确性契约——其中多条是 memory 记录的「rollout 0%」元凶：
  - action[t] = vector[t+1]（下一帧绝对 qpos 目标，错一帧会让离线 demo 与在线动作约定不符）；
  - 三相机 head/left/right → cam_high/cam_left_wrist/cam_right_wrist **不串**；
  - 每集随机模板指令来自 sibling `instructions/episode{N}.json`（用固定 task 串会 rollout 0%）。

构造**格式真实**的临时 RoboTwin episode hdf5（joint_action/vector + 三相机逐帧 JPEG 字节），
用真实 cv2.imencode/imdecode 往返，不 mock —— 这样测的是 loader 的真实行为。

⚠️ 本地显存不足、不跑 pytest；本测试由维护人在**云端 expo-ft venv**执行：
    python -m pytest test/robotwin_utils_test.py -v
"""

import json
import os
import types

import cv2
import h5py
import numpy as np
import pytest

from expo_ft.env.robotwin_utils import (
    _decode_rgb_chw,
    _discover_episode_files,
    _episode_instruction,
    process_robotwin_dataset,
)


# ---------------- fixtures / helpers ----------------
def _cfg(instruction_type="seen", language_instruction="fallback instruction"):
    """最小 task_config 替身（process_robotwin_dataset 只 getattr 这两个字段）。"""
    return types.SimpleNamespace(
        instruction_type=instruction_type,
        language_instruction=language_instruction,
    )


def _make_episode_hdf5(path, T, cam_values=(50, 150, 250), H=16, W=24):
    """写一个最小但格式真实的 RoboTwin episode hdf5。

    - joint_action/vector: (T, 14)，vector[t] 整行 = t（便于断言 state[t]=t / action[t]=t+1）。
    - observation/{head,left,right}_camera/rgb: (T,) 定长 S 字节，每帧一张 JPEG（灰度纯色，
      head=50/left=150/right=250，用不同亮度验证 _CAM_MAP 不串）。
    """
    vec = np.tile(np.arange(T, dtype=np.float32).reshape(T, 1), (1, 14))  # (T,14), row t == t
    with h5py.File(path, "w") as f:
        f.create_dataset("joint_action/vector", data=vec)
        for cam_name, val in zip(("head_camera", "left_camera", "right_camera"), cam_values):
            jpegs = []
            for _t in range(T):
                img = np.full((H, W, 3), val, dtype=np.uint8)
                ok, enc = cv2.imencode(".jpg", img)
                assert ok, "cv2.imencode failed building test fixture"
                jpegs.append(enc.tobytes())
            maxlen = max(len(j) for j in jpegs)
            arr = np.array(jpegs, dtype=f"S{maxlen}")  # numpy 右填 \0 到定长，复刻 RoboTwin 存储
            f.create_dataset(f"observation/{cam_name}/rgb", data=arr)


# ---------------- _discover_episode_files ----------------
def test_discover_sorts_numerically(tmp_path):
    # 数字序而非字典序：episode10 必须排在 episode2 之后。
    for name in ("episode2.hdf5", "episode10.hdf5", "episode0.hdf5"):
        (tmp_path / name).touch()
    got = [os.path.basename(p) for p in _discover_episode_files(str(tmp_path))]
    assert got == ["episode0.hdf5", "episode2.hdf5", "episode10.hdf5"]


def test_discover_ignores_non_episode_files(tmp_path):
    # 只挑 episode{纯数字}.hdf5："episode.hdf5"(空后缀非数字)/"episodeX.hdf5"/非 hdf5/无关名 全排除。
    for name in ("episode0.hdf5", "foo.hdf5", "episodeX.hdf5", "episode1.txt", "episode.hdf5"):
        (tmp_path / name).touch()
    got = [os.path.basename(p) for p in _discover_episode_files(str(tmp_path))]
    assert got == ["episode0.hdf5"]


def test_discover_finds_in_subdirectories(tmp_path):
    # os.walk 递归发现子目录下的 episode 文件。
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "episode0.hdf5").touch()
    (tmp_path / "b" / "episode1.hdf5").touch()
    got = [os.path.basename(p) for p in _discover_episode_files(str(tmp_path))]
    assert got == ["episode0.hdf5", "episode1.hdf5"]


# ---------------- _decode_rgb_chw ----------------
def test_decode_returns_chw_uint8_contiguous():
    img = np.full((16, 24, 3), 90, dtype=np.uint8)  # H=16 != W=24，证明轴序未颠倒
    enc = cv2.imencode(".jpg", img)[1].tobytes()
    out = _decode_rgb_chw(enc)
    assert out.shape == (3, 16, 24)  # CHW
    assert out.dtype == np.uint8
    assert out.flags["C_CONTIGUOUS"]


def test_decode_strips_trailing_null_padding():
    # hdf5 定长 S 会在短帧尾部填 \0；rstrip(b"\0") 后仍应能正常 decode。
    img = np.full((16, 24, 3), 123, dtype=np.uint8)
    enc = cv2.imencode(".jpg", img)[1].tobytes()
    out = _decode_rgb_chw(enc + b"\x00" * 32)
    assert out.shape == (3, 16, 24)


def test_decode_raises_on_corrupt_bytes():
    with pytest.raises(ValueError):
        _decode_rgb_chw(b"definitely not a jpeg\x00\x00")


# ---------------- process_robotwin_dataset ----------------
def test_process_raises_when_no_episodes(tmp_path):
    with pytest.raises(ValueError):
        process_robotwin_dataset(str(tmp_path), _cfg())


def test_process_frame_alignment_state_t_action_t_plus_1(tmp_path):
    # 核心契约：state[t]=vector[t]，action[t]=vector[t+1]（下一帧绝对 qpos），共 T-1 行。
    _make_episode_hdf5(str(tmp_path / "episode0.hdf5"), T=4)  # vector[t] 整行 == t
    data = process_robotwin_dataset(str(tmp_path), _cfg())
    assert len(data) == 3  # T-1，丢末帧
    for t in range(3):
        assert np.all(data[t]["observations"]["observation.state"] == float(t))
        assert np.all(data[t]["actions"] == float(t + 1))


def test_process_sparse_terminal_reward_done_mask(tmp_path):
    # 成功专家 demo → 稀疏终局：仅末个 transition reward=1/done=1；mask=1-done。
    _make_episode_hdf5(str(tmp_path / "episode0.hdf5"), T=4)
    data = process_robotwin_dataset(str(tmp_path), _cfg())
    assert [float(d["rewards"]) for d in data] == [0.0, 0.0, 1.0]
    assert [float(d["dones"]) for d in data] == [0.0, 0.0, 1.0]
    assert [float(d["masks"]) for d in data] == [1.0, 1.0, 0.0]
    for d in data:
        assert float(d["masks"]) == 1.0 - float(d["dones"])


def test_process_maps_three_cameras_without_crosswiring(tmp_path):
    # head→cam_high(50) / left→cam_left_wrist(150) / right→cam_right_wrist(250) 不串，且 CHW uint8。
    _make_episode_hdf5(str(tmp_path / "episode0.hdf5"), T=3, cam_values=(50, 150, 250))
    data = process_robotwin_dataset(str(tmp_path), _cfg())
    obs = data[0]["observations"]
    for key in (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ):
        assert obs[key].shape == (3, 16, 24)
        assert obs[key].dtype == np.uint8
    assert abs(float(obs["observation.images.cam_high"].mean()) - 50) < 25
    assert abs(float(obs["observation.images.cam_left_wrist"].mean()) - 150) < 25
    assert abs(float(obs["observation.images.cam_right_wrist"].mean()) - 250) < 25


def test_process_skips_episode_shorter_than_two_frames(tmp_path):
    # T<2 的 episode 无可用 transition（obs[t]→action=state[t+1]），应被跳过。
    _make_episode_hdf5(str(tmp_path / "episode0.hdf5"), T=1)  # 跳过
    _make_episode_hdf5(str(tmp_path / "episode1.hdf5"), T=3)  # 2 transitions
    data = process_robotwin_dataset(str(tmp_path), _cfg())
    assert len(data) == 2  # 只来自 episode1
    assert np.all(data[0]["observations"]["observation.state"] == 0.0)
    assert np.all(data[1]["actions"] == 2.0)


def test_process_uses_per_episode_instruction_as_prompt(tmp_path):
    # loader 用 sibling instructions/episode{N}.json 的模板指令作 prompt，而非固定 language_instruction。
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    instr_dir = tmp_path / "instructions"
    instr_dir.mkdir()
    _make_episode_hdf5(str(data_dir / "episode0.hdf5"), T=3)
    (instr_dir / "episode0.json").write_text(json.dumps({"seen": ["STACK_SEEN"]}), encoding="utf-8")
    data = process_robotwin_dataset(
        str(data_dir), _cfg(instruction_type="seen", language_instruction="FALLBACK")
    )
    assert data and all(d["observations"]["prompt"] == "STACK_SEEN" for d in data)


def test_process_num_data_limits_episode_count(tmp_path):
    for i in range(4):
        _make_episode_hdf5(str(tmp_path / f"episode{i}.hdf5"), T=3)  # 每个 2 transitions
    data = process_robotwin_dataset(str(tmp_path), _cfg(), num_data=2)
    assert len(data) == 4  # 2 episodes × 2


def test_process_episode_indices_selects_subset(tmp_path):
    for i in range(4):
        _make_episode_hdf5(str(tmp_path / f"episode{i}.hdf5"), T=3)
    data = process_robotwin_dataset(str(tmp_path), _cfg(), episode_indices=[0, 2])
    assert len(data) == 4  # 2 selected × 2


# ---------------- _episode_instruction ----------------
def test_episode_instruction_reads_sibling_json_by_type(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    instr_dir = tmp_path / "instructions"
    instr_dir.mkdir()
    ep = data_dir / "episode0.hdf5"
    ep.touch()
    (instr_dir / "episode0.json").write_text(
        json.dumps({"seen": ["SEEN_ONLY"], "unseen": ["UNSEEN_ONLY"]}), encoding="utf-8"
    )
    assert _episode_instruction(str(ep), "seen", "FB") == "SEEN_ONLY"
    assert _episode_instruction(str(ep), "unseen", "FB") == "UNSEEN_ONLY"


def test_episode_instruction_fallback_when_json_missing(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    ep = data_dir / "episode0.hdf5"
    ep.touch()  # 无 instructions/ 目录 → 回退 fallback
    assert _episode_instruction(str(ep), "seen", "FALLBACK") == "FALLBACK"


def test_episode_instruction_falls_back_to_seen_when_type_pool_empty(tmp_path):
    # pool = d.get(instruction_type) or d.get("seen") or []：unseen 池空 → 回退 seen 池。
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    instr_dir = tmp_path / "instructions"
    instr_dir.mkdir()
    ep = data_dir / "episode0.hdf5"
    ep.touch()
    (instr_dir / "episode0.json").write_text(
        json.dumps({"seen": ["SEEN_X"], "unseen": []}), encoding="utf-8"
    )
    assert _episode_instruction(str(ep), "unseen", "FB") == "SEEN_X"
