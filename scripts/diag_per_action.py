"""诊断 per_action 真机软 PD 下变慢：monkey-patch 记录每段 seg_len/sd_start/sd_end/T，
对比 sim PD(1000/200) vs 真机 ARX5 PD(kp/kd)。两者都挂同一 force_limit。

假设：真机软 PD 跟踪滞后 → 逐段实测 q_current 落后命令 → seg_len 变大 + sd_start 低
     → TOPPRA 每段规划更长(T 大) → dense_steps 累加暴增。

Run (cwd=RoboTwin):
    cd /home/xukainan/RoboTwin
    PYTHONPATH=. python /home/xukainan/expo-ft/scripts/diag_per_action.py
"""
import importlib
import os
import sys

import h5py
import numpy as np
import yaml

sys.path.insert(0, os.path.abspath("."))
from envs import CONFIGS_PATH
import envs.robot.toppra_chunk_executor as tce

REAL_KP = [80., 70., 70., 30., 30., 20.]
REAL_KD = [2., 2., 2., 1., 1., 0.7]
REAL_TAU = [30., 40., 30., 15., 10., 10.]
CHUNK = 50

LOG = []
_orig_sd = tce.compute_segment_sd_bounds
_orig_rt = tce.retime_chunk


def _sd(*a, **k):
    r = _orig_sd(*a, **k)
    q_current, target = np.asarray(a[0]), np.asarray(a[2])
    LOG.append({"seg_len": r[3], "sd_start": r[0], "sd_end": r[1], "degenerate": r[2] is None,
                "q_lag": float(np.linalg.norm(target - q_current)), "T": 0,
                "status": "degenerate" if r[2] is None else "?"})
    return r


def _rt(*a, **k):
    r = _orig_rt(*a, **k)
    if LOG:
        LOG[-1]["T"] = 0 if r["dense_arm_pos"] is None else int(r["dense_arm_pos"].shape[0])
        LOG[-1]["status"] = r["status"]
        LOG[-1]["reason"] = str(r.get("fallback_reason"))[:50]
    return r


tce.compute_segment_sd_bounds = _sd
tce.retime_chunk = _rt


def build_env(task_name, task_config):
    mod = importlib.import_module(f"envs.{task_name}")
    env = getattr(mod, task_name)()
    with open(f"./task_config/{task_config}.yml") as f:
        args = yaml.safe_load(f)
    args["task_name"] = task_name; args["task_config"] = task_config
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml")) as f:
        emb = yaml.safe_load(f)
    et = args["embodiment"]
    args["left_robot_file"] = emb[et[0]]["file_path"]
    args["right_robot_file"] = emb[et[0]]["file_path"]
    args["dual_arm_embodied"] = True
    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml")) as f:
        cam = yaml.safe_load(f)
    h = args["camera"]["head_camera_type"]
    args["head_camera_h"] = cam[h]["h"]; args["head_camera_w"] = cam[h]["w"]
    def _ec(fp):
        with open(os.path.join(fp, "config.yml")) as f:
            return yaml.safe_load(f)
    args["left_embodiment_config"] = _ec(args["left_robot_file"])
    args["right_embodiment_config"] = _ec(args["right_robot_file"])
    args["eval_mode"] = True
    return env, args


def apply_pd(robot, real):
    for joints in (robot.left_arm_joints, robot.right_arm_joints):
        for i, j in enumerate(joints):
            kp = REAL_KP[i] if real else robot.left_joint_stiffness
            kd = REAL_KD[i] if real else robot.left_joint_damping
            j.set_drive_property(stiffness=float(kp), damping=float(kd), force_limit=float(REAL_TAU[i]))


def run_per_action(env, actions):
    for i in range(0, len(actions), CHUNK):
        chunk = actions[i:i + CHUNK]
        if chunk.shape[0] < 2:
            break
        info = env.take_chunk_action_per_action(chunk, vel_limit=5.0, acc_limit=8.0, v=1.0)
        if info.get("status") == "truncated" or env.eval_success:
            break


def main():
    task, cfg = "shake_bottle", "smoke_test"
    env, env_args = build_env(task, cfg)
    with h5py.File(f"./data/{task}/{cfg}/data/episode0.hdf5") as f:
        actions = f["joint_action"]["vector"][...]
    seed = 2

    for real in (False, True):
        LOG.clear()
        env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **env_args)
        apply_pd(env.robot, real)
        run_per_action(env, actions)
        ok = [d for d in LOG if d["status"] == "success"]
        degen = [d for d in LOG if d.get("degenerate")]
        fb = [d for d in LOG if d["status"] not in ("success", "degenerate")]
        segs = np.array([d["seg_len"] for d in LOG]) if LOG else np.array([0.])
        sds = np.array([d["sd_start"] for d in LOG]) if LOG else np.array([0.])
        Ts = np.array([d["T"] for d in LOG]) if LOG else np.array([0])
        lags = np.array([d["q_lag"] for d in LOG]) if LOG else np.array([0.])
        tag = "真机 PD(kp80/kd2)" if real else "sim PD(1000/200)"
        print(f"\n===== {tag} =====")
        print(f"  段数={len(LOG)}  success={len(ok)} 退化(seg≈0跳过)={len(degen)} 真fallback={len(fb)}")
        print(f"  seg_len   均值={segs.mean():.4f}  max={segs.max():.4f}  (实测q_current→target 距离)")
        print(f"  q_lag     均值={lags.mean():.4f}  max={lags.max():.4f}  (同 seg_len, 越大=实测越落后命令)")
        print(f"  sd_start  均值={sds.mean():.3f}  (实测起速投影, 越小=PD 越跟不上)")
        print(f"  T/段      均值={Ts.mean():.1f}   总 dense_steps≈{int(Ts.sum())}")
        if fb:
            from collections import Counter
            print(f"  fallback reasons: {dict(Counter(d.get('reason', '?') for d in fb))}")
            print(f"  fallback 段 seg_len(前8): {[round(d['seg_len'], 3) for d in fb[:8]]}")
        try: env.close_env()
        except Exception: pass


if __name__ == "__main__":
    main()
