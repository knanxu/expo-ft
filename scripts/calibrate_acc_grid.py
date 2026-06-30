"""无负载 acc 上界标定（任务无关）：q̈_j,max = (τ_j,max − |g_j(q)|) / M_robot_jj(q)。

抓取负载不在此标定——由执行层 force_limit + RL 吸收（见 spec §3#3）。扫机器人工作空间
随机位形，统计 per-joint q̈_max 范围，给 acc grid 建议值（回填 exec_backends._DEFAULT_ACC_LIMIT
与 toppra_chunk_executor.PHYS_ACC_CEIL）。任务无关：只依赖 robot embodiment 与 τ_max，
机器人/τ_max 不变则换任务不重标。

API（本地已验证，RoboTwin venv）：
  - 质量矩阵：art.create_pinocchio_model().compute_generalized_mass_matrix(qpos_full)
  - 重力力矩：art.compute_passive_force(gravity=True, coriolis_and_centrifugal=False)

Run（cwd 必须 = RoboTwin 根，因 import envs）：
    cd /home/xukainan/RoboTwin   # 云端 /home/chenlu/RoboTwin
    python /home/xukainan/expo-ft/scripts/calibrate_acc_grid.py \
        --tau_max 40,40,20,20,10,10 --task stack_blocks_two --task_config demo_clean
"""
import argparse
import importlib
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath("."))   # cwd 必须 = RoboTwin 根，使 import envs 生效
from envs import CONFIGS_PATH


def build_task_env(task_name: str, task_config: str):
    """构造 RoboTwin 任务 env + setup_demo 一个 seed（参考 speedtune/tests/smoke_replay_expert）。"""
    mod = importlib.import_module(f"envs.{task_name}")
    env = getattr(mod, task_name)()

    with open(f"./task_config/{task_config}.yml", "r") as f:
        args = yaml.safe_load(f)
    args["task_name"] = task_name
    args["task_config"] = task_config

    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r") as f:
        emb = yaml.safe_load(f)
    et = args["embodiment"]
    if len(et) != 1:
        raise NotImplementedError("标定仅支持单 embodiment（aloha-agilex）")
    args["left_robot_file"] = emb[et[0]]["file_path"]
    args["right_robot_file"] = emb[et[0]]["file_path"]
    args["dual_arm_embodied"] = True

    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), "r") as f:
        cam = yaml.safe_load(f)
    head = args["camera"]["head_camera_type"]
    args["head_camera_h"] = cam[head]["h"]
    args["head_camera_w"] = cam[head]["w"]

    def _emb_cfg(fp):
        with open(os.path.join(fp, "config.yml"), "r") as f:
            return yaml.safe_load(f)

    args["left_embodiment_config"] = _emb_cfg(args["left_robot_file"])
    args["right_embodiment_config"] = _emb_cfg(args["right_robot_file"])
    args["eval_mode"] = True
    env.setup_demo(now_ep_num=0, seed=0, is_test=True, **args)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau_max", required=True,
                    help="per-joint 单臂力矩上限 N·m，逗号分隔，如 '40,40,20,20,10,10'")
    ap.add_argument("--task", default="stack_blocks_two")
    ap.add_argument("--task_config", default="demo_clean")
    ap.add_argument("--n_samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tau = np.array([float(x) for x in args.tau_max.split(",")], dtype=np.float64)
    env = build_task_env(args.task, args.task_config)
    robot = env.robot
    art = robot.left_entity                    # 单臂 sapien PhysxArticulation
    arm_joints = robot.left_arm_joints
    dof = len(arm_joints)
    assert tau.shape[0] == dof, f"--tau_max 长度 {tau.shape[0]} != 单臂 arm dof {dof}"

    pin = art.create_pinocchio_model()
    active = art.get_active_joints()
    arm_idx = [active.index(j) for j in arm_joints]
    qlim = np.asarray(art.get_qlimits())       # (n_active, 2)
    lo = np.where(np.isfinite(qlim[arm_idx, 0]), qlim[arm_idx, 0], -np.pi)
    hi = np.where(np.isfinite(qlim[arm_idx, 1]), qlim[arm_idx, 1], np.pi)

    rng = np.random.default_rng(args.seed)
    qdd = []
    for _ in range(args.n_samples):
        qfull = np.asarray(art.get_qpos(), dtype=np.float64).copy()
        for k, idx in enumerate(arm_idx):
            qfull[idx] = rng.uniform(lo[k], hi[k])
        art.set_qpos(qfull)
        M = np.asarray(pin.compute_generalized_mass_matrix(qfull))
        g = np.asarray(art.compute_passive_force(gravity=True, coriolis_and_centrifugal=False))
        Mjj = np.array([M[i, i] for i in arm_idx])
        gj = np.array([g[i] for i in arm_idx])
        qdd.append(np.maximum((tau - np.abs(gj)) / np.maximum(Mjj, 1e-9), 0.0))
    arr = np.stack(qdd)                          # (n, dof)

    print(f"\ntask={args.task} dof={dof} tau_max={tau.tolist()} n={args.n_samples}")
    print("per-joint q̈_max (rad/s^2):")
    print("  min   :", np.round(arr.min(0), 2))
    print("  p10   :", np.round(np.percentile(arr, 10, 0), 2))
    print("  median:", np.round(np.median(arr, 0), 2))
    print("  max   :", np.round(arr.max(0), 2))
    weakest = arr.min(1)                         # 每样本最弱关节的 q̈_max（标量 grid 受其约束）
    upper = float(np.percentile(weakest, 90))
    lower = float(np.percentile(weakest, 10))
    grid = tuple(float(x) for x in np.round(np.linspace(max(lower, 0.5), upper, 4), 2))
    print(f"\n建议 _DEFAULT_ACC_LIMIT = {grid}")
    print(f"建议 PHYS_ACC_CEIL >= {round(upper * 1.2, 2)}")

    try:
        env.close_env()
    except Exception:
        pass


if __name__ == "__main__":
    main()
