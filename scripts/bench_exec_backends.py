"""三种执行后端完整对比测试（本地，真机 ARX5 参数）。

50 条正常速度专家 demo(stack_blocks_two) → 各后端回放，真机 PD(kp/kd) + per-joint
force_limit(τ_max)，**不 k_skip**(整 chunk)，记 success_rate + dense_steps(执行时间)。
每 config 第一个 episode 录 head_camera 视频。

测试组:
  fixed_time : v = 1.0,1.5,2.0,2.5,3.0,3.5,4.0 (chunk 压缩比扫描, streaming 无 vel/acc 约束)
  per_action / per_action_zero / whole_chunk : 1-D 受控 sweep, baseline=v1/vel5/acc8
    v   ∈ {1,2,3}        (vel5/acc8 固定)
    vel ∈ {1,2,3,4,5}    (v1/acc8 固定, 含下端 1.0)
    acc ∈ {1,2,3,4,6,8}  (v1/vel5 固定, 含下端 1.0)
  per_action_zero = per_action 执行核 + 段间边界速度强制为 0 (还原 RoboTwin 原版逐 action 两点零速
    stop-and-go); 与 per_action(方法B 非零边界) 唯一差异 = 边界速度 → 干净 A/B (见文末对照表),
    回答"零边界 TOPPRA 完成任务需多少物理步"。注: 走同一 retime_chunk 执行核(非 mplib take_action),
    步数口径诚实(250Hz 逐步采样)、与 method B 可比。

真机 ARX5(arx5-sdk config.h): kp[80,70,70,30,30,20] kd[2,2,2,1,1,0.7]
                              tau[30,40,30,15,10,10] vel[5,5,5.5,5.5,5,5]

Run (cwd=RoboTwin):
    cd /home/xukainan/RoboTwin
    PATH=$CONDA/bin:$PATH PYTHONPATH=. python /home/xukainan/expo-ft/scripts/bench_exec_backends.py \
        --task stack_blocks_two --rollout demo_clean --task_config demo_clean --episodes 50
"""
import argparse
import importlib
import os
import subprocess
import sys

import h5py
import numpy as np
import yaml

sys.path.insert(0, os.path.abspath("."))
from envs import CONFIGS_PATH

CHUNK = 50
VSF = 25                       # 视频每 25 物理步 1 帧 → 250/25=10fps
OUT_DIR = "/home/xukainan/expo-ft/scripts/bench_out"
REAL_KP = [80., 70., 70., 30., 30., 20.]
REAL_KD = [2., 2., 2., 1., 1., 0.7]
REAL_TAU = [30., 40., 30., 15., 10., 10.]

# fixed_time: v 扫描
FIXED_V = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# per_action / per_action_zero / whole_chunk: 三条 1-D 受控 sweep, 共享 baseline=v1/vel5/acc8.
# 相比旧 sweep: vel/acc 下探到 1.0(默认下端) + 加密粒度(旧 sweep 缺 1.0 且过粗, 看不出 vel_limit
# 何时真 binding). 每条只动一个变量, 其余固定 baseline.
BASE_V, BASE_VEL, BASE_ACC = 1.0, 5.0, 8.0
V_GRID = [1.0, 2.0, 3.0]                    # 压缩比      (vel5/acc8 固定)
VEL_GRID = [1.0, 2.0, 3.0, 4.0, 5.0]        # 绝对速度上限 (v1/acc8 固定, 含 1.0)
ACC_GRID = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]   # 绝对加速度上限(v1/vel5 固定, 含 1.0)


def build_sweep():
    """三条 1-D sweep 合并 + 去重(baseline 在三条里各出现一次, 只保留一行)。label 与旧版同格式。"""
    seen, out = set(), []

    def add(sp):
        key = (sp["v"], sp["vel_limit"], sp["acc_limit"])
        if key in seen:
            return
        seen.add(key)
        lab = f"v{sp['v']:.1f}_vel{sp['vel_limit']:g}_acc{sp['acc_limit']:g}"
        out.append((lab, sp))

    for v in V_GRID:
        add(dict(v=v, vel_limit=BASE_VEL, acc_limit=BASE_ACC))
    for vl in VEL_GRID:
        add(dict(v=BASE_V, vel_limit=vl, acc_limit=BASE_ACC))
    for al in ACC_GRID:
        add(dict(v=BASE_V, vel_limit=BASE_VEL, acc_limit=al))
    return out


SWEEP = build_sweep()


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


def apply_real_params(robot):
    """位置 PD 刚度用 sim 校准值(robot.joint_stiffness=1000/damping=200, ≈真机硬位置伺服),
    力矩约束用真机 ARX5 per-joint force_limit(τ_max) → 激进指令力矩饱和→失败。
    注: arx5-sdk kp=80/kd=2 是力矩控制层 PD, 非位置伺服刚度; 当 sapien stiffness 用太软
    (stack 放不准→success 0%, 已实测), 故位置刚度保留 sim 值, 只把真机力矩上限加为 force_limit."""
    for joints, st, dp in [(robot.left_arm_joints, robot.left_joint_stiffness, robot.left_joint_damping),
                           (robot.right_arm_joints, robot.right_joint_stiffness, robot.right_joint_damping)]:
        for i, j in enumerate(joints):
            j.set_drive_property(stiffness=float(st), damping=float(dp), force_limit=REAL_TAU[i])


def start_video(env, h, w, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
         "-video_size", f"{w}x{h}", "-framerate", f"{250.0/VSF}", "-i", "-",
         "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "23", out_path],
        stdin=subprocess.PIPE)
    env.eval_video_path = os.path.dirname(out_path)
    env._set_eval_video_ffmpeg(ff, fps=250.0 / VSF, phys_hz=250, video_save_freq=VSF)


def stop_video(env):
    try:
        if getattr(env, "eval_video_ffmpeg", None):
            env._del_eval_video_ffmpeg()
    except Exception:
        pass
    env.eval_video_path = None


def _zero_sd_bounds(q_current, qd_actual, target, vel_limit=None, acc_limit=None, **kw):
    """还原 RoboTwin 原版「段间零速 TOPPRA」: 强制 sd_start=sd_end=0, 仅保留退化判定 + 单位切向。

    用于 monkeypatch take_chunk_action_per_action 的 compute_segment_sd_bounds, 使 per_action
    执行核退化为逐 action 两点零速(stop-and-go), 与 method B(非零边界) 形成「唯一变量=边界速度」
    的干净 A/B: 同一 retime_chunk 执行核、同一绝对 vel_limit/acc_limit、同一开环前馈, 只差边界速度。
    签名须兼容原 compute_segment_sd_bounds(q_current, qd_actual, target, vel_limit=, acc_limit=)。
    """
    q_current = np.asarray(q_current, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    seg = target - q_current
    seg_len = float(np.linalg.norm(seg))
    if seg_len < 1e-6:                       # 退化段(current≈target): tangent=None, 与原版一致
        return 0.0, 0.0, None, seg_len
    return 0.0, 0.0, seg / seg_len, seg_len  # sd_start=sd_end=0


def run_backend(env, actions, backend, sp, vsf):
    # per_action_zero: 临时 monkeypatch 段间边界速度为 0(还原原版 stop-and-go), 走同一 per_action
    # 执行核; finally 还原, 不污染其它 config(env/进程跨 config 复用). 依赖 take_chunk_action_per_action
    # 每次调用都 `from .robot.toppra_chunk_executor import compute_segment_sd_bounds`(读模块属性) → 生效。
    _patched = None
    if backend == "per_action_zero":
        import envs.robot.toppra_chunk_executor as tce
        _patched = (tce, tce.compute_segment_sd_bounds)
        tce.compute_segment_sd_bounds = _zero_sd_bounds
        backend = "per_action"
    try:
        total = 0
        for i in range(0, len(actions), CHUNK):
            chunk = actions[i:i + CHUNK]
            if chunk.shape[0] < 2:
                break
            if backend == "streaming":
                info = env.take_chunk_action_streaming(chunk, v=sp["v"], hold_steps=15,
                                                       max_actions=None, video_save_freq=vsf)
            elif backend == "per_action":
                info = env.take_chunk_action_per_action(chunk, vel_limit=sp["vel_limit"],
                                                        acc_limit=sp["acc_limit"], v=sp["v"],
                                                        max_actions=None, video_save_freq=vsf)
            else:
                info = env.take_chunk_action(chunk, vel_limit=sp["vel_limit"], acc_limit=sp["acc_limit"],
                                             v=sp["v"], video_save_freq=vsf)
            total += int(info.get("dense_steps", 0) or 0)
            if info.get("status") == "truncated" or env.eval_success:
                break
        return {"dense_steps": total, "success": bool(env.eval_success)}
    finally:
        if _patched is not None:
            _mod, _orig = _patched
            _mod.compute_segment_sd_bounds = _orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="stack_blocks_two")
    ap.add_argument("--rollout", default="demo_clean")
    ap.add_argument("--task_config", default="demo_clean")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--quick", action="store_true", help="小测: 少 config 少 ep, 验证修复")
    args = ap.parse_args()

    data_dir = f"./data/{args.task}/{args.rollout}/data"
    with open(f"./data/{args.task}/{args.rollout}/seed.txt") as f:
        seeds = [int(s) for s in f.read().split()]
    files = sorted([x for x in os.listdir(data_dir) if x.endswith(".hdf5")],
                   key=lambda x: int(x.replace("episode", "").replace(".hdf5", "")))
    files, seeds = files[:args.episodes], seeds[:args.episodes]
    os.makedirs(OUT_DIR, exist_ok=True)

    env, env_args = build_env(args.task, args.task_config)
    hh, hw = int(env_args["head_camera_h"]), int(env_args["head_camera_w"])

    # 测试组: (backend, label, speed_params)
    if args.quick:
        _qb = dict(v=1.0, vel_limit=5.0, acc_limit=8.0)
        _qa1 = dict(v=1.0, vel_limit=5.0, acc_limit=1.0)   # acc 下端, 验证 acc_limit binding + 零边界
        configs = [("streaming", f"fixed_time_v{v}", dict(v=v, vel_limit=5.0, acc_limit=8.0)) for v in (1.0, 2.0, 4.0)]
        configs += [("per_action", "per_action_v1.0_vel5_acc8", _qb),
                    ("per_action_zero", "per_action_zero_v1.0_vel5_acc8", _qb),
                    ("per_action", "per_action_v1.0_vel5_acc1", _qa1),
                    ("per_action_zero", "per_action_zero_v1.0_vel5_acc1", _qa1),
                    ("whole_chunk", "whole_chunk_v1.0_vel5_acc8", _qb)]
    else:
        configs = [("streaming", f"fixed_time_v{v}", dict(v=v, vel_limit=5.0, acc_limit=8.0)) for v in FIXED_V]
        configs += [("per_action", f"per_action_{lab}", sp) for lab, sp in SWEEP]
        # 零边界 A/B(段间速度归零): 与上面 per_action 同 sweep 一一对应, 供文末对照表算 zero/B 步数比.
        # 若云端机时紧, 可注释下一行(A/B 表会自动跳过缺失项).
        configs += [("per_action_zero", f"per_action_zero_{lab}", sp) for lab, sp in SWEEP]
        configs += [("whole_chunk", f"whole_chunk_{lab}", sp) for lab, sp in SWEEP]

    rows = []
    for backend, label, sp in configs:
        recs = []
        for ep, (fname, seed) in enumerate(zip(files, seeds)):
            with h5py.File(os.path.join(data_dir, fname)) as f:
                actions = f["joint_action"]["vector"][...]
            vsf = -1
            try:
                env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **env_args)
                apply_real_params(env.robot)
                if ep == 0:   # 每 config 首个 episode 录视频
                    start_video(env, hh, hw, os.path.join(OUT_DIR, "videos", f"{label}.mp4"))
                    vsf = VSF
                r = run_backend(env, actions, backend, sp, vsf)
            except Exception as e:
                import traceback; traceback.print_exc(); r = {"dense_steps": -1, "success": False}
            finally:
                if ep == 0:
                    stop_video(env)
                try: env.close_env()
                except Exception: pass
            recs.append(r)
        ok = [r for r in recs if r["dense_steps"] >= 0]
        sr = sum(r["success"] for r in ok) / max(len(ok), 1)
        succ = [r["dense_steps"] for r in ok if r["success"]]
        ds_succ = sum(succ) / max(len(succ), 1)
        ds_all = sum(r["dense_steps"] for r in ok) / max(len(ok), 1)
        rows.append((label, sr, ds_succ, ds_all, len(succ)))
        print(f"[done] {label:<26} success={sr*100:.0f}%  ds_succ={ds_succ:.0f}  ds_all={ds_all:.0f}", flush=True)

    # ---- 主表 (所有 config) ----
    md = [f"# 三后端执行对比 (stack_blocks_two, {args.episodes}ep, 真机 PD+force_limit, 不 k_skip)\n",
          "| config | success_rate | dense_steps(成功,÷250=s) | dense_steps(全部) | n_success |",
          "|---|---|---|---|---|"]
    for label, sr, dss, dsa, ns in rows:
        md.append(f"| {label} | {sr*100:.0f}% | {dss:.0f} | {dsa:.0f} | {ns}/{args.episodes} |")

    # ---- A/B 对照表: per_action(方法B 非零边界) vs per_action_zero(段间零速) ----
    # 唯一变量=边界速度. zero/B = 零边界步数 / method B 步数; >1 表示零边界更慢(stop-and-go 代价).
    by_label = {r[0]: r for r in rows}
    ab = ["\n## per_action 边界速度 A/B (方法B 非零边界 vs 原版段间零速; 唯一变量=边界速度)\n",
          "> zero/B = 零边界 ÷ 方法B 的成功 dense_steps; **>1 = 零边界更慢**(stop-and-go 代价), <1 反之。\n",
          "| config | B_steps(成功) | zero_steps(成功) | zero/B | B_sr | zero_sr |",
          "|---|---|---|---|---|---|"]
    n_ab = 0
    for lab, _sp in SWEEP:
        rb, rz = by_label.get(f"per_action_{lab}"), by_label.get(f"per_action_zero_{lab}")
        if rb is None or rz is None:
            continue
        b_steps, z_steps = rb[2], rz[2]   # dense_steps(成功)
        ratio = f"{z_steps / b_steps:.2f}×" if b_steps > 0 else "—"
        ab.append(f"| {lab} | {b_steps:.0f} | {z_steps:.0f} | {ratio} | "
                  f"{rb[1]*100:.0f}% | {rz[1]*100:.0f}% |")
        n_ab += 1
    if n_ab > 0:
        md += ab

    out_md = os.path.join(OUT_DIR, "results.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    try:
        print("\n".join(md))
    except UnicodeEncodeError:
        print(f"(表格已写 {out_md})")
    print(f"\n表格: {out_md}\n视频: {OUT_DIR}/videos/")


if __name__ == "__main__":
    main()
