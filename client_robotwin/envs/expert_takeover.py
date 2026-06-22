"""RoboTwin 自动专家介入：触发判定 + 在线专家录播 + 回放状态机。

替代 DROID 人类在环（spacemouse），用 RoboTwin 自带 expert(``play_once``) 在 agent rollout
失败/卡住时从**当前状态**接管，产生覆盖 agent 真实失败状态的 on-policy 纠正样本，打破
EXPO rollout 0% 死锁。算法无关（EXPO/BC 通用），零侵入：

  - 不 import RoboTwin / robotwin_env；``record_expert_tape`` 通过参数接收 env 与 obs 扁平化回调。
  - 不改 RoboTwin 本体：录播期只在运行时 monkey-patch ``env._take_picture``（跑完恢复）。
  - learner 端零改动：RoboTwinEnv 接管期 step 返回 ``action_type="human"``，复用既有 is_hil 通路。

三个可独立测试的单元：
  - ``should_takeover``     纯判定（何时该接管，替代人眼）。
  - ``TapeReplayer``        纯回放状态机（逐帧吐 expert 动作 + 终局信号）。
  - ``record_expert_tape``  录播（跑 play_once 收 (obs, qpos) 带；可 mock env 测）。
"""

import logging

import numpy as np


def should_takeover(*, steps_since_reset, step_budget, success_once, step_frac, enabled):
    """agent 用到 ``step_frac·step_budget`` 预算仍未 success → 该让 expert 接管。

    替代 DROID 的人眼判断（何时按 spacemouse）。纯函数、无副作用，便于单测。

    Args:
      steps_since_reset: 本 episode 已执行的 env-step（take_action）数。
      step_budget:       本 episode 的 step 预算（RoboTwin step_lim）。
      success_once:      本 episode 是否已成功过（成功就不必接管）。
      step_frac:         触发阈值比例 ∈(0,1]，用到该比例预算仍未成功即接管。
      enabled:           总开关（False 时退化为纯 offline-demo 行为，对照实验用）。

    Returns:
      是否应触发 expert 接管。
    """
    if not enabled or success_once or step_budget <= 0:
        return False
    return steps_since_reset >= step_frac * step_budget


class TapeReplayer:
    """逐帧回放专家录播带。纯状态机，无 RoboTwin 依赖，便于单测。

    ``frames`` = list of ``(obs_flat: dict, action_qpos: np.ndarray)``。一个 learner step 的
    调用时序（对齐 ``train_pi_robo_async`` 主循环）::

        get_observation()   -> current_obs()    # 看当前 cursor
        get_info_for_step() -> current_info()   # 看当前 cursor
        step()              -> next_action()     # 取当前帧 qpos 并推进 cursor

    终局语义：末帧 ``current_info()=(done=True, success=True, reward=1.0, mask=0.0)``，其余
    ``(False, False, 0.0, 1.0)`` —— 整条专家带是一条通向 success 的稀疏终局轨迹。
    """

    def __init__(self, frames):
        assert frames is not None and len(frames) > 0, "TapeReplayer frames 不能为空"
        self._frames = frames
        self._cursor = 0

    @property
    def exhausted(self):
        return self._cursor >= len(self._frames)

    def _idx(self):
        # 钳到最后一帧，避免末帧 step 推进后 current_* 越界（learner 在 done 当步仍各取一次）。
        return min(self._cursor, len(self._frames) - 1)

    def current_obs(self):
        return self._frames[self._idx()][0]

    def current_info(self):
        """(done, success, reward, mask) for 当前 cursor（在 step 推进之前）。"""
        is_last = self._cursor >= len(self._frames) - 1
        done = bool(is_last)
        success = bool(is_last)
        reward = 1.0 if is_last else 0.0
        mask = 0.0 if done else 1.0
        return done, success, reward, mask

    def next_action(self):
        """返回当前帧的 expert qpos 目标，并推进 cursor。"""
        qpos = self._frames[self._idx()][1]
        self._cursor += 1
        return qpos


def record_expert_tape(env, flatten_obs_fn, *, save_freq=15, n_real_dims=14):
    """从 env 当前状态跑 ``play_once`` 录专家带（= 在线版专家采集）。

    在 agent rollout 的真实（失败）状态下调用，得到「从该状态通向 success」的 on-policy 专家
    轨迹。实现：临时把 ``env._take_picture`` monkey-patch 成「内存收集
    ``(flatten_obs_fn(env.get_obs()), vector[:n_real_dims])``」，跑完恢复——不改 RoboTwin 源码。

    Args:
      env:            RoboTwin task env（含 play_once / get_obs / check_success / plan_success）。
      flatten_obs_fn: raw get_obs dict -> openpi 扁平 obs dict 的回调（RoboTwinEnv._flatten_obs）。
      save_freq:      录播采样率（物理步），对齐 offline demo / streaming 节奏。
      n_real_dims:    真实关节维度（双臂 aloha=14）。

    Returns:
      frames = list of ``(obs_flat, action_qpos)``（``action[t]=vector[t+1]``，对齐
      ``process_robotwin_dataset``）；规划失败 / 未成功 / 帧数<2 → ``None``（episode 作废）。
    """
    raw_frames = []
    orig_take_picture = env._take_picture
    orig_save_freq = getattr(env, "save_freq", None)

    def _collect():
        obs = env.get_obs()
        vec = np.asarray(obs["joint_action"]["vector"], dtype=np.float64)[:n_real_dims]
        raw_frames.append((flatten_obs_fn(obs), vec))

    try:
        env.save_freq = save_freq
        env._take_picture = _collect
        env.plan_success = True
        env.play_once()
        success = bool(env.check_success())
    except Exception:
        logging.exception("[RoboTwin] expert takeover play_once failed")
        success = False
    finally:
        env._take_picture = orig_take_picture
        env.save_freq = orig_save_freq

    if not bool(getattr(env, "plan_success", False)) or not success or len(raw_frames) < 2:
        return None
    # action[t] = vector[t+1]（下一帧绝对 qpos），对齐 process_robotwin_dataset。
    return [(raw_frames[t][0], raw_frames[t + 1][1]) for t in range(len(raw_frames) - 1)]
