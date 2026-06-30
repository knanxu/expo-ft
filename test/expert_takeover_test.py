"""client_robotwin.envs.expert_takeover 单元测试（纯逻辑，不依赖 sapien/RoboTwin）。

⚠️ 本地显存不足、不跑 pytest；本测试由维护人在**云端**执行：
    python -m pytest test/expert_takeover_test.py -v
"""

import numpy as np
import pytest

from client_robotwin.envs.expert_takeover import (
    should_takeover,
    TapeReplayer,
    record_expert_tape,
)


# ---------------- should_takeover ----------------
def _kw(**over):
    base = dict(steps_since_reset=0, step_budget=800, success_once=False,
                step_frac=0.5, enabled=True)
    base.update(over)
    return base


def test_trigger_fires_at_threshold():
    assert should_takeover(**_kw(steps_since_reset=400)) is True   # 0.5*800
    assert should_takeover(**_kw(steps_since_reset=399)) is False


def test_trigger_blocked_when_disabled_or_success_or_bad_budget():
    assert should_takeover(**_kw(steps_since_reset=800, enabled=False)) is False
    assert should_takeover(**_kw(steps_since_reset=800, success_once=True)) is False
    assert should_takeover(**_kw(steps_since_reset=800, step_budget=0)) is False


# ---------------- TapeReplayer ----------------
def _frames(n):
    return [({"observation.state": np.full(14, i, np.float32)},
             np.full(14, i + 100, np.float64)) for i in range(n)]


def test_replayer_timing_and_terminal():
    r = TapeReplayer(_frames(3))
    # frame 0（非末帧）
    assert r.current_obs()["observation.state"][0] == 0
    assert r.current_info() == (False, False, 0.0, 1.0)
    assert r.next_action()[0] == 100        # 推进到 cursor=1
    # frame 1
    assert r.current_obs()["observation.state"][0] == 1
    assert r.current_info() == (False, False, 0.0, 1.0)
    assert r.next_action()[0] == 101        # cursor=2
    # frame 2（末帧）
    assert r.current_info() == (True, True, 1.0, 0.0)
    assert r.exhausted is False
    assert r.next_action()[0] == 102        # cursor=3
    assert r.exhausted is True


def test_replayer_rejects_empty():
    with pytest.raises(AssertionError):
        TapeReplayer([])


# ---------------- record_expert_tape ----------------
class _FakeEnv:
    """模拟 RoboTwin task env 的录播相关接口。play_once 内按 n 次调 _take_picture。"""

    def __init__(self, n, success=True, plan_ok=True):
        self.save_freq = None
        self.plan_success = True
        self._n, self._success, self._plan_ok = n, success, plan_ok
        self._t = 0

    def get_obs(self):
        self._t += 1
        return {"joint_action": {"vector": list(np.full(14, self._t, np.float64))}}

    def play_once(self):
        self.plan_success = self._plan_ok
        for _ in range(self._n):
            self._take_picture()      # 被 record_expert_tape monkey-patch

    def check_success(self):
        return self._success

    def _take_picture(self):          # 原始实现（会被替换，跑完应恢复）
        pass


def _flat(raw):
    return {"observation.state": np.asarray(raw["joint_action"]["vector"], np.float32)}


def test_record_success_builds_frames():
    env = _FakeEnv(n=5)
    frames = record_expert_tape(env, _flat, save_freq=1)
    assert frames is not None and len(frames) == 4            # n-1
    # action[t] = vector[t+1]：frame0 的 obs 是 vector=1、action 是 vector=2
    assert frames[0][1][0] == 2 and frames[0][0]["observation.state"][0] == 1
    assert env._take_picture.__name__ == "_take_picture"     # monkey-patch 已恢复


def test_record_returns_none_on_failure():
    assert record_expert_tape(_FakeEnv(n=5, success=False), _flat, save_freq=1) is None
    assert record_expert_tape(_FakeEnv(n=5, plan_ok=False), _flat, save_freq=1) is None
    assert record_expert_tape(_FakeEnv(n=1), _flat, save_freq=1) is None    # <2 帧
