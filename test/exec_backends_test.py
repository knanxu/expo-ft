"""SpeedTune exec_backends 测试：动作空间 + success-gated reward（防 reward hacking）。

    python -m expo_ft.speedtune.exec_backends_test
"""

from expo_ft.speedtune.exec_backends import build_backend, list_backends, parse_force_limit


def test_head_sizes():
    assert build_backend("fixed_time").head_sizes == (5,)
    assert build_backend("per_action_toppra").n_heads == 3
    assert build_backend("chunk_toppra").n_heads == 3
    for name in list_backends():
        b = build_backend(name)
        assert len(b.head_sizes) == b.n_heads
        assert all(n >= 1 for n in b.head_sizes)


def test_per_action_vars_renamed_absolute():
    # 绝对值制: vel_scale/acc_scale -> vel_limit/acc_limit（直接 rad/s, rad/s^2）
    assert [v.name for v in build_backend("per_action_toppra").vars] == ["v", "vel_limit", "acc_limit"]
    assert [v.name for v in build_backend("chunk_toppra").vars] == ["v", "vel_limit", "acc_limit"]
    assert [v.name for v in build_backend("fixed_time").vars] == ["v"]   # fixed_time 不变


def test_decode_lengths():
    b = build_backend("per_action_toppra")
    idxs = [0] * b.n_heads
    sp, v = b.decode(idxs)
    assert set(sp.keys()) == {"v", "vel_limit", "acc_limit"}   # 绝对值制: 直接传 take_chunk_action_per_action
    assert len(v) == b.n_heads


def test_aggressiveness_direction():
    b = build_backend("per_action_toppra")
    vmap = {var.name: var for var in b.vars}
    # 全部 faster_is="larger"（v/vel/acc 越大越快）：最大档 aggressiveness=1，最小档=0
    for name in ("v", "vel_limit", "acc_limit"):
        var = vmap[name]
        assert var.faster_is == "larger"
        assert var.aggressiveness(int(max(range(var.n_bins), key=lambda i: var.grid[i]))) == 1.0
        assert var.aggressiveness(int(min(range(var.n_bins), key=lambda i: var.grid[i]))) == 0.0


def test_reward_gating_no_hacking():
    """核心：失败 episode 速度项必须归零（任何盲目加速都拿不到额外奖励）。"""
    for name in list_backends():
        b = build_backend(name)
        most_aggressive = [max(range(v.n_bins), key=lambda i: v.aggressiveness(i)) for v in b.vars]
        _, v_list = b.decode(most_aggressive)
        # 失败：r_task=0，最激进 → reward 必须 == 0
        assert b.total_reward(r_task=0.0, success=False, v_list=v_list) == 0.0
        # 失败但 r_task 也给（不该发生，但即便给也只有任务项，无速度加成）
        assert b.total_reward(r_task=0.0, success=False, v_list=v_list) == 0.0
        # speed_reward 单独也应为 0
        assert b.speed_reward(False, v_list) == 0.0


def test_reward_monotone_when_success():
    """成功时：更激进 → reward 单调更高，且总 ≥ 仅任务奖励。"""
    for name in list_backends():
        b = build_backend(name)
        aggr = [max(range(v.n_bins), key=lambda i: v.aggressiveness(i)) for v in b.vars]
        cons = [max(range(v.n_bins), key=lambda i: -v.aggressiveness(i)) for v in b.vars]
        _, v_a = b.decode(aggr)
        _, v_c = b.decode(cons)
        r_task = 1.0
        r_aggr = b.total_reward(r_task, True, v_a)
        r_cons = b.total_reward(r_task, True, v_c)
        assert r_cons == r_task            # 不加速 → 只拿任务奖励
        assert r_aggr > r_cons             # 加速 → 更高（成功前提下）


def test_reward_override():
    b = build_backend("fixed_time", overrides={"v": {"alpha": 3.0, "beta": 2.0}})
    var = b.vars[0]
    assert var.name == "v" and var.alpha == 3.0 and var.beta == 2.0
    # v=1（最激进）→ speed = 3*1^2 = 3
    _, v = b.decode([max(range(var.n_bins), key=lambda i: var.aggressiveness(i))])
    assert abs(b.speed_reward(True, v) - 3.0) < 1e-9


def test_parse_force_limit_str():
    assert parse_force_limit("30,40,30,15,10,10") == [30.0, 40.0, 30.0, 15.0, 10.0, 10.0]


def test_parse_force_limit_empty_and_none():
    assert parse_force_limit("") is None
    assert parse_force_limit("   ") is None
    assert parse_force_limit(None) is None


def test_parse_force_limit_list_tuple_and_spaces():
    assert parse_force_limit([30, 40]) == [30.0, 40.0]
    assert parse_force_limit((30.0, 40.0)) == [30.0, 40.0]
    assert parse_force_limit("30, 40 ,30") == [30.0, 40.0, 30.0]


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} exec_backends tests passed.")


if __name__ == "__main__":
    main()
