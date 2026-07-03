from expo_ft.speedtune.curriculum import SpeedCurriculum


GRID = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)


def _state():
    return SpeedCurriculum(
        speed_grid=GRID, window_size=20, success_threshold=0.7
    )


def test_curriculum_starts_with_only_slowest_bin():
    state = _state()
    assert state.max_unlocked_idx == 0
    assert state.max_unlocked_speed == 1.0
    assert state.action_limits == (0,)


def test_curriculum_requires_full_window_then_unlocks_exactly_one_bin():
    state = _state()
    for index, success in enumerate([True] * 14 + [False] * 5):
        assert not state.record_episode(success, episode=index + 1, decision=index + 1)
    assert state.max_unlocked_idx == 0

    assert state.record_episode(False, episode=20, decision=100)
    assert state.max_unlocked_idx == 1
    assert state.max_unlocked_speed == 1.5
    assert state.recent_results == []
    assert state.unlock_events == [
        {"episode": 20, "decision": 100, "unlocked_idx": 1, "speed": 1.5}
    ]


def test_failed_window_slides_until_recent_success_rate_reaches_threshold():
    state = _state()
    for success in [False] * 7 + [True] * 13:
        state.record_episode(success, episode=20, decision=20)
    assert state.max_unlocked_idx == 0
    assert len(state.recent_results) == 20
    assert state.window_success_rate == 0.65

    assert state.record_episode(True, episode=21, decision=21)
    assert state.max_unlocked_idx == 1
    assert state.recent_results == []


def test_curriculum_never_unlocks_past_last_bin():
    state = _state()
    episode = 0
    for _ in range(len(GRID) + 2):
        for _ in range(20):
            episode += 1
            state.record_episode(True, episode=episode, decision=episode)
    assert state.max_unlocked_idx == len(GRID) - 1
    assert len(state.unlock_events) == len(GRID) - 1


def test_action_counts_and_state_round_trip():
    state = _state()
    state.record_action([0])
    state.record_action([0])
    for success in [True, False, True]:
        state.record_episode(success, episode=3, decision=30)

    restored = SpeedCurriculum.from_state_dict(state.state_dict())

    assert restored.state_dict() == state.state_dict()
    assert restored.action_counts == [2, 0, 0, 0, 0, 0, 0]


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} SpeedTune curriculum tests passed")


if __name__ == "__main__":
    main()
