import numpy as np

from client_robotwin.envs.robotwin_env import RoboTwinEnv


class FakeWholeChunkEnv:
    def __init__(self):
        self.kwargs = None
        self.eval_video_path = None

    def take_chunk_action(
        self, chunk, *, vel_limit, execution_steps, video_save_freq
    ):
        self.kwargs = {
            "vel_limit": vel_limit,
            "execution_steps": execution_steps,
            "video_save_freq": video_save_freq,
        }
        return {
            "status": "success",
            "dense_steps": 25,
            "duration": 0.1,
            "take_action_cnt_delta": execution_steps,
            "vel_limit": vel_limit,
            "acc_limit": 4.0 * vel_limit ** 2,
            "execution_steps": execution_steps,
            "planned_cruise_fraction": 0.75,
        }


class FakeOldWholeChunkEnv:
    eval_video_path = None

    def take_chunk_action(self, chunk, *, vel_limit, video_save_freq):
        del chunk, vel_limit, video_save_freq
        return {"status": "success", "dense_steps": 1}


class FakeOldStreamingEnv:
    eval_video_path = None

    def take_chunk_action_streaming(
        self, chunk, *, v, hold_steps, max_actions, video_save_freq
    ):
        del chunk, v, hold_steps, max_actions, video_save_freq
        return {
            "status": "success",
            "dense_steps": 1,
            "take_action_cnt_delta": 1,
        }


def _wrapper(env, backend, k_skip):
    wrapper = object.__new__(RoboTwinEnv)
    wrapper.env = env
    wrapper.n_real_dims = 14
    wrapper.exec_backend = backend
    wrapper._k_skip = k_skip
    wrapper._stream_hold_steps = 15
    wrapper._eval_vsf = 25
    wrapper._steps_since_reset = 0
    wrapper._contact_info = lambda: (0.0, 0.0, False, False)
    return wrapper


def _assert_runtime_error(fn, text):
    try:
        fn()
    except RuntimeError as exc:
        assert text in str(exc), exc
    else:
        raise AssertionError("expected RuntimeError")


def test_whole_chunk_step_passes_vel_and_execution_steps_only():
    wrapper = _wrapper(FakeWholeChunkEnv(), "chunk_toppra", 20)

    info = wrapper.step_chunk(
        np.zeros((50, 14), dtype=np.float64),
        {"vel_limit": 2.5},
        "chunk_toppra",
    )

    assert wrapper.env.kwargs == {
        "vel_limit": 2.5,
        "execution_steps": 20,
        "video_save_freq": -1,
    }
    assert info["acc_limit"] == 25.0
    assert info["planned_cruise_fraction"] == 0.75
    assert info["execution_steps"] == 20
    assert info["input_chunk_steps"] == 50
    assert info["requested_execution_steps"] == 20


def test_whole_chunk_rejects_old_robotwin_without_execution_steps_parameter():
    wrapper = _wrapper(FakeOldWholeChunkEnv(), "chunk_toppra", 20)
    _assert_runtime_error(
        lambda: wrapper.step_chunk(
            np.zeros((50, 14), dtype=np.float64),
            {"vel_limit": 2.5},
            "chunk_toppra",
        ),
        "execution_steps",
    )


def test_fixed_time_rejects_old_robotwin_without_speed_gate_telemetry():
    wrapper = _wrapper(FakeOldStreamingEnv(), "fixed_time", 10)
    _assert_runtime_error(
        lambda: wrapper.step_chunk(
            np.zeros((50, 14), dtype=np.float64), {"v": 2.0}, "fixed_time"
        ),
        "fixed_time_speed_violation",
    )


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} RoboTwin step_chunk tests passed")


if __name__ == "__main__":
    main()
