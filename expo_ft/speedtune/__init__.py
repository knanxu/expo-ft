"""SpeedTune 加速模块（RainbowDQN）：在冻结 VLA 之上学「该不该加速 / 加速多少」。

零侵入增量包（与 EXPO/BC/DBPO 算法平级）：读 pi0.5 action expert 的 detached
``suffix_feat``，用 branching RainbowDQN 输出三种执行方式各自的速度控制参数。
详见 ``docs`` 与计划文件。
"""

from expo_ft.speedtune.exec_backends import (
    ExecBackend,
    SpeedVar,
    build_backend,
    list_backends,
)

__all__ = ["ExecBackend", "SpeedVar", "build_backend", "list_backends"]
