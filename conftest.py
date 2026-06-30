"""Repo-root conftest —— 把仓库根钉进 sys.path，让 test/ 下的测试能 import 顶层包。

测试已从各包目录集中到 test/（见 CLAUDE.md「验证不回归」）。test/ 下用绝对 import
（from expo_ft / configs / client_robotwin ...）；这里显式把本文件所在的仓库根插入 sys.path，
确保 `python -m pytest` 或裸 `pytest` 都能解析这些顶层包，不依赖 pytest 的 rootdir / import-mode 细节。

本地不跑 pytest（显存不足，见 CLAUDE.md）；测试由维护人在云端执行，例如：
    python -m pytest test/ -v
    python -m pytest test/robotwin_utils_test.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
