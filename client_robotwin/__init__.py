"""RoboTwin rollout client (env-server side) for EXPO-FT online RL.

独立包，与 `client/`（DROID）平级、零污染：env 执行侧（分离①）。speak 与 `client/run_client.py`
完全相同的 websocket 协议，learner 端 `expo_ft/env/env_client.py` 不变。重 sim 依赖
（sapien/robotwin/curobo）隔离在本包自己的运行环境（建议独立 venv）。
"""
