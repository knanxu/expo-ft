"""RoboTwin rollout server for DBPO×EXPO-FT online RL.

与 `client/run_client.py` **完全相同的 websocket 协议**（create_env / reset / step /
get_observation / get_info_for_step，msgpack_numpy），learner 端 `env_client` 不变。
差异：① 用 RoboTwin task config + `RoboTwinEnv`；② 去掉 DROID 专有的 spacemouse human-override；
③ 注入 RoboTwin 仓库根路径，使 `import envs.{task_name}` 可用。

运行（在 RoboTwin venv，含 sapien/curobo）：
    python -m client_robotwin.run_robotwin_client \
        --config_task_path configs/task/robotwin_stack_blocks.py \
        --robotwin_root /home/xukainan/RoboTwin --server_port 8102
"""

import asyncio
import dataclasses
import logging
import os
import sys
from typing import Any, Dict, Optional

import numpy as np
import websockets
import websockets.asyncio.server as _server
from openpi_client import msgpack_numpy

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYOPENGL_PLATFORM"] = "egl"  # headless 渲染（sapien）

import tyro


@dataclasses.dataclass
class Args:
    server_host: str = "0.0.0.0"
    server_port: int = 8102
    config_task_path: str = "configs/task/robotwin_stack_blocks.py"
    robotwin_root: str = "/home/xukainan/RoboTwin"  # 含 envs/ 的 RoboTwin 仓库根


_env_storage: Dict[str, Any] = {}
_config_task_path: Optional[str] = None
_robotwin_root: Optional[str] = None


def load_task_config(config_path: Optional[str]):
    """Import configs.task.X.get_config()（同 run_client.load_task_config）。"""
    if config_path is None:
        return None
    if "/" in config_path or ".py" in config_path:
        config_path = config_path.replace(".py", "").replace("/", ".")
    module = __import__(config_path, fromlist=["get_config"])
    return module.get_config()


async def _handle_environment_request(websocket: _server.ServerConnection):
    logger = logging.getLogger(__name__)
    packer = msgpack_numpy.Packer()
    try:
        while True:
            try:
                request = msgpack_numpy.unpackb(await websocket.recv())
                operation = request.get("operation")

                if operation == "create_env":
                    task_config = load_task_config(_config_task_path)
                    env_name = task_config.env_name
                    env_usage = request["env_usage"]
                    env_id = f"{env_name}_{env_usage}"
                    logger.info("Creating RoboTwin env %s ...", env_id)
                    # 通用建 env（同 run_client）：config-as-kwargs；RoboTwinEnv 按名提取、**kwargs 吸收余项。
                    env_kwargs = dict(task_config)
                    env_kwargs["video_dir"] = request.get("video_dir") or ""
                    env_kwargs["is_eval"] = (env_usage == "eval")
                    env = task_config.env(**env_kwargs)
                    _env_storage[env_id] = env
                    response = {
                        "status": "success",
                        "env_id": env_id,
                        "task_description": task_config.language_instruction,
                    }
                    await websocket.send(packer.pack(response))
                    logger.info("RoboTwin env %s created.", env_id)

                elif operation == "reset":
                    env = _env_storage.get(request["env_id"])
                    if env is None:
                        response = {"status": "error", "message": f"Env {request['env_id']} not found"}
                    else:
                        response = {"status": "success", "observation": env.reset(), "done": False}
                    await websocket.send(packer.pack(response))

                elif operation == "step":
                    env = _env_storage.get(request["env_id"])
                    if env is None:
                        response = {"status": "error", "message": f"Env {request['env_id']} not found"}
                    else:
                        sent_action = np.array(request["action"], dtype=np.float64)
                        if not np.isfinite(sent_action).all():
                            logger.warning("Action has NaN/Inf; zero-filling. Check policy/training stability.")
                            sent_action = np.where(np.isfinite(sent_action), sent_action, 0.0)
                        # invalid sentinel（全 -1）跳过执行，同 run_client。
                        if np.allclose(sent_action, -1.0):
                            executed_action = sent_action
                        else:
                            executed_action = np.array(env.step(sent_action)["executed_action"], dtype=np.float64)
                        response = {
                            "status": "success",
                            "action": executed_action.tolist(),
                            "action_type": "policy",   # RoboTwin 仿真无 human override
                        }
                    await websocket.send(packer.pack(response))

                elif operation == "get_observation":
                    env = _env_storage.get(request["env_id"])
                    if env is None:
                        response = {"status": "error", "message": f"Env {request['env_id']} not found"}
                    else:
                        response = {"status": "success", "observation": env.get_observation()}
                    await websocket.send(packer.pack(response))

                elif operation == "get_info_for_step":
                    env = _env_storage.get(request["env_id"])
                    if env is None:
                        response = {"status": "error", "message": f"Env {request['env_id']} not found"}
                    else:
                        done, success, reward, mask = env.get_info_for_step()
                        response = {
                            "status": "success",
                            "done": bool(done), "success": bool(success),
                            "reward": float(reward), "mask": float(mask),
                        }
                    await websocket.send(packer.pack(response))

                else:
                    response = {"status": "error", "message": f"Unknown operation: {operation}"}
                    await websocket.send(packer.pack(response))

            except websockets.exceptions.ConnectionClosed:
                logger.debug("Connection closed by %s", websocket.remote_address)
                break
            except Exception as e:
                logger.error("Error handling request: %s", e, exc_info=True)
                try:
                    await websocket.send(packer.pack({"status": "error", "message": str(e)}))
                except websockets.exceptions.ConnectionClosed:
                    break
    except Exception as e:
        logger.error("Unexpected handler error: %s", e, exc_info=True)


async def _run_server(args: Args):
    global _config_task_path, _robotwin_root
    _config_task_path = args.config_task_path
    _robotwin_root = os.path.abspath(args.robotwin_root) if args.robotwin_root else None

    # 先把 expo-ft 仓库根（含 configs/ 与 client_robotwin/）以**绝对路径**钉进 sys.path，
    # 这样即便下面 chdir 到 RoboTwin，`import configs.task.X` / `client_robotwin.*` 仍可 import。
    _expo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _expo_root not in sys.path:
        sys.path.insert(0, _expo_root)
    if _robotwin_root and _robotwin_root not in sys.path:
        sys.path.insert(0, _robotwin_root)  # 使 RoboTwinEnv 内 `import envs.{task}` 可用
    # RoboTwin 任务/资产代码大量用**相对 cwd**的路径：import 期就读（如
    # envs/utils/rand_create_cluttered_actor.py 的 ./assets/objects/objaverse/list.json），
    # 运行期 setup_demo/get_obs 加载 mesh 同理；RoboTwin 自家脚本均假定 cwd=仓库根。这里持久
    # chdir 对齐，否则从 expo-ft 目录启动 server 会 FileNotFoundError。
    if _robotwin_root:
        os.chdir(_robotwin_root)
    logger = logging.getLogger(__name__)
    async with _server.serve(
        _handle_environment_request, args.server_host, args.server_port,
        compression=None, max_size=None,
        ping_interval=None, ping_timeout=None, close_timeout=100,
    ) as server:
        logger.info("RoboTwin env server on %s:%d", args.server_host, args.server_port)
        await server.serve_forever()


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.getLogger("websockets.server").setLevel(logging.WARNING)
    asyncio.run(_run_server(args))


if __name__ == "__main__":
    main(tyro.cli(Args))
