#!/usr/bin/env python3
"""
[INPUT]: 依赖 CARLA PythonAPI、冻结 run plan/attempt、隔离式子进程和 runner 的 Python/动态库环境。
[OUTPUT]: 对外提供 build_python_environment、run_carla_startup_rpc、wait_for_carla 与 build_evaluator_command，封装 CARLA 启动探针和 Leaderboard 命令构造。
[POS]: tools.evaluation 的 CARLA 进程适配层，使 run_thesis_baseline 专注 attempt 编排与资源生命周期，不承载版本敏感的 RPC/环境细节。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import os
import subprocess
import sys
import time
from pathlib import Path

from tools.evaluation.runtime_resources import RunnerError


CARLA_MAP_MARKER = "CARLA_STARTUP_MAP="
CARLA_STARTUP_RPC = r"""
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
for path in (
    repo_root / "carla" / "PythonAPI",
    repo_root / "carla" / "PythonAPI" / "carla",
):
    sys.path.insert(0, str(path))

import carla

client = carla.Client("127.0.0.1", int(sys.argv[2]))
client.set_timeout(float(sys.argv[3]))
world = client.load_world(sys.argv[4]) if len(sys.argv) == 5 else client.get_world()
print("CARLA_STARTUP_MAP=" + world.get_map().name, flush=True)
"""


def build_python_environment(repo_root, plan, attempt, save_path):
    env = os.environ.copy()
    runtime = plan["runtime"]
    env.update({str(key): str(value) for key, value in plan["environment"].items()})
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(runtime["agent_cuda_visible_device"]),
            "INTERFUSER_MODEL_PATH": plan["checkpoint_path"],
            "INTERFUSER_BG_VEHICLES": str(attempt["background_vehicles"]),
            "SAVE_PATH": str(save_path),
            "CARLA_ROOT": str(repo_root / "carla"),
            "SCENARIO_RUNNER_ROOT": str(repo_root / "scenario_runner"),
            "LEADERBOARD_ROOT": str(repo_root / "leaderboard"),
            "PYTHONUNBUFFERED": "1",
            "PYGAME_HIDE_SUPPORT_PROMPT": "1",
            "MALLOC_TRIM_THRESHOLD_": "100000",
        }
    )
    python_paths = [
        repo_root / "interfuser",
        repo_root / "carla" / "PythonAPI",
        repo_root / "carla" / "PythonAPI" / "examples",
        repo_root / "carla" / "PythonAPI" / "carla",
        repo_root / "leaderboard",
        repo_root / "leaderboard" / "team_code",
        repo_root / "scenario_runner",
        repo_root,
    ]
    if env.get("PYTHONPATH"):
        python_paths.append(Path(env["PYTHONPATH"]))
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in python_paths)
    prefix = Path(sys.prefix)
    library_paths = [prefix / "lib"]
    library_paths.extend(
        (prefix / "lib" / "python3.10" / "site-packages" / "nvidia").glob(
            "*/lib"
        )
    )
    if env.get("LD_LIBRARY_PATH"):
        library_paths.append(Path(env["LD_LIBRARY_PATH"]))
    env["LD_LIBRARY_PATH"] = os.pathsep.join(str(path) for path in library_paths)
    return env


def run_carla_startup_rpc(repo_root, port, timeout_seconds, runtime_map=None):
    """Keep CARLA native client aborts outside the long-lived runner process."""
    command = [
        sys.executable,
        "-c",
        CARLA_STARTUP_RPC,
        str(repo_root),
        str(port),
        str(timeout_seconds),
    ]
    if runtime_map is not None:
        command.append(str(runtime_map))
    try:
        result = subprocess.run(
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=max(10.0, float(timeout_seconds) + 5.0),
        )
    except subprocess.TimeoutExpired as exc:
        raise RunnerError(
            f"CARLA startup RPC exceeded {timeout_seconds} seconds"
        ) from exc
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        suffix = detail[-1] if detail else "no stderr"
        raise RunnerError(
            f"CARLA startup RPC exited with code {result.returncode}: {suffix}"
        )
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(CARLA_MAP_MARKER):
            return line[len(CARLA_MAP_MARKER) :]
    raise RunnerError("CARLA startup RPC returned no map name")


def wait_for_carla(repo_root, process, port, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    last_error = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RunnerError(
                f"CARLA exited during startup with code {process.returncode}"
            )
        try:
            return run_carla_startup_rpc(repo_root, port, 2)
        except Exception as exc:  # CARLA raises version-specific RPC exceptions.
            last_error = exc
            time.sleep(2)
    raise RunnerError(f"CARLA did not become ready: {last_error}")


def build_evaluator_command(repo_root, plan, attempt, route_path, result_path):
    runtime = plan["runtime"]
    return [
        sys.executable,
        str(repo_root / "leaderboard" / "leaderboard" / "leaderboard_evaluator.py"),
        f"--scenarios={plan['scenarios_path']}",
        f"--routes={route_path}",
        "--repetitions=1",
        "--track=SENSORS",
        f"--checkpoint={result_path}",
        f"--agent={plan['agent_path']}",
        f"--agent-config={plan['agent_config_path']}",
        "--debug=0",
        "--record=",
        "--resume=False",
        f"--port={runtime['carla_port']}",
        f"--trafficManagerPort={runtime['traffic_manager_port']}",
        f"--trafficManagerSeed={attempt['traffic_manager_seed']}",
        f"--carlaProviderSeed={attempt['carla_provider_seed']}",
        f"--timeout={runtime['carla_client_timeout_seconds']}",
    ]
