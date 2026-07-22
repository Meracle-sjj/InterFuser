#!/usr/bin/env python3
"""
[INPUT]: 依赖通过 P0 的 baseline_eval JSON、CARLA/Leaderboard 运行时、冻结 checkpoint，以及空闲端口和 GPU。
[OUTPUT]: 对外提供 RunnerError、build_run_plan、write_single_route_xml、parse_leaderboard_result、wait_for_ports_free、execute_run_plan 与 CLI，生成隔离的 route/seed 原始结果和 manifest。
[POS]: tools/evaluation 的 M0 配置驱动 runner，位于静态预检之后、统计汇总之前；默认 dry-run，显式 --execute 才启动外部进程。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluation.preflight_thesis_baseline import (  # noqa: E402
    DEFAULT_CONFIG,
    PreflightError,
    preflight_baseline,
)


RUN_PLAN_SCHEMA_VERSION = 1
RUN_MANIFEST_SCHEMA_VERSION = 1
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
PORT_RELEASE_TIMEOUT_SECONDS = 60
PORT_RELEASE_POLL_SECONDS = 0.5


class RunnerError(RuntimeError):
    """Raised when a run cannot be planned or executed without ambiguity."""


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"unable to read JSON {path}: {exc}") from exc


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _positive_int(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RunnerError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RunnerError(f"{name} must be a nonnegative integer")
    return value


def _route_index(path):
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise RunnerError(f"unable to parse route XML {path}: {exc}") from exc
    routes = {}
    for element in root.findall("route"):
        try:
            route_id = int(element.attrib["id"])
            town = element.attrib["town"]
        except (KeyError, TypeError, ValueError) as exc:
            raise RunnerError("route XML contains invalid id/town") from exc
        if route_id in routes:
            raise RunnerError(f"route XML contains duplicate id {route_id}")
        routes[route_id] = {"town": town, "element": element}
    return routes


def write_single_route_xml(source_path, route_id, output_path):
    """Write one unchanged route under a fresh routes root."""
    routes = _route_index(source_path)
    if route_id not in routes:
        raise RunnerError(f"route {route_id} is absent from {source_path}")
    root = ET.Element("routes")
    root.append(copy.deepcopy(routes[route_id]["element"]))
    tree = ET.ElementTree(root)
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def _selected_values(config_values, requested, name):
    available = list(config_values)
    if requested is None:
        return available
    if len(requested) != len(set(requested)):
        raise RunnerError(f"{name} contains duplicates")
    unavailable = sorted(set(requested) - set(available))
    if unavailable:
        raise RunnerError(f"{name} contains values outside config: {unavailable}")
    return list(requested)


def _resolved_result_root(repo_root, relative):
    candidate = (repo_root / relative).resolve()
    try:
        candidate.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise RunnerError("result_root escapes repository") from exc
    return candidate


def build_run_plan(
    config_path=DEFAULT_CONFIG,
    repo_root=REPO_ROOT,
    run_id=None,
    route_set="development_d7",
    route_ids=None,
    seeds=None,
    agent_gpu=None,
    carla_graphics_adapter=None,
    timeout_seconds=None,
    check_git=True,
):
    """Build an immutable route/seed plan after the P0 gate passes."""
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise RunnerError(
            "run_id must be 1-96 characters using letters, digits, dot, dash or underscore"
        )
    repo_root = Path(repo_root).resolve()
    config_path = Path(config_path).resolve()
    try:
        preflight = preflight_baseline(config_path, repo_root, check_git=check_git)
    except PreflightError as exc:
        raise RunnerError(f"P0 preflight could not run: {exc}") from exc
    if not preflight["valid"]:
        raise RunnerError("P0 preflight failed: " + "; ".join(preflight["errors"]))

    config = _read_json(config_path)
    route_sets = config.get("route_sets", {})
    if route_set not in route_sets:
        raise RunnerError(f"unknown route set: {route_set}")
    selected_routes = _selected_values(route_sets[route_set], route_ids, "route_ids")
    selected_seeds = _selected_values(config["random_seeds"], seeds, "seeds")
    if not selected_routes or not selected_seeds:
        raise RunnerError("run plan must contain at least one route and one seed")

    routes_path = (repo_root / config["inputs"]["routes"]["path"]).resolve()
    routes = _route_index(routes_path)
    runtime = config["runtime"]
    agent_gpu = _nonnegative_int(
        runtime["agent_cuda_visible_device"] if agent_gpu is None else agent_gpu,
        "agent_gpu",
    )
    carla_graphics_adapter = _nonnegative_int(
        runtime["carla_graphics_adapter"]
        if carla_graphics_adapter is None
        else carla_graphics_adapter,
        "carla_graphics_adapter",
    )
    timeout_seconds = _positive_int(
        runtime["external_route_timeout_seconds"]
        if timeout_seconds is None
        else timeout_seconds,
        "timeout_seconds",
    )
    background = config["background_vehicles_by_town"]
    allow_opt = bool(config["map_policy"].get("allow_opt_runtime_equivalent"))
    provider_offset = int(runtime["carla_provider_seed_offset"])

    attempts = []
    for route_id in selected_routes:
        if route_id not in routes:
            raise RunnerError(f"selected route {route_id} is absent from route XML")
        town = routes[route_id]["town"]
        if town not in background:
            raise RunnerError(f"no background vehicle count for {town}")
        runtime_map = f"{town}_Opt" if allow_opt else town
        for seed in selected_seeds:
            attempts.append(
                {
                    "attempt_id": f"route_{route_id:02d}_seed_{seed}",
                    "route_id": route_id,
                    "town": town,
                    "runtime_map": runtime_map,
                    "traffic_manager_seed": seed,
                    "carla_provider_seed": provider_offset + seed,
                    "background_vehicles": int(background[town]),
                    "external_timeout_seconds": timeout_seconds,
                }
            )

    result_root = _resolved_result_root(repo_root, config["result_root"])
    return {
        "run_plan_schema_version": RUN_PLAN_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": _utc_now(),
        "config_path": str(config_path),
        "config_sha256": preflight["config_sha256"],
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": _sha256(__file__),
        "python_executable": sys.executable,
        "code_anchor": config["code_anchor"],
        "git_head": preflight["git_head"],
        "route_set": route_set,
        "routes_path": str(routes_path),
        "scenarios_path": str(
            (repo_root / config["inputs"]["scenarios"]["path"]).resolve()
        ),
        "checkpoint_path": config["checkpoint"]["path"],
        "agent_path": str((repo_root / config["inputs"]["agent"]["path"]).resolve()),
        "agent_config_path": str(
            (repo_root / config["inputs"]["agent_config"]["path"]).resolve()
        ),
        "result_root": str(result_root),
        "run_directory": str(result_root / run_id),
        "runtime": {
            **runtime,
            "agent_cuda_visible_device": agent_gpu,
            "carla_graphics_adapter": carla_graphics_adapter,
        },
        "environment": dict(config.get("environment", {})),
        "attempts": attempts,
    }


def parse_leaderboard_result(path):
    """Separate a valid driving failure from a broken evaluator run."""
    path = Path(path)
    if not path.is_file():
        return {"valid": False, "error": "result JSON is missing"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"valid": False, "error": f"unable to parse result JSON: {exc}"}
    records = data.get("_checkpoint", {}).get("records", [])
    if not isinstance(records, list) or len(records) != 1:
        return {"valid": False, "error": "result must contain exactly one route record"}
    record = records[0]
    scores = record.get("scores")
    required_scores = ("score_composed", "score_route", "score_penalty")
    if not isinstance(scores, dict) or any(
        not isinstance(scores.get(name), (int, float)) for name in required_scores
    ):
        return {"valid": False, "error": "route record has invalid scores"}
    infractions = record.get("infractions", {})
    infraction_counts = {
        name: len(values) if isinstance(values, list) else None
        for name, values in sorted(infractions.items())
    }
    parsed = {
        "valid": True,
        "status": record.get("status"),
        "route_id": record.get("route_id"),
        "scores": {name: float(scores[name]) for name in required_scores},
        "infraction_counts": infraction_counts,
        "meta": record.get("meta", {}),
        "entry_status": data.get("entry_status"),
        "eligible": data.get("eligible"),
    }
    status = str(parsed["status"] or "")
    setup_failure_markers = (
        "couldn't be set up",
        "could not be set up",
        "agent crashed",
        "simulation crashed",
    )
    duration_game = parsed["meta"].get("duration_game")
    invalid_reasons = []
    if any(marker in status.lower() for marker in setup_failure_markers):
        invalid_reasons.append(f"infrastructure status: {status}")
    if not isinstance(duration_game, (int, float)) or duration_game <= 0:
        invalid_reasons.append("route has no positive game duration")
    if invalid_reasons:
        parsed["valid"] = False
        parsed["error"] = "; ".join(invalid_reasons)
    return parsed


def _port_is_open(port):
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=0.25):
            return True
    except OSError:
        return False


def ensure_ports_free(ports):
    occupied = [int(port) for port in ports if _port_is_open(port)]
    if occupied:
        raise RunnerError(f"required ports are already in use: {occupied}")


def wait_for_ports_free(
    ports,
    timeout_seconds=PORT_RELEASE_TIMEOUT_SECONDS,
    poll_seconds=PORT_RELEASE_POLL_SECONDS,
):
    """Wait for a runner-owned CARLA process to finish releasing its sockets."""
    ports = [int(port) for port in ports]
    started = time.monotonic()
    deadline = started + max(0.0, float(timeout_seconds))
    while True:
        occupied = [port for port in ports if _port_is_open(port)]
        if not occupied:
            return round(time.monotonic() - started, 3)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RunnerError(f"ports did not become free after shutdown: {occupied}")
        time.sleep(min(float(poll_seconds), remaining))


def _gpu_memory_usage():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RunnerError(f"unable to query GPU usage: {result.stderr.strip()}")
    usage = {}
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 2:
            usage[int(fields[0])] = int(fields[1])
    return usage


def ensure_gpus_available(indices, threshold_mb):
    usage = _gpu_memory_usage()
    failures = []
    for index in sorted(set(indices)):
        if index not in usage:
            failures.append(f"GPU {index} is unavailable")
        elif usage[index] > threshold_mb:
            failures.append(
                f"GPU {index} uses {usage[index]} MiB, above {threshold_mb} MiB"
            )
    if failures:
        raise RunnerError("; ".join(failures))
    return usage


class _GpuMemoryMonitor:
    def __init__(self, indices):
        self.indices = sorted(set(indices))
        self.peaks = {index: 0 for index in self.indices}
        self.error = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self._stop.is_set():
            try:
                usage = _gpu_memory_usage()
                for index in self.indices:
                    self.peaks[index] = max(self.peaks[index], usage.get(index, 0))
            except RunnerError as exc:
                self.error = str(exc)
            self._stop.wait(1.0)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)
        return {str(index): value for index, value in self.peaks.items()}


def _python_environment(repo_root, plan, attempt, save_path):
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
    library_paths.extend((prefix / "lib" / "python3.10" / "site-packages" / "nvidia").glob("*/lib"))
    if env.get("LD_LIBRARY_PATH"):
        library_paths.append(Path(env["LD_LIBRARY_PATH"]))
    env["LD_LIBRARY_PATH"] = os.pathsep.join(str(path) for path in library_paths)
    return env


def _stop_process_group(process, grace_seconds=20):
    if process is None or process.poll() is not None:
        return None if process is None else process.returncode
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=grace_seconds)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
    return process.returncode


def _carla_client(repo_root, port, timeout_seconds):
    for path in (
        repo_root / "carla" / "PythonAPI",
        repo_root / "carla" / "PythonAPI" / "carla",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import carla

    client = carla.Client("127.0.0.1", int(port))
    client.set_timeout(float(timeout_seconds))
    return client


def _wait_for_carla(repo_root, process, port, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    last_error = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RunnerError(f"CARLA exited during startup with code {process.returncode}")
        try:
            client = _carla_client(repo_root, port, 2)
            return client.get_world().get_map().name
        except Exception as exc:  # CARLA raises version-specific RPC exceptions.
            last_error = exc
            time.sleep(2)
    raise RunnerError(f"CARLA did not become ready: {last_error}")


def _evaluator_command(repo_root, plan, attempt, route_path, result_path):
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


def _execute_attempt(repo_root, plan, run_dir, attempt):
    runtime = plan["runtime"]
    attempt_dir = run_dir / "attempts" / attempt["attempt_id"]
    attempt_dir.mkdir(parents=True, exist_ok=False)
    route_path = write_single_route_xml(
        plan["routes_path"],
        attempt["route_id"],
        attempt_dir / "route.xml",
    )
    result_path = attempt_dir / "leaderboard_result.json"
    attempt_manifest_path = attempt_dir / "attempt_manifest.json"
    attempt_manifest = {
        **attempt,
        "started_at": _utc_now(),
        "finished_at": None,
        "pipeline_valid": False,
        "process_exit_code": None,
        "external_timeout": False,
        "runtime_map_loaded": None,
        "host": socket.gethostname(),
        "carla_command": None,
        "carla_pid": None,
        "carla_exit_code": None,
        "evaluator_pid": None,
        "duration_seconds": None,
        "gpu_memory_before_mb": None,
        "gpu_peak_memory_mb": None,
        "gpu_monitor_error": None,
        "port_release_wait_seconds": None,
        "cleanup_error": None,
        "error": None,
        "leaderboard_result": None,
        "evaluator_command": None,
    }
    _write_json_atomic(attempt_manifest_path, attempt_manifest)

    carla_process = None
    evaluator_process = None
    started_monotonic = time.monotonic()
    gpu_indices = [
        runtime["agent_cuda_visible_device"],
        runtime["carla_graphics_adapter"],
    ]
    gpu_monitor = _GpuMemoryMonitor(gpu_indices)
    carla_log = (attempt_dir / "carla.log").open("w", encoding="utf-8")
    evaluator_log = (attempt_dir / "evaluator.log").open("w", encoding="utf-8")
    try:
        ensure_ports_free([runtime["carla_port"], runtime["traffic_manager_port"]])
        attempt_manifest["gpu_memory_before_mb"] = {
            str(index): value
            for index, value in ensure_gpus_available(
                gpu_indices, runtime["gpu_busy_memory_threshold_mb"]
            ).items()
            if index in gpu_indices
        }
        gpu_monitor.start()
        carla_command = [
            str(repo_root / "carla" / "CarlaUE4.sh"),
            f"--world-port={runtime['carla_port']}",
            f"-graphicsadapter={runtime['carla_graphics_adapter']}",
            "-quality-level=Low",
            "-RenderOffScreen",
        ]
        attempt_manifest["carla_command"] = carla_command
        carla_process = subprocess.Popen(
            carla_command,
            cwd=repo_root,
            stdout=carla_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        attempt_manifest["carla_pid"] = carla_process.pid
        _wait_for_carla(
            repo_root,
            carla_process,
            runtime["carla_port"],
            runtime["carla_start_timeout_seconds"],
        )
        client = _carla_client(
            repo_root, runtime["carla_port"], runtime["carla_client_timeout_seconds"]
        )
        world = client.load_world(attempt["runtime_map"])
        attempt_manifest["runtime_map_loaded"] = world.get_map().name

        save_path = attempt_dir / "sensor_data"
        save_path.mkdir()
        env = _python_environment(repo_root, plan, attempt, save_path)
        env["ROUTES"] = str(route_path)
        command = _evaluator_command(repo_root, plan, attempt, route_path, result_path)
        attempt_manifest["evaluator_command"] = command
        _write_json_atomic(attempt_manifest_path, attempt_manifest)
        evaluator_process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=env,
            stdout=evaluator_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        attempt_manifest["evaluator_pid"] = evaluator_process.pid
        try:
            attempt_manifest["process_exit_code"] = evaluator_process.wait(
                timeout=attempt["external_timeout_seconds"]
            )
        except subprocess.TimeoutExpired:
            attempt_manifest["external_timeout"] = True
            _stop_process_group(evaluator_process, grace_seconds=10)
            attempt_manifest["process_exit_code"] = 124

        parsed = parse_leaderboard_result(result_path)
        attempt_manifest["leaderboard_result"] = parsed
        attempt_manifest["pipeline_valid"] = bool(
            parsed["valid"]
            and attempt_manifest["process_exit_code"] == 0
            and not attempt_manifest["external_timeout"]
        )
    except KeyboardInterrupt:
        attempt_manifest["error"] = "interrupted"
        raise
    except Exception as exc:
        attempt_manifest["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _stop_process_group(evaluator_process, grace_seconds=10)
        attempt_manifest["carla_exit_code"] = _stop_process_group(carla_process)
        if carla_process is not None:
            try:
                attempt_manifest["port_release_wait_seconds"] = wait_for_ports_free(
                    [runtime["carla_port"], runtime["traffic_manager_port"]]
                )
            except RunnerError as exc:
                attempt_manifest["cleanup_error"] = str(exc)
                attempt_manifest["pipeline_valid"] = False
        evaluator_log.close()
        carla_log.close()
        attempt_manifest["finished_at"] = _utc_now()
        if gpu_monitor._thread.is_alive():
            attempt_manifest["gpu_peak_memory_mb"] = gpu_monitor.stop()
        else:
            attempt_manifest["gpu_peak_memory_mb"] = {
                str(index): value for index, value in gpu_monitor.peaks.items()
            }
        attempt_manifest["gpu_monitor_error"] = gpu_monitor.error
        attempt_manifest["duration_seconds"] = round(
            time.monotonic() - started_monotonic, 3
        )
        _write_json_atomic(attempt_manifest_path, attempt_manifest)
    return attempt_manifest


def execute_run_plan(plan, repo_root=REPO_ROOT, resume=False):
    """Execute a prepared plan without touching unrelated CARLA processes."""
    repo_root = Path(repo_root).resolve()
    runtime = plan["runtime"]
    threshold = _positive_int(
        runtime["gpu_busy_memory_threshold_mb"], "gpu_busy_memory_threshold_mb"
    )
    selected_gpu_indices = sorted(
        {
            runtime["agent_cuda_visible_device"],
            runtime["carla_graphics_adapter"],
        }
    )
    initial_gpu_usage = ensure_gpus_available(selected_gpu_indices, threshold)
    ensure_ports_free([runtime["carla_port"], runtime["traffic_manager_port"]])

    run_dir = Path(plan["run_directory"])
    manifest_path = run_dir / "run_manifest.json"
    if run_dir.exists() and not resume:
        raise RunnerError(f"run directory already exists: {run_dir}")
    if run_dir.exists():
        manifest = _read_json(manifest_path)
        old_plan = manifest.get("run_plan", {})
        if old_plan.get("config_sha256") != plan["config_sha256"]:
            raise RunnerError("resume config hash does not match existing run")
        if [item["attempt_id"] for item in old_plan.get("attempts", [])] != [
            item["attempt_id"] for item in plan["attempts"]
        ]:
            raise RunnerError("resume attempt list does not match existing run")
    else:
        run_dir.mkdir(parents=True)
        shutil.copy2(plan["config_path"], run_dir / "baseline_eval_config.json")
        manifest = {
            "run_manifest_schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "run_plan": plan,
            "attempts": [],
            "hardware_preflight": {
                "host": socket.gethostname(),
                "gpu_memory_used_mb": {
                    str(index): initial_gpu_usage[index] for index in selected_gpu_indices
                },
            },
            "summary": {},
        }
        _write_json_atomic(manifest_path, manifest)

    completed = {item["attempt_id"]: item for item in manifest.get("attempts", [])}
    for attempt in plan["attempts"]:
        previous = completed.get(attempt["attempt_id"])
        if previous and previous.get("pipeline_valid"):
            continue
        attempt_dir = run_dir / "attempts" / attempt["attempt_id"]
        if attempt_dir.exists():
            raise RunnerError(
                f"incomplete attempt directory requires manual review: {attempt_dir}"
            )
        result = _execute_attempt(repo_root, plan, run_dir, attempt)
        completed[attempt["attempt_id"]] = result
        manifest["attempts"] = [
            completed[item["attempt_id"]]
            for item in plan["attempts"]
            if item["attempt_id"] in completed
        ]
        valid_count = sum(item.get("pipeline_valid", False) for item in completed.values())
        manifest["summary"] = {
            "planned_attempts": len(plan["attempts"]),
            "recorded_attempts": len(completed),
            "pipeline_valid_attempts": valid_count,
            "pipeline_invalid_attempts": len(completed) - valid_count,
        }
        manifest["updated_at"] = _utc_now()
        _write_json_atomic(manifest_path, manifest)
        if result.get("cleanup_error"):
            raise RunnerError(
                f"{attempt['attempt_id']} cleanup failed: {result['cleanup_error']}"
            )

    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Plan or execute a config-driven thesis baseline evaluation"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--route-set", default="development_d7")
    parser.add_argument("--route-id", type=int, action="append", dest="route_ids")
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--agent-gpu", type=int)
    parser.add_argument("--carla-graphics-adapter", type=int)
    parser.add_argument("--timeout-seconds", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    try:
        plan = build_run_plan(
            config_path=args.config,
            repo_root=args.repo_root,
            run_id=args.run_id,
            route_set=args.route_set,
            route_ids=args.route_ids,
            seeds=args.seeds,
            agent_gpu=args.agent_gpu,
            carla_graphics_adapter=args.carla_graphics_adapter,
            timeout_seconds=args.timeout_seconds,
        )
        if not args.execute:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        manifest = execute_run_plan(plan, repo_root=args.repo_root, resume=args.resume)
        print(json.dumps(manifest["summary"], indent=2, sort_keys=True))
        return 0 if manifest["summary"]["pipeline_invalid_attempts"] == 0 else 1
    except RunnerError as exc:
        print(f"runner error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
