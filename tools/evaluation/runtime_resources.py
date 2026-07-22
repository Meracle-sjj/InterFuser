#!/usr/bin/env python3
"""
[INPUT]: 依赖 POSIX 进程组、TCP socket、nvidia-smi GPU UUID/计算进程与 Python subprocess/threading，观察 runner 独占的端口和 GPU。
[OUTPUT]: 对外提供 RunnerError、端口/GPU 门禁与释放等待，拒绝任何已有 GPU 计算 owner，并提供 GPU 峰值监控和完整进程组回收能力。
[POS]: tools/evaluation 的底层运行时资源守卫，将外部进程生命周期、外来 GPU owner 与硬件归零从实验编排中隔离。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import os
import signal
import socket
import subprocess
import threading
import time


PORT_RELEASE_TIMEOUT_SECONDS = 60
PORT_RELEASE_POLL_SECONDS = 0.5
GPU_RELEASE_TIMEOUT_SECONDS = 60
GPU_RELEASE_POLL_SECONDS = 1.0
PROCESS_GROUP_POLL_SECONDS = 0.25


class RunnerError(RuntimeError):
    """Raised when a run cannot proceed without ambiguous resource ownership."""


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


def _gpu_compute_processes():
    gpu_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if gpu_result.returncode != 0:
        raise RunnerError(f"unable to query GPU UUIDs: {gpu_result.stderr.strip()}")
    uuid_to_index = {}
    for line in gpu_result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",", 1)]
        if len(fields) == 2:
            uuid_to_index[fields[1]] = int(fields[0])

    process_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if process_result.returncode != 0:
        raise RunnerError(
            f"unable to query GPU compute processes: {process_result.stderr.strip()}"
        )
    processes = {}
    for line in process_result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",", 3)]
        if len(fields) != 4 or fields[0] not in uuid_to_index:
            continue
        index = uuid_to_index[fields[0]]
        processes.setdefault(index, []).append(
            {
                "pid": fields[1],
                "process_name": fields[2],
                "used_memory_mb": fields[3],
            }
        )
    return processes


def ensure_gpus_available(indices, threshold_mb):
    usage = _gpu_memory_usage()
    compute_processes = _gpu_compute_processes()
    failures = []
    for index in sorted(set(indices)):
        if index not in usage:
            failures.append(f"GPU {index} is unavailable")
        elif usage[index] > threshold_mb:
            failures.append(
                f"GPU {index} uses {usage[index]} MiB, above {threshold_mb} MiB"
            )
        owners = compute_processes.get(index, [])
        if owners:
            details = ", ".join(
                f"{item['pid']} ({item['process_name']}, "
                f"{item['used_memory_mb']} MiB)"
                for item in owners
            )
            failures.append(f"GPU {index} has active compute processes: {details}")
    if failures:
        raise RunnerError("; ".join(failures))
    return usage


def wait_for_gpus_available(
    indices,
    threshold_mb,
    timeout_seconds=GPU_RELEASE_TIMEOUT_SECONDS,
    poll_seconds=GPU_RELEASE_POLL_SECONDS,
):
    """Wait for runner-owned CUDA contexts to release their memory."""
    started = time.monotonic()
    deadline = started + max(0.0, float(timeout_seconds))
    while True:
        try:
            ensure_gpus_available(indices, threshold_mb)
            return round(time.monotonic() - started, 3)
        except RunnerError as exc:
            last_error = exc
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RunnerError(f"GPUs did not become available: {last_error}")
        time.sleep(min(float(poll_seconds), remaining))


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


def _process_group_exists(process_group_id):
    try:
        os.killpg(process_group_id, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _wait_for_process_group_exit(process, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    while True:
        process.poll()  # Reap an exited group leader while children are still stopping.
        if not _process_group_exists(process.pid):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(PROCESS_GROUP_POLL_SECONDS, remaining))


def _stop_process_group(process, grace_seconds=20):
    """Stop every member of a start_new_session process group, not only its leader."""
    if process is None:
        return None
    if _process_group_exists(process.pid):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if not _wait_for_process_group_exit(process, grace_seconds):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if not _wait_for_process_group_exit(process, 10):
                raise RunnerError(f"process group {process.pid} did not exit")
    process.poll()
    return process.returncode
