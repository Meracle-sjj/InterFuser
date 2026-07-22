"""
[INPUT]: 依赖 tools.evaluation.runtime_resources 的进程组回收、GPU 计算 owner 门禁与释放等待 API，并用独立 POSIX session 构造忽略 SIGTERM 的子进程。
[OUTPUT]: 提供外来 GPU 计算进程拒绝、CARLA 包装进程先退出时整组清理、CUDA 显存延迟归零的生命周期回归测试。
[POS]: tests 的 M0 外部资源回收测试，覆盖启动前独占性与运行后完整回收两个资源边界。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import os
import signal
import subprocess
import sys
import unittest
from unittest.mock import patch

from tools.evaluation.runtime_resources import (
    RunnerError,
    _process_group_exists,
    _stop_process_group,
    ensure_gpus_available,
    wait_for_gpus_available,
)


class EvaluationRuntimeResourceTests(unittest.TestCase):
    def test_gpu_gate_rejects_existing_compute_process_below_memory_threshold(self):
        with patch(
            "tools.evaluation.runtime_resources._gpu_memory_usage",
            return_value={6: 81, 7: 706},
        ), patch(
            "tools.evaluation.runtime_resources._gpu_compute_processes",
            return_value={
                7: [
                    {
                        "pid": "1903747",
                        "process_name": "/external/python",
                        "used_memory_mb": "2778",
                    }
                ]
            },
        ):
            with self.assertRaisesRegex(
                RunnerError, "GPU 7 has active compute processes: 1903747"
            ):
                ensure_gpus_available([6, 7], threshold_mb=1024)

    def test_stop_process_group_kills_child_after_leader_exits(self):
        child_code = (
            "import os,signal,time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "print(os.getpid(), flush=True); time.sleep(60)"
        )
        leader_code = (
            "import subprocess,sys,time; "
            f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
            "time.sleep(60)"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", leader_code],
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        child_pid = int(process.stdout.readline().strip())
        try:
            return_code = _stop_process_group(process, grace_seconds=0.05)

            self.assertIsNotNone(return_code)
            self.assertFalse(_process_group_exists(process.pid))
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
            process.stdout.close()

    def test_gpu_release_waits_until_memory_is_below_threshold(self):
        with patch(
            "tools.evaluation.runtime_resources.ensure_gpus_available",
            side_effect=[RunnerError("GPU 7 is busy"), {6: 81, 7: 45}],
        ), patch(
            "tools.evaluation.runtime_resources.time.monotonic",
            side_effect=[0.0, 0.0, 0.2],
        ), patch("tools.evaluation.runtime_resources.time.sleep") as sleep:
            elapsed = wait_for_gpus_available(
                [6, 7], threshold_mb=1024, timeout_seconds=1, poll_seconds=0.1
            )

        self.assertEqual(elapsed, 0.2)
        sleep.assert_called_once_with(0.1)


if __name__ == "__main__":
    unittest.main()
