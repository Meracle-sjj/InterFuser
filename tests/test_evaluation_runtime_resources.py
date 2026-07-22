"""
[INPUT]: 依赖 tools.evaluation.runtime_resources 的进程组回收与 GPU 释放等待 API，并使用独立 POSIX session 构造忽略 SIGTERM 的子进程。
[OUTPUT]: 提供 CARLA 包装进程先退出时仍能清理整个进程组、CUDA 显存延迟归零的生命周期回归测试。
[POS]: tests 的 M0 外部资源回收测试，复现真实 runner 中 shell leader 退出但 CARLA binary 继续存活的故障。
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
    wait_for_gpus_available,
)


class EvaluationRuntimeResourceTests(unittest.TestCase):
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
