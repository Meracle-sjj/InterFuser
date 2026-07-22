"""
[INPUT]: 依赖 tools.evaluation.summarize_thesis_baseline 的契约校验、矩阵完整性、指标归约与原子式禁止覆盖输出 API。
[OUTPUT]: 提供 M0 三种子汇总对完整矩阵、pipeline-invalid、输入漂移和确定性输出的纯离线回归测试。
[POS]: tests 的闭环结果汇总门禁，保证缺失或不可比运行不会进入论文统计。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.evaluation.summarize_thesis_baseline import (
    SummaryError,
    build_summary,
    write_summary,
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ThesisBaselineSummaryTests(unittest.TestCase):
    def _write_run(self, root, run_id, seeds, evaluator_hash="eval-a", invalid=False):
        run_dir = Path(root) / run_id
        run_dir.mkdir()
        inputs = {
            name: {"path": name, "sha256": f"hash-{name}"}
            for name in (
                "routes",
                "scenarios",
                "agent",
                "agent_config",
                "controller",
                "model_definition",
                "leaderboard_evaluator",
                "leaderboard_route_scenario",
                "scenario_runner_route_scenario",
            )
        }
        inputs["leaderboard_evaluator"]["sha256"] = evaluator_hash
        config = {
            "checkpoint": {"path": "model.tar", "sha256": "checkpoint"},
            "inputs": inputs,
            "route_sets": {"development_d7": [0, 6]},
            "random_seeds": [0, 1, 2],
        }
        config_path = run_dir / "baseline_eval_config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        attempts = []
        for route_id in [0, 6]:
            for seed in seeds:
                valid = not invalid or (route_id, seed) != (6, seeds[-1])
                attempts.append(
                    {
                        "attempt_id": f"route_{route_id:02d}_seed_{seed}",
                        "route_id": route_id,
                        "traffic_manager_seed": seed,
                        "pipeline_valid": valid,
                        "process_exit_code": 0 if valid else -6,
                        "cleanup_error": None,
                        "duration_seconds": 10 + seed,
                        "port_release_wait_seconds": 0.0,
                        "gpu_release_wait_seconds": 0.2,
                        "gpu_peak_memory_mb": {"6": 100 + seed, "7": 200 + seed},
                        "leaderboard_result": {
                            "valid": valid,
                            "status": "Completed" if valid else "Crashed",
                            "scores": {
                                "score_composed": route_id + seed,
                                "score_route": route_id + seed + 10,
                                "score_penalty": 1.0,
                            },
                            "infraction_counts": {"red_light": seed},
                        },
                    }
                )
        valid_count = sum(item["pipeline_valid"] for item in attempts)
        manifest = {
            "run_plan": {
                "run_id": run_id,
                "route_set": "development_d7",
                "git_head": f"git-{run_id}",
                "code_anchor": f"anchor-{run_id}",
                "runner_sha256": f"runner-{run_id}",
                "config_sha256": _sha256(config_path),
                "runtime": {"carla_port": 2155},
                "environment": {"PYTHONOPTIMIZE": "1"},
            },
            "attempts": attempts,
            "summary": {
                "planned_attempts": len(attempts),
                "recorded_attempts": len(attempts),
                "pipeline_valid_attempts": valid_count,
                "pipeline_invalid_attempts": len(attempts) - valid_count,
            },
        }
        manifest_path = run_dir / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path

    def test_summarizes_complete_matrix_with_route_then_seed_reduction(self):
        with tempfile.TemporaryDirectory() as root:
            seed0 = self._write_run(root, "seed0", [0])
            seeds12 = self._write_run(root, "seeds12", [1, 2])

            summary = build_summary([seed0, seeds12])

        self.assertTrue(summary["valid"])
        self.assertEqual(summary["attempt_count"], 6)
        self.assertEqual(summary["aggregate"]["driving_score"]["mean"], 4.0)
        self.assertAlmostEqual(
            summary["aggregate"]["driving_score"][
                "population_stddev_across_seed_macro_means"
            ],
            0.816496580927726,
        )
        self.assertEqual(summary["infraction_counts"]["red_light"], 6)
        self.assertEqual(summary["resources"]["gpu_peak_memory_mb"], {"6": 102, "7": 202})

    def test_rejects_missing_seed(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = self._write_run(root, "seed0", [0])
            with self.assertRaisesRegex(SummaryError, "matrix mismatch"):
                build_summary([manifest])

    def test_rejects_pipeline_invalid_run(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = self._write_run(root, "all", [0, 1, 2], invalid=True)
            with self.assertRaisesRegex(SummaryError, "pipeline-invalid"):
                build_summary([manifest])

    def test_requires_explicit_approval_for_input_hash_drift(self):
        with tempfile.TemporaryDirectory() as root:
            seed0 = self._write_run(root, "seed0", [0], evaluator_hash="eval-a")
            seeds12 = self._write_run(root, "seeds12", [1, 2], evaluator_hash="eval-b")
            with self.assertRaisesRegex(SummaryError, "leaderboard_evaluator"):
                build_summary([seed0, seeds12])

            summary = build_summary(
                [seed0, seeds12], allowed_input_drift=["leaderboard_evaluator"]
            )
            reversed_summary = build_summary(
                [seeds12, seed0], allowed_input_drift=["leaderboard_evaluator"]
            )

        self.assertEqual(summary, reversed_summary)
        self.assertEqual(summary["allowed_input_drift"], ["leaderboard_evaluator"])
        self.assertEqual(
            summary["input_hash_variants"]["leaderboard_evaluator"],
            ["eval-a", "eval-b"],
        )

    def test_output_is_deterministic_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = self._write_run(root, "all", [0, 1, 2])
            first = build_summary([manifest])
            second = build_summary([manifest])
            self.assertEqual(first, second)
            output = Path(root) / "summary.json"
            write_summary(first, output)
            with self.assertRaisesRegex(SummaryError, "overwrite"):
                write_summary(second, output)


if __name__ == "__main__":
    unittest.main()
