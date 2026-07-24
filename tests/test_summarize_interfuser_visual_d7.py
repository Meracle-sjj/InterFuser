"""
[INPUT]: 依赖 summarize_interfuser_visual_d7 的双组 M0 汇总复用、checkpoint-only 配置差异、冻结 test 方向票和禁止覆盖输出 API。
[OUTPUT]: 验证 21×2 完整矩阵、固定 attempt 顺序、单变量契约、配对差值与 H1 预注册分类。
[POS]: tests 的 M2 H1 最终归约纯文件回归；使用合成 manifest，不启动 CARLA、GPU 或训练进程。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.evaluation.summarize_interfuser_visual_d7 import (
    PairedSummaryError,
    ROUTE_ORDER,
    SEEDS,
    build_paired_summary,
    write_paired_summary,
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class InterfuserVisualD7SummaryTests(unittest.TestCase):
    def _write_json(self, path, value):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(value), encoding="utf-8")

    def _fixture(self, root, driving_delta=1.0):
        root = Path(root)
        checkpoints = {}
        for variant in ("b0", "v"):
            checkpoint = root / f"{variant}.pth"
            checkpoint.write_bytes(variant.encode("ascii"))
            checkpoints[variant] = checkpoint

        formal_manifest = root / "formal" / "run_manifest.json"
        self._write_json(
            formal_manifest,
            {
                "status": "completed",
                "pipeline_valid": True,
                "variants": [
                    {
                        "variant": variant,
                        "artifacts": {
                            "best_checkpoint": {
                                "sha256": _sha256(checkpoints[variant])
                            }
                        },
                    }
                    for variant in ("b0", "v")
                ],
            },
        )
        formal_hash = _sha256(formal_manifest)

        test_manifest = root / "test" / "test_manifest.json"
        test_value = {
            "status": "completed",
            "pipeline_valid": True,
            "formal_training_manifest_sha256": formal_hash,
            "variants": [
                {
                    "variant": variant,
                    "pipeline_valid": True,
                    "worker_result": {
                        "checkpoint_sha256": _sha256(checkpoints[variant])
                    },
                }
                for variant in ("b0", "v")
            ],
            "v_minus_b0": {
                "traffic_average_precision": 0.1,
                "traffic_roc_auc": 0.1,
                "traffic_occupied_iou": 0.1,
                "waypoint_ade": -0.1,
                "waypoint_fde_horizon_10": 0.1,
            },
        }
        self._write_json(test_manifest, test_value)
        test_hash = _sha256(test_manifest)

        manifests = {}
        for variant in ("b0", "v"):
            run_dir = root / f"d7-{variant}"
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
            config = {
                "checkpoint": {
                    "path": str(checkpoints[variant]),
                    "sha256": _sha256(checkpoints[variant]),
                    "architecture": "interfuser_baseline",
                    "epoch": 10 if variant == "b0" else 11,
                },
                "inputs": inputs,
                "route_sets": {
                    "development_d7": [0, 6, 12, 18, 30, 36, 39]
                },
                "random_seeds": list(SEEDS),
                "runtime": {"carla_port": 2155, "traffic_manager_port": 2255},
                "environment": {"PYTHONOPTIMIZE": "1"},
                "comparison": {
                    "schema_version": 1,
                    "variant": variant,
                    "formal_checkpoint_sha256": _sha256(checkpoints[variant]),
                    "formal_training_manifest": str(formal_manifest),
                    "formal_training_manifest_sha256": formal_hash,
                    "visual_test_manifest": str(test_manifest),
                    "visual_test_manifest_sha256": test_hash,
                },
            }
            config_path = run_dir / "baseline_eval_config.json"
            self._write_json(config_path, config)

            attempts = []
            offset = 0.0 if variant == "b0" else driving_delta
            for route_id in ROUTE_ORDER:
                for seed in SEEDS:
                    attempts.append(
                        {
                            "attempt_id": f"route_{route_id:02d}_seed_{seed}",
                            "route_id": route_id,
                            "traffic_manager_seed": seed,
                            "pipeline_valid": True,
                            "process_exit_code": 0,
                            "cleanup_error": None,
                            "duration_seconds": 10.0,
                            "port_release_wait_seconds": 0.0,
                            "gpu_release_wait_seconds": 0.1,
                            "gpu_peak_memory_mb": {"6": 100, "7": 200},
                            "leaderboard_result": {
                                "valid": True,
                                "status": "Completed",
                                "scores": {
                                    "score_composed": route_id + seed + offset,
                                    "score_route": route_id + seed + 10 + offset,
                                    "score_penalty": 1.0 + offset,
                                },
                                "infraction_counts": {"red_light": seed},
                            },
                        }
                    )
            manifest = {
                "run_plan": {
                    "run_id": f"d7-{variant}",
                    "route_set": "development_d7",
                    "git_head": "git-head",
                    "code_anchor": "code-anchor",
                    "runner_sha256": "runner-hash",
                    "config_sha256": _sha256(config_path),
                    "checkpoint_path": str(checkpoints[variant]),
                    "runtime": config["runtime"],
                    "environment": config["environment"],
                },
                "attempts": attempts,
                "summary": {
                    "planned_attempts": 21,
                    "recorded_attempts": 21,
                    "pipeline_valid_attempts": 21,
                    "pipeline_invalid_attempts": 0,
                },
            }
            manifest_path = run_dir / "run_manifest.json"
            self._write_json(manifest_path, manifest)
            manifests[variant] = manifest_path
        return manifests

    def test_builds_supported_h1_from_complete_paired_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            manifests = self._fixture(root)
            summary = build_paired_summary(
                manifests["b0"], manifests["v"], repo_root=root
            )

        self.assertTrue(summary["valid"])
        self.assertEqual(len(summary["paired_d7"]["per_attempt"]), 21)
        driving = summary["paired_d7"]["aggregate"]["driving_score"]
        self.assertEqual(driving["macro_mean_v_minus_b0"], 1.0)
        self.assertEqual(driving["improved_pairs"], 21)
        self.assertEqual(
            summary["offline_test"]["decision_metrics"]["improved_count"], 4
        )
        self.assertEqual(summary["h1"]["classification"], "supported")
        self.assertFalse(summary["h1"]["noninferiority_claim_allowed"])

    def test_rejects_attempt_order_drift(self):
        with tempfile.TemporaryDirectory() as root:
            manifests = self._fixture(root)
            value = json.loads(manifests["v"].read_text(encoding="utf-8"))
            value["attempts"][0], value["attempts"][1] = (
                value["attempts"][1],
                value["attempts"][0],
            )
            self._write_json(manifests["v"], value)
            with self.assertRaisesRegex(PairedSummaryError, "attempt order"):
                build_paired_summary(
                    manifests["b0"], manifests["v"], repo_root=root
                )

    def test_rejects_non_checkpoint_config_drift(self):
        with tempfile.TemporaryDirectory() as root:
            manifests = self._fixture(root)
            config_path = manifests["v"].parent / "baseline_eval_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["runtime"]["carla_port"] = 2999
            self._write_json(config_path, config)
            manifest = json.loads(manifests["v"].read_text(encoding="utf-8"))
            manifest["run_plan"]["config_sha256"] = _sha256(config_path)
            manifest["run_plan"]["runtime"] = config["runtime"]
            self._write_json(manifests["v"], manifest)
            with self.assertRaisesRegex(PairedSummaryError, "configs differ"):
                build_paired_summary(
                    manifests["b0"], manifests["v"], repo_root=root
                )

    def test_rejects_pipeline_invalid_child(self):
        with tempfile.TemporaryDirectory() as root:
            manifests = self._fixture(root)
            value = json.loads(manifests["v"].read_text(encoding="utf-8"))
            value["attempts"][-1]["pipeline_valid"] = False
            value["summary"]["pipeline_valid_attempts"] = 20
            value["summary"]["pipeline_invalid_attempts"] = 1
            self._write_json(manifests["v"], value)
            with self.assertRaisesRegex(PairedSummaryError, "pipeline-invalid"):
                build_paired_summary(
                    manifests["b0"], manifests["v"], repo_root=root
                )

    def test_rejects_run_plan_checkpoint_drift(self):
        with tempfile.TemporaryDirectory() as root:
            manifests = self._fixture(root)
            value = json.loads(manifests["v"].read_text(encoding="utf-8"))
            value["run_plan"]["checkpoint_path"] = str(Path(root) / "b0.pth")
            self._write_json(manifests["v"], value)
            with self.assertRaisesRegex(PairedSummaryError, "run plan checkpoint"):
                build_paired_summary(
                    manifests["b0"], manifests["v"], repo_root=root
                )

    def test_output_is_deterministic_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            manifests = self._fixture(root)
            first = build_paired_summary(
                manifests["b0"], manifests["v"], repo_root=root
            )
            second = build_paired_summary(
                manifests["b0"], manifests["v"], repo_root=root
            )
            self.assertEqual(first, second)
            output = Path(root) / "paired.json"
            write_paired_summary(first, output)
            with self.assertRaisesRegex(PairedSummaryError, "overwrite"):
                write_paired_summary(second, output)


if __name__ == "__main__":
    unittest.main()
