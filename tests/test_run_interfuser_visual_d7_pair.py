"""
[INPUT]: 依赖 run_interfuser_visual_d7_pair 的冻结 test/child config 契约、固定计划构造和父级串行执行 API。
[OUTPUT]: 验证 test 准入、B0→V 顺序、42/42 完成门槛、失败即停、目录幂等与非 checkpoint 配置漂移拒绝。
[POS]: tests 的 M2 H1 D7 父级 runner 纯文件/模拟进程回归；mock 单组 runner，不启动 CARLA 或 GPU。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.evaluation.run_interfuser_visual_d7_pair import (
    D7PairError,
    ROUTE_ORDER,
    SEEDS,
    build_d7_pair_plans,
    execute_d7_pair,
    load_d7_pair_contract,
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RunInterfuserVisualD7PairTests(unittest.TestCase):
    def _write_json(self, path, value):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(value), encoding="utf-8")

    def _fixture(self, root, test_valid=True):
        root = Path(root)
        result_root = root / "results"
        checkpoints = {}
        for variant in ("b0", "v"):
            checkpoint = root / f"{variant}.pth"
            checkpoint.write_bytes(variant.encode("ascii"))
            checkpoints[variant] = checkpoint

        test_manifest = root / "test_manifest.json"
        self._write_json(
            test_manifest,
            {
                "status": "completed" if test_valid else "running",
                "pipeline_valid": test_valid,
                "variants": [
                    {
                        "variant": variant,
                        "pipeline_valid": test_valid,
                        "worker_result": {
                            "checkpoint_sha256": _sha256(checkpoints[variant])
                        },
                    }
                    for variant in ("b0", "v")
                ],
            },
        )
        test_hash = _sha256(test_manifest)

        child_specs = {}
        for variant in ("b0", "v"):
            config = {
                "schema_version": 1,
                "status": "frozen",
                "runtime": {
                    "carla_port": 2155,
                    "traffic_manager_port": 2255,
                },
                "route_sets": {
                    "development_d7": [0, 6, 12, 18, 30, 36, 39]
                },
                "random_seeds": list(SEEDS),
                "result_root": str(result_root),
                "checkpoint": {
                    "path": str(checkpoints[variant]),
                    "sha256": _sha256(checkpoints[variant]),
                    "architecture": "interfuser_baseline",
                    "epoch": 10 if variant == "b0" else 11,
                },
                "comparison": {
                    "schema_version": 1,
                    "variant": variant,
                    "formal_checkpoint_sha256": _sha256(checkpoints[variant]),
                    "formal_training_manifest": str(root / "formal.json"),
                    "formal_training_manifest_sha256": "f" * 64,
                    "visual_test_manifest": str(test_manifest),
                    "visual_test_manifest_sha256": test_hash,
                },
            }
            config_path = root / f"d7-{variant}.json"
            self._write_json(config_path, config)
            child_specs[variant] = {
                "run_id": f"d7-{variant}-run",
                "config": str(config_path),
                "config_sha256": _sha256(config_path),
            }

        pair_config = root / "pair.json"
        self._write_json(
            pair_config,
            {
                "schema_version": 1,
                "status": "preregistered",
                "pair_run_id": "d7-pair-run",
                "visual_test_manifest": str(test_manifest),
                "visual_test_manifest_sha256": test_hash,
                "result_root": str(result_root),
                "variants": child_specs,
            },
        )
        return pair_config, result_root

    @staticmethod
    def _fake_build_run_plan(**kwargs):
        config_path = Path(kwargs["config_path"])
        config = json.loads(config_path.read_text(encoding="utf-8"))
        result_root = Path(config["result_root"])
        attempts = [
            {"route_id": route_id, "traffic_manager_seed": seed}
            for route_id in kwargs["route_ids"]
            for seed in kwargs["seeds"]
        ]
        return {
            "run_id": kwargs["run_id"],
            "config_sha256": _sha256(config_path),
            "checkpoint_path": config["checkpoint"]["path"],
            "result_root": str(result_root),
            "run_directory": str(result_root / kwargs["run_id"]),
            "attempts": attempts,
        }

    def test_incomplete_visual_test_blocks_pair(self):
        with tempfile.TemporaryDirectory() as root:
            config, _ = self._fixture(root, test_valid=False)
            with self.assertRaisesRegex(D7PairError, "visual test must be completed"):
                load_d7_pair_contract(config, repo_root=root)

    def test_preflight_builds_fixed_b0_then_v_plans(self):
        with tempfile.TemporaryDirectory() as root:
            config, _ = self._fixture(root)
            contract = load_d7_pair_contract(config, repo_root=root)
            with mock.patch(
                "tools.evaluation.run_interfuser_visual_d7_pair.build_run_plan",
                side_effect=self._fake_build_run_plan,
            ) as build:
                plans = build_d7_pair_plans(contract)

        self.assertEqual(list(plans), ["b0", "v"])
        self.assertEqual(build.call_count, 2)
        self.assertEqual(
            [
                (item["route_id"], item["traffic_manager_seed"])
                for item in plans["b0"]["attempts"]
            ],
            [
                (route_id, seed)
                for route_id in ROUTE_ORDER
                for seed in SEEDS
            ],
        )

    def test_execute_runs_b0_then_v_and_records_42_valid_attempts(self):
        with tempfile.TemporaryDirectory() as root:
            config, result_root = self._fixture(root)
            execution_order = []

            def execute(plan, repo_root, resume):
                execution_order.append(plan["run_id"])
                run_dir = Path(plan["run_directory"])
                run_dir.mkdir(parents=True)
                manifest = {
                    "summary": {
                        "planned_attempts": 21,
                        "recorded_attempts": 21,
                        "pipeline_valid_attempts": 21,
                        "pipeline_invalid_attempts": 0,
                    }
                }
                self._write_json(run_dir / "run_manifest.json", manifest)
                return manifest

            with mock.patch(
                "tools.evaluation.run_interfuser_visual_d7_pair.build_run_plan",
                side_effect=self._fake_build_run_plan,
            ), mock.patch(
                "tools.evaluation.run_interfuser_visual_d7_pair.execute_run_plan",
                side_effect=execute,
            ):
                manifest = execute_d7_pair(
                    config, "d7-pair-run", repo_root=root
                )
            self.assertTrue(
                (result_root / "d7-pair-run" / "pair_manifest.json").is_file()
            )

            self.assertEqual(execution_order, ["d7-b0-run", "d7-v-run"])
            self.assertEqual(manifest["status"], "completed")
            self.assertTrue(manifest["pipeline_valid"])
            self.assertEqual(
                manifest["comparability"]["total_pipeline_valid_attempts"], 42
            )

    def test_invalid_b0_stops_before_v(self):
        with tempfile.TemporaryDirectory() as root:
            config, result_root = self._fixture(root)
            execution_order = []

            def execute(plan, repo_root, resume):
                execution_order.append(plan["run_id"])
                run_dir = Path(plan["run_directory"])
                run_dir.mkdir(parents=True)
                manifest = {
                    "summary": {
                        "planned_attempts": 21,
                        "recorded_attempts": 1,
                        "pipeline_valid_attempts": 0,
                        "pipeline_invalid_attempts": 1,
                    }
                }
                self._write_json(run_dir / "run_manifest.json", manifest)
                return manifest

            with mock.patch(
                "tools.evaluation.run_interfuser_visual_d7_pair.build_run_plan",
                side_effect=self._fake_build_run_plan,
            ), mock.patch(
                "tools.evaluation.run_interfuser_visual_d7_pair.execute_run_plan",
                side_effect=execute,
            ):
                with self.assertRaisesRegex(D7PairError, "recorded attempts"):
                    execute_d7_pair(config, "d7-pair-run", repo_root=root)

            parent = json.loads(
                (result_root / "d7-pair-run" / "pair_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(execution_order, ["d7-b0-run"])
        self.assertEqual(parent["status"], "failed")
        self.assertFalse(parent["pipeline_valid"])

    def test_existing_pair_directory_requires_resume(self):
        with tempfile.TemporaryDirectory() as root:
            config, result_root = self._fixture(root)
            (result_root / "d7-pair-run").mkdir(parents=True)
            with mock.patch(
                "tools.evaluation.run_interfuser_visual_d7_pair.build_run_plan",
                side_effect=self._fake_build_run_plan,
            ):
                with self.assertRaisesRegex(D7PairError, "already exists"):
                    execute_d7_pair(config, "d7-pair-run", repo_root=root)

    def test_non_checkpoint_child_config_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            config, _ = self._fixture(root)
            pair = json.loads(config.read_text(encoding="utf-8"))
            v_config_path = Path(pair["variants"]["v"]["config"])
            v_config = json.loads(v_config_path.read_text(encoding="utf-8"))
            v_config["runtime"]["carla_port"] = 2999
            self._write_json(v_config_path, v_config)
            pair["variants"]["v"]["config_sha256"] = _sha256(v_config_path)
            self._write_json(config, pair)
            with self.assertRaisesRegex(D7PairError, "configs differ"):
                load_d7_pair_contract(config, repo_root=root)


if __name__ == "__main__":
    unittest.main()
