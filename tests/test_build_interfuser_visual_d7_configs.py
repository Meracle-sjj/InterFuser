"""
[INPUT]: 依赖 build_interfuser_visual_d7_configs 的静态 build 契约、formal/test 双重 checkpoint provenance 和禁止覆盖输出 API。
[OUTPUT]: 验证 test 前只能 preflight、test 有效后三配置确定性生成、checkpoint-only 差异、hash 绑定与 overwrite 门禁。
[POS]: tests 的 M2 H1 D7 配置冻结器纯文件回归；使用小型合成 checkpoint/manifest，不启动模型、CARLA 或 GPU。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.evaluation.build_interfuser_visual_d7_configs import (
    D7ConfigBuildError,
    build_d7_configs,
    load_d7_build_contract,
)
from tools.evaluation.run_interfuser_visual_d7_pair import load_d7_pair_contract
from tools.evaluation.summarize_interfuser_visual_d7 import (
    normalize_d7_config_for_pair,
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class BuildInterfuserVisualD7ConfigsTests(unittest.TestCase):
    def _write_json(self, path, value):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(value), encoding="utf-8")

    def _fixture(self, root, with_results=False, test_valid=True):
        root = Path(root)
        baseline = root / "configs" / "baseline.json"
        self._write_json(
            baseline,
            {
                "schema_version": 1,
                "status": "frozen",
                "code_anchor": "anchor",
                "runtime_code_roots": [
                    "interfuser",
                    "leaderboard",
                    "scenario_runner",
                ],
                "runtime": {
                    "agent_cuda_visible_device": 6,
                    "carla_graphics_adapter": 7,
                    "carla_port": 2155,
                    "traffic_manager_port": 2255,
                },
                "checkpoint": {
                    "path": "/old/model.pth",
                    "sha256": "0" * 64,
                    "architecture": "interfuser_baseline",
                },
                "inputs": {},
                "route_sets": {
                    "development_d7": [0, 6, 12, 18, 30, 36, 39]
                },
                "random_seeds": [0, 1, 2],
                "result_root": "results/old",
            },
        )
        formal_config = root / "configs" / "formal.json"
        formal_run_id = "formal-run"
        self._write_json(formal_config, {"status": "formal"})
        test_config = root / "configs" / "test.json"
        test_run_id = "test-run"
        self._write_json(
            test_config,
            {
                "run_id": test_run_id,
                "training_run_id": formal_run_id,
                "training_config_sha256": _sha256(formal_config),
            },
        )

        result_root = root / "results" / "thesis_m2"
        build_config = root / "configs" / "build.json"
        self._write_json(
            build_config,
            {
                "schema_version": 1,
                "status": "preregistered",
                "baseline_template": str(baseline),
                "baseline_template_sha256": _sha256(baseline),
                "formal_training_config": str(formal_config),
                "formal_training_config_sha256": _sha256(formal_config),
                "formal_training_run_id": formal_run_id,
                "visual_test_config": str(test_config),
                "visual_test_config_sha256": _sha256(test_config),
                "visual_test_run_id": test_run_id,
                "visual_test_manifest": str(
                    result_root / test_run_id / "test_manifest.json"
                ),
                "route_order": [18, 6, 12, 30, 36, 39, 0],
                "seeds": [0, 1, 2],
                "result_root": str(result_root),
                "run_ids": {
                    "pair": "pair-run",
                    "b0": "b0-run",
                    "v": "v-run",
                },
                "output_configs": {
                    "b0": str(root / "configs" / "d7-b0.json"),
                    "v": str(root / "configs" / "d7-v.json"),
                    "pair": str(root / "configs" / "d7-pair.json"),
                },
            },
        )

        if with_results:
            checkpoints = {}
            for variant in ("b0", "v"):
                checkpoint = root / "artifacts" / f"{variant}.pth"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(variant.encode("ascii"))
                checkpoints[variant] = checkpoint
            formal_manifest = result_root / formal_run_id / "run_manifest.json"
            self._write_json(
                formal_manifest,
                {
                    "run_id": formal_run_id,
                    "status": "completed",
                    "pipeline_valid": True,
                    "config_sha256": _sha256(formal_config),
                    "variants": [
                        {
                            "variant": variant,
                            "pipeline_valid": True,
                            "state_tensors": 1132,
                            "errors": [],
                            "best_epoch": 10 if variant == "b0" else 11,
                            "best_metric": 0.5 if variant == "b0" else 0.4,
                            "artifacts": {
                                "best_checkpoint": {
                                    "path": str(checkpoints[variant]),
                                    "sha256": _sha256(checkpoints[variant]),
                                }
                            },
                        }
                        for variant in ("b0", "v")
                    ],
                },
            )
            test_manifest = result_root / test_run_id / "test_manifest.json"
            self._write_json(
                test_manifest,
                {
                    "run_id": test_run_id,
                    "status": "completed" if test_valid else "running",
                    "pipeline_valid": test_valid,
                    "config_sha256": _sha256(test_config),
                    "formal_training_manifest": str(formal_manifest),
                    "formal_training_manifest_sha256": _sha256(formal_manifest),
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
        return build_config

    def test_static_preflight_succeeds_before_test_manifest_exists(self):
        with tempfile.TemporaryDirectory() as root:
            config = self._fixture(root)
            contract = load_d7_build_contract(config, repo_root=root)
        self.assertEqual(contract["raw"]["run_ids"]["pair"], "pair-run")
        self.assertFalse(contract["test_manifest_path"].exists())

    def test_build_requires_visual_test_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            config = self._fixture(root)
            with self.assertRaisesRegex(D7ConfigBuildError, "manifest is absent"):
                build_d7_configs(config, repo_root=root)

    def test_builds_three_hash_bound_configs_after_valid_test(self):
        with tempfile.TemporaryDirectory() as root:
            config = self._fixture(root, with_results=True)
            result = build_d7_configs(config, repo_root=root)
            contract = load_d7_build_contract(config, repo_root=root)
            b0 = json.loads(
                contract["output_paths"]["b0"].read_text(encoding="utf-8")
            )
            v = json.loads(
                contract["output_paths"]["v"].read_text(encoding="utf-8")
            )
            pair = json.loads(
                contract["output_paths"]["pair"].read_text(encoding="utf-8")
            )
            pair_contract = load_d7_pair_contract(
                contract["output_paths"]["pair"], repo_root=root
            )

        self.assertNotEqual(b0["checkpoint"]["sha256"], v["checkpoint"]["sha256"])
        self.assertEqual(
            normalize_d7_config_for_pair(b0), normalize_d7_config_for_pair(v)
        )
        self.assertEqual(
            pair["variants"]["b0"]["config_sha256"],
            result["outputs"]["b0"]["sha256"],
        )
        self.assertEqual(pair_contract["pair_run_id"], "pair-run")

    def test_incomplete_visual_test_blocks_build(self):
        with tempfile.TemporaryDirectory() as root:
            config = self._fixture(root, with_results=True, test_valid=False)
            with self.assertRaisesRegex(D7ConfigBuildError, "not completed"):
                build_d7_configs(config, repo_root=root)

    def test_refuses_to_overwrite_generated_configs(self):
        with tempfile.TemporaryDirectory() as root:
            config = self._fixture(root, with_results=True)
            build_d7_configs(config, repo_root=root)
            with self.assertRaisesRegex(D7ConfigBuildError, "overwrite"):
                build_d7_configs(config, repo_root=root)

    def test_route_order_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            config = self._fixture(root)
            value = json.loads(config.read_text(encoding="utf-8"))
            value["route_order"] = sorted(value["route_order"])
            self._write_json(config, value)
            with self.assertRaisesRegex(D7ConfigBuildError, "route_order"):
                load_d7_build_contract(config, repo_root=root)


if __name__ == "__main__":
    unittest.main()
