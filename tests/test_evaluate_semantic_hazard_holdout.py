"""
[INPUT]: 依赖 tools.training.evaluate_semantic_hazard_holdout 的专项 holdout 投影与路由组隔离契约。
[OUTPUT]: 提供 validation/test 样本数归约、训练契约不可变性与 holdout 泄漏拒绝的回归测试。
[POS]: tests 的 M2 行人危险专项评估测试，保证定向难例不会反向污染模型训练或 checkpoint 选择。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from tools.training.evaluate_semantic_hazard_holdout import (
    HazardHoldoutEvaluationError,
    prepare_hazard_holdout_contract,
)


class HazardHoldoutEvaluationTests(unittest.TestCase):
    def _fixture(self, root):
        root = Path(root)
        dataset = root / "dataset"
        dataset.mkdir()
        contract = {
            "class_config_sha256": "a" * 64,
            "split_manifest_loaded": {"dataset_root": str(dataset)},
            "data": {
                "cameras": ["front", "left", "right"],
                "max_validation_samples": 5,
                "expected_available_validation_samples": 5,
            },
        }
        manifest = {
            "valid": True,
            "dataset_root": str(dataset),
            "cameras": ["front", "left", "right"],
            "source": {"class_config_sha256": "a" * 64},
            "hazard_holdout": {
                "validation": [
                    {
                        "path": "town01/route1/sequence",
                        "route_group": "Town01:route001",
                        "declared_frames": 2,
                        "split": "hazard_validation",
                    }
                ],
                "test": [
                    {
                        "path": "town03/route2/sequence",
                        "route_group": "Town03:route002",
                        "declared_frames": 3,
                        "split": "hazard_test",
                    }
                ],
            },
        }
        path = root / "hazard.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return contract, path

    def test_projects_holdout_without_mutating_training_contract(self):
        with tempfile.TemporaryDirectory() as root:
            contract, path = self._fixture(root)
            original = copy.deepcopy(contract)
            projected, source = prepare_hazard_holdout_contract(
                contract, path, "validation"
            )

        self.assertEqual(contract, original)
        self.assertEqual(source["expected_camera_samples"], 6)
        self.assertEqual(projected["data"]["max_validation_samples"], 6)
        self.assertEqual(
            projected["split_manifest_loaded"]["sequences"][0]["split"],
            "validation",
        )

    def test_rejects_validation_test_route_group_overlap(self):
        with tempfile.TemporaryDirectory() as root:
            contract, path = self._fixture(root)
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["hazard_holdout"]["test"][0]["route_group"] = (
                "Town01:route001"
            )
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                HazardHoldoutEvaluationError, "route groups overlap"
            ):
                prepare_hazard_holdout_contract(contract, path, "validation")


if __name__ == "__main__":
    unittest.main()
