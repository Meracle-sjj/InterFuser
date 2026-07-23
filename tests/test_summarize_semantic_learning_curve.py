"""
[INPUT]: 依赖 tools.training.summarize_semantic_learning_curve，并以临时 pilot 配置、run manifest 和带哈希 checkpoint/骨干文件构造学习曲线矩阵。
[OUTPUT]: 提供完整预算、嵌套 train、相同 validation、pipeline/provenance 与产物完整性的汇总拒绝测试。
[POS]: tests 的 M2 学习曲线门禁测试，保证数据量结论只来自完整、可比且原始产物仍可验证的三个 pilot run。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.training.summarize_semantic_learning_curve import (
    LearningCurveError,
    summarize_learning_curve,
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _metrics(value):
    return {
        "loss": value,
        "pixel_accuracy": value,
        "mean_iou": value,
        "macro_f1": value,
        "samples": 1,
        "batches": 1,
        "duration_seconds": 0.1,
        "confusion_matrix": [[1]],
        "per_class": [
            {
                "train_id": 0,
                "name": "road",
                "support_pixels": 1,
                "predicted_pixels": 1,
                "iou": value,
                "f1": value,
            }
        ],
    }


class SemanticLearningCurveSummaryTests(unittest.TestCase):
    def _fixture(self, root):
        root = Path(root)
        config = {
            "status": "pilot",
            "data": {
                "learning_curve_train_samples": [1, 2, 3],
                "expected_available_validation_samples": 2,
            },
            "training": {"epochs": 1},
        }
        config_path = root / "pilot.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        config_sha256 = _sha256(config_path)
        manifests = []
        common = {
            "git_head": "abc123",
            "config": str(config_path),
            "config_sha256": config_sha256,
            "class_config_sha256": "class-sha",
            "split_manifest_sha256": "split-sha",
            "pretrained_source": "imagenet",
            "pretrained_checkpoint_sha256": "pretrained-sha",
            "experiment_status": "pilot",
            "status": "completed",
            "pipeline_valid": True,
        }
        all_train_keys = ["train-a", "train-b", "train-c"]
        for sample_count in (1, 2, 3):
            run_dir = root / f"run-{sample_count}"
            run_dir.mkdir()
            checkpoint = run_dir / "checkpoint.pth"
            backbone = run_dir / "backbone.pth"
            checkpoint.write_bytes(f"checkpoint-{sample_count}".encode())
            backbone.write_bytes(f"backbone-{sample_count}".encode())
            manifest = dict(common)
            manifest.update(
                {
                    "run_id": f"run-{sample_count}",
                    "data": {
                        "train_samples": sample_count,
                        "validation_samples": 2,
                        "train_sample_limit_requested": sample_count,
                        "train_sample_keys": all_train_keys[:sample_count],
                        "validation_sample_keys": ["validation-a", "validation-b"],
                    },
                    "epochs": [
                        {
                            "epoch": 1,
                            "train": _metrics(0.1 * sample_count),
                            "validation": _metrics(0.2 * sample_count),
                        }
                    ],
                    "artifacts": {
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": _sha256(checkpoint),
                        "backbone_export": str(backbone),
                        "backbone_export_sha256": _sha256(backbone),
                    },
                    "resources": {"peak_memory_allocated_mb": sample_count},
                }
            )
            manifest_path = run_dir / "run_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            manifests.append(manifest_path)
        return manifests

    def test_summarizes_complete_nested_curve(self):
        with tempfile.TemporaryDirectory() as root:
            manifests = self._fixture(root)

            summary = summarize_learning_curve(reversed(manifests))

        self.assertTrue(summary["valid"])
        self.assertTrue(summary["nested_train_samples"])
        self.assertTrue(summary["identical_validation_samples"])
        self.assertEqual(
            [point["train_samples"] for point in summary["points"]], [1, 2, 3]
        )

    def test_rejects_validation_sample_drift(self):
        with tempfile.TemporaryDirectory() as root:
            manifests = self._fixture(root)
            manifest = json.loads(manifests[1].read_text(encoding="utf-8"))
            manifest["data"]["validation_sample_keys"].reverse()
            manifests[1].write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(LearningCurveError, "validation sample keys differ"):
                summarize_learning_curve(manifests)

    def test_rejects_artifact_hash_drift(self):
        with tempfile.TemporaryDirectory() as root:
            manifests = self._fixture(root)
            manifest = json.loads(manifests[2].read_text(encoding="utf-8"))
            Path(manifest["artifacts"]["backbone_export"]).write_bytes(b"drift")

            with self.assertRaisesRegex(LearningCurveError, "artifact hash mismatch"):
                summarize_learning_curve(manifests)


if __name__ == "__main__":
    unittest.main()
