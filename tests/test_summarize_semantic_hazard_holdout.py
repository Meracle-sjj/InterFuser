"""
[INPUT]: 依赖 tools.training.summarize_semantic_hazard_holdout 的两份专项评估报告配对契约。
[OUTPUT]: 提供指标/逐类差值、样本 key 一致性与口径漂移拒绝的回归测试。
[POS]: tests 的 M2 行人危险归约测试，保证论文差值只来自模型，而不是样本或 loss 切换。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import json
import tempfile
import unittest
from pathlib import Path

from tools.training.summarize_semantic_hazard_holdout import (
    HazardHoldoutSummaryError,
    summarize_semantic_hazard_holdout,
)


def _metrics(iou):
    return {
        "samples": 3,
        "sample_keys_sha256": "k" * 64,
        "mean_iou": iou,
        "macro_f1": iou + 0.1,
        "loss": 1.0 - iou,
        "per_class": [
            {
                "name": "pedestrian",
                "train_id": 5,
                "support_pixels": 20,
                "iou": iou,
                "f1": iou + 0.1,
            }
        ],
    }


def _report(iou):
    sources = {}
    for split in ("validation", "test"):
        sources[split] = {
            "manifest_sha256": "m" * 64,
            "holdout_split": split,
            "route_groups": [f"Town01:{split}"],
            "sequences": 1,
            "expected_camera_samples": 3,
        }
    return {
        "valid": True,
        "contract": {
            "privileged_hazard_used_as_model_input": False,
            "checkpoint_selection_uses_hazard_holdout": False,
            "class_weights": [1.0],
        },
        "sources": {
            "checkpoint_sha256": "c" * 64,
            "hazard_holdouts": sources,
        },
        "evaluations": {
            "validation": _metrics(iou),
            "test": _metrics(iou - 0.1),
        },
    }


class HazardHoldoutSummaryTests(unittest.TestCase):
    def _write(self, root, name, value):
        path = Path(root) / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_computes_strict_paired_deltas(self):
        with tempfile.TemporaryDirectory() as root:
            baseline = self._write(root, "baseline.json", _report(0.2))
            augmented = self._write(root, "augmented.json", _report(0.5))
            summary = summarize_semantic_hazard_holdout(
                baseline, augmented
            )

        self.assertTrue(summary["valid"])
        self.assertAlmostEqual(
            summary["splits"]["validation"]["augmented_minus_baseline"][
                "mean_iou"
            ],
            0.3,
        )
        self.assertEqual(
            summary["splits"]["test"]["classes_improved_iou"], 1
        )

    def test_rejects_different_sample_keys(self):
        with tempfile.TemporaryDirectory() as root:
            baseline_report = _report(0.2)
            augmented_report = _report(0.5)
            augmented_report["evaluations"]["test"][
                "sample_keys_sha256"
            ] = "x" * 64
            baseline = self._write(root, "baseline.json", baseline_report)
            augmented = self._write(root, "augmented.json", augmented_report)
            with self.assertRaisesRegex(
                HazardHoldoutSummaryError, "evaluated samples differ"
            ):
                summarize_semantic_hazard_holdout(baseline, augmented)


if __name__ == "__main__":
    unittest.main()
