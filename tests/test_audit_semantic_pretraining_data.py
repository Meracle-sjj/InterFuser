"""
[INPUT]: 依赖 tools.data.audit_semantic_pretraining_data 的类别配置校验与数据审计 API，并使用临时 RGB/语义图构造最小 route sequence。
[OUTPUT]: 提供 source-tag 唯一映射、像素/有效 mask 统计、帧对齐错误和 readiness 门槛的回归测试。
[POS]: tests 的 M1 数据准入契约测试，保证统计报告能阻止类别缺失或结构损坏的数据进入 ResNet-50 预训练。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.data.audit_semantic_pretraining_data import (
    AuditError,
    audit_semantic_dataset,
    load_class_config,
)


def _config(traffic_light_masks=1):
    return {
        "schema_version": 1,
        "source_labels": {
            "0": "NONE",
            "1": "Roads",
            "7": "TrafficLight",
            "255": "Any",
        },
        "ignore_source_tags": [255],
        "dataset_readiness": {
            "minimum_sequences": 1,
            "minimum_towns": 1,
            "minimum_logical_frames": 1,
        },
        "classes": [
            {
                "train_id": 0,
                "name": "background",
                "source_tags": [0],
                "minimum_pixels_per_mask": 1,
                "minimum_qualified_masks": 0,
                "minimum_sequences": 0,
            },
            {
                "train_id": 1,
                "name": "road",
                "source_tags": [1],
                "minimum_pixels_per_mask": 2,
                "minimum_qualified_masks": 1,
                "minimum_sequences": 1,
            },
            {
                "train_id": 2,
                "name": "traffic_light",
                "source_tags": [7],
                "minimum_pixels_per_mask": 2,
                "minimum_qualified_masks": traffic_light_masks,
                "minimum_sequences": 1,
            },
        ],
    }


class SemanticPretrainingAuditTests(unittest.TestCase):
    def _write_config(self, root, config=None):
        path = Path(root) / "classes.json"
        path.write_text(json.dumps(config or _config()), encoding="utf-8")
        return path

    def _write_frame(self, root, write_rgb=True):
        route = Path(root) / "route_00_Town01" / "sequence_01"
        seg_dir = route / "seg_front"
        rgb_dir = route / "rgb_front"
        seg_dir.mkdir(parents=True)
        rgb_dir.mkdir(parents=True)
        labels = np.array(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 0, 7, 7],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        Image.fromarray(labels).save(seg_dir / "0000.png")
        if write_rgb:
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
                rgb_dir / "0000.jpg"
            )

    def test_counts_grouped_pixels_and_passes_ready_gate(self):
        with tempfile.TemporaryDirectory() as root:
            config_path = self._write_config(root)
            self._write_frame(root)

            summary = audit_semantic_dataset(
                root, config_path=config_path, cameras=("front",)
            )

        by_name = {item["name"]: item for item in summary["classes"]}
        self.assertTrue(summary["valid"])
        self.assertTrue(summary["readiness"]["ready"])
        self.assertEqual(summary["sequence_count"], 1)
        self.assertEqual(summary["logical_frame_count"], 1)
        self.assertEqual(summary["semantic_mask_count"], 1)
        self.assertEqual(by_name["road"]["pixels"], 8)
        self.assertEqual(by_name["traffic_light"]["qualified_masks"], 1)

    def test_missing_rgb_frame_is_structural_failure(self):
        with tempfile.TemporaryDirectory() as root:
            config_path = self._write_config(root)
            self._write_frame(root, write_rgb=False)

            summary = audit_semantic_dataset(
                root, config_path=config_path, cameras=("front",)
            )

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["readiness"]["ready"])
        self.assertTrue(any("without RGB" in item for item in summary["errors"]))

    def test_rejects_source_tag_mapped_to_multiple_classes(self):
        config = _config()
        config["classes"][2]["source_tags"] = [1, 7]
        with tempfile.TemporaryDirectory() as root:
            config_path = self._write_config(root, config)

            with self.assertRaisesRegex(AuditError, "mapped more than once"):
                load_class_config(config_path)

    def test_readiness_reports_underrepresented_class(self):
        with tempfile.TemporaryDirectory() as root:
            config_path = self._write_config(root, _config(traffic_light_masks=2))
            self._write_frame(root)

            summary = audit_semantic_dataset(
                root, config_path=config_path, cameras=("front",)
            )

        self.assertTrue(summary["valid"])
        self.assertFalse(summary["readiness"]["ready"])
        self.assertIn(
            "class traffic_light: qualified_masks=1 < 2",
            summary["readiness"]["failures"],
        )


if __name__ == "__main__":
    unittest.main()
