"""
[INPUT]: 依赖 tools.data.audit_semantic_pretraining_data 的类别配置、dataset_index 抽样与数据审计 API，并使用临时 RGB/语义图构造最小 sequence。
[OUTPUT]: 提供 source-tag 映射、像素统计、索引分层抽样、帧数对账、对齐错误和 readiness 门槛的回归测试。
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

    def _write_frame(
        self,
        root,
        write_rgb=True,
        relative_route="route_00_Town01/sequence_01",
    ):
        route = Path(root) / relative_route
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

    def _write_index(self, root, entries):
        path = Path(root) / "dataset_index.txt"
        path.write_text(
            "".join(f"{relative} {count}\n" for relative, count in entries),
            encoding="utf-8",
        )
        return path

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

    def test_index_sampling_is_reproducible_per_town_weather(self):
        entries = [
            ("town01/town01_route00_w0_ClearNoon/sequence_a", 1),
            ("town01/town01_route01_w0_ClearNoon/sequence_b", 1),
            ("town03/town03_route00_w1_ClearSunset/sequence_c", 1),
            ("town03/town03_route01_w1_ClearSunset/sequence_d", 1),
        ]
        with tempfile.TemporaryDirectory() as root:
            config_path = self._write_config(root)
            for relative, _ in entries:
                self._write_frame(root, relative_route=relative)
            index_path = self._write_index(root, entries)

            first = audit_semantic_dataset(
                root,
                config_path=config_path,
                cameras=("front",),
                index_path=index_path,
                sample_per_town_weather=1,
                sample_seed=20260722,
            )
            second = audit_semantic_dataset(
                root,
                config_path=config_path,
                cameras=("front",),
                index_path=index_path,
                sample_per_town_weather=1,
                sample_seed=20260722,
            )

        selection = first["sequence_selection"]
        self.assertTrue(first["valid"])
        self.assertEqual(first["report_schema_version"], 2)
        self.assertEqual(first["sequence_count"], 2)
        self.assertEqual(first["towns"], ["Town01", "Town03"])
        self.assertEqual(selection["available_sequence_count"], 4)
        self.assertEqual(
            selection["selected_sequences"],
            second["sequence_selection"]["selected_sequences"],
        )

    def test_index_declared_frame_count_mismatch_is_structural_failure(self):
        relative = "town01/town01_route00_w0_ClearNoon/sequence_a"
        with tempfile.TemporaryDirectory() as root:
            config_path = self._write_config(root)
            self._write_frame(root, relative_route=relative)
            index_path = self._write_index(root, [(relative, 2)])

            summary = audit_semantic_dataset(
                root,
                config_path=config_path,
                cameras=("front",),
                index_path=index_path,
            )

        self.assertFalse(summary["valid"])
        self.assertTrue(any("index declares 2" in item for item in summary["errors"]))

    def test_sampling_requires_dataset_index(self):
        with tempfile.TemporaryDirectory() as root:
            config_path = self._write_config(root)
            self._write_frame(root)

            with self.assertRaisesRegex(AuditError, "requires index_path"):
                audit_semantic_dataset(
                    root,
                    config_path=config_path,
                    cameras=("front",),
                    sample_per_town_weather=1,
                )


if __name__ == "__main__":
    unittest.main()
