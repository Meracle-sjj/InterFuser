"""
[INPUT]: 依赖 M1 split 构建器与对齐复核渲染器，并以临时 route-group、RGB、语义 mask 和 pilot provenance 构造最小数据集。
[OUTPUT]: 提供 route 级无泄漏、确定性类别覆盖、内容哈希、结构失败和九类式复核证据渲染的回归测试。
[POS]: tests 的 M1 划分与人工复核契约测试，阻止同一路线跨 split、稀缺类别缺失或 RGB/mask 尺寸漂移进入预训练。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
from PIL import Image

from tools.data.build_semantic_split_manifest import (
    build_split_manifest,
    main as split_main,
)
from tools.data.render_semantic_alignment_review import (
    ReviewError,
    render_alignment_review,
)


def _write_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _class_config():
    return {
        "schema_version": 1,
        "source_labels": {
            "0": "NONE",
            "1": "Roads",
            "12": "Pedestrians",
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
                "name": "pedestrian",
                "source_tags": [12],
                "minimum_pixels_per_mask": 2,
                "minimum_qualified_masks": 1,
                "minimum_sequences": 1,
            },
        ],
    }


def _split_config():
    return {
        "schema_version": 1,
        "assignment_seed": 20260723,
        "split_unit": "town_route",
        "splits": {"train": 0.5, "validation": 0.25, "test": 0.25},
        "excluded_classes": ["background"],
        "minimum_towns_per_split": 1,
        "minimum_sequences_per_class_per_split": 1,
        "maximum_sequence_ratio_deviation": 0.4,
    }


class SemanticSplitAndReviewTests(unittest.TestCase):
    def _write_sequence(self, root, relative, rgb_size=(4, 4)):
        sequence = Path(root) / relative
        rgb_dir = sequence / "rgb_front"
        seg_dir = sequence / "seg_front"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        seg_dir.mkdir(parents=True, exist_ok=True)
        labels = np.array(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 0, 12, 12],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        Image.fromarray(np.zeros((rgb_size[1], rgb_size[0], 3), dtype=np.uint8)).save(
            rgb_dir / "0000.jpg"
        )
        Image.fromarray(labels).save(seg_dir / "0000.png")

    def _fixture(self, root):
        root = Path(root)
        class_path = _write_json(root / "classes.json", _class_config())
        split_path = _write_json(root / "split.json", _split_config())
        sequence_specs = [
            ("town01/town01_tiny_route1_w0_ClearNoon/route1_seq_a", "Town01", 0),
            ("town01/town01_tiny_route1_w1_ClearSunset/route1_seq_b", "Town01", 1),
            ("town01/town01_tiny_route2_w0_ClearNoon/route2_seq", "Town01", 0),
            ("town03/town03_tiny_route3_w0_ClearNoon/route3_seq", "Town03", 0),
            ("town03/town03_tiny_route4_w1_ClearSunset/route4_seq", "Town03", 1),
            ("town04/town04_tiny_route5_w0_ClearNoon/route5_seq", "Town04", 0),
            ("town05/town05_tiny_route6_w1_ClearSunset/route6_seq", "Town05", 1),
        ]
        selected = []
        for relative, town, weather in sequence_specs:
            self._write_sequence(root, relative)
            selected.append(
                {
                    "path": relative,
                    "declared_frames": 1,
                    "town": town,
                    "weather": weather,
                }
            )
        pilot = {
            "valid": True,
            "readiness": {"ready": True},
            "dataset_root": str(root.resolve()),
            "cameras": ["front"],
            "class_config_sha256": _sha256(class_path),
            "sequence_selection": {
                "dataset_index": str(root / "dataset_index.txt"),
                "dataset_index_sha256": "index-sha",
                "selected_sequences": selected,
            },
        }
        pilot_path = _write_json(root / "pilot.json", pilot)
        return class_path, split_path, pilot_path

    def test_builds_deterministic_route_group_split_with_core_coverage(self):
        with tempfile.TemporaryDirectory() as root:
            class_path, split_path, pilot_path = self._fixture(root)
            first = build_split_manifest(
                root,
                pilot_path,
                class_config_path=class_path,
                split_config_path=split_path,
                cameras=("front",),
            )
            second = build_split_manifest(
                root,
                pilot_path,
                class_config_path=class_path,
                split_config_path=split_path,
                cameras=("front",),
            )

        self.assertTrue(first["valid"])
        self.assertEqual(first, second)
        self.assertEqual(first["summary"]["sequences"], 7)
        self.assertEqual(first["summary"]["route_groups"], 6)
        self.assertTrue(
            first["leakage_check"]["all_selected_sequences_assigned_once"]
        )
        for split in ("train", "validation", "test"):
            coverage = first["summary"]["splits"][split][
                "sequences_with_qualified_class"
            ]
            self.assertGreaterEqual(coverage["road"], 1)
            self.assertGreaterEqual(coverage["pedestrian"], 1)
        route_one_splits = {
            item["split"]
            for item in first["sequences"]
            if item["route_group"] == "Town01:route001"
        }
        self.assertEqual(len(route_one_splits), 1)
        self.assertEqual(len(first["sequences"][0]["content_sha256"]["rgb"]), 64)

    def test_dimension_mismatch_blocks_split_admission(self):
        with tempfile.TemporaryDirectory() as root:
            class_path, split_path, pilot_path = self._fixture(root)
            relative = (
                "town01/town01_tiny_route2_w0_ClearNoon/route2_seq"
            )
            self._write_sequence(root, relative, rgb_size=(5, 4))

            manifest = build_split_manifest(
                root,
                pilot_path,
                class_config_path=class_path,
                split_config_path=split_path,
                cameras=("front",),
            )

        self.assertFalse(manifest["valid"])
        self.assertTrue(any("dimensions differ" in item for item in manifest["errors"]))

    def test_duplicate_pilot_sequence_blocks_split_admission(self):
        with tempfile.TemporaryDirectory() as root:
            class_path, split_path, pilot_path = self._fixture(root)
            pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
            pilot["sequence_selection"]["selected_sequences"].append(
                dict(pilot["sequence_selection"]["selected_sequences"][0])
            )
            pilot_path.write_text(json.dumps(pilot), encoding="utf-8")

            manifest = build_split_manifest(
                root,
                pilot_path,
                class_config_path=class_path,
                split_config_path=split_path,
                cameras=("front",),
            )

        self.assertFalse(manifest["valid"])
        self.assertFalse(
            manifest["leakage_check"]["all_selected_sequences_assigned_once"]
        )
        self.assertEqual(manifest["leakage_check"]["sequence_overlap_count"], 1)
        self.assertTrue(any("duplicate sequence" in item for item in manifest["errors"]))

    def test_split_cli_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as root:
            class_path, split_path, pilot_path = self._fixture(root)
            output = Path(root) / "existing.json"
            output.write_text("keep", encoding="utf-8")
            argv = [
                root,
                str(pilot_path),
                "--class-config",
                str(class_path),
                "--split-config",
                str(split_path),
                "--cameras",
                "front",
                "--output",
                str(output),
            ]

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                exit_code = split_main(argv)
            content = output.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 2)
        self.assertEqual(content, "keep")

    def test_renders_pending_manual_review_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            class_path, split_path, pilot_path = self._fixture(root)
            manifest = build_split_manifest(
                root,
                pilot_path,
                class_config_path=class_path,
                split_config_path=split_path,
                cameras=("front",),
            )
            manifest_path = _write_json(Path(root) / "manifest.json", manifest)

            report = render_alignment_review(
                manifest_path,
                Path(root) / "review",
                class_config_path=class_path,
            )

        self.assertEqual(report["status"], "pending_manual_review")
        self.assertEqual([item["class"] for item in report["items"]], ["road", "pedestrian"])
        for item in report["items"]:
            self.assertEqual(item["review"]["status"], "pending")
            self.assertEqual(len(item["render_sha256"]), 64)

    def test_review_refuses_nonempty_output_directory(self):
        with tempfile.TemporaryDirectory() as root:
            class_path, split_path, pilot_path = self._fixture(root)
            manifest = build_split_manifest(
                root,
                pilot_path,
                class_config_path=class_path,
                split_config_path=split_path,
                cameras=("front",),
            )
            manifest_path = _write_json(Path(root) / "manifest.json", manifest)
            output_dir = Path(root) / "review"
            output_dir.mkdir()
            (output_dir / "existing.txt").write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(ReviewError, "non-empty output directory"):
                render_alignment_review(
                    manifest_path,
                    output_dir,
                    class_config_path=class_path,
                )


if __name__ == "__main__":
    unittest.main()
