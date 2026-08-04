"""
[INPUT]: 依赖 tools.data.build_pedestrian_hazard_training_manifest 的 M1 split、威胁审计与内容哈希扫描 API。
[OUTPUT]: 提供 train-only 扩充、Town+route holdout 隔离、validation/test 不变和特权真值准入的回归测试。
[POS]: tests 的 M1.1 行人危险数据构建测试，防止为提升小类而引入 route 泄漏或改动原始验证口径。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.data.build_pedestrian_hazard_training_manifest import (
    PedestrianHazardManifestError,
    build_pedestrian_hazard_training_manifest,
)


def _write_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PedestrianHazardTrainingManifestTests(unittest.TestCase):
    def _write_sequence(self, root, relative):
        sequence = Path(root) / relative
        for directory in ("rgb_front", "seg_front"):
            (sequence / directory).mkdir(parents=True)
        labels = np.array([[0, 12], [1, 1]], dtype=np.uint8)
        for index, frame in enumerate(("0000", "0001")):
            rgb = np.full((2, 2, 3), 40 + index, dtype=np.uint8)
            Image.fromarray(rgb).save(sequence / "rgb_front" / f"{frame}.jpg")
            Image.fromarray(labels).save(sequence / "seg_front" / f"{frame}.png")

    def _fixture(self, root):
        root = Path(root)
        dataset = root / "dataset"
        base_paths = {
            "train": "town01/town01_tiny_route1_w0_ClearNoon/base_train",
            "validation": "town01/town01_tiny_route2_w0_ClearNoon/base_validation",
            "test": "town01/town01_tiny_route3_w0_ClearNoon/base_test",
        }
        candidate_paths = {
            1: "town01/town01_tiny_route1_w1_CloudyNoon/hazard_train_weather",
            4: "town01/town01_tiny_route4_w0_ClearNoon/hazard_route4",
            5: "town01/town01_tiny_route5_w0_ClearNoon/hazard_route5",
            6: "town01/town01_tiny_route6_w0_ClearNoon/hazard_route6",
        }
        for relative in list(base_paths.values()) + list(candidate_paths.values()):
            self._write_sequence(dataset, relative)
        class_config = {
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
                    "minimum_pixels_per_mask": 1,
                    "minimum_qualified_masks": 1,
                    "minimum_sequences": 1,
                },
                {
                    "train_id": 2,
                    "name": "pedestrian",
                    "source_tags": [12],
                    "minimum_pixels_per_mask": 1,
                    "minimum_qualified_masks": 1,
                    "minimum_sequences": 1,
                },
            ],
        }
        class_path = _write_json(root / "classes.json", class_config)
        group_by_split = {
            "train": "Town01:route001",
            "validation": "Town01:route002",
            "test": "Town01:route003",
        }
        base_sequences = []
        route_groups = []
        for split, relative in base_paths.items():
            group = group_by_split[split]
            base_sequences.append(
                {
                    "path": relative,
                    "split": split,
                    "route_group": group,
                    "town": "Town01",
                    "declared_frames": 2,
                }
            )
            route_groups.append(
                {
                    "route_group": group,
                    "split": split,
                    "assignment_reason": "fixture",
                    "sequence_paths": [relative],
                }
            )
        base = {
            "split_manifest_schema_version": 1,
            "valid": True,
            "dataset_root": str(dataset),
            "cameras": ["front"],
            "source": {"class_config_sha256": _sha256(class_path)},
            "policy": {"split_unit": "town_route"},
            "review_candidates": {},
            "route_groups": route_groups,
            "sequences": base_sequences,
        }
        base_path = _write_json(root / "base.json", base)
        audit_sequences = []
        for route, relative in candidate_paths.items():
            audit_sequences.append(
                {
                    "valid": True,
                    "has_privileged_collision_hazard": True,
                    "has_visible_hazard_frame": True,
                    "sequence": relative,
                    "town": "Town01",
                    "weather": 0,
                    "semantic_logical_frames": 2,
                    "town_route_group": f"Town01:route{route:03d}",
                    "route_id": route,
                    "hazard_frames": 2,
                    "hazard_visible_frames": 2,
                    "hazard_frame_visibility_ratio": 1.0,
                    "trajectory_visibility_signature": f"signature-{route}",
                    "camera_stats": {
                        "front": {
                            "qualified_masks": 2,
                            "pixels": 20,
                        }
                    },
                }
            )
        audit = {
            "valid": True,
            "dataset_root": str(dataset),
            "cameras": ["front"],
            "summary": {
                "all_sequences_have_privileged_collision_hazard": True,
                "all_sequences_have_visible_hazard_frame": True,
            },
            "sequences": audit_sequences,
        }
        audit_path = _write_json(root / "audit.json", audit)
        return base_path, audit_path, class_path

    def test_adds_train_only_and_reserves_route_group_holdouts(self):
        with tempfile.TemporaryDirectory() as root:
            base_path, audit_path, class_path = self._fixture(root)
            first = build_pedestrian_hazard_training_manifest(
                base_path,
                audit_path,
                class_config_path=class_path,
                selection_seed=9,
                holdout_ratio=0.15,
                cameras=("front",),
            )
            second = build_pedestrian_hazard_training_manifest(
                base_path,
                audit_path,
                class_config_path=class_path,
                selection_seed=9,
                holdout_ratio=0.15,
                cameras=("front",),
            )

        self.assertTrue(first["valid"])
        self.assertEqual(first, second)
        self.assertEqual(first["summary"]["augmentation"]["train_sequences_added"], 2)
        self.assertEqual(len(first["hazard_holdout"]["validation"]), 1)
        self.assertEqual(len(first["hazard_holdout"]["test"]), 1)
        self.assertTrue(first["leakage_check"]["base_validation_test_unchanged"])
        self.assertEqual(
            first["summary"]["splits"]["validation"]["sequences"], 1
        )
        self.assertEqual(first["summary"]["splits"]["test"]["sequences"], 1)
        train_groups = {
            item["route_group"]
            for item in first["sequences"]
            if item["split"] == "train"
        }
        holdout_groups = {
            item["route_group"]
            for split in ("validation", "test")
            for item in first["hazard_holdout"][split]
        }
        self.assertFalse(train_groups & holdout_groups)

    def test_rejects_audit_without_global_visibility_gate(self):
        with tempfile.TemporaryDirectory() as root:
            base_path, audit_path, class_path = self._fixture(root)
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["summary"]["all_sequences_have_visible_hazard_frame"] = False
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            with self.assertRaisesRegex(
                PedestrianHazardManifestError, "collision hazard and visibility"
            ):
                build_pedestrian_hazard_training_manifest(
                    base_path,
                    audit_path,
                    class_config_path=class_path,
                    cameras=("front",),
                )


if __name__ == "__main__":
    unittest.main()
