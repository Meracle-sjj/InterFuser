"""
[INPUT]: 依赖 build_interfuser_downstream_indexes 的 v1/v2 冻结配置、全量索引投影与 measurements 场景真值 API，以临时 Town+route 数据构造可控划分。
[OUTPUT]: 验证 M1 holdout 归属、v2 行人分层扩充、语义 train 隔离、确定性选择、真值 census 门禁、哈希漂移/覆盖拒绝及显式 index 消费。
[POS]: tests 的 M2 下游数据隔离回归，阻止预训练已见 route group 泄漏到 B0/V 最终评价并防止稀缺交通场景塌缩到单一路线。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from timm.data.carla_dataset import CarlaMVDetDataset
from tools.data.build_interfuser_downstream_indexes import (
    DownstreamSplitError,
    build_downstream_indexes,
    write_downstream_indexes,
)


def _write_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class InterfuserDownstreamIndexTests(unittest.TestCase):
    def _fixture(self, root):
        root = Path(root)
        dataset = root / "dataset"
        dataset.mkdir()
        index = dataset / "dataset_index.txt"
        index.write_text(
            "\n".join(
                [
                    "town01/town01_tiny_route01_w0_Clear/a 2",
                    "town01/town01_tiny_route02_w1_Clear/b 3",
                    "town03/town03_tiny_route03_w2_Clear/c 5",
                    "town04/town04_tiny_route04_w3_Clear/d 7",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        semantic = _write_json(
            root / "semantic.json",
            {
                "valid": True,
                "dataset_root": str(dataset),
                "source": {"dataset_index_sha256": _sha256(index)},
                "leakage_check": {
                    "route_group_overlap_count": 0,
                    "all_selected_sequences_assigned_once": True,
                },
                "route_groups": [
                    {"route_group": "Town01:route001", "split": "train"},
                    {"route_group": "Town01:route002", "split": "validation"},
                    {"route_group": "Town03:route003", "split": "test"},
                ],
            },
        )
        config = {
            "schema_version": 1,
            "status": "frozen",
            "dataset_root": str(dataset),
            "dataset_index": str(index),
            "dataset_index_sha256": _sha256(index),
            "semantic_split_manifest": str(semantic),
            "semantic_split_manifest_sha256": _sha256(semantic),
            "split_unit": "town_route",
            "expansion_policy": {
                "frozen_validation_route_groups": "validation",
                "frozen_test_route_groups": "test",
                "frozen_train_route_groups": "train",
                "unassigned_route_groups": "train",
            },
        }
        return _write_json(root / "config.json", config), dataset, index

    def _scene_fixture(self, root):
        root = Path(root)
        dataset = root / "dataset"
        dataset.mkdir()
        records = [
            ("town01/town01_tiny_route01_w0_Clear/a", "Town01:route001"),
            ("town03/town03_tiny_route02_w1_Clear/b", "Town03:route002"),
            ("town04/town04_tiny_route03_w2_Clear/c", "Town04:route003"),
            ("town04/town04_tiny_route04_w3_Clear/d", "Town04:route004"),
            ("town03/town03_tiny_route05_w4_Clear/e", "Town03:route005"),
            ("town01/town01_tiny_route06_w5_Clear/f", "Town01:route006"),
            ("town05/town05_tiny_route07_w6_Clear/g", "Town05:route007"),
            ("town05/town05_tiny_route08_w7_Clear/h", "Town05:route008"),
        ]
        index = dataset / "dataset_index.txt"
        index.write_text(
            "".join(f"{relative_path} 1\n" for relative_path, _ in records),
            encoding="utf-8",
        )
        for relative_path, _ in records:
            measurements = dataset / relative_path / "measurements"
            measurements.mkdir(parents=True)
            _write_json(
                measurements / "0000.json",
                {
                    "command": 4,
                    "x_command": 1.0,
                    "y_command": 2.0,
                    "future_waypoints": [[1.0, 2.0]],
                    "theta": 0.0,
                    "is_pedestrian_present": [42],
                },
            )
        semantic = _write_json(
            root / "semantic.json",
            {
                "valid": True,
                "dataset_root": str(dataset),
                "source": {"dataset_index_sha256": _sha256(index)},
                "leakage_check": {
                    "route_group_overlap_count": 0,
                    "all_selected_sequences_assigned_once": True,
                },
                "route_groups": [
                    {"route_group": records[0][1], "split": "train"},
                    {"route_group": records[1][1], "split": "validation"},
                    {"route_group": records[2][1], "split": "test"},
                ],
            },
        )
        config = {
            "schema_version": 2,
            "status": "frozen",
            "dataset_root": str(dataset),
            "dataset_index": str(index),
            "dataset_index_sha256": _sha256(index),
            "semantic_split_manifest": str(semantic),
            "semantic_split_manifest_sha256": _sha256(semantic),
            "split_unit": "town_route",
            "expansion_policy": {
                "frozen_validation_route_groups": "validation",
                "frozen_test_route_groups": "test",
                "frozen_train_route_groups": "train",
                "unassigned_pedestrian_route_groups": (
                    "deterministic_stratified_holdout_then_train"
                ),
                "remaining_unassigned_route_groups": "train",
            },
            "scene_stratification": {
                "selection_method": "seeded_town_coverage_balanced_frames_v1",
                "selection_seed": 20260816,
                "pedestrian_measurement_field": "is_pedestrian_present",
                "required_navigation_fields": [
                    "command",
                    "x_command",
                    "y_command",
                    "future_waypoints",
                    "theta",
                ],
                "expected_scene_totals": {
                    "effective_frames": 8,
                    "dropped_frames": 0,
                    "pedestrian_frames": 8,
                    "pedestrian_sequences": 8,
                    "pedestrian_route_groups": 8,
                },
                "target_pedestrian_route_groups": {
                    "train": 4,
                    "validation": 2,
                    "test": 2,
                },
                "required_pedestrian_towns_per_holdout": ["Town03", "Town04"],
                "minimum_pedestrian_frames_per_holdout": 2,
                "minimum_pedestrian_sequences_per_holdout": 2,
                "minimum_pedestrian_weathers_per_holdout": 2,
                "maximum_holdout_pedestrian_frame_ratio": 1.0,
            },
        }
        return _write_json(root / "scene_config.json", config)

    def test_projects_holdouts_and_sends_unassigned_groups_only_to_train(self):
        with tempfile.TemporaryDirectory() as root:
            config, _, _ = self._fixture(root)
            _, splits, manifest = build_downstream_indexes(config)

        self.assertEqual(
            [item["route_group"] for item in splits["validation"]],
            ["Town01:route002"],
        )
        self.assertEqual(
            [item["route_group"] for item in splits["test"]],
            ["Town03:route003"],
        )
        self.assertEqual(
            {item["route_group"] for item in splits["train"]},
            {"Town01:route001", "Town04:route004"},
        )
        self.assertEqual(manifest["summary"]["train"]["logical_frames"], 9)
        self.assertTrue(manifest["leakage_check"]["all_source_sequences_assigned_once"])
        self.assertTrue(
            all(
                not groups
                for groups in manifest["leakage_check"]["route_group_overlaps"].values()
            )
        )

    def test_writes_indexes_and_dataset_consumes_explicit_index(self):
        with tempfile.TemporaryDirectory() as root:
            config, dataset, _ = self._fixture(root)
            output = Path(root) / "output"
            manifest, _, _ = write_downstream_indexes(config, output)
            train_dataset = CarlaMVDetDataset(
                dataset,
                towns=[1, 4],
                weathers=[0, 3],
                dataset_index=manifest["artifacts"]["train"]["path"],
            )
            validation_dataset = CarlaMVDetDataset(
                dataset,
                towns=[1],
                weathers=[1],
                dataset_index=manifest["artifacts"]["validation"]["path"],
            )

        self.assertEqual(len(train_dataset), 9)
        self.assertEqual(len(validation_dataset), 3)

    def test_rejects_dataset_index_hash_drift(self):
        with tempfile.TemporaryDirectory() as root:
            config, _, index = self._fixture(root)
            index.write_text("corrupt\n", encoding="utf-8")
            with self.assertRaisesRegex(DownstreamSplitError, "SHA-256 mismatch"):
                build_downstream_indexes(config)

    def test_rejects_missing_frozen_holdout_group(self):
        with tempfile.TemporaryDirectory() as root:
            config, _, index = self._fixture(root)
            lines = index.read_text(encoding="utf-8").splitlines()
            index.write_text("\n".join(lines[:1] + lines[2:]) + "\n", encoding="utf-8")
            value = json.loads(config.read_text(encoding="utf-8"))
            value["dataset_index_sha256"] = _sha256(index)
            semantic_path = Path(value["semantic_split_manifest"])
            semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
            semantic["source"]["dataset_index_sha256"] = _sha256(index)
            semantic_path.write_text(json.dumps(semantic), encoding="utf-8")
            value["semantic_split_manifest_sha256"] = _sha256(semantic_path)
            config.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(DownstreamSplitError, "absent from full index"):
                build_downstream_indexes(config)

    def test_v2_expands_only_unseen_pedestrian_groups_into_holdouts(self):
        with tempfile.TemporaryDirectory() as root:
            config = self._scene_fixture(root)
            _, splits, manifest = build_downstream_indexes(config)

        groups = {
            split: {item["route_group"] for item in records}
            for split, records in splits.items()
        }
        self.assertEqual(
            groups["validation"], {"Town03:route002", "Town04:route004"}
        )
        self.assertEqual(groups["test"], {"Town04:route003", "Town03:route005"})
        self.assertIn("Town01:route001", groups["train"])
        self.assertEqual(
            manifest["scene_stratification"]["split_scene_summary"]["validation"][
                "pedestrian_route_groups"
            ],
            2,
        )
        self.assertEqual(
            manifest["leakage_check"]["semantic_train_holdout_overlap"], []
        )
        self.assertEqual(
            manifest["scene_stratification"]["selection"][
                "semantic_train_holdout_overlap"
            ],
            [],
        )

    def test_v2_selection_is_deterministic(self):
        with tempfile.TemporaryDirectory() as root:
            config = self._scene_fixture(root)
            first = build_downstream_indexes(config)[2]["scene_stratification"]
            second = build_downstream_indexes(config)[2]["scene_stratification"]

        self.assertEqual(first, second)

    def test_v2_rejects_scene_census_drift(self):
        with tempfile.TemporaryDirectory() as root:
            config = self._scene_fixture(root)
            value = json.loads(config.read_text(encoding="utf-8"))
            value["scene_stratification"]["expected_scene_totals"][
                "pedestrian_frames"
            ] = 9
            config.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(DownstreamSplitError, "scene census differs"):
                build_downstream_indexes(config)


if __name__ == "__main__":
    unittest.main()
