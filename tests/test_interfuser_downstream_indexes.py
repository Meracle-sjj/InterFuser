"""
[INPUT]: 依赖 build_interfuser_downstream_indexes 的冻结配置、全量索引投影 API，以临时 Town+route 数据构造可控划分。
[OUTPUT]: 验证 M1 holdout route group 在全量索引中保持归属、未见组只进 train、哈希漂移/覆盖被拒绝，以及 CarlaMVDetDataset 消费显式 index。
[POS]: tests 的 M2 下游数据隔离回归，阻止预训练已见 route group 泄漏到 B0/V 最终评价。
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


if __name__ == "__main__":
    unittest.main()
