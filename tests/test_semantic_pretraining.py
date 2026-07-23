"""
[INPUT]: 依赖 tools.training.semantic_pretraining 的配置、数据集、模型、指标和骨干导出 API，并以临时 M1 split/RGB/mask 构造最小训练契约。
[OUTPUT]: 提供 provenance 哈希门禁、确定性样本选择、CARLA 标签映射、指标计算、ResNet50d 前向与 InterFuser 严格迁移兼容测试。
[POS]: tests 的 M2 训练契约测试，阻止数据划分漂移、标签静默忽略、指标错误或不可迁移的视觉骨干进入真实训练。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

from tools.training.semantic_pretraining import (
    ConfusionMetrics,
    DeterministicCrossEntropyLoss,
    SemanticFrameDataset,
    SemanticPretrainingModel,
    TrainingContractError,
    load_training_contract,
    make_backbone_export,
    resolve_train_sample_limit,
    validate_backbone_export,
)


def _write_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SemanticPretrainingTests(unittest.TestCase):
    def _write_sequence(self, root, relative, frame_ids=("0000", "0001")):
        sequence = Path(root) / relative
        rgb_dir = sequence / "rgb_front"
        seg_dir = sequence / "seg_front"
        rgb_dir.mkdir(parents=True)
        seg_dir.mkdir(parents=True)
        labels = np.array(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 0, 12, 12],
                [0, 0, 255, 255],
            ],
            dtype=np.uint8,
        )
        for index, frame_id in enumerate(frame_ids):
            rgb = np.full((4, 4, 3), 32 + index, dtype=np.uint8)
            Image.fromarray(rgb).save(rgb_dir / f"{frame_id}.jpg")
            Image.fromarray(labels).save(seg_dir / f"{frame_id}.png")

    def _fixture(self, root):
        root = Path(root)
        dataset_root = root / "dataset"
        train_path = "town01/town01_tiny_route1_w0_ClearNoon/train_sequence"
        validation_path = "town03/town03_tiny_route2_w1_ClearSunset/validation_sequence"
        self._write_sequence(dataset_root, train_path)
        self._write_sequence(dataset_root, validation_path)
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
        split_manifest = {
            "valid": True,
            "dataset_root": str(dataset_root),
            "cameras": ["front"],
            "source": {"class_config_sha256": _sha256(class_path)},
            "sequences": [
                {"path": train_path, "split": "train"},
                {"path": validation_path, "split": "validation"},
            ],
        }
        split_path = _write_json(root / "split.json", split_manifest)
        pretrained_path = root / "pretrained.pth"
        pretrained_path.write_bytes(b"test checkpoint")
        config = {
            "schema_version": 1,
            "status": "smoke",
            "class_config": str(class_path),
            "class_config_sha256": _sha256(class_path),
            "split_manifest": str(split_path),
            "split_manifest_sha256": _sha256(split_path),
            "backbone": {
                "name": "resnet50d",
                "feature_indices": [1, 2, 3, 4],
                "pretrained_source": "test",
                "pretrained_checkpoint": str(pretrained_path),
                "pretrained_checkpoint_sha256": _sha256(pretrained_path),
            },
            "model": {"num_classes": 3, "decoder_channels": 8, "dropout": 0.0},
            "data": {
                "cameras": ["front"],
                "input_width": 32,
                "input_height": 24,
                "sample_seed": 7,
                "max_train_samples": 2,
                "max_validation_samples": 1,
                "image_mean": [0.485, 0.456, 0.406],
                "image_std": [0.229, 0.224, 0.225],
            },
            "training": {
                "seed": 7,
                "epochs": 1,
                "batch_size": 1,
                "num_workers": 0,
                "optimizer": "adamw",
                "learning_rate": 0.0001,
                "weight_decay": 0.0,
                "ignore_index": 255,
                "deterministic": True,
                "physical_gpu_index": 0,
                "gpu_busy_memory_threshold_mb": 1024,
                "require_clean_git": False,
            },
        }
        config_path = _write_json(root / "training.json", config)
        return config_path

    def test_loads_hashed_contract_and_maps_carla_labels(self):
        with tempfile.TemporaryDirectory() as root:
            contract = load_training_contract(self._fixture(root))
            first = SemanticFrameDataset(contract, "train")
            second = SemanticFrameDataset(contract, "train")
            sample = first[0]

        self.assertEqual([item["key"] for item in first.records], [item["key"] for item in second.records])
        self.assertEqual(tuple(sample["image"].shape), (3, 24, 32))
        self.assertEqual(tuple(sample["label"].shape), (24, 32))
        self.assertEqual(set(sample["label"].unique().tolist()), {0, 1, 2, 255})

    def test_rejects_split_manifest_hash_drift(self):
        with tempfile.TemporaryDirectory() as root:
            config_path = self._fixture(root)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            Path(config["split_manifest"]).write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(TrainingContractError, "SHA-256 mismatch"):
                load_training_contract(config_path)

    def test_pilot_only_allows_configured_nested_train_budgets(self):
        with tempfile.TemporaryDirectory() as root:
            config_path = self._fixture(root)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["status"] = "pilot"
            config["data"].update(
                {
                    "expected_available_train_samples": 2,
                    "expected_available_validation_samples": 2,
                    "learning_curve_train_samples": [1, 2],
                    "validation_mode": "full_split",
                    "max_validation_samples": 2,
                }
            )
            config_path.write_text(json.dumps(config), encoding="utf-8")
            contract = load_training_contract(config_path)

            small = SemanticFrameDataset(contract, "train", sample_limit=1)
            full = SemanticFrameDataset(contract, "train", sample_limit=2)

            self.assertEqual(resolve_train_sample_limit(contract, 1), 1)
            self.assertEqual(resolve_train_sample_limit(contract, 2), 2)
            with self.assertRaisesRegex(TrainingContractError, "not in configured"):
                resolve_train_sample_limit(contract, 3)
            self.assertLess(
                {item["key"] for item in small.records},
                {item["key"] for item in full.records},
            )

    def test_optimization_probe_requires_full_train_and_validation(self):
        with tempfile.TemporaryDirectory() as root:
            config_path = self._fixture(root)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["status"] = "optimization_probe"
            config["data"].update(
                {
                    "expected_available_train_samples": 2,
                    "expected_available_validation_samples": 2,
                    "validation_mode": "full_split",
                    "max_validation_samples": 2,
                }
            )
            config_path.write_text(json.dumps(config), encoding="utf-8")

            contract = load_training_contract(config_path)

            self.assertEqual(resolve_train_sample_limit(contract), 2)
            with self.assertRaisesRegex(TrainingContractError, "must remain 2"):
                resolve_train_sample_limit(contract, 1)
            self.assertEqual(
                SemanticFrameDataset(contract, "validation").available_samples, 2
            )

    def test_confusion_metrics_report_exact_values(self):
        metrics = ConfusionMetrics(3, ignore_index=255)
        predictions = torch.tensor([[[0, 1], [2, 0]]])
        logits = torch.nn.functional.one_hot(predictions, num_classes=3).permute(0, 3, 1, 2).float()
        labels = torch.tensor([[[0, 1], [1, 255]]])

        metrics.update(logits, labels)
        summary = metrics.summary(["background", "road", "pedestrian"])

        self.assertAlmostEqual(summary["pixel_accuracy"], 2 / 3)
        self.assertAlmostEqual(summary["per_class"][0]["iou"], 1.0)
        self.assertAlmostEqual(summary["per_class"][1]["iou"], 0.5)
        self.assertAlmostEqual(summary["per_class"][2]["iou"], 0.0)

    def test_deterministic_loss_matches_standard_cross_entropy(self):
        logits = torch.tensor(
            [
                [
                    [[2.0, 0.1], [0.5, -1.0]],
                    [[0.0, 1.5], [1.0, 0.0]],
                    [[-1.0, 0.0], [0.0, 2.0]],
                ]
            ],
            requires_grad=True,
        )
        labels = torch.tensor([[[0, 1], [2, 255]]])
        expected = torch.nn.functional.cross_entropy(logits, labels, ignore_index=255)
        actual = DeterministicCrossEntropyLoss(ignore_index=255)(logits, labels)

        self.assertTrue(torch.allclose(actual, expected))
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_model_output_and_backbone_export_match_interfuser(self):
        with tempfile.TemporaryDirectory() as root:
            contract = load_training_contract(self._fixture(root))
            with mock.patch(
                "tools.training.semantic_pretraining._load_pretrained_backbone"
            ):
                model = SemanticPretrainingModel(contract)
            model.eval()
            with torch.no_grad():
                output = model(torch.zeros(1, 3, 64, 64))
            export = make_backbone_export(model, contract)
            validation = validate_backbone_export(export)

        self.assertEqual(tuple(output.shape), (1, 3, 64, 64))
        self.assertTrue(validation["strict_load"])
        self.assertGreater(validation["parameter_tensors"], 300)
        self.assertTrue(all(key.startswith("backbone.") for key in export["state_dict"]))


if __name__ == "__main__":
    unittest.main()
