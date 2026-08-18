#!/usr/bin/env python3
"""
[INPUT]: 依赖 semantic_pretraining 的蒸馏契约/教师构造/归一化L2、run_semantic_pretraining._run_epoch 训练步与最小合成训练契约 fixture。
[OUTPUT]: 提供 v7 特征蒸馏的契约校验、教师冻结、归一化L2数学、训练步蒸馏接线和关闭等价性测试。
[POS]: tests 的 M2 H1 蒸馏单元测试；阻止可调系数、未冻结教师或改变既有配方的蒸馏实现进入真实训练。
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
from timm.models.resnet import resnet50d

from tools.training.run_semantic_pretraining import _run_epoch
from tools.training.semantic_pretraining import (
    DeterministicCrossEntropyLoss,
    SemanticPretrainingModel,
    TrainingContractError,
    load_training_contract,
    make_frozen_feature_teacher,
    normalized_feature_l2,
)


def _write_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _interfuser_checkpoint(path):
    source = resnet50d(pretrained=False, in_chans=3, features_only=True, out_indices=[1, 2, 3, 4])
    torch.save(
        {
            "state_dict": {
                f"rgb_backbone.{key}": value for key, value in source.state_dict().items()
            }
        },
        path,
    )
    return source


class NormalizedFeatureL2Test(unittest.TestCase):
    def test_exact_math_on_crafted_stages(self):
        teacher = [torch.full((1, 2, 2, 2), 2.0), torch.full((1, 4, 1, 1), 1.0)]
        student = [torch.full((1, 2, 2, 2), 3.0), torch.full((1, 4, 1, 1), 3.0)]
        # stage1: diff^2=1 per element, 8 elements -> 8; ||t||^2 = 4*8=32 -> 0.25
        # stage2: diff^2=4 per element, 4 elements -> 16; ||t||^2 = 4 -> 4.0
        penalty = normalized_feature_l2(student, teacher)
        self.assertAlmostEqual(float(penalty), 4.25, places=6)

    def test_zero_when_identical(self):
        features = [torch.randn(1, 3, 4, 4)]
        self.assertAlmostEqual(float(normalized_feature_l2(features, features)), 0.0, places=6)

    def test_misaligned_stage_lists_rejected(self):
        with self.assertRaises(TrainingContractError):
            normalized_feature_l2([torch.zeros(1, 2, 2, 2)], [])
        with self.assertRaises(TrainingContractError):
            normalized_feature_l2(
                [torch.zeros(1, 2, 2, 2)], [torch.zeros(1, 4, 2, 2)]
            )


class FeatureTeacherTest(unittest.TestCase):
    def test_teacher_frozen_eval_and_matches_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "teacher.pth"
            source = _interfuser_checkpoint(checkpoint)
            contract = {
                "backbone": {"feature_indices": [1, 2, 3, 4]},
                "training": {
                    "feature_distillation": {
                        "teacher_checkpoint_format": "interfuser_checkpoint",
                        "teacher_state_prefix": "rgb_backbone.",
                    }
                },
                "feature_distillation_teacher_path": checkpoint,
            }
            teacher = make_frozen_feature_teacher(contract)
            self.assertFalse(teacher.training)
            self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
            for key, value in source.state_dict().items():
                torch.testing.assert_close(value, teacher.state_dict()[key])
            with torch.no_grad():
                features = teacher(torch.zeros(1, 3, 32, 32))
            self.assertEqual(len(features), 4)

    def test_unconfigured_distillation_rejected(self):
        with self.assertRaises(TrainingContractError):
            make_frozen_feature_teacher({"backbone": {"feature_indices": [1, 2, 3, 4]}, "training": {}})


class DistillationContractTest(unittest.TestCase):
    def _base_config(self, root):
        root = Path(root)
        class_config = {
            "schema_version": 1,
            "source_labels": {"0": "NONE", "1": "Roads"},
            "ignore_source_tags": [],
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
                    "minimum_qualified_masks": 0,
                    "minimum_sequences": 0,
                },
            ],
        }
        class_path = _write_json(root / "classes.json", class_config)
        split_path = _write_json(
            root / "split.json",
            {
                "valid": True,
                "dataset_root": str(root / "dataset"),
                "cameras": ["front"],
                "source": {"class_config_sha256": _sha256(class_path)},
                "sequences": [],
            },
        )
        pretrained_path = root / "pretrained.pth"
        pretrained_path.write_bytes(b"pretrained")
        teacher_path = root / "teacher.pth"
        teacher_path.write_bytes(b"teacher")
        return {
            "schema_version": 1,
            "status": "optimization_probe",
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
            "model": {"num_classes": 2, "decoder_channels": 8, "dropout": 0.0},
            "data": {
                "cameras": ["front"],
                "input_width": 32,
                "input_height": 24,
                "sample_seed": 7,
                "max_train_samples": 2,
                "max_validation_samples": 1,
                "expected_available_train_samples": 2,
                "expected_available_validation_samples": 1,
                "validation_mode": "full_split",
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
                "gpu_resource_policy": "shared_capacity",
                "gpu_minimum_free_memory_mb": 1024,
                "require_clean_git": False,
                "feature_distillation": {
                    "teacher_checkpoint": str(teacher_path),
                    "teacher_checkpoint_sha256": _sha256(teacher_path),
                    "teacher_checkpoint_format": "interfuser_checkpoint",
                    "teacher_state_prefix": "rgb_backbone.",
                    "coefficient": 1.0,
                },
            },
        }

    def test_accepts_valid_distillation_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._base_config(tmp)
            contract = load_training_contract(_write_json(Path(tmp) / "c.json", config))
            self.assertEqual(
                Path(contract["feature_distillation_teacher_path"]),
                Path(config["training"]["feature_distillation"]["teacher_checkpoint"]),
            )

    def test_rejects_teacher_hash_drift_and_mutual_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._base_config(tmp)
            config["training"]["feature_distillation"]["teacher_checkpoint_sha256"] = "0" * 64
            with self.assertRaises(TrainingContractError):
                load_training_contract(_write_json(Path(tmp) / "bad.json", config))
        with tempfile.TemporaryDirectory() as tmp:
            config = self._base_config(tmp)
            config["training"]["source_parameter_regularization"] = {
                "method": "l2_sp",
                "coefficient": 1e-5,
                "reference": "backbone_initialization",
            }
            with self.assertRaises(TrainingContractError):
                load_training_contract(_write_json(Path(tmp) / "bad2.json", config))


class DistillationRunEpochTest(unittest.TestCase):
    def _model_and_loader(self):
        contract = {
            "backbone": {"feature_indices": [1, 2, 3, 4]},
            "model": {"num_classes": 2, "decoder_channels": 8, "dropout": 0.0},
            "pretrained_path": "unused",
        }
        with mock.patch(
            "tools.training.semantic_pretraining._load_pretrained_backbone",
            lambda backbone, path, contract=None: None,
        ):
            model = SemanticPretrainingModel(contract)
        rng = np.random.RandomState(0)
        batches = [
            {
                "image": torch.from_numpy(rng.rand(2, 3, 64, 64).astype(np.float32)),
                "label": torch.from_numpy(rng.randint(0, 2, (2, 64, 64)).astype(np.int64)),
            }
            for _ in range(2)
        ]
        return model, batches

    def test_distillation_off_matches_plain_forward(self):
        model, batches = self._model_and_loader()
        criterion = DeterministicCrossEntropyLoss(ignore_index=255)
        summary = _run_epoch(
            model, batches, criterion, torch.device("cpu"), ["a", "b"]
        )
        self.assertEqual(summary["feature_distillation_penalty"], 0.0)
        self.assertEqual(summary["weighted_feature_distillation_penalty"], 0.0)
        self.assertAlmostEqual(summary["loss"], summary["task_loss"], places=10)

    def test_distillation_adds_penalty_during_training(self):
        model, batches = self._model_and_loader()
        criterion = DeterministicCrossEntropyLoss(ignore_index=255)
        teacher = resnet50d(
            pretrained=False, in_chans=3, features_only=True, out_indices=[1, 2, 3, 4]
        )
        teacher.eval()
        teacher.requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)
        summary = _run_epoch(
            model,
            batches,
            criterion,
            torch.device("cpu"),
            ["a", "b"],
            optimizer=optimizer,
            feature_teacher=teacher,
            distill_coefficient=1.0,
        )
        self.assertGreater(summary["feature_distillation_penalty"], 0.0)
        # 2026-08-18: 值域 ~1e3 时多核浮点归约噪声 ~1e-5，places=5 过严，改用绝对容差
        self.assertAlmostEqual(
            summary["loss"],
            summary["task_loss"] + summary["feature_distillation_penalty"],
            delta=1e-2,
        )


if __name__ == "__main__":
    unittest.main()
