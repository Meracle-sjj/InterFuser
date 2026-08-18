"""
 * [INPUT]: 依赖 tools/training/functional_distillation 的纯函数与数据集包装、interfuser/timm 变换工厂。
 * [OUTPUT]: 对外提供 v8b 功能蒸馏的单元测试：几何对齐、逐头归一化损失、LUT、掩码解码、
 *   互斥契约、冻结不变量与骨干导出。
 * [POS]: tests 的 v8b 守护测试；不加载真实大 checkpoint，配对构建由实际 run 验证。
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""
import json
import sys
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
INTERFUSER_ROOT = REPO_ROOT / "interfuser"
for import_root in (REPO_ROOT, INTERFUSER_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tools.training.functional_distillation import (  # noqa: E402
    FunctionalContractError,
    FunctionalFrameDataset,
    assert_pair_invariants,
    build_label_lut,
    export_student_backbone,
    front_mask_geometry,
    front_mask_geometry as _geometry,
    functional_output_l2,
    load_functional_contract,
)


class FrontGeometryTests(unittest.TestCase):
    def test_geometry_matches_rgb_transform_output(self):
        from timm.data.transforms_carla_factory import create_carla_rgb_transform
        from torchvision import transforms as tv_transforms

        for input_size in ((3, 224, 224), (3, 120, 160)):
            transform = create_carla_rgb_transform(input_size, is_training=False)
            probe = Image.new("RGB", (800, 600), color=(10, 20, 30))
            tensor = transform(probe)
            geometry = front_mask_geometry(input_size)
            pil_stage = tv_transforms.Compose(transform.transforms[:2])
            cropped = pil_stage(probe)
            crop = geometry["crop_size"]
            expected_size = (crop[1], crop[0]) if isinstance(crop, tuple) else (crop, crop)
            self.assertEqual(cropped.size, expected_size)
            self.assertEqual(tuple(tensor.shape[-2:]), expected_size[::-1])
            self.assertEqual(np.asarray(cropped).shape[:2], expected_size[::-1])

    def test_table_entries(self):
        self.assertEqual(front_mask_geometry((3, 224, 224))["resize_wh"], (341, 256))
        self.assertEqual(front_mask_geometry(128)["resize_wh"], (195, 146))


class FunctionalOutputL2Tests(unittest.TestCase):
    def test_zero_when_equal(self):
        outputs = [torch.zeros(2, 4, 5), torch.ones(2, 3, 2)]
        total, per_head = functional_output_l2(
            [outputs[0], outputs[1], outputs[0], outputs[0], outputs[0]],
            [outputs[0], outputs[1], outputs[0], outputs[0], outputs[0]],
        )
        self.assertAlmostEqual(float(total), 0.0)
        self.assertTrue(all(float(value) == 0.0 for value in per_head.values()))

    def test_normalized_ratio(self):
        student = [torch.ones(1, 4) for _ in range(5)]
        teacher = [torch.full((1, 4), 2.0) for _ in range(5)]
        total, per_head = functional_output_l2(student, teacher)
        expected = 4.0 / 16.0
        self.assertAlmostEqual(float(total), 5 * expected, places=5)
        for value in per_head.values():
            self.assertAlmostEqual(float(value), expected, places=5)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(FunctionalContractError):
            functional_output_l2(
                [torch.zeros(1, 4)] * 5, [torch.zeros(2, 4)] * 5
            )


class LabelLutTests(unittest.TestCase):
    def test_lut_maps_and_ignores(self):
        class_config = {
            "classes": [
                {"name": "background", "train_id": 0, "source_tags": [0]},
                {"name": "pedestrian", "train_id": 1, "source_tags": [4]},
            ],
            "ignore_tags": [22],
        }
        label_lut, known_lut = build_label_lut(class_config, 255)
        self.assertEqual(label_lut[0], 0)
        self.assertEqual(label_lut[4], 1)
        self.assertEqual(label_lut[22], 255)
        self.assertEqual(label_lut[9], 255)
        self.assertTrue(known_lut[22])
        self.assertFalse(known_lut[9])


class _StubBase:
    def __init__(self, route_dir, frame_id, data=None):
        self.route_frames = [(route_dir, frame_id)]
        self.data = data or {"rgb": torch.zeros(3, 4, 4)}

    def __len__(self):
        return len(self.route_frames)

    def _get_item_impl(self, index):
        return self.data, (torch.zeros(1), 0)


class FunctionalFrameDatasetTests(unittest.TestCase):
    CLASS_CONFIG = {
        "classes": [
            {"name": "background", "train_id": 0, "source_tags": [0]},
            {"name": "pedestrian", "train_id": 1, "source_tags": [4]},
        ],
        "ignore_tags": [22],
    }

    def _write_mask(self, directory, array):
        directory.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array.astype(np.uint8), mode="L").save(directory / "0000.png")

    def test_decodes_and_crops(self):
        with tempfile.TemporaryDirectory() as tmp:
            sequence = Path(tmp)
            raw = np.zeros((60, 80), dtype=np.uint8)
            raw[20:30, 30:40] = 4
            raw[35:45, 50:58] = 22
            self._write_mask(sequence / "seg_front", raw)
            dataset = FunctionalFrameDataset(
                _StubBase(sequence, "0000"),
                self.CLASS_CONFIG,
                255,
                {"resize_wh": (32, 24), "crop_size": (16, 16)},
            )
            sample = dataset[0]
            labels = sample["label"]
            self.assertEqual(labels.dtype, torch.int64)
            self.assertEqual(tuple(labels.shape), (16, 16))
            values = set(int(v) for v in torch.unique(labels))
            self.assertTrue(values <= {0, 1, 255})
            self.assertIn(1, values)

    def test_unknown_tag_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            sequence = Path(tmp)
            raw = np.zeros((60, 80), dtype=np.uint8)
            raw[25:30, 32:36] = 9
            self._write_mask(sequence / "seg_front", raw)
            dataset = FunctionalFrameDataset(
                _StubBase(sequence, "0000"),
                self.CLASS_CONFIG,
                255,
                {"resize_wh": (32, 24), "crop_size": (16, 16)},
            )
            with self.assertRaises(FunctionalContractError):
                dataset[0]


class _SurrogateModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 2, 1), torch.nn.BatchNorm2d(2)
        )
        self.other = torch.nn.Linear(4, 4)


class InvariantTests(unittest.TestCase):
    def _checkpoint(self, path, model):
        torch.save({"state_dict": model.state_dict()}, path)

    def test_non_rgb_change_raises(self):
        model = _SurrogateModel()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "init.pth"
            self._checkpoint(path, model)
            model.other.weight.data.add_(1.0)
            with self.assertRaises(FunctionalContractError):
                assert_pair_invariants(model, path)

    def test_rgb_change_and_bn_frozen_pass(self):
        model = _SurrogateModel()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "init.pth"
            self._checkpoint(path, model)
            model.rgb_backbone[0].weight.data.add_(1.0)
            report = assert_pair_invariants(model, path)
            self.assertEqual(report["changed_tensors"], 1)
            self.assertTrue(report["changed_tensors_all_rgb"])

    def test_tied_alias_keys_count_as_rgb(self):
        class _TiedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rgb_backbone = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 2, 1), torch.nn.BatchNorm2d(2)
                )
                self.duplicate = self.rgb_backbone
                self.other = torch.nn.Linear(4, 4)

        model = _TiedModel()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "init.pth"
            self._checkpoint(path, model)
            model.rgb_backbone[0].weight.data.add_(1.0)
            report = assert_pair_invariants(model, path)
            self.assertTrue(report["changed_tensors_all_rgb"])
            self.assertGreater(report["rgb_alias_key_count"], 0)

    def test_rgb_bn_buffer_change_raises(self):
        model = _SurrogateModel()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "init.pth"
            self._checkpoint(path, model)
            model.rgb_backbone[1].running_mean.data.add_(1.0)
            with self.assertRaises(FunctionalContractError):
                assert_pair_invariants(model, path)


class ExportTests(unittest.TestCase):
    def test_export_excludes_fc(self):
        from unittest.mock import patch

        model = _SurrogateModel()
        with patch(
            "tools.training.functional_distillation.validate_backbone_export",
            lambda export: None,
        ):
            export = export_student_backbone(model, "a" * 64)
        self.assertEqual(export["architecture"], "resnet50d")
        self.assertEqual(export["source_training_config_sha256"], "a" * 64)
        self.assertIn("backbone.0.weight", export["state_dict"])
        self.assertNotIn("backbone.fc.weight", export["state_dict"])


class ContractTests(unittest.TestCase):
    def test_feature_distillation_mutex(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "training": {"feature_distillation": {"coefficient": 1.0}},
                        "functional_distillation": {"coefficient": 1.0},
                    }
                )
            )
            with self.assertRaises(FunctionalContractError):
                load_functional_contract(path)

    def test_negative_coefficient_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {"functional_distillation": {"coefficient": -1.0, "matched_heads": [
                        "traffic", "waypoints", "is_junction", "traffic_light_state", "stop_sign"
                    ]}}
                )
            )
            with self.assertRaises(FunctionalContractError):
                load_functional_contract(path)


if __name__ == "__main__":
    unittest.main()
