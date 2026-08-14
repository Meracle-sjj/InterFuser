"""
[INPUT]: 依赖 CarlaMVDetDataset 的显式 LiDAR y 轴契约，以及模态审计器的注册干预、配对输出累加器和预注册依赖判定。
[OUTPUT]: 验证 LiDAR 坐标乘数拒绝非法值、RGB/LiDAR 单变量注入、输出效应量分母及 weak/material 阈值方向。
[POS]: tests 的模态因果审计回归，阻止消融同时污染非目标输入或在结果后漂移判定口径。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from timm.data.carla_dataset import CarlaMVDetDataset

from tools.evaluation.run_interfuser_modality_ablation import (
    OutputSensitivityAccumulator,
    _variant_verdict,
    perturb_inputs,
)


class PerturbInputsTest(unittest.TestCase):
    def setUp(self):
        self.inputs = {
            "rgb": torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2),
            "rgb_center": torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2) + 100,
            "rgb_left": torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2) + 200,
            "rgb_right": torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2) + 300,
            "lidar": torch.ones(2, 2, 2, 2),
            "measurements": torch.ones(2, 7),
        }

    def test_rgb_mean_fill_only_changes_rgb(self):
        result = perturb_inputs(self.inputs, "rgb_mean_fill")
        for key in ("rgb", "rgb_center", "rgb_left", "rgb_right"):
            self.assertEqual(torch.count_nonzero(result[key]), 0)
        self.assertIs(result["lidar"], self.inputs["lidar"])
        self.assertIs(result["measurements"], self.inputs["measurements"])

    def test_rgb_shuffle_keeps_views_aligned(self):
        result = perturb_inputs(self.inputs, "rgb_shuffle")
        for key in ("rgb", "rgb_center", "rgb_left", "rgb_right"):
            torch.testing.assert_close(result[key][0], self.inputs[key][1])
            torch.testing.assert_close(result[key][1], self.inputs[key][0])
        self.assertIs(result["lidar"], self.inputs["lidar"])

    def test_lidar_zero_only_changes_lidar(self):
        result = perturb_inputs(self.inputs, "lidar_zero")
        self.assertEqual(torch.count_nonzero(result["lidar"]), 0)
        for key in ("rgb", "rgb_center", "rgb_left", "rgb_right"):
            self.assertIs(result[key], self.inputs[key])


class LidarAxisContractTest(unittest.TestCase):
    def test_explicit_axis_multiplier_and_invalid_value(self):
        with TemporaryDirectory() as temporary:
            index = Path(temporary) / "dataset_index.txt"
            index.write_text("", encoding="utf-8")
            dataset = CarlaMVDetDataset(
                temporary,
                towns=[1],
                weathers=[0],
                dataset_index=str(index),
                lidar_y_axis_multiplier=1.0,
            )
            self.assertEqual(dataset.lidar_y_axis_multiplier, 1.0)
            with self.assertRaisesRegex(ValueError, "lidar_y_axis_multiplier"):
                CarlaMVDetDataset(
                    temporary,
                    towns=[1],
                    weathers=[0],
                    dataset_index=str(index),
                    lidar_y_axis_multiplier=0.0,
                )


class OutputSensitivityAccumulatorTest(unittest.TestCase):
    @staticmethod
    def outputs(candidate=False):
        traffic = torch.zeros(2, 400, 7)
        waypoints = torch.zeros(2, 10, 2)
        class_logits = [torch.tensor([[2.0, 0.0], [2.0, 0.0]]) for _ in range(3)]
        feature = torch.ones(2, 400, 256)
        if candidate:
            traffic[:, :, 0] = 0.1
            waypoints[:, :, 0] = 3.0
            waypoints[:, :, 1] = 4.0
            class_logits = [torch.tensor([[0.0, 2.0], [0.0, 2.0]]) for _ in range(3)]
            feature *= 2.0
        return (traffic, waypoints, *class_logits, feature)

    def test_exact_paired_effect_sizes(self):
        accumulator = OutputSensitivityAccumulator()
        accumulator.update(self.outputs(False), self.outputs(True))
        result = accumulator.finalize()
        self.assertAlmostEqual(result["traffic_probability_mae_from_normal"], 0.1)
        self.assertAlmostEqual(result["traffic_full_output_mae_from_normal"], 0.1 / 7)
        self.assertAlmostEqual(result["waypoint_shift_ade"], 5.0)
        self.assertAlmostEqual(result["waypoint_shift_fde_horizon_10"], 5.0)
        self.assertAlmostEqual(result["feature_relative_l2_global"], 1.0)
        self.assertAlmostEqual(result["feature_cosine_distance_mean"], 0.0)
        for value in result["class_flip_rate"].values():
            self.assertAlmostEqual(value, 1.0)


class DependencyVerdictTest(unittest.TestCase):
    THRESHOLDS = {
        "weak_max_rgb_to_lidar_output_ratio": 0.25,
        "weak_max_relative_waypoint_ade_degradation": 0.05,
        "material_min_rgb_to_lidar_output_ratio": 0.5,
        "material_min_relative_waypoint_ade_degradation": 0.1,
    }

    @staticmethod
    def condition(ade, waypoint_shift, traffic_shift):
        return {
            "task_metrics": {"waypoints": {"ade": ade}},
            "sensitivity": {
                "waypoint_shift_ade": waypoint_shift,
                "traffic_probability_mae_from_normal": traffic_shift,
            },
        }

    def test_weak_and_material_threshold_directions(self):
        conditions = {
            "normal": self.condition(1.0, 0.0, 0.0),
            "rgb_mean_fill": self.condition(1.01, 0.1, 0.01),
            "rgb_shuffle": self.condition(1.02, 0.2, 0.02),
            "lidar_zero": self.condition(2.0, 1.0, 0.1),
        }
        self.assertEqual(
            _variant_verdict(conditions, self.THRESHOLDS)["verdict"],
            "weak_rgb_dependency",
        )
        conditions["rgb_shuffle"] = self.condition(1.02, 0.6, 0.06)
        self.assertEqual(
            _variant_verdict(conditions, self.THRESHOLDS)["verdict"],
            "material_rgb_dependency",
        )


if __name__ == "__main__":
    unittest.main()
