"""
[INPUT]: 依赖 interfuser_offline_metrics 的二分类曲线、混淆矩阵、五头 dataset-level 累加器与连续帧累加器。
[OUTPUT]: 验证 AP/AUC/IoU、逐类指标、正确 stop-sign head 映射、逐时域 waypoint 归约、目标条件连续帧残差与无支持门禁。
[POS]: tests 的 M2 H1 冻结 test 纯指标回归；不加载模型、数据集或 GPU。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import math
import unittest

import numpy as np

from tools.evaluation.interfuser_offline_metrics import (
    InterfuserMetricAccumulator,
    InterfuserTemporalAccumulator,
    MetricError,
    binary_confusion_metrics,
    binary_score_metrics,
)


class InterfuserOfflineMetricsTests(unittest.TestCase):
    def test_binary_confusion_reports_exact_macro_metrics(self):
        metrics = binary_confusion_metrics(
            np.array([0, 0, 1, 1]),
            np.array([0, 1, 1, 0]),
            ("negative", "positive"),
        )
        self.assertEqual(metrics["confusion_matrix_target_rows"], [[1, 1], [1, 1]])
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["macro_f1"], 0.5)
        self.assertEqual(metrics["per_class"]["positive"]["support"], 2)

    def test_binary_scores_report_perfect_threshold_and_ranking(self):
        metrics = binary_score_metrics(
            np.array([0, 0, 1, 1]),
            np.array([0.1, 0.4, 0.8, 0.7]),
            0.5,
        )
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["average_precision"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["occupied_iou"], 1.0)

    def _batch(self):
        traffic_target = np.zeros((2, 400, 7), dtype=np.float32)
        traffic_output = np.zeros((2, 400, 7), dtype=np.float32)
        traffic_target[0, 0] = [1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        traffic_target[1, 1] = [0.5, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        traffic_output[0, 0] = traffic_target[0, 0]
        traffic_output[1, 1] = traffic_target[1, 1]
        waypoint_target = np.zeros((2, 10, 2), dtype=np.float32)
        waypoint_output = np.ones((2, 10, 2), dtype=np.float32)
        labels = np.array([0, 1], dtype=np.int64)
        correct_logits = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        wrong_logits = correct_logits[::-1].copy()
        outputs = (
            traffic_output,
            waypoint_output,
            correct_logits,
            wrong_logits,
            correct_logits,
        )
        targets = (
            np.zeros((2, 1), dtype=np.float32),
            waypoint_target,
            labels,
            labels,
            traffic_target,
            np.zeros((2, 1), dtype=np.float32),
            labels,
        )
        return outputs, targets

    def test_accumulator_uses_correct_heads_and_exact_dataset_denominators(self):
        outputs, targets = self._batch()
        accumulator = InterfuserMetricAccumulator()
        accumulator.update(outputs, targets)
        metrics = accumulator.finalize()

        self.assertEqual(metrics["samples"], 2)
        self.assertEqual(metrics["traffic"]["positive_cells"], 2)
        self.assertEqual(metrics["traffic"]["negative_cells"], 798)
        self.assertEqual(metrics["traffic"]["occupancy"]["average_precision"], 1.0)
        self.assertAlmostEqual(metrics["waypoints"]["ade"], math.sqrt(2))
        self.assertEqual(metrics["waypoints"]["coordinate_mae"], 1.0)
        self.assertEqual(metrics["junction"]["accuracy"], 1.0)
        self.assertEqual(metrics["red_light"]["accuracy"], 0.0)
        self.assertEqual(metrics["stop_sign"]["accuracy"], 1.0)

    def test_accumulator_rejects_a_waypoint_horizon_without_support(self):
        outputs, targets = self._batch()
        targets = list(targets)
        targets[1] = targets[1].copy()
        targets[1][:, -1] = 10000.0
        accumulator = InterfuserMetricAccumulator()
        accumulator.update(outputs, targets)
        with self.assertRaisesRegex(MetricError, "every waypoint horizon"):
            accumulator.finalize()

    def test_binary_scores_reject_nonfinite_values(self):
        with self.assertRaisesRegex(MetricError, "non-finite"):
            binary_score_metrics([0, 1], [0.1, float("nan")], 0.5)

    def _temporal_batch(self):
        traffic_target = np.zeros((3, 400, 7), dtype=np.float32)
        traffic_output = traffic_target.copy()
        traffic_target[1, 0, 0] = 1.0
        traffic_output[1, 0, 0] = 0.75

        waypoint_target = np.zeros((3, 10, 2), dtype=np.float32)
        waypoint_output = waypoint_target.copy()
        waypoint_target[1, :, 0] = 1.0
        waypoint_output[1, :, 0] = 1.5

        def logits(predictions):
            return np.asarray(
                [[2.0, 0.0] if value == 0 else [0.0, 2.0] for value in predictions],
                dtype=np.float32,
            )

        junction_targets = np.array([0, 1, 0], dtype=np.int64)
        red_light_targets = np.array([0, 1, 0], dtype=np.int64)
        stop_sign_targets = np.array([0, 0, 0], dtype=np.int64)
        outputs = (
            traffic_output,
            waypoint_output,
            logits([0, 1, 0]),
            logits([0, 0, 0]),
            logits([0, 1, 0]),
        )
        targets = (
            np.zeros((3, 1), dtype=np.float32),
            waypoint_target,
            junction_targets,
            red_light_targets,
            traffic_target,
            np.zeros((3, 1), dtype=np.float32),
            stop_sign_targets,
        )
        return outputs, targets

    def test_temporal_accumulator_uses_only_adjacent_sequence_frames(self):
        outputs, targets = self._temporal_batch()
        accumulator = InterfuserTemporalAccumulator()
        accumulator.update(outputs, targets, ["sequence-a", "sequence-a", "sequence-b"], [0, 1, 0])
        metrics = accumulator.finalize()

        self.assertEqual(metrics["adjacent_pairs"], 1)
        self.assertEqual(metrics["sequences_with_pairs"], 1)
        self.assertAlmostEqual(
            metrics["traffic_probability_delta_residual_mae"], 0.25 / 400
        )
        self.assertAlmostEqual(metrics["waypoint_delta_residual_ade"], 0.5)
        self.assertEqual(metrics["binary_transition_error_rate"]["junction"], 0.0)
        self.assertEqual(metrics["binary_transition_error_rate"]["red_light"], 1.0)
        self.assertEqual(metrics["binary_transition_error_rate"]["stop_sign"], 1.0)

    def test_temporal_accumulator_requires_an_adjacent_pair(self):
        outputs, targets = self._temporal_batch()
        accumulator = InterfuserTemporalAccumulator()
        accumulator.update(
            tuple(value[:2] for value in outputs),
            tuple(value[:2] for value in targets),
            ["sequence-a", "sequence-b"],
            [0, 0],
        )
        with self.assertRaisesRegex(MetricError, "adjacent frame pairs"):
            accumulator.finalize()


if __name__ == "__main__":
    unittest.main()
