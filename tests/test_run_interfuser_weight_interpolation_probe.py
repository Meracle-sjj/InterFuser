"""
[INPUT]: 依赖 run_interfuser_weight_interpolation_probe 的张量插值与稳定性-可塑性门禁 API，以浮点参数、离散BN缓冲和固定方向差值构造最小证据。
[OUTPUT]: 验证浮点线性插值、非浮点B0保留、schema拒绝及行人收益/普通保持/时序保持四项联合准入。
[POS]: tests 的 M2 H1 负迁移修复回归；阻止错误插值或只看行人聚合收益的候选进入完整重训。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import unittest

import torch

from tools.evaluation.run_interfuser_weight_interpolation_probe import (
    InterpolationProbeError,
    evaluate_candidate_gate,
    interpolate_state_dicts,
)


class WeightInterpolationProbeTests(unittest.TestCase):
    def test_interpolates_floats_and_copies_b0_discrete_buffers(self):
        b0 = {
            "weight": torch.tensor([0.0, 2.0]),
            "num_batches_tracked": torch.tensor(11, dtype=torch.int64),
        }
        v = {
            "weight": torch.tensor([2.0, 4.0]),
            "num_batches_tracked": torch.tensor(99, dtype=torch.int64),
        }

        state, summary = interpolate_state_dicts(b0, v, 0.5)

        torch.testing.assert_close(state["weight"], torch.tensor([1.0, 3.0]))
        self.assertEqual(state["num_batches_tracked"].item(), 11)
        self.assertEqual(summary["floating_tensors_interpolated"], 1)
        self.assertEqual(summary["nonfloating_tensors_copied_from_b0"], 1)
        self.assertEqual(summary["nonfloating_source_differences"], 1)

    def test_rejects_state_schema_drift(self):
        with self.assertRaisesRegex(InterpolationProbeError, "keys/order"):
            interpolate_state_dicts(
                {"a": torch.tensor([1.0])},
                {"b": torch.tensor([1.0])},
                0.5,
            )

    @staticmethod
    def _metric(relative, higher=True, improved=True):
        return {
            "relative_percent": relative,
            "higher_is_better": higher,
            "v_improved": improved,
        }

    def _comparison(self, nonpedestrian_harm=2.0, temporal_harm=5.0):
        pedestrian = {
            "traffic_average_precision": self._metric(5.0, True, True),
            "traffic_roc_auc": self._metric(-1.0, True, False),
            "traffic_occupied_iou": self._metric(4.0, True, True),
            "waypoint_ade": self._metric(-6.0, False, True),
            "waypoint_fde_horizon_10": self._metric(1.0, False, False),
            "traffic_delta_residual_mae": self._metric(
                temporal_harm, False, temporal_harm < 0
            ),
            "waypoint_delta_residual_ade": self._metric(2.0, False, False),
        }
        overall = {
            "traffic_average_precision": self._metric(-2.0, True, False),
            "traffic_roc_auc": self._metric(-1.0, True, False),
            "traffic_occupied_iou": self._metric(-1.5, True, False),
        }
        nonpedestrian = {
            "traffic_average_precision": self._metric(
                -nonpedestrian_harm, True, False
            ),
            "traffic_roc_auc": self._metric(-1.0, True, False),
            "traffic_occupied_iou": self._metric(-1.5, True, False),
            "waypoint_ade": self._metric(1.0, False, False),
            "waypoint_fde_horizon_10": self._metric(2.0, False, False),
        }
        return {
            "cohorts": {
                "pedestrian": pedestrian,
                "overall": overall,
                "nonpedestrian": nonpedestrian,
            }
        }

    def test_requires_pedestrian_gain_and_retention_gates_together(self):
        acceptance = {
            "minimum_pedestrian_core_improvements": 3,
            "maximum_nonpedestrian_core_degradation_percent": 3.0,
            "maximum_overall_traffic_degradation_percent": 3.0,
            "maximum_pedestrian_temporal_degradation_percent": 10.0,
        }

        passed = evaluate_candidate_gate(self._comparison(), acceptance)
        failed = evaluate_candidate_gate(
            self._comparison(nonpedestrian_harm=7.0), acceptance
        )

        self.assertTrue(passed["passed"])
        self.assertEqual(passed["pedestrian_core_improvements"], 3)
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["checks"]["nonpedestrian_retention"])


if __name__ == "__main__":
    unittest.main()
