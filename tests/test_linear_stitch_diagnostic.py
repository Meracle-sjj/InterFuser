#!/usr/bin/env python3
"""
[INPUT]: 依赖 linear_stitch_diagnostic 的纯函数/累加器接口与合成张量。
[OUTPUT]: 对外提供 v8a 线性缝合诊断的单元测试：确定性划分、岭回归闭式解、误差消减口径、判定规则。
[POS]: tests 的 v8a 诊断组件测试；不触碰 GPU、数据划分与真实 checkpoint。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import hashlib
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPO_ROOT, REPO_ROOT / "interfuser"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import torch  # noqa: E402

from tools.evaluation.linear_stitch_diagnostic import (  # noqa: E402
    LinearStitchDiagnosticError,
    StageEvalAccumulator,
    StageFitAccumulator,
    deterministic_frame_split,
    evaluate_stitch_decision,
    solve_ridge,
)


class DeterministicFrameSplitTest(unittest.TestCase):
    def test_split_is_deterministic_and_disjoint(self):
        keys = [f"seq{i:03d}:front:{j:05d}" for i in range(8) for j in range(12)]
        fit_a, eval_a = deterministic_frame_split(keys, 20260817)
        fit_b, eval_b = deterministic_frame_split(keys, 20260817)
        self.assertEqual(fit_a, fit_b)
        self.assertEqual(eval_a, eval_b)
        self.assertEqual(set(fit_a) & set(eval_a), set())
        self.assertEqual(sorted(fit_a + eval_a), sorted(keys))

    def test_split_ratio_is_roughly_balanced(self):
        keys = [f"key-{index}" for index in range(1725)]
        fit, evaluation = deterministic_frame_split(keys, 20260817)
        ratio = len(fit) / len(keys)
        self.assertGreater(ratio, 0.4)
        self.assertLess(ratio, 0.6)

    def test_seed_change_flips_assignment(self):
        keys = [f"key-{index}" for index in range(64)]
        fit_a, _ = deterministic_frame_split(keys, 1)
        fit_b, _ = deterministic_frame_split(keys, 2)
        self.assertNotEqual(fit_a, fit_b)

    def test_empty_keys_raise(self):
        with self.assertRaises(LinearStitchDiagnosticError):
            deterministic_frame_split([], 20260817)


class RidgeClosedFormTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.device = torch.device("cpu")

    def _make_stage_pair(self, channels, transform=None, noise=0.0):
        student = torch.randn(2, channels, 6, 8, dtype=torch.float32)
        flat = student.permute(0, 2, 3, 1).reshape(-1, channels)
        if transform is None:
            teacher_flat = flat.clone()
        else:
            teacher_flat = flat @ transform
        teacher_flat = teacher_flat + noise * torch.randn_like(teacher_flat)
        teacher = teacher_flat.reshape(2, 6, 8, channels).permute(0, 3, 1, 2).contiguous()
        return student, teacher

    def test_exact_linear_transform_is_fully_recovered(self):
        channels = 16
        transform = torch.randn(channels, channels, dtype=torch.float64)
        student, teacher = self._make_stage_pair(
            channels, transform=transform.to(torch.float32)
        )
        fit = StageFitAccumulator(channels, self.device)
        fit.update(student, teacher)
        weight, lam = solve_ridge(fit.gram_xx, fit.gram_xy)
        evaluation = StageEvalAccumulator(self.device)
        evaluation.update(student, teacher, weight)
        result = evaluation.result()
        self.assertAlmostEqual(result["stitched_relative_error"], 0.0, places=5)
        self.assertAlmostEqual(result["error_reduction_ratio"], 1.0, places=4)
        self.assertGreater(lam, 0.0)

    def test_identity_baseline_preserved_when_unrelated(self):
        channels = 4
        student = torch.randn(2, channels, 24, 24)
        teacher = torch.randn(2, channels, 24, 24)
        fit = StageFitAccumulator(channels, self.device)
        fit.update(student, teacher)
        weight, _ = solve_ridge(fit.gram_xx, fit.gram_xy)
        evaluation = StageEvalAccumulator(self.device)
        evaluation.update(student, teacher, weight)
        result = evaluation.result()
        self.assertAlmostEqual(result["identity_relative_error"], 2.0, delta=0.6)
        self.assertLess(abs(result["error_reduction_ratio"]), 0.5)

    def test_shape_mismatch_raises(self):
        fit = StageFitAccumulator(4, self.device)
        with self.assertRaises(LinearStitchDiagnosticError):
            fit.update(torch.randn(1, 4, 4, 4), torch.randn(1, 8, 4, 4))

    def test_cosine_improves_under_exact_transform(self):
        channels = 8
        transform = torch.eye(channels) * 2.0
        student, teacher = self._make_stage_pair(channels, transform=transform)
        fit = StageFitAccumulator(channels, self.device)
        fit.update(student, teacher)
        weight, _ = solve_ridge(fit.gram_xx, fit.gram_xy)
        evaluation = StageEvalAccumulator(self.device)
        evaluation.update(student, teacher, weight)
        result = evaluation.result()
        self.assertGreater(result["cosine_stitched_mean"], 0.99)


class StitchDecisionTest(unittest.TestCase):
    def _arms(self, reduction):
        return {
            "v7_distilled": {
                "stages": [
                    {"stage_index": 1, "error_reduction_ratio": 0.1},
                    {"stage_index": 4, "error_reduction_ratio": reduction},
                ]
            }
        }

    def test_viable_at_or_above_threshold(self):
        decision = evaluate_stitch_decision(self._arms(0.5), "v7_distilled", 0.5)
        self.assertTrue(decision["linear_stitch_viable"])
        self.assertEqual(decision["final_stage"], 4)

    def test_not_viable_below_threshold(self):
        decision = evaluate_stitch_decision(self._arms(0.4999), "v7_distilled", 0.5)
        self.assertFalse(decision["linear_stitch_viable"])

    def test_missing_primary_arm_raises(self):
        with self.assertRaises(LinearStitchDiagnosticError):
            evaluate_stitch_decision(self._arms(0.9), "other_arm", 0.5)


if __name__ == "__main__":
    unittest.main()
