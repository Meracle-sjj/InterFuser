#!/usr/bin/env python3
"""
[INPUT]: 依赖 calibrate_interfuser_occupancy_threshold 的划分/搜索/诊断入口、_write_score_dump 落盘与 binary_score_metrics 冻结口径。
[OUTPUT]: 对外提供 v6 纯校准诊断的划分确定性、约束检查、阈值搜索一致性与端到端合成回归测试。
[POS]: tests 的 M2 H1 校准诊断单元测试；不加载模型或数据集，保证诊断数学与冻结指标口径一致。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.evaluation.calibrate_interfuser_occupancy_threshold import (
    FIXED_THRESHOLD,
    CalibrationDiagnosticError,
    _check_split_constraints,
    run_calibration_diagnostic,
    split_route_groups,
    sweep_occupied_iou,
)
from tools.evaluation.interfuser_offline_metrics import binary_score_metrics
from tools.evaluation.run_interfuser_scene_validation import (
    SceneValidationError,
    _write_score_dump,
)


def _metadata(counts, pedestrian_groups):
    samples = []
    for group, frames in counts.items():
        for frame in range(frames):
            samples.append(
                {
                    "sequence_id": f"{group}/seq",
                    "frame_id": frame,
                    "pedestrian": group in pedestrian_groups,
                    "route_group": group,
                }
            )
    return samples


def _synthetic_groups():
    counts = {
        "Town01:route001": 40,
        "Town01:route013": 35,
        "Town03:route002": 45,
        "Town03:route318": 30,
        "Town04:route003": 42,
        "Town04:route094": 38,
        "Town05:route004": 44,
        "Town05:route025": 36,
    }
    pedestrian = {
        "Town01:route013",
        "Town03:route318",
        "Town04:route094",
        "Town05:route025",
    }
    return counts, pedestrian


class SplitRouteGroupsTest(unittest.TestCase):
    def test_same_seed_byte_identical(self):
        counts, pedestrian = _synthetic_groups()
        metadata = _metadata(counts, pedestrian)
        first = split_route_groups(metadata, 20260817)
        second = split_route_groups(metadata, 20260817)
        self.assertEqual(first, second)
        self.assertEqual(
            json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True)
        )

    def test_constraints_hold_on_accepted_split(self):
        counts, pedestrian = _synthetic_groups()
        split = split_route_groups(_metadata(counts, pedestrian), 20260817)
        towns = {"Town01", "Town03", "Town04", "Town05"}
        for subset in (split["calibration"], split["evaluation"]):
            self.assertEqual({group.split(":")[0] for group in subset}, towns)
            self.assertEqual(len(set(subset) & pedestrian), 2)
        frames_a = sum(counts[group] for group in split["calibration"])
        frames_b = sum(counts[group] for group in split["evaluation"])
        self.assertTrue(0.75 <= frames_a / frames_b <= 4.0 / 3.0)

    def test_constraint_checker_rejects_bad_assignment(self):
        counts, pedestrian = _synthetic_groups()
        towns = {"Town01", "Town03", "Town04", "Town05"}
        ped_sorted = sorted(pedestrian)
        missing_town = {
            "calibration": ["Town01:route001", ped_sorted[0]],
            "evaluation": [group for group in counts if group not in {"Town01:route001", ped_sorted[0]}],
        }
        self.assertFalse(
            _check_split_constraints(missing_town, counts, ped_sorted, towns)
        )
        unbalanced_ped = {
            "calibration": ped_sorted[:3] + ["Town01:route001"],
            "evaluation": ped_sorted[3:] + ["Town03:route002", "Town04:route003", "Town05:route004"],
        }
        self.assertFalse(
            _check_split_constraints(unbalanced_ped, counts, ped_sorted, towns)
        )
        skewed_frames = {
            "calibration": ["Town01:route001", "Town03:route002", ped_sorted[2], ped_sorted[3]],
            "evaluation": ["Town04:route003", "Town05:route004", ped_sorted[0], ped_sorted[1]],
        }
        self.assertTrue(
            _check_split_constraints(skewed_frames, counts, ped_sorted, towns)
        )
        broken_ratio = dict(counts)
        broken_ratio["Town01:route001"] = 1000
        assignment = {
            "calibration": ["Town01:route001", ped_sorted[0], ped_sorted[1]],
            "evaluation": [
                "Town03:route002",
                "Town04:route003",
                "Town05:route004",
                ped_sorted[2],
                ped_sorted[3],
            ],
        }
        self.assertFalse(
            _check_split_constraints(assignment, broken_ratio, ped_sorted, towns)
        )

    def test_odd_pedestrian_group_count_rejected(self):
        counts, pedestrian = _synthetic_groups()
        pedestrian = set(sorted(pedestrian)[:3])
        with self.assertRaises(CalibrationDiagnosticError):
            split_route_groups(_metadata(counts, pedestrian), 20260817)


class SweepOccupiedIoUTest(unittest.TestCase):
    def test_matches_frozen_metric_at_every_distinct_threshold(self):
        rng = np.random.RandomState(7)
        scores = rng.rand(500)
        targets = (rng.rand(500) > 0.7).astype(np.int64)
        sweep = sweep_occupied_iou(targets, scores)
        for threshold, iou in zip(
            sweep["thresholds"][:20], sweep["occupied_iou"][:20]
        ):
            expected = binary_score_metrics(targets, scores, float(threshold))[
                "occupied_iou"
            ]
            self.assertAlmostEqual(float(iou), expected, places=12)

    def test_best_threshold_on_crafted_example(self):
        targets = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.4, 0.8, 0.1])
        sweep = sweep_occupied_iou(targets, scores)
        self.assertAlmostEqual(sweep["best_threshold"], 0.4, places=12)
        self.assertAlmostEqual(sweep["best_occupied_iou"], 2.0 / 3.0, places=12)

    def test_single_class_rejected(self):
        with self.assertRaises(CalibrationDiagnosticError):
            sweep_occupied_iou(np.ones(10), np.linspace(0.1, 0.9, 10))


class ScoreDumpRoundTripTest(unittest.TestCase):
    def test_write_and_reload_with_hashes(self):
        rng = np.random.RandomState(3)
        dump = {
            "scores": [rng.rand(4, 400).astype(np.float32)],
            "targets": [(rng.rand(4, 400) > 0.5).astype(np.int8)],
            "metadata": _metadata({"Town01:route001": 2, "Town03:route002": 2}, set()),
        }
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = _write_score_dump(tmp, "b0", dump, 4)
            scores = np.load(artifacts["traffic_scores"]["path"])
            targets = np.load(artifacts["traffic_targets"]["path"])
            self.assertTrue(np.array_equal(scores, dump["scores"][0]))
            self.assertTrue(np.array_equal(targets, dump["targets"][0]))
            self.assertEqual(artifacts["traffic_scores"]["shape"], [4, 400])
            meta = json.loads(Path(artifacts["meta"]["path"]).read_text())
            self.assertEqual(len(meta["samples"]), 4)
            self.assertEqual(artifacts["meta"]["samples"], 4)
            for artifact in artifacts.values():
                self.assertEqual(len(artifact["sha256"]), 64)

    def test_sample_count_mismatch_rejected(self):
        dump = {
            "scores": [np.zeros((3, 400), dtype=np.float32)],
            "targets": [np.zeros((3, 400), dtype=np.int8)],
            "metadata": _metadata({"Town01:route001": 3}, set()),
        }
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SceneValidationError):
                _write_score_dump(tmp, "b0", dump, 4)


class CalibrationDiagnosticEndToEndTest(unittest.TestCase):
    def test_synthetic_run(self):
        counts, pedestrian = _synthetic_groups()
        metadata = _metadata(counts, pedestrian)
        rng = np.random.RandomState(11)
        samples = len(metadata)
        base_scores = rng.rand(samples, 400).astype(np.float32)
        targets = (rng.rand(samples, 400) > 0.95).astype(np.int8)
        candidate_scores = np.clip(base_scores + 0.05, 0.0, 1.0).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            scores_dir = Path(tmp) / "scores"
            for variant, scores in (("b0", base_scores), ("candidate", candidate_scores)):
                dump = {
                    "scores": [scores],
                    "targets": [targets],
                    "metadata": metadata,
                }
                _write_score_dump(scores_dir, variant, dump, samples)
            output = Path(tmp) / "calibration_manifest.json"
            manifest = run_calibration_diagnostic(scores_dir, output, 20260817)
            self.assertTrue(output.is_file())
            self.assertIsNone(manifest["stage2_admission"])
            self.assertFalse(manifest["test_accessed"])
            self.assertEqual(manifest["fixed_threshold"], FIXED_THRESHOLD)
            for variant in ("b0", "candidate"):
                entry = manifest["models"][variant]
                self.assertGreaterEqual(entry["t_star"], 0.0)
                self.assertLessEqual(entry["t_star"], 1.0)
                for subset in ("calibration", "evaluation", "full_validation"):
                    self.assertIn("fixed_threshold", entry[subset])
                    self.assertIn("calibrated_threshold", entry[subset])
            self.assertEqual(
                manifest["split"]["pedestrian_route_groups"], sorted(pedestrian)
            )

    def test_order_mismatch_rejected(self):
        counts, pedestrian = _synthetic_groups()
        metadata = _metadata(counts, pedestrian)
        shuffled = list(reversed(metadata))
        rng = np.random.RandomState(5)
        samples = len(metadata)
        with tempfile.TemporaryDirectory() as tmp:
            scores_dir = Path(tmp) / "scores"
            for variant, meta in (("b0", metadata), ("candidate", shuffled)):
                dump = {
                    "scores": [rng.rand(samples, 400).astype(np.float32)],
                    "targets": [(rng.rand(samples, 400) > 0.9).astype(np.int8)],
                    "metadata": meta,
                }
                _write_score_dump(scores_dir, variant, dump, samples)
            with self.assertRaises(CalibrationDiagnosticError):
                run_calibration_diagnostic(
                    scores_dir, Path(tmp) / "out.json", 20260817
                )


if __name__ == "__main__":
    unittest.main()
