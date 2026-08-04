"""
[INPUT]: 依赖 tools.evaluation.analyze_interfuser_visual_d7_failures 的配对 manifest、Leaderboard 事件与连续控制帧归约 API。
[OUTPUT]: 提供 attempt 矩阵一致性、早期碰撞回归、起步窗口偏移与状态/指标差值的回归测试。
[POS]: tests 的 H1 闭环失败归因测试，保证分数退化能追溯到同路线同 seed 的原始违规事件。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import json
import csv
import tempfile
import unittest
from pathlib import Path

from tools.evaluation.analyze_interfuser_visual_d7_failures import (
    VisualD7FailureAnalysisError,
    analyze_visual_d7_failures,
)


def _attempt(attempt_id, route_id, seed, score, completion, status):
    return {
        "attempt_id": attempt_id,
        "route_id": route_id,
        "traffic_manager_seed": seed,
        "pipeline_valid": True,
        "leaderboard_result": {
            "valid": True,
            "status": status,
            "scores": {
                "score_composed": score,
                "score_route": completion,
                "score_penalty": 0.65,
            },
        },
    }


class VisualD7FailureAnalysisTests(unittest.TestCase):
    def _write_run(
        self,
        root,
        variant,
        attempts,
        event_by_attempt,
        control_lane_offset=0.1,
    ):
        run = Path(root) / variant
        run.mkdir()
        manifest = run / "run_manifest.json"
        manifest.write_text(json.dumps({"attempts": attempts}), encoding="utf-8")
        for attempt in attempts:
            attempt_dir = run / "attempts" / attempt["attempt_id"]
            attempt_dir.mkdir(parents=True)
            infractions = {
                "collisions_layout": event_by_attempt.get(
                    attempt["attempt_id"], []
                ),
                "collisions_vehicle": [],
                "collisions_pedestrian": [],
                "vehicle_blocked": [],
            }
            result = {
                "_checkpoint": {
                    "records": [
                        {
                            "infractions": infractions,
                            "meta": {"duration_game": 20.0},
                        }
                    ]
                }
            }
            (attempt_dir / "leaderboard_result.json").write_text(
                json.dumps(result), encoding="utf-8"
            )
            control_dir = attempt_dir / "sensor_data" / "route_fixture"
            control_dir.mkdir(parents=True)
            with (control_dir / "control.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                fields = [
                    "step",
                    "speed",
                    "steer",
                    "lane_offset",
                    "net_is_junction",
                    "aux_junction",
                    "ctrl0_x",
                    "ctrl1_x",
                    "lane_center_correction",
                ]
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for step in range(6):
                    writer.writerow(
                        {
                            "step": step,
                            "speed": 0.0 if step < 2 else 1.0,
                            "steer": control_lane_offset / 2,
                            "lane_offset": control_lane_offset,
                            "net_is_junction": 0.9 if variant == "v" else 0.1,
                            "aux_junction": 0.1 if variant == "v" else 0.9,
                            "ctrl0_x": control_lane_offset,
                            "ctrl1_x": control_lane_offset * 2,
                            "lane_center_correction": min(
                                control_lane_offset, 0.25
                            ),
                        }
                    )
        return manifest

    def test_finds_repeatable_early_layout_collision_regression(self):
        b0_attempts = [
            _attempt("route_39_seed_0", 39, 0, 40.0, 70.0, "timed out"),
            _attempt("route_39_seed_1", 39, 1, 39.0, 69.0, "timed out"),
        ]
        v_attempts = [
            _attempt("route_39_seed_0", 39, 0, 2.0, 3.0, "blocked"),
            _attempt("route_39_seed_1", 39, 1, 2.0, 3.0, "blocked"),
        ]
        b0_events = {
            item["attempt_id"]: [
                "Agent collided against object at (x=1.0, y=2.0, z=0.0), at time=35.0"
            ]
            for item in b0_attempts
        }
        v_events = {
            item["attempt_id"]: [
                "Agent collided against object at (x=3.0, y=4.0, z=0.0), at time=3.05"
            ]
            for item in v_attempts
        }
        with tempfile.TemporaryDirectory() as root:
            b0 = self._write_run(
                root, "b0", b0_attempts, b0_events, control_lane_offset=0.1
            )
            v = self._write_run(
                root, "v", v_attempts, v_events, control_lane_offset=1.0
            )
            report = analyze_visual_d7_failures(
                b0, v, early_control_steps=4, require_control=True
            )

        self.assertTrue(report["valid"])
        self.assertEqual(
            report["summary"]["routes_with_repeatable_early_layout_regression"],
            [39],
        )
        self.assertEqual(
            report["summary"]["early_layout_collision_regression_pairs"], 2
        )
        self.assertEqual(
            report["per_route"][0]["mean_v_minus_b0"]["route_completion"],
            -66.5,
        )
        self.assertEqual(
            report["pairs"][0]["v"]["first_layout_collision"]["location"]["x"],
            3.0,
        )
        self.assertAlmostEqual(
            report["pairs"][0]["early_control_v_minus_b0"][
                "mean_abs_lane_offset"
            ],
            0.9,
        )
        self.assertEqual(
            report["pairs"][0]["v"]["early_control"][
                "junction_positive_fraction"
            ],
            1.0,
        )
        self.assertEqual(
            report["pairs"][0]["v"]["early_control"][
                "raw_aux_junction_disagreement_fraction"
            ],
            1.0,
        )

    def test_rejects_mismatched_attempt_matrix(self):
        b0_attempts = [
            _attempt("route_39_seed_0", 39, 0, 1.0, 1.0, "blocked")
        ]
        v_attempts = [
            _attempt("route_39_seed_1", 39, 1, 1.0, 1.0, "blocked")
        ]
        with tempfile.TemporaryDirectory() as root:
            b0 = self._write_run(root, "b0", b0_attempts, {})
            v = self._write_run(root, "v", v_attempts, {})
            with self.assertRaisesRegex(
                VisualD7FailureAnalysisError, "attempt matrices differ"
            ):
                analyze_visual_d7_failures(b0, v)


if __name__ == "__main__":
    unittest.main()
