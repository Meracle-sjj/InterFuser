"""
[INPUT]: 依赖 run_interfuser_scene_validation 的 provenance contract 与固定方向指标差值 API，以临时 manifest/checkpoint/index 构造隔离证据链。
[OUTPUT]: 验证只读 validation 的哈希绑定、test index 拒绝、B0/V best checkpoint 归属和行人核心多数票方向。
[POS]: tests 的 M2 H1 scene validation 回归；阻止未完成训练、测试集泄漏或指标方向错误进入视觉预训练结论。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.evaluation.run_interfuser_scene_validation import (
    SceneValidationError,
    build_metric_comparison,
    load_scene_validation_contract,
)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SceneValidationTests(unittest.TestCase):
    def _fixture(self, root):
        root = Path(root)
        dataset = root / "dataset"
        dataset.mkdir()
        validation_index = dataset / "validation.txt"
        validation_index.write_text(
            "town01/town01_tiny_route01_w0_Clear/a 4\n", encoding="utf-8"
        )
        split = _write_json(
            root / "split.json",
            {
                "valid": True,
                "manifest_schema_version": 2,
                "artifacts": {
                    "validation": {"sha256": _sha256(validation_index)}
                },
                "scene_stratification": {
                    "split_scene_summary": {
                        "validation": {
                            "effective_frames": 4,
                            "pedestrian_frames": 2,
                            "pedestrian_sequences": 1,
                            "pedestrian_route_groups": 1,
                        }
                    }
                },
            },
        )
        training = _write_json(root / "training.json", {"status": "pilot"})
        checkpoints = {}
        for name in ("b0", "v"):
            path = root / f"{name}.pth"
            path.write_bytes(name.encode("utf-8"))
            checkpoints[name] = path
        pair = _write_json(
            root / "pair.json",
            {
                "status": "completed",
                "pipeline_valid": True,
                "config": str(training.resolve()),
                "config_sha256": _sha256(training),
                "comparability": {
                    "normalized_training_args_identical": True,
                    "only_initial_checkpoint_differs": True,
                    "variant_order": ["b0", "v"],
                },
                "inputs": {
                    "downstream_split_manifest": {
                        "path": str(split.resolve()),
                        "sha256": _sha256(split),
                    }
                },
                "variants": [
                    {
                        "variant": name,
                        "artifacts": {
                            "best_checkpoint": {
                                "path": str(checkpoints[name].resolve()),
                                "sha256": _sha256(checkpoints[name]),
                            }
                        },
                    }
                    for name in ("b0", "v")
                ],
            },
        )
        config = {
            "schema_version": 1,
            "status": "frozen",
            "run_id": "scene-validation-fixture",
            "pair_run_manifest": {"path": str(pair), "sha256": _sha256(pair)},
            "pair_training_config": {
                "path": str(training),
                "sha256": _sha256(training),
            },
            "downstream_split_manifest": {
                "path": str(split),
                "sha256": _sha256(split),
            },
            "model": "interfuser_baseline",
            "dataset": {
                "root": str(dataset),
                "validation_index": str(validation_index),
                "validation_index_sha256": _sha256(validation_index),
                "towns": [1],
                "weathers": [0],
                "lidar_y_axis_multiplier": 1.0,
                "navigation_frame": "carla0916_compass",
                "missing_navigation_policy": "drop",
            },
            "scene": {
                "pedestrian_measurement_field": "is_pedestrian_present",
                "effective_frames": 4,
                "pedestrian_frames": 2,
                "pedestrian_sequences": 1,
                "pedestrian_route_groups": 1,
                "temporal": {
                    "overall_adjacent_pairs": 3,
                    "overall_sequences_with_pairs": 1,
                    "pedestrian_adjacent_pairs": 1,
                    "pedestrian_sequences_with_pairs": 1,
                },
            },
            "variants": {
                name: {
                    "checkpoint": str(checkpoints[name]),
                    "checkpoint_sha256": _sha256(checkpoints[name]),
                }
                for name in ("b0", "v")
            },
            "metrics": {
                "traffic_positive_target_threshold": 0.01,
                "traffic_prediction_threshold": 0.5,
                "invalid_waypoint_threshold": 1000.0,
            },
            "runtime": {
                "seed": 1,
                "gpu": 0,
                "batch_size": 2,
                "workers": 0,
                "log_interval_batches": 1,
                "gpu_minimum_free_memory_mb": 1,
                "require_clean_git": False,
            },
            "result_root": str(root / "results"),
        }
        return _write_json(root / "config.json", config)

    def test_loads_completed_pair_and_bound_validation_only(self):
        with tempfile.TemporaryDirectory() as root:
            config = self._fixture(root)
            contract = load_scene_validation_contract(config, repo_root=root)

        self.assertEqual(tuple(contract["resolved_variants"]), ("b0", "v"))
        self.assertEqual(contract["scene"]["pedestrian_frames"], 2)

    def test_rejects_test_index_exposure(self):
        with tempfile.TemporaryDirectory() as root:
            config = self._fixture(root)
            value = json.loads(config.read_text(encoding="utf-8"))
            value["dataset"]["test_index"] = "forbidden.txt"
            config.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(SceneValidationError, "test_index"):
                load_scene_validation_contract(config, repo_root=root)

    @staticmethod
    def _cohort(scale):
        return {
            "traffic": {
                "occupancy": {
                    "average_precision": 0.50 * scale,
                    "roc_auc": 0.60 * scale,
                    "occupied_iou": 0.40 * scale,
                }
            },
            "waypoints": {"ade": 0.30 / scale, "fde_horizon_10": 0.50 / scale},
            "junction": {"macro_f1": 0.80 * scale},
            "red_light": {"macro_f1": 0.70 * scale},
            "temporal": {
                "traffic_probability_delta_residual_mae": 0.10 / scale,
                "waypoint_delta_residual_ade": 0.20 / scale,
            },
        }

    def test_builds_fixed_direction_pedestrian_majority_gate(self):
        b0 = {
            "metrics": {
                "overall": self._cohort(1.0),
                "pedestrian": self._cohort(1.0),
                "nonpedestrian": self._cohort(1.0),
            },
            "pedestrian_route_group_metrics": {
                "Town01:route001": self._cohort(1.0)
            },
        }
        v = {
            "metrics": {
                "overall": self._cohort(1.1),
                "pedestrian": self._cohort(1.1),
                "nonpedestrian": self._cohort(1.1),
            },
            "pedestrian_route_group_metrics": {
                "Town01:route001": self._cohort(1.1)
            },
        }

        comparison = build_metric_comparison(b0, v)

        self.assertEqual(comparison["gate"]["pedestrian_core_improved"], 5)
        self.assertTrue(comparison["gate"]["pedestrian_majority_supports_v"])
        self.assertTrue(comparison["gate"]["test_remains_frozen"])
        self.assertLess(
            comparison["cohorts"]["pedestrian"]["waypoint_ade"]["v_minus_b0"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
