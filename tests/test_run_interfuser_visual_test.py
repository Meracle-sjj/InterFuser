"""
[INPUT]: 依赖 run_interfuser_visual_test 的预注册配置、formal manifest/checkpoint/summary 门禁与配对差值 API。
[OUTPUT]: 验证静态 test 契约、哈希漂移拒绝、训练未完成阻断、完整 B0/V 输入准入和固定方向差值。
[POS]: tests 的 M2 H1 冻结 test runner 纯文件回归；不启动模型、GPU 或外部进程。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import json
import tempfile
import unittest
from pathlib import Path

from tools.evaluation.run_interfuser_visual_test import (
    VisualTestError,
    _metric_delta,
    _worker_evaluate,
    load_visual_test_contract,
    sha256_file,
)


class RunInterfuserVisualTestTests(unittest.TestCase):
    def _write_json(self, path, value):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(value), encoding="utf-8")

    def _fixture(self, root, complete=True):
        root = Path(root)
        dataset_root = root / "dataset"
        dataset_root.mkdir()
        test_index = root / "test.txt"
        test_index.write_text("sequence 2\n", encoding="utf-8")
        test_hash = sha256_file(test_index)
        split_manifest = root / "split.json"
        self._write_json(
            split_manifest,
            {
                "valid": True,
                "artifacts": {"test": {"sha256": test_hash}},
                "summary": {"test": {"logical_frames": 2}},
            },
        )
        formal_config = root / "formal.json"
        self._write_json(
            formal_config,
            {
                "status": "formal",
                "training": {"epochs": 2},
                "dataset": {
                    "test_index_sha256": test_hash,
                    "towns": [1],
                    "weathers": [0],
                },
            },
        )
        formal_hash = sha256_file(formal_config)
        run_id = "formal-run"
        result_root = root / "results"
        run_dir = result_root / run_id
        run_dir.mkdir(parents=True)
        manifest_path = run_dir / "run_manifest.json"
        if complete:
            variants = []
            for name in ("b0", "v"):
                checkpoint = run_dir / f"{name}.pth"
                checkpoint.write_bytes(name.encode("ascii"))
                summary = run_dir / f"{name}.csv"
                summary.write_text(
                    "epoch,train_loss,eval_loss,eval_l1_error\n"
                    "0,1.0,0.9,0.8\n"
                    "1,0.7,0.6,0.5\n",
                    encoding="utf-8",
                )
                variants.append(
                    {
                        "variant": name,
                        "pipeline_valid": True,
                        "process_exit_code": 0,
                        "errors": [],
                        "state_tensors": 1132,
                        "state_schema_sha256": "a" * 64,
                        "best_epoch": 1,
                        "best_metric": 0.5,
                        "artifacts": {
                            "best_checkpoint": {
                                "path": str(checkpoint),
                                "sha256": sha256_file(checkpoint),
                            },
                            "summary": {
                                "path": str(summary),
                                "sha256": sha256_file(summary),
                            },
                        },
                    }
                )
            manifest = {
                "run_id": run_id,
                "status": "completed",
                "pipeline_valid": True,
                "mode": "formal",
                "errors": [],
                "git_status": "",
                "config_sha256": formal_hash,
                "comparability": {
                    "normalized_training_args_identical": True,
                    "only_initial_checkpoint_differs": True,
                },
                "variants": variants,
            }
        else:
            manifest = {
                "run_id": run_id,
                "status": "running",
                "pipeline_valid": False,
                "mode": "formal",
                "errors": [],
                "git_status": "",
                "config_sha256": formal_hash,
                "variants": [],
            }
        self._write_json(manifest_path, manifest)
        config = root / "test.json"
        self._write_json(
            config,
            {
                "schema_version": 1,
                "status": "preregistered",
                "run_id": "frozen-test-run",
                "training_config": str(formal_config),
                "training_config_sha256": formal_hash,
                "training_run_id": run_id,
                "downstream_split_manifest": str(split_manifest),
                "downstream_split_manifest_sha256": sha256_file(split_manifest),
                "dataset": {
                    "root": str(dataset_root),
                    "test_index": str(test_index),
                    "test_index_sha256": test_hash,
                    "logical_frames": 2,
                    "towns": [1],
                    "weathers": [0],
                },
                "model": {
                    "name": "interfuser_baseline",
                    "multi_view_input_size": [3, 128, 128],
                    "with_lidar": True,
                },
                "metrics": {
                    "traffic_positive_target_threshold": 0.01,
                    "traffic_prediction_threshold": 0.5,
                    "invalid_waypoint_threshold": 1000.0,
                    "require_both_binary_classes": True,
                },
                "runtime": {
                    "seed": 7,
                    "gpu": 0,
                    "batch_size": 2,
                    "workers": 0,
                    "log_interval_batches": 1,
                    "timeout_seconds_per_variant": 60,
                    "gpu_busy_memory_threshold_mb": 10,
                    "require_clean_git": True,
                },
                "result_root": str(result_root),
            },
        )
        return config, test_index

    def test_static_contract_is_valid_before_training_completes(self):
        with tempfile.TemporaryDirectory() as root:
            config, _ = self._fixture(root, complete=False)
            contract = load_visual_test_contract(
                config, require_training_complete=False, repo_root=root
            )
        self.assertEqual(contract["formal_epochs"], 2)
        self.assertEqual(contract["dataset"]["logical_frames"], 2)
        self.assertEqual(contract["run_id"], "frozen-test-run")

    def test_test_index_hash_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            config, test_index = self._fixture(root, complete=False)
            test_index.write_text("changed 2\n", encoding="utf-8")
            with self.assertRaisesRegex(VisualTestError, "SHA-256 mismatch"):
                load_visual_test_contract(
                    config, require_training_complete=False, repo_root=root
                )

    def test_incomplete_formal_training_blocks_test(self):
        with tempfile.TemporaryDirectory() as root:
            config, _ = self._fixture(root, complete=False)
            with self.assertRaisesRegex(VisualTestError, "completed and pipeline valid"):
                load_visual_test_contract(
                    config, require_training_complete=True, repo_root=root
                )

    def test_complete_formal_pair_resolves_checkpoints_and_schema(self):
        with tempfile.TemporaryDirectory() as root:
            config, _ = self._fixture(root, complete=True)
            contract = load_visual_test_contract(
                config, require_training_complete=True, repo_root=root
            )
        self.assertEqual(tuple(contract["resolved_variants"]), ("b0", "v"))
        self.assertEqual(contract["state_schema_sha256"], "a" * 64)

    def test_metric_delta_uses_v_minus_b0_for_fixed_fields(self):
        def metrics(offset):
            return {
                "traffic": {
                    "occupancy": {
                        "average_precision": 1 + offset,
                        "roc_auc": 2 + offset,
                        "occupied_iou": 3 + offset,
                    },
                    "probability_mae": 4 + offset,
                },
                "waypoints": {"ade": 5 + offset, "fde_horizon_10": 6 + offset},
                "junction": {"macro_f1": 7 + offset},
                "red_light": {"macro_f1": 8 + offset},
                "stop_sign": {"macro_f1": 9 + offset},
            }

        deltas = _metric_delta(metrics(0), metrics(0.25))
        self.assertTrue(all(value == 0.25 for value in deltas.values()))

    def test_worker_refuses_to_overwrite_before_loading_gpu_runtime(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "worker.json"
            output.write_text("existing", encoding="utf-8")
            with self.assertRaisesRegex(VisualTestError, "refusing to overwrite"):
                _worker_evaluate({}, "b0", output)


if __name__ == "__main__":
    unittest.main()
