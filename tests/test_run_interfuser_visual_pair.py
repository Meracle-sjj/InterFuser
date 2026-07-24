"""
[INPUT]: 依赖 run_interfuser_visual_pair 的 smoke 索引抽样、训练命令、summary 与 args 可比性 API。
[OUTPUT]: 验证 B0/V 命令共享同一训练参数、smoke 抽样确定性、formal test index 强制绑定、允许的 provenance 字段被归一且真实预算漂移可见。
[POS]: tests 的 M2 H1 配对训练编排回归，不启动 GPU 或外部训练进程。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from tools.training.run_interfuser_visual_pair import (
    PairRunError,
    _normalized_args_hash,
    _parse_summary,
    _write_smoke_index,
    build_training_command,
    load_pair_run_contract,
    sha256_file,
)


class RunInterfuserVisualPairTests(unittest.TestCase):
    def _contract(self, root):
        root = Path(root)
        return {
            "status": "smoke",
            "dataset": {"towns": [1, 3], "weathers": [0, 1]},
            "training": {
                "gpus": [6, 7],
                "master_port": 29655,
                "model": "interfuser_baseline",
                "scheduler": "cosine",
                "epochs": 1,
                "warmup_epochs": 0,
                "cooldown_epochs": 0,
                "learning_rate": 0.0005,
                "batch_size_per_gpu": 2,
                "workers_per_process": 1,
                "seed": 7,
                "optimizer": "adamw",
                "weight_decay": 0.05,
                "scale": [0.9, 1.1],
                "color_jitter": 0.1,
                "clip_grad": 10.0,
                "backbone_learning_rate": 0.0002,
            },
            "resolved": {
                "dataset_root": root,
                "b0_initial_checkpoint": root / "b0.pth",
                "v_initial_checkpoint": root / "v.pth",
            },
        }

    def test_b0_v_commands_share_all_arguments_before_variant_provenance(self):
        with tempfile.TemporaryDirectory() as root:
            contract = self._contract(root)
            b0 = build_training_command(
                contract, "b0", "train.txt", "validation.txt", Path(root) / "b0"
            )
            v = build_training_command(
                contract, "v", "train.txt", "validation.txt", Path(root) / "v"
            )

        marker = b0.index("--initial-checkpoint")
        self.assertEqual(b0[:marker], v[:marker])
        self.assertNotEqual(b0[marker + 1], v[marker + 1])
        with self.assertRaisesRegex(PairRunError, "unknown variant"):
            build_training_command(
                contract, "x", "train.txt", "validation.txt", Path(root) / "x"
            )

    def test_smoke_index_sampling_is_deterministic(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "source.txt"
            source.write_text("c 3\na 1\nb 2\n", encoding="utf-8")
            first = Path(root) / "first.txt"
            second = Path(root) / "second.txt"
            first_info = _write_smoke_index(source, first, 2, 17, "train")
            second_info = _write_smoke_index(source, second, 2, 17, "train")
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first_info["sha256"], second_info["sha256"])
            self.assertEqual(first_info["sequences"], 2)

    def test_normalized_args_ignore_only_variant_provenance(self):
        with tempfile.TemporaryDirectory() as root:
            first = Path(root) / "first.yaml"
            second = Path(root) / "second.yaml"
            first.write_text(
                yaml.safe_dump(
                    {
                        "batch_size": 2,
                        "initial_checkpoint": "b0.pth",
                        "experiment": "b0",
                        "output": "b0-output",
                        "rank": 0,
                    }
                ),
                encoding="utf-8",
            )
            second.write_text(
                yaml.safe_dump(
                    {
                        "batch_size": 2,
                        "initial_checkpoint": "v.pth",
                        "experiment": "v",
                        "output": "v-output",
                        "rank": 0,
                    }
                ),
                encoding="utf-8",
            )
            first_hash, _ = _normalized_args_hash(first)
            second_hash, _ = _normalized_args_hash(second)
            self.assertEqual(first_hash, second_hash)
            value = yaml.safe_load(second.read_text(encoding="utf-8"))
            value["batch_size"] = 4
            second.write_text(yaml.safe_dump(value), encoding="utf-8")
            changed_hash, _ = _normalized_args_hash(second)

        self.assertNotEqual(first_hash, changed_hash)

    def test_formal_contract_requires_frozen_test_index(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            train = root / "train.txt"
            validation = root / "validation.txt"
            test = root / "test.txt"
            for path in (train, validation, test):
                path.write_text("sequence 1\n", encoding="utf-8")
            split_manifest = root / "split.json"
            split_manifest.write_text(
                json.dumps(
                    {
                        "valid": True,
                        "artifacts": {
                            name: {"sha256": sha256_file(path)}
                            for name, path in (
                                ("train", train),
                                ("validation", validation),
                                ("test", test),
                            )
                        },
                    }
                ),
                encoding="utf-8",
            )
            initialization_manifest = root / "initialization.json"
            initialization_manifest.write_text(
                json.dumps({"pipeline_valid": True}), encoding="utf-8"
            )
            config = root / "formal.json"
            config.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "formal",
                        "downstream_split_manifest": str(split_manifest),
                        "downstream_split_manifest_sha256": sha256_file(split_manifest),
                        "initialization_manifest": str(initialization_manifest),
                        "initialization_manifest_sha256": sha256_file(
                            initialization_manifest
                        ),
                        "dataset": {
                            "root": str(root),
                            "train_index": str(train),
                            "train_index_sha256": sha256_file(train),
                            "validation_index": str(validation),
                            "validation_index_sha256": sha256_file(validation),
                            "towns": [1],
                            "weathers": [0],
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PairRunError, "dataset.test_index"):
                load_pair_run_contract(config)

    def test_summary_requires_complete_finite_epoch_rows(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "summary.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["epoch", "train_loss", "eval_loss"]
                )
                writer.writeheader()
                writer.writerow({"epoch": 0, "train_loss": 1.0, "eval_loss": 0.5})
            rows = _parse_summary(path, 1)
            self.assertEqual(rows[0]["eval_loss"], 0.5)
            with self.assertRaisesRegex(PairRunError, "rows=1"):
                _parse_summary(path, 2)
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["epoch", "train_loss", "eval_loss"]
                )
                writer.writeheader()
                writer.writerow({"epoch": 1, "train_loss": 1.0, "eval_loss": 0.5})
            with self.assertRaisesRegex(PairRunError, "expected 0..0"):
                _parse_summary(path, 1)


if __name__ == "__main__":
    unittest.main()
