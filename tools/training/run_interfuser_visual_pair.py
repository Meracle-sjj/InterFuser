#!/usr/bin/env python3
"""
[INPUT]: 依赖版本化 B0/V 训练配置、无泄漏下游 train/validation/test 索引、strict 初始 checkpoint 对、InterFuser train.py 与 GPU/端口资源守卫。
[OUTPUT]: 对外提供 PairRunError、load_pair_run_contract、build_training_command、execute_pair_run 与 CLI，串行生成 B0/V 训练产物、指标和配对 manifest。
[POS]: tools/training 的 M2 H1 下游训练编排器；复用上游 train.py 而不重写训练循环，任一 variant 基础设施失败即停止配对准入。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERFUSER_ROOT = REPO_ROOT / "interfuser"
for import_root in (REPO_ROOT, INTERFUSER_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tools.evaluation.runtime_resources import (  # noqa: E402
    RunnerError,
    _GpuMemoryMonitor,
    _stop_process_group,
    ensure_gpus_available,
    ensure_ports_free,
    wait_for_gpus_available,
    wait_for_ports_free,
)


SCHEMA_VERSION = 1
MANIFEST_VERSION = 1
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
VARIANTS = ("b0", "v")


class PairRunError(RuntimeError):
    """Raised when paired downstream training loses comparability or provenance."""


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _read_json(path, label):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PairRunError(f"unable to read {label} JSON {path}: {exc}") from exc


def _resolve_path(value, label):
    if not isinstance(value, str) or not value:
        raise PairRunError(f"{label} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _verify_hash(path, expected, label):
    if not isinstance(expected, str) or len(expected) != 64:
        raise PairRunError(f"{label} SHA-256 must contain 64 hex characters")
    actual = sha256_file(path)
    if actual != expected:
        raise PairRunError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def _positive_int(value, label, allow_zero=False):
    valid = isinstance(value, int) and not isinstance(value, bool)
    valid = valid and (value >= 0 if allow_zero else value > 0)
    if not valid:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise PairRunError(f"{label} must be a {qualifier} integer")
    return value


def _positive_number(value, label, allow_zero=False):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise PairRunError(f"{label} must be numeric")
    value = float(value)
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise PairRunError(f"{label} must be finite and nonnegative/positive")
    return value


def load_pair_run_contract(path):
    """Validate one paired training contract and every referenced artifact."""
    path = Path(path).resolve()
    raw = _read_json(path, "pair run config")
    if raw.get("schema_version") != SCHEMA_VERSION or raw.get("status") not in {
        "smoke",
        "formal",
    }:
        raise PairRunError("pair run config must be smoke/formal schema v1")
    resolved = {}
    for field in ("downstream_split_manifest", "initialization_manifest"):
        resolved[field] = _resolve_path(raw.get(field), field).resolve()
        _verify_hash(resolved[field], raw.get(f"{field}_sha256"), field)
    split_manifest = _read_json(resolved["downstream_split_manifest"], "downstream split")
    init_manifest = _read_json(resolved["initialization_manifest"], "initialization")
    if not split_manifest.get("valid"):
        raise PairRunError("downstream split manifest must be valid")
    if not init_manifest.get("pipeline_valid"):
        raise PairRunError("initialization manifest must be pipeline valid")

    dataset = raw.get("dataset")
    if not isinstance(dataset, dict):
        raise PairRunError("dataset must be an object")
    resolved["dataset_root"] = _resolve_path(dataset.get("root"), "dataset.root").resolve()
    if not resolved["dataset_root"].is_dir():
        raise PairRunError("dataset.root is not a directory")
    contract_splits = (
        ("train", "validation", "test")
        if raw["status"] == "formal"
        else ("train", "validation")
    )
    for split in contract_splits:
        field = f"{split}_index"
        resolved[field] = _resolve_path(dataset.get(field), f"dataset.{field}").resolve()
        _verify_hash(resolved[field], dataset.get(f"{field}_sha256"), field)
        manifest_hash = split_manifest.get("artifacts", {}).get(split, {}).get("sha256")
        if manifest_hash != dataset.get(f"{field}_sha256"):
            raise PairRunError(f"{field} differs from downstream split manifest")
    for field in ("towns", "weathers"):
        values = dataset.get(field)
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise PairRunError(f"dataset.{field} must be unique and non-empty")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise PairRunError(f"dataset.{field} must contain integers")

    variants = raw.get("variants")
    if not isinstance(variants, dict) or tuple(variants) != VARIANTS:
        raise PairRunError("variants must define b0 then v")
    for name in VARIANTS:
        value = variants[name]
        resolved[f"{name}_initial_checkpoint"] = _resolve_path(
            value.get("initial_checkpoint"), f"variants.{name}.initial_checkpoint"
        ).resolve()
        _verify_hash(
            resolved[f"{name}_initial_checkpoint"],
            value.get("initial_checkpoint_sha256"),
            f"{name} initial checkpoint",
        )
        init_hash = init_manifest.get("variants", {}).get(name, {}).get(
            "checkpoint_sha256"
        )
        if init_hash != value.get("initial_checkpoint_sha256"):
            raise PairRunError(f"{name} checkpoint differs from initialization manifest")

    sampling = raw.get("smoke_sampling")
    if raw["status"] == "smoke":
        if not isinstance(sampling, dict):
            raise PairRunError("smoke config requires smoke_sampling")
        _positive_int(sampling.get("seed"), "smoke_sampling.seed", allow_zero=True)
        for field in ("train_sequences", "validation_sequences"):
            _positive_int(sampling.get(field), f"smoke_sampling.{field}")
    elif sampling is not None:
        raise PairRunError("formal config must not define smoke_sampling")

    training = raw.get("training")
    if not isinstance(training, dict) or training.get("model") != "interfuser_baseline":
        raise PairRunError("training.model must be interfuser_baseline")
    for field in (
        "seed",
        "epochs",
        "batch_size_per_gpu",
        "workers_per_process",
        "warmup_epochs",
        "cooldown_epochs",
        "master_port",
        "timeout_seconds_per_variant",
        "gpu_busy_memory_threshold_mb",
    ):
        _positive_int(
            training.get(field),
            f"training.{field}",
            allow_zero=field in {"seed", "workers_per_process", "warmup_epochs", "cooldown_epochs"},
        )
    _positive_int(training.get("log_interval", 1), "training.log_interval")
    if training.get("optimizer") != "adamw" or training.get("scheduler") != "cosine":
        raise PairRunError("training optimizer/scheduler must be adamw/cosine")
    for field in (
        "learning_rate",
        "backbone_learning_rate",
        "weight_decay",
        "color_jitter",
        "clip_grad",
    ):
        _positive_number(
            training.get(field), f"training.{field}", allow_zero=field == "color_jitter"
        )
    scale = training.get("scale")
    if not isinstance(scale, list) or len(scale) != 2 or any(
        not isinstance(value, (int, float)) or value <= 0 for value in scale
    ):
        raise PairRunError("training.scale must contain two positive numbers")
    gpus = training.get("gpus")
    if not isinstance(gpus, list) or len(gpus) < 2 or len(gpus) != len(set(gpus)):
        raise PairRunError("training.gpus must contain at least two unique indices")
    if any(not isinstance(value, int) or value < 0 for value in gpus):
        raise PairRunError("training.gpus must contain nonnegative integers")
    if not isinstance(training.get("require_clean_git"), bool):
        raise PairRunError("training.require_clean_git must be boolean")
    result_root = _resolve_path(raw.get("result_root"), "result_root").resolve()
    try:
        result_root.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise PairRunError("result_root escapes repository") from exc
    normalized = dict(raw)
    normalized.update(
        {
            "path": path,
            "sha256": sha256_file(path),
            "resolved": resolved,
            "split_manifest_loaded": split_manifest,
            "initialization_manifest_loaded": init_manifest,
            "result_root_path": result_root,
        }
    )
    return normalized


def _read_index(path):
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        relative, frames = line.rsplit(maxsplit=1)
        records.append((relative, int(frames)))
    return records


def _write_smoke_index(source_path, output_path, limit, seed, split):
    records = _read_index(source_path)
    ranked = sorted(
        records,
        key=lambda item: (
            hashlib.sha256(f"{seed}:{split}:{item[0]}".encode()).hexdigest(),
            item[0],
        ),
    )
    selected = ranked[:limit]
    if len(selected) != limit:
        raise PairRunError(f"{split} smoke index has fewer than {limit} sequences")
    Path(output_path).write_text(
        "".join(f"{path} {frames}\n" for path, frames in selected), encoding="utf-8"
    )
    return {
        "path": str(Path(output_path).resolve()),
        "sha256": sha256_file(output_path),
        "sequences": len(selected),
        "logical_frames": sum(frames for _, frames in selected),
        "source_index_sha256": sha256_file(source_path),
        "sequence_paths": [path for path, _ in selected],
    }


def _shared_training_args(contract, train_index, validation_index):
    dataset = contract["dataset"]
    training = contract["training"]
    args = [
        str(contract["resolved"]["dataset_root"]),
        "--dataset",
        "carla",
        "--train-towns",
        *map(str, dataset["towns"]),
        "--val-towns",
        *map(str, dataset["towns"]),
        "--train-weathers",
        *map(str, dataset["weathers"]),
        "--val-weathers",
        *map(str, dataset["weathers"]),
        "--train-dataset-index",
        str(train_index),
        "--val-dataset-index",
        str(validation_index),
        "--model",
        training["model"],
        "--sched",
        training["scheduler"],
        "--epochs",
        str(training["epochs"]),
        "--warmup-epochs",
        str(training["warmup_epochs"]),
        "--cooldown-epochs",
        str(training["cooldown_epochs"]),
        "--lr",
        str(training["learning_rate"]),
        "--batch-size",
        str(training["batch_size_per_gpu"]),
        "-j",
        str(training["workers_per_process"]),
        "--seed",
        str(training["seed"]),
        "--no-prefetcher",
        "--eval-metric",
        "l1_error",
        "--opt",
        training["optimizer"],
        "--opt-eps",
        "1e-8",
        "--weight-decay",
        str(training["weight_decay"]),
        "--scale",
        *map(str, training["scale"]),
        "--color-jitter",
        str(training["color_jitter"]),
        "--saver-decreasing",
        "--clip-grad",
        str(training["clip_grad"]),
        "--freeze-num",
        "-1",
        "--with-backbone-lr",
        "--backbone-lr",
        str(training["backbone_learning_rate"]),
        "--multi-view",
        "--with-lidar",
        "--multi-view-input-size",
        "3",
        "128",
        "128",
        "--checkpoint-hist",
        "1",
        "--log-interval",
        str(training.get("log_interval", 1)),
    ]
    return args


def build_training_command(contract, variant, train_index, validation_index, output_root):
    if variant not in VARIANTS:
        raise PairRunError(f"unknown variant: {variant}")
    training = contract["training"]
    command = [
        sys.executable,
        "-m",
        "torch.distributed.launch",
        f"--nproc_per_node={len(training['gpus'])}",
        f"--master_port={training['master_port']}",
        "train.py",
        *_shared_training_args(contract, train_index, validation_index),
        "--initial-checkpoint",
        str(contract["resolved"][f"{variant}_initial_checkpoint"]),
        "--experiment",
        f"{variant}-{contract['status']}",
        "--output",
        str(output_root),
    ]
    return command


def _git_output(*args):
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise PairRunError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _parse_summary(path, expected_epochs):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected_epochs:
        raise PairRunError(
            f"summary rows={len(rows)} differ from epochs={expected_epochs}"
        )
    parsed = []
    for row in rows:
        values = {}
        for key, value in row.items():
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise PairRunError(f"summary field {key} is not numeric") from exc
            if not math.isfinite(number):
                raise PairRunError(f"summary field {key} is not finite")
            values[key] = number
        parsed.append(values)
    epochs = [row.get("epoch") for row in parsed]
    if epochs != [float(epoch) for epoch in range(expected_epochs)]:
        raise PairRunError(
            f"summary epochs {epochs} differ from expected 0..{expected_epochs - 1}"
        )
    return parsed


def _find_training_output(output_root):
    candidates = sorted(
        path.parent for path in Path(output_root).glob("*/summary.csv") if path.is_file()
    )
    if len(candidates) != 1:
        raise PairRunError(f"expected one training output directory, found {len(candidates)}")
    return candidates[0]


def _normalized_args_hash(args_path):
    value = yaml.safe_load(Path(args_path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PairRunError("args.yaml must contain an object")
    for field in ("initial_checkpoint", "experiment", "output", "local_rank", "rank"):
        value.pop(field, None)
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest(), value


def _state_schema(checkpoint, label):
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise PairRunError(f"{label} checkpoint has no state_dict")
    schema = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise PairRunError(f"{label} checkpoint state_dict is not tensor-only")
        schema[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    return schema


def _schema_sha256(schema):
    serialized = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _collect_variant_artifacts(output_root, contract, variant):
    output_dir = _find_training_output(output_root)
    required = {
        "summary": output_dir / "summary.csv",
        "args": output_dir / "args.yaml",
        "best_checkpoint": output_dir / "model_best.pth.tar",
        "last_checkpoint": output_dir / "last.pth.tar",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise PairRunError(f"training artifacts are missing: {missing}")
    summary = _parse_summary(required["summary"], contract["training"]["epochs"])
    args_hash, normalized_args = _normalized_args_hash(required["args"])
    checkpoint = torch.load(
        required["best_checkpoint"], map_location="cpu", weights_only=False
    )
    if checkpoint.get("arch") != contract["training"]["model"]:
        raise PairRunError("best checkpoint architecture differs")
    initial_checkpoint = torch.load(
        contract["resolved"][f"{variant}_initial_checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_schema = _state_schema(checkpoint, "best")
    initial_schema = _state_schema(initial_checkpoint, "initial")
    if checkpoint_schema != initial_schema:
        raise PairRunError("best checkpoint state schema differs from initial checkpoint")
    best_epoch = checkpoint.get("epoch")
    if (
        not isinstance(best_epoch, int)
        or isinstance(best_epoch, bool)
        or not 0 <= best_epoch < contract["training"]["epochs"]
    ):
        raise PairRunError("best checkpoint epoch is outside the training budget")
    try:
        best_metric = float(checkpoint["metric"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PairRunError("best checkpoint metric is missing or nonnumeric") from exc
    if not math.isfinite(best_metric):
        raise PairRunError("best checkpoint metric is not finite")
    return {
        "output_directory": str(output_dir),
        "summary_rows": summary,
        "normalized_args_sha256": args_hash,
        "normalized_args": normalized_args,
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in required.items()
        },
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "state_tensors": len(checkpoint_schema),
        "state_schema_sha256": _schema_sha256(checkpoint_schema),
    }


def _run_variant(contract, variant, train_index, validation_index, run_dir):
    training = contract["training"]
    variant_dir = run_dir / variant
    variant_dir.mkdir()
    output_root = variant_dir / "output"
    command = build_training_command(
        contract, variant, train_index, validation_index, output_root
    )
    log_path = variant_dir / "launcher.log"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, training["gpus"]))
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), str(INTERFUSER_ROOT), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    environment.setdefault("OMP_NUM_THREADS", "1")
    ensure_gpus_available(
        training["gpus"], training["gpu_busy_memory_threshold_mb"]
    )
    ensure_ports_free([training["master_port"]])
    monitor = _GpuMemoryMonitor(training["gpus"])
    process = None
    started = time.monotonic()
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log:
        try:
            monitor.start()
            process = subprocess.Popen(
                command,
                cwd=INTERFUSER_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                exit_code = process.wait(timeout=training["timeout_seconds_per_variant"])
            except subprocess.TimeoutExpired:
                timed_out = True
                exit_code = 124
        finally:
            if process is not None:
                _stop_process_group(process)
            peaks = monitor.stop()
    port_release = wait_for_ports_free([training["master_port"]])
    gpu_release = wait_for_gpus_available(
        training["gpus"], training["gpu_busy_memory_threshold_mb"]
    )
    result = {
        "variant": variant,
        "command": command,
        "launcher_log": str(log_path),
        "launcher_log_sha256": sha256_file(log_path),
        "process_exit_code": exit_code,
        "external_timeout": timed_out,
        "duration_seconds": round(time.monotonic() - started, 3),
        "gpu_peak_memory_mb": peaks,
        "gpu_monitor_error": monitor.error,
        "gpu_release_wait_seconds": gpu_release,
        "port_release_wait_seconds": port_release,
        "pipeline_valid": False,
        "errors": [],
    }
    if exit_code != 0 or timed_out:
        result["errors"].append(
            "training timed out" if timed_out else f"training exited with {exit_code}"
        )
        return result
    if monitor.error:
        result["errors"].append(f"GPU monitor failed: {monitor.error}")
        return result
    try:
        result.update(_collect_variant_artifacts(output_root, contract, variant))
        result["pipeline_valid"] = True
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    return result


def execute_pair_run(config_path, run_id):
    """Run B0 then V and admit the pair only when both pipelines are valid."""
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise PairRunError("run_id must use letters, digits, dot, dash or underscore")
    contract = load_pair_run_contract(config_path)
    git_status = _git_output("status", "--porcelain")
    if contract["training"]["require_clean_git"] and git_status:
        raise PairRunError("Git worktree must be clean")
    git_head = _git_output("rev-parse", "HEAD")
    run_dir = contract["result_root_path"] / run_id
    if run_dir.exists():
        raise PairRunError(f"refusing to overwrite run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "run_manifest.json"
    manifest = {
        "manifest_schema_version": MANIFEST_VERSION,
        "run_id": run_id,
        "status": "running",
        "pipeline_valid": False,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "git_head": git_head,
        "git_status": git_status,
        "config": str(contract["path"]),
        "config_sha256": contract["sha256"],
        "mode": contract["status"],
        "inputs": {
            "downstream_split_manifest": {
                "path": str(contract["resolved"]["downstream_split_manifest"]),
                "sha256": sha256_file(
                    contract["resolved"]["downstream_split_manifest"]
                ),
            },
            "initialization_manifest": {
                "path": str(contract["resolved"]["initialization_manifest"]),
                "sha256": sha256_file(
                    contract["resolved"]["initialization_manifest"]
                ),
            },
            "initial_checkpoints": {
                name: {
                    "path": str(
                        contract["resolved"][f"{name}_initial_checkpoint"]
                    ),
                    "sha256": contract["variants"][name][
                        "initial_checkpoint_sha256"
                    ],
                }
                for name in VARIANTS
            },
        },
        "errors": [],
        "variants": [],
    }
    _write_json(manifest_path, manifest)
    try:
        if contract["status"] == "smoke":
            sampling = contract["smoke_sampling"]
            train_info = _write_smoke_index(
                contract["resolved"]["train_index"],
                run_dir / "train_smoke_dataset_index.txt",
                sampling["train_sequences"],
                sampling["seed"],
                "train",
            )
            validation_info = _write_smoke_index(
                contract["resolved"]["validation_index"],
                run_dir / "validation_smoke_dataset_index.txt",
                sampling["validation_sequences"],
                sampling["seed"],
                "validation",
            )
            train_index = Path(train_info["path"])
            validation_index = Path(validation_info["path"])
        else:
            train_index = contract["resolved"]["train_index"]
            validation_index = contract["resolved"]["validation_index"]
            train_info = {
                "path": str(train_index),
                "sha256": sha256_file(train_index),
            }
            validation_info = {
                "path": str(validation_index),
                "sha256": sha256_file(validation_index),
            }
        manifest["data"] = {"train": train_info, "validation": validation_info}
        _write_json(manifest_path, manifest)

        for variant in VARIANTS:
            result = _run_variant(
                contract, variant, train_index, validation_index, run_dir
            )
            manifest["variants"].append(result)
            manifest["updated_at"] = _utc_now()
            _write_json(manifest_path, manifest)
            if not result["pipeline_valid"]:
                raise PairRunError(
                    f"{variant} pipeline invalid: {'; '.join(result['errors'])}"
                )
        args_hashes = {
            result["normalized_args_sha256"] for result in manifest["variants"]
        }
        if len(args_hashes) != 1:
            raise PairRunError("B0/V normalized training arguments differ")
        manifest.update(
            {
                "status": "completed",
                "pipeline_valid": True,
                "updated_at": _utc_now(),
                "completed_at": _utc_now(),
                "comparability": {
                    "normalized_training_args_identical": True,
                    "normalized_training_args_sha256": next(iter(args_hashes)),
                    "variant_order": list(VARIANTS),
                    "only_initial_checkpoint_differs": True,
                },
            }
        )
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "pipeline_valid": False,
                "updated_at": _utc_now(),
                "completed_at": _utc_now(),
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
        )
        _write_json(manifest_path, manifest)
        raise
    _write_json(manifest_path, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run provenance-locked InterFuser B0/V paired training"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    try:
        manifest = execute_pair_run(args.config, args.run_id)
    except (PairRunError, RunnerError) as exc:
        print(f"pair run error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"pair run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(manifest["comparability"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
