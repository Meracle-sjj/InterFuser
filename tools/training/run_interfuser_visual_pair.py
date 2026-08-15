#!/usr/bin/env python3
"""
[INPUT]: 依赖版本化 B0/V 训练配置、无泄漏下游索引、strict 初始 checkpoint 对、显式传感器坐标契约、InterFuser train.py 与独占/共享 GPU 资源守卫。
[OUTPUT]: 对外提供 PairRunError、load_pair_run_contract、build_training_command、execute_pair_run 与 CLI，串行生成 smoke/pilot/formal 的 B0/V 训练产物和配对 manifest。
[POS]: tools.training 的 M2 H1 Stage 2 编排器；复用 train.py 并冻结唯一初始化变量，任一数据、资源或产物漂移即停止配对准入。
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
    ensure_gpus_have_free_memory,
    ensure_ports_free,
    gpu_processes_in_process_groups,
    wait_for_gpu_processes_exit,
    wait_for_gpus_available,
    wait_for_ports_free,
)
from tools.training.interfuser_pair_contract import (  # noqa: E402
    PairRunError,
    VARIANTS,
    load_pair_run_contract,
    sha256_file,
)


MANIFEST_VERSION = 1
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


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
        "--lidar-y-axis-multiplier",
        str(dataset.get("lidar_y_axis_multiplier", -1.0)),
        "--navigation-frame",
        dataset.get("navigation_frame", "carla0916_standard_ego"),
        "--missing-navigation-policy",
        dataset.get("missing_navigation_policy", "fallback"),
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
    expected_samples = dataset.get("expected_effective_samples")
    if expected_samples is not None:
        args.extend(
            [
                "--expected-train-samples",
                str(expected_samples["train"]),
                "--expected-val-samples",
                str(expected_samples["validation"]),
            ]
        )
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
    gpu_resource_policy = training.get("gpu_resource_policy", "exclusive")
    if gpu_resource_policy == "shared_capacity":
        ensure_gpus_have_free_memory(
            training["gpus"], training["gpu_minimum_free_memory_mb"]
        )
    else:
        ensure_gpus_available(
            training["gpus"], training["gpu_busy_memory_threshold_mb"]
        )
    ensure_ports_free([training["master_port"]])
    monitor = _GpuMemoryMonitor(training["gpus"])
    process = None
    owned_gpu_processes = []
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
            if process is not None and gpu_resource_policy == "shared_capacity":
                owned_gpu_processes = gpu_processes_in_process_groups(
                    training["gpus"], [process.pid]
                )
            if process is not None:
                _stop_process_group(process)
            peaks = monitor.stop()
    port_release = wait_for_ports_free([training["master_port"]])
    if gpu_resource_policy == "shared_capacity":
        gpu_release = wait_for_gpu_processes_exit(
            [int(item["pid"]) for item in owned_gpu_processes]
        )
    else:
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
        "gpu_resource_policy": gpu_resource_policy,
        "gpu_owned_compute_processes": owned_gpu_processes,
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
