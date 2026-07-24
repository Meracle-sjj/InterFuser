#!/usr/bin/env python3
"""
[INPUT]: 依赖预注册 visual-pair test 配置、完成且 pipeline-valid 的 formal B0/V manifest、冻结 test index、best checkpoint 与 runtime_resources。
[OUTPUT]: 对外提供 VisualTestError、load_visual_test_contract、execute_visual_test 与 CLI，以隔离单 GPU worker 串行生成 B0/V test 指标、差值和资源 manifest。
[POS]: tools/evaluation 的 M2 H1 冻结 test runner；位于正式配对训练之后、D7 闭环之前，训练未完整归约时拒绝读取 test。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERFUSER_ROOT = REPO_ROOT / "interfuser"
for import_root in (REPO_ROOT, INTERFUSER_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tools.evaluation.interfuser_offline_metrics import (  # noqa: E402
    InterfuserMetricAccumulator,
    MetricError,
)
from tools.evaluation.runtime_resources import (  # noqa: E402
    RunnerError,
    _GpuMemoryMonitor,
    _stop_process_group,
    ensure_gpus_available,
    wait_for_gpus_available,
)


CONFIG_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
VARIANTS = ("b0", "v")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


class VisualTestError(RuntimeError):
    """Raised when frozen test provenance or pair comparability is invalid."""


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path, label):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VisualTestError(f"unable to read {label} JSON {path}: {exc}") from exc


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _resolve_path(repo_root, value, label, require_file=True):
    if not isinstance(value, str) or not value:
        raise VisualTestError(f"{label} must be a non-empty path")
    path = Path(value)
    path = path if path.is_absolute() else Path(repo_root) / path
    path = path.resolve()
    if require_file and not path.is_file():
        raise VisualTestError(f"{label} is not a file: {path}")
    return path


def _verify_hash(path, expected, label):
    if not isinstance(expected, str) or len(expected) != 64:
        raise VisualTestError(f"{label} SHA-256 must have 64 characters")
    actual = sha256_file(path)
    if actual != expected:
        raise VisualTestError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def _positive_int(value, label, allow_zero=False):
    valid = isinstance(value, int) and not isinstance(value, bool)
    valid = valid and (value >= 0 if allow_zero else value > 0)
    if not valid:
        raise VisualTestError(f"{label} must be a valid integer")
    return value


def _finite_positive(value, label):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise VisualTestError(f"{label} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise VisualTestError(f"{label} must be finite and positive")
    return value


def _validate_summary(path, expected_epochs):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected_epochs:
        raise VisualTestError(
            f"formal summary rows={len(rows)} differ from epochs={expected_epochs}"
        )
    for expected_epoch, row in enumerate(rows):
        if float(row.get("epoch", -1)) != float(expected_epoch):
            raise VisualTestError("formal summary epoch sequence is incomplete")
        if any(not math.isfinite(float(value)) for value in row.values()):
            raise VisualTestError("formal summary contains non-finite values")
    return rows


def _validate_training_manifest(manifest, contract, repo_root):
    if manifest.get("run_id") != contract["training_run_id"]:
        raise VisualTestError("formal manifest run_id differs from test contract")
    if manifest.get("status") != "completed" or manifest.get("pipeline_valid") is not True:
        raise VisualTestError("formal training must be completed and pipeline valid")
    if manifest.get("mode") != "formal" or manifest.get("errors"):
        raise VisualTestError("formal training manifest has an invalid mode or errors")
    if manifest.get("git_status") != "":
        raise VisualTestError("formal training did not start from a clean worktree")
    if manifest.get("config_sha256") != contract["training_config_sha256"]:
        raise VisualTestError("formal training config hash differs")
    comparability = manifest.get("comparability") or {}
    if not all(
        comparability.get(field) is True
        for field in ("normalized_training_args_identical", "only_initial_checkpoint_differs")
    ):
        raise VisualTestError("formal B0/V comparability gate is not valid")

    variants = manifest.get("variants")
    if not isinstance(variants, list) or [item.get("variant") for item in variants] != list(VARIANTS):
        raise VisualTestError("formal manifest must contain B0 then V")
    resolved_variants = {}
    schemas = set()
    for item in variants:
        name = item["variant"]
        if item.get("pipeline_valid") is not True or item.get("process_exit_code") != 0:
            raise VisualTestError(f"formal {name} pipeline is invalid")
        if item.get("errors") or item.get("state_tensors") != 1132:
            raise VisualTestError(f"formal {name} checkpoint structure is invalid")
        schemas.add(item.get("state_schema_sha256"))
        artifacts = item.get("artifacts") or {}
        best = artifacts.get("best_checkpoint") or {}
        checkpoint = _resolve_path(repo_root, best.get("path"), f"formal {name} best")
        _verify_hash(checkpoint, best.get("sha256"), f"formal {name} best")
        summary_spec = artifacts.get("summary") or {}
        summary = _resolve_path(repo_root, summary_spec.get("path"), f"formal {name} summary")
        _verify_hash(summary, summary_spec.get("sha256"), f"formal {name} summary")
        _validate_summary(summary, contract["formal_epochs"])
        resolved_variants[name] = {
            "checkpoint": checkpoint,
            "checkpoint_sha256": best["sha256"],
            "summary": summary,
            "summary_sha256": summary_spec["sha256"],
            "best_epoch": item.get("best_epoch"),
            "best_metric": item.get("best_metric"),
        }
    if len(schemas) != 1 or None in schemas:
        raise VisualTestError("formal B0/V checkpoint schemas differ")
    return resolved_variants, next(iter(schemas))


def load_visual_test_contract(config_path, require_training_complete=True, repo_root=REPO_ROOT):
    """Validate pre-registered metrics and optionally require complete formal inputs."""
    repo_root = Path(repo_root).resolve()
    config_path = Path(config_path).resolve()
    raw = _read_json(config_path, "visual test config")
    if raw.get("schema_version") != CONFIG_SCHEMA_VERSION or raw.get("status") != "preregistered":
        raise VisualTestError("visual test config must be preregistered schema v1")

    training_config = _resolve_path(repo_root, raw.get("training_config"), "training_config")
    _verify_hash(training_config, raw.get("training_config_sha256"), "training_config")
    formal = _read_json(training_config, "formal training config")
    if formal.get("status") != "formal":
        raise VisualTestError("referenced training config is not formal")
    formal_epochs = _positive_int(
        (formal.get("training") or {}).get("epochs"), "formal training epochs"
    )

    split_manifest = _resolve_path(
        repo_root, raw.get("downstream_split_manifest"), "downstream_split_manifest"
    )
    _verify_hash(
        split_manifest,
        raw.get("downstream_split_manifest_sha256"),
        "downstream_split_manifest",
    )
    split = _read_json(split_manifest, "downstream split manifest")
    if split.get("valid") is not True:
        raise VisualTestError("downstream split manifest is not valid")

    dataset = raw.get("dataset") or {}
    dataset_root = _resolve_path(
        repo_root, dataset.get("root"), "dataset.root", require_file=False
    )
    if not dataset_root.is_dir():
        raise VisualTestError("dataset.root is not a directory")
    test_index = _resolve_path(repo_root, dataset.get("test_index"), "dataset.test_index")
    test_hash = _verify_hash(
        test_index, dataset.get("test_index_sha256"), "dataset.test_index"
    )
    if test_hash != (split.get("artifacts", {}).get("test", {}) or {}).get("sha256"):
        raise VisualTestError("test index differs from downstream split manifest")
    if test_hash != (formal.get("dataset") or {}).get("test_index_sha256"):
        raise VisualTestError("test index differs from formal training contract")
    logical_frames = _positive_int(dataset.get("logical_frames"), "dataset.logical_frames")
    if logical_frames != (split.get("summary", {}).get("test", {}) or {}).get("logical_frames"):
        raise VisualTestError("test logical frame count differs from split manifest")
    for field in ("towns", "weathers"):
        values = dataset.get(field)
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise VisualTestError(f"dataset.{field} must be unique and non-empty")
        if values != (formal.get("dataset") or {}).get(field):
            raise VisualTestError(f"dataset.{field} differs from formal training contract")

    model = raw.get("model") or {}
    if model.get("name") != "interfuser_baseline" or model.get("with_lidar") is not True:
        raise VisualTestError("test model must be lidar-enabled interfuser_baseline")
    if model.get("multi_view_input_size") != [3, 128, 128]:
        raise VisualTestError("test multi-view input size must be [3, 128, 128]")
    metrics = raw.get("metrics") or {}
    if metrics.get("require_both_binary_classes") is not True:
        raise VisualTestError("test metrics must require both binary classes")
    for field in (
        "traffic_positive_target_threshold",
        "traffic_prediction_threshold",
        "invalid_waypoint_threshold",
    ):
        _finite_positive(metrics.get(field), f"metrics.{field}")

    runtime = raw.get("runtime") or {}
    for field in (
        "seed",
        "gpu",
        "batch_size",
        "workers",
        "log_interval_batches",
        "timeout_seconds_per_variant",
        "gpu_busy_memory_threshold_mb",
    ):
        _positive_int(runtime.get(field), f"runtime.{field}", allow_zero=field in {"seed", "gpu", "workers"})
    if not isinstance(runtime.get("require_clean_git"), bool):
        raise VisualTestError("runtime.require_clean_git must be boolean")

    result_root = _resolve_path(repo_root, raw.get("result_root"), "result_root", require_file=False)
    try:
        result_root.relative_to(repo_root)
    except ValueError as exc:
        raise VisualTestError("result_root escapes repository") from exc
    training_run_id = raw.get("training_run_id")
    if not isinstance(training_run_id, str) or not RUN_ID_PATTERN.fullmatch(training_run_id):
        raise VisualTestError("training_run_id is invalid")
    training_manifest = result_root / training_run_id / "run_manifest.json"

    contract = dict(raw)
    contract.update(
        {
            "path": config_path,
            "sha256": sha256_file(config_path),
            "repo_root": repo_root,
            "training_config_path": training_config,
            "training_config_sha256": raw["training_config_sha256"],
            "formal_epochs": formal_epochs,
            "split_manifest_path": split_manifest,
            "dataset_root_path": dataset_root,
            "test_index_path": test_index,
            "result_root_path": result_root,
            "training_manifest_path": training_manifest,
        }
    )
    if require_training_complete:
        manifest = _read_json(training_manifest, "formal training manifest")
        variants, schema = _validate_training_manifest(manifest, contract, repo_root)
        contract["training_manifest"] = manifest
        contract["training_manifest_sha256"] = sha256_file(training_manifest)
        contract["resolved_variants"] = variants
        contract["state_schema_sha256"] = schema
    return contract


def _git_output(repo_root, *args):
    result = subprocess.run(
        ["git", *args], cwd=repo_root, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise VisualTestError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _worker_evaluate(contract, variant, output_path):
    output_path = Path(output_path)
    if output_path.exists():
        raise VisualTestError(f"refusing to overwrite worker output: {output_path}")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from timm import create_model
    from timm.data import create_carla_dataset, create_carla_loader, resolve_data_config

    runtime = contract["runtime"]
    torch.manual_seed(runtime["seed"])
    np.random.seed(runtime["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = create_model(
        contract["model"]["name"],
        pretrained=False,
        checkpoint_path=str(contract["resolved_variants"][variant]["checkpoint"]),
        freeze_num=-1,
    )
    model.cuda().eval()
    data_config = resolve_data_config({}, model=model, verbose=False)
    dataset = create_carla_dataset(
        "carla",
        root=str(contract["dataset_root_path"]),
        towns=contract["dataset"]["towns"],
        weathers=contract["dataset"]["weathers"],
        batch_size=runtime["batch_size"],
        with_lidar=True,
        multi_view=True,
        augment_prob=0.0,
        dataset_index=str(contract["test_index_path"]),
    )
    loader = create_carla_loader(
        dataset,
        input_size=data_config["input_size"],
        batch_size=runtime["batch_size"],
        multi_view_input_size=contract["model"]["multi_view_input_size"],
        is_training=False,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=runtime["workers"],
        distributed=False,
        pin_memory=False,
        persistent_workers=runtime["workers"] > 0,
    )
    if len(dataset) != contract["dataset"]["logical_frames"]:
        raise VisualTestError(
            f"test dataset length={len(dataset)} differs from frozen logical frames"
        )
    accumulator = InterfuserMetricAccumulator(
        contract["metrics"]["traffic_positive_target_threshold"],
        contract["metrics"]["traffic_prediction_threshold"],
        contract["metrics"]["invalid_waypoint_threshold"],
    )
    started = time.monotonic()
    with torch.inference_mode():
        for batch_index, (inputs, targets) in enumerate(loader):
            inputs = {name: value.cuda(non_blocking=False) for name, value in inputs.items()}
            outputs = model(inputs)
            accumulator.update(outputs, targets)
            if batch_index % runtime["log_interval_batches"] == 0:
                print(
                    f"variant={variant} batch={batch_index}/{len(loader)} ",
                    f"samples={accumulator.samples}",
                    flush=True,
                )
    metrics = accumulator.finalize()
    result = {
        "worker_schema_version": 1,
        "variant": variant,
        "pipeline_valid": True,
        "checkpoint": str(contract["resolved_variants"][variant]["checkpoint"]),
        "checkpoint_sha256": contract["resolved_variants"][variant]["checkpoint_sha256"],
        "test_index": str(contract["test_index_path"]),
        "test_index_sha256": contract["dataset"]["test_index_sha256"],
        "samples": metrics["samples"],
        "duration_seconds": round(time.monotonic() - started, 3),
        "metrics": metrics,
    }
    _write_json(output_path, result)
    return result


def _run_worker(contract, variant, run_dir):
    runtime = contract["runtime"]
    gpu = runtime["gpu"]
    variant_dir = run_dir / variant
    variant_dir.mkdir()
    worker_output = variant_dir / "worker_result.json"
    log_path = variant_dir / "worker.log"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(contract["path"]),
        "--worker-variant",
        variant,
        "--worker-output",
        str(worker_output),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), str(INTERFUSER_ROOT), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    environment.setdefault("OMP_NUM_THREADS", "1")
    ensure_gpus_available([gpu], runtime["gpu_busy_memory_threshold_mb"])
    monitor = _GpuMemoryMonitor([gpu])
    started = time.monotonic()
    process = None
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log:
        try:
            monitor.start()
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                exit_code = process.wait(timeout=runtime["timeout_seconds_per_variant"])
            except subprocess.TimeoutExpired:
                timed_out = True
                exit_code = 124
        finally:
            if process is not None:
                _stop_process_group(process)
            peaks = monitor.stop()
    gpu_release = wait_for_gpus_available(
        [gpu], runtime["gpu_busy_memory_threshold_mb"]
    )
    result = {
        "variant": variant,
        "command": command,
        "process_exit_code": exit_code,
        "external_timeout": timed_out,
        "duration_seconds": round(time.monotonic() - started, 3),
        "worker_log": str(log_path),
        "worker_log_sha256": sha256_file(log_path),
        "gpu_peak_memory_mb": peaks,
        "gpu_monitor_error": monitor.error,
        "gpu_release_wait_seconds": gpu_release,
        "pipeline_valid": False,
        "errors": [],
    }
    if exit_code != 0 or timed_out:
        result["errors"].append(
            "evaluation timed out" if timed_out else f"worker exited with {exit_code}"
        )
        return result
    if monitor.error:
        result["errors"].append(f"GPU monitor failed: {monitor.error}")
        return result
    try:
        worker = _read_json(worker_output, f"{variant} worker result")
        if worker.get("pipeline_valid") is not True or worker.get("variant") != variant:
            raise VisualTestError("worker result identity or status is invalid")
        if worker.get("samples") != contract["dataset"]["logical_frames"]:
            raise VisualTestError("worker test sample count differs")
        if worker.get("checkpoint_sha256") != contract["resolved_variants"][variant]["checkpoint_sha256"]:
            raise VisualTestError("worker checkpoint hash differs")
        result["worker_result"] = worker
        result["worker_result_sha256"] = sha256_file(worker_output)
        result["pipeline_valid"] = True
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    return result


def _metric_delta(b0, v):
    paths = {
        "traffic_average_precision": ("traffic", "occupancy", "average_precision"),
        "traffic_roc_auc": ("traffic", "occupancy", "roc_auc"),
        "traffic_occupied_iou": ("traffic", "occupancy", "occupied_iou"),
        "traffic_probability_mae": ("traffic", "probability_mae"),
        "waypoint_ade": ("waypoints", "ade"),
        "waypoint_fde_horizon_10": ("waypoints", "fde_horizon_10"),
        "junction_macro_f1": ("junction", "macro_f1"),
        "red_light_macro_f1": ("red_light", "macro_f1"),
        "stop_sign_macro_f1": ("stop_sign", "macro_f1"),
    }

    def value_at(root, path):
        value = root
        for field in path:
            value = value[field]
        return float(value)

    return {
        name: value_at(v, path) - value_at(b0, path) for name, path in paths.items()
    }


def execute_visual_test(config_path, run_id):
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise VisualTestError("run_id is invalid")
    contract = load_visual_test_contract(config_path, require_training_complete=True)
    git_status = _git_output(REPO_ROOT, "status", "--porcelain")
    if contract["runtime"]["require_clean_git"] and git_status:
        raise VisualTestError("Git worktree must be clean")
    run_dir = contract["result_root_path"] / run_id
    if run_dir.exists():
        raise VisualTestError(f"refusing to overwrite run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "test_manifest.json"
    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "running",
        "pipeline_valid": False,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "git_head": _git_output(REPO_ROOT, "rev-parse", "HEAD"),
        "git_status": git_status,
        "config": str(contract["path"]),
        "config_sha256": contract["sha256"],
        "formal_training_manifest": str(contract["training_manifest_path"]),
        "formal_training_manifest_sha256": contract["training_manifest_sha256"],
        "formal_training_git_head": contract["training_manifest"].get("git_head"),
        "test_index": str(contract["test_index_path"]),
        "test_index_sha256": contract["dataset"]["test_index_sha256"],
        "state_schema_sha256": contract["state_schema_sha256"],
        "errors": [],
        "variants": [],
    }
    _write_json(manifest_path, manifest)
    try:
        for variant in VARIANTS:
            result = _run_worker(contract, variant, run_dir)
            manifest["variants"].append(result)
            manifest["updated_at"] = _utc_now()
            _write_json(manifest_path, manifest)
            if not result["pipeline_valid"]:
                raise VisualTestError(
                    f"{variant} test invalid: {'; '.join(result['errors'])}"
                )
        b0_metrics = manifest["variants"][0]["worker_result"]["metrics"]
        v_metrics = manifest["variants"][1]["worker_result"]["metrics"]
        manifest.update(
            {
                "status": "completed",
                "pipeline_valid": True,
                "updated_at": _utc_now(),
                "completed_at": _utc_now(),
                "comparability": {
                    "variant_order": list(VARIANTS),
                    "same_test_index": True,
                    "same_metric_contract": True,
                    "same_checkpoint_schema": True,
                },
                "v_minus_b0": _metric_delta(b0_metrics, v_metrics),
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
        description="Run frozen-test evaluation for a formal InterFuser B0/V pair"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--worker-variant", choices=VARIANTS)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.worker_variant:
            if args.worker_output is None:
                raise VisualTestError("worker mode requires --worker-output")
            contract = load_visual_test_contract(
                args.config, require_training_complete=True
            )
            _worker_evaluate(contract, args.worker_variant, args.worker_output)
        elif args.preflight_only:
            contract = load_visual_test_contract(
                args.config, require_training_complete=False
            )
            print(json.dumps({"config_sha256": contract["sha256"]}, indent=2))
        else:
            if not args.run_id:
                raise VisualTestError("parent mode requires --run-id")
            manifest = execute_visual_test(args.config, args.run_id)
            print(json.dumps(manifest["v_minus_b0"], indent=2, sort_keys=True))
    except (VisualTestError, MetricError, RunnerError) as exc:
        print(f"visual test error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"visual test failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
