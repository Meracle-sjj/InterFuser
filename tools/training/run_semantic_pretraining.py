#!/usr/bin/env python3
"""
[INPUT]: 依赖版本化 M2 配置、semantic_pretraining 领域 API、GPU owner 门禁、干净 Git 工作树与冻结 M1 数据。
[OUTPUT]: 对外提供 TrainingRunError、run_training 与 CLI，原子生成训练/验证指标、完整 checkpoint、可迁移骨干权重和 run manifest。
[POS]: tools/training 的 M2 单机运行编排器；只协调资源、训练生命周期与 provenance，不定义标签语义或修改 InterFuser 推理代码。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import argparse
import json
import math
import os
import platform
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

import numpy as np  # noqa: E402
import PIL  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from tools.evaluation.runtime_resources import (  # noqa: E402
    RunnerError,
    ensure_gpus_available,
)
from tools.training.semantic_pretraining import (  # noqa: E402
    ConfusionMetrics,
    DeterministicCrossEntropyLoss,
    SemanticFrameDataset,
    SemanticPretrainingModel,
    TrainingContractError,
    load_training_contract,
    make_backbone_export,
    resolve_train_sample_limit,
    set_reproducible_seed,
    sha256_file,
    validate_backbone_export,
)


RUN_MANIFEST_SCHEMA_VERSION = 1
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class TrainingRunError(RuntimeError):
    """Raised when a smoke run cannot preserve its resource or evidence contract."""


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _git_output(*args):
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise TrainingRunError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _atomic_write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    serialized = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)


def _worker_seed(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)


def _make_loader(dataset, contract, shuffle):
    generator = torch.Generator()
    generator.manual_seed(contract["training"]["seed"])
    return DataLoader(
        dataset,
        batch_size=contract["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=contract["training"]["num_workers"],
        pin_memory=True,
        drop_last=False,
        worker_init_fn=_worker_seed,
        generator=generator,
    )


def _cpu_model_state(model):
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def _run_epoch(model, loader, criterion, device, class_names, optimizer=None):
    training = optimizer is not None
    model.train(training)
    metrics = ConfusionMetrics(len(class_names), criterion.ignore_index)
    total_loss = 0.0
    sample_count = 0
    started = time.monotonic()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        batch_size = images.shape[0]
        total_loss += float(loss.detach().cpu().item()) * batch_size
        sample_count += batch_size
        metrics.update(logits, labels)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    summary = metrics.summary(class_names)
    summary.update(
        {
            "loss": total_loss / sample_count,
            "samples": sample_count,
            "batches": len(loader),
            "duration_seconds": round(time.monotonic() - started, 3),
        }
    )
    return summary


def _dependency_versions():
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
    }


def _validate_finite_metrics(epoch_record):
    for phase in ("train", "validation"):
        metrics = epoch_record[phase]
        for field in ("loss", "pixel_accuracy", "mean_iou", "macro_f1"):
            value = metrics[field]
            if value is None or not math.isfinite(value):
                raise TrainingRunError(
                    f"epoch {epoch_record['epoch']} {phase} {field} is not finite"
                )
        for item in metrics["per_class"]:
            for field in ("iou", "f1"):
                value = item[field]
                if value is not None and not math.isfinite(value):
                    raise TrainingRunError(
                        f"epoch {epoch_record['epoch']} {phase} class "
                        f"{item['name']} {field} is not finite"
                    )


def _prepare_run(config_path, run_id, result_root):
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise TrainingRunError(f"invalid run ID: {run_id}")
    contract = load_training_contract(config_path)
    git_head = _git_output("rev-parse", "HEAD")
    git_status = _git_output("status", "--porcelain")
    if contract["training"]["require_clean_git"] and git_status:
        raise TrainingRunError("training requires a clean Git worktree")
    physical_gpu = contract["training"]["physical_gpu_index"]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(physical_gpu):
        raise TrainingRunError(
            f"CUDA_VISIBLE_DEVICES must be {physical_gpu}, got {visible!r}"
        )
    try:
        gpu_usage = ensure_gpus_available(
            [physical_gpu],
            contract["training"]["gpu_busy_memory_threshold_mb"],
        )
    except RunnerError as exc:
        raise TrainingRunError(str(exc)) from exc
    if not torch.cuda.is_available():
        raise TrainingRunError("CUDA is unavailable after GPU preflight")
    result_root = Path(result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    run_directory = result_root / run_id
    try:
        run_directory.mkdir()
    except FileExistsError as exc:
        raise TrainingRunError(f"refusing to reuse run directory: {run_directory}") from exc
    return contract, git_head, git_status, gpu_usage, run_directory


def run_training(config_path, run_id, result_root, train_sample_limit=None):
    """Execute one clean, resource-gated semantic pretraining run."""
    contract, git_head, git_status, gpu_usage, run_directory = _prepare_run(
        config_path, run_id, result_root
    )
    manifest = {
        "run_manifest_schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "running",
        "pipeline_valid": False,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "git_head": git_head,
        "git_status": git_status,
        "config": str(contract["path"]),
        "config_sha256": contract["sha256"],
        "class_config_sha256": contract["class_config_sha256"],
        "split_manifest_sha256": contract["split_manifest_sha256"],
        "pretrained_source": contract["backbone"]["pretrained_source"],
        "experiment_status": contract["status"],
        "pretrained_checkpoint_sha256": contract["backbone"][
            "pretrained_checkpoint_sha256"
        ],
        "dependencies": _dependency_versions(),
        "resources": {
            "physical_gpu_index": contract["training"]["physical_gpu_index"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_memory_before_mb": gpu_usage,
        },
        "errors": [],
    }
    manifest_path = run_directory / "run_manifest.json"
    _atomic_write_json(manifest_path, manifest)
    try:
        set_reproducible_seed(
            contract["training"]["seed"],
            deterministic=contract["training"]["deterministic"],
        )
        resolved_train_limit = resolve_train_sample_limit(
            contract, train_sample_limit
        )
        train_dataset = SemanticFrameDataset(
            contract, "train", sample_limit=resolved_train_limit
        )
        validation_dataset = SemanticFrameDataset(contract, "validation")
        train_loader = _make_loader(train_dataset, contract, shuffle=True)
        validation_loader = _make_loader(validation_dataset, contract, shuffle=False)
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        model = SemanticPretrainingModel(contract).to(device)
        criterion = DeterministicCrossEntropyLoss(
            ignore_index=contract["training"]["ignore_index"]
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=contract["training"]["learning_rate"],
            weight_decay=contract["training"]["weight_decay"],
        )
        class_names = [
            item["name"] for item in contract["class_config_loaded"]["classes"]
        ]
        epochs = []
        best_epoch = None
        best_validation_miou = float("-inf")
        best_model_state = None
        for epoch_index in range(contract["training"]["epochs"]):
            train_metrics = _run_epoch(
                model,
                train_loader,
                criterion,
                device,
                class_names,
                optimizer=optimizer,
            )
            with torch.no_grad():
                validation_metrics = _run_epoch(
                    model,
                    validation_loader,
                    criterion,
                    device,
                    class_names,
                )
            epoch_record = {
                "epoch": epoch_index + 1,
                "train": train_metrics,
                "validation": validation_metrics,
            }
            _validate_finite_metrics(epoch_record)
            epochs.append(epoch_record)
            validation_miou = validation_metrics["mean_iou"]
            if validation_miou > best_validation_miou:
                best_epoch = epoch_index + 1
                best_validation_miou = validation_miou
                best_model_state = _cpu_model_state(model)

        checkpoint_path = run_directory / "checkpoint_last.pth"
        torch.save(
            {
                "format_version": 1,
                "epoch": contract["training"]["epochs"],
                "training_config_sha256": contract["sha256"],
                "model_state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_path,
        )
        if best_model_state is None or best_epoch is None:
            raise TrainingRunError("training produced no best model state")
        best_checkpoint_path = run_directory / "checkpoint_best.pth"
        torch.save(
            {
                "format_version": 1,
                "epoch": best_epoch,
                "selection_metric": "validation.mean_iou",
                "selection_metric_value": best_validation_miou,
                "training_config_sha256": contract["sha256"],
                "model_state_dict": best_model_state,
            },
            best_checkpoint_path,
        )
        model.load_state_dict(best_model_state, strict=True)
        backbone_export = make_backbone_export(model, contract)
        transfer_validation = validate_backbone_export(backbone_export)
        backbone_path = run_directory / "backbone_resnet50d.pth"
        torch.save(backbone_export, backbone_path)
        torch.cuda.synchronize(device)
        manifest.update(
            {
                "status": "completed",
                "pipeline_valid": True,
                "updated_at": _utc_now(),
                "completed_at": _utc_now(),
                "data": {
                    "train_samples": len(train_dataset),
                    "validation_samples": len(validation_dataset),
                    "train_samples_available": train_dataset.available_samples,
                    "validation_samples_available": validation_dataset.available_samples,
                    "train_sample_limit_requested": resolved_train_limit,
                    "train_sample_keys": [item["key"] for item in train_dataset.records],
                    "validation_sample_keys": [
                        item["key"] for item in validation_dataset.records
                    ],
                },
                "epochs": epochs,
                "artifacts": {
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "best_checkpoint": str(best_checkpoint_path),
                    "best_checkpoint_sha256": sha256_file(best_checkpoint_path),
                    "best_epoch": best_epoch,
                    "best_selection_metric": "validation.mean_iou",
                    "best_selection_metric_value": best_validation_miou,
                    "backbone_export": str(backbone_path),
                    "backbone_export_sha256": sha256_file(backbone_path),
                    "transfer_validation": transfer_validation,
                },
            }
        )
        manifest["resources"].update(
            {
                "cuda_device_name": torch.cuda.get_device_name(device),
                "peak_memory_allocated_mb": round(
                    torch.cuda.max_memory_allocated(device) / (1024**2), 3
                ),
                "peak_memory_reserved_mb": round(
                    torch.cuda.max_memory_reserved(device) / (1024**2), 3
                ),
            }
        )
        _atomic_write_json(manifest_path, manifest)
        return manifest
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
        _atomic_write_json(manifest_path, manifest)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run one provenance-locked semantic pretraining job"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--train-samples", type=int)
    parser.add_argument(
        "--result-root", type=Path, default=REPO_ROOT / "results" / "thesis_m2"
    )
    args = parser.parse_args(argv)
    try:
        manifest = run_training(
            args.config,
            args.run_id,
            args.result_root,
            train_sample_limit=args.train_samples,
        )
    except (TrainingContractError, TrainingRunError) as exc:
        print(f"training error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"training failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
