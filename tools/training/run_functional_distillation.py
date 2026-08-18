#!/usr/bin/env python3
"""
 * [INPUT]: 依赖 v8.1 协议、functional_distillation 库、runtime_resources 的 GPU 共享容量门禁、
 *   哈希绑定的 functional_distillation 配置。
 * [OUTPUT]: 对外提供 run_functional_distillation CLI——执行 v8b 五轮功能蒸馏 probe，产出 run manifest、
 *   checkpoint_best、backbone_resnet50d.pth 导出与不变量断言记录；OOM 时按预注册回退 bs8×累积2。
 * [POS]: tools/training 的 v8b 训练入口；镜像 run_semantic_pretraining 的资源纪律与产物契约。
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
INTERFUSER_ROOT = REPO_ROOT / "interfuser"
for import_root in (REPO_ROOT, INTERFUSER_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tools.evaluation.runtime_resources import (  # noqa: E402
    RunnerError,
    ensure_gpus_have_free_memory,
)
from tools.training.functional_distillation import (  # noqa: E402
    FunctionalContractError,
    FunctionalFrameDataset,
    assert_pair_invariants,
    attach_carla_transforms,
    build_base_datasets,
    build_functional_pair,
    export_student_backbone,
    front_mask_geometry,
    load_functional_contract,
    run_functional_epoch,
)
from tools.training.semantic_pretraining import (  # noqa: E402
    DeterministicCrossEntropyLoss,
    set_reproducible_seed,
    sha256_file,
)

from torch.utils.data import DataLoader  # noqa: E402

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class FunctionalRunError(RuntimeError):
    """Raised when the v8b training run violates its contract."""


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _git_output(*args):
    result = subprocess.run(
        ["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True, check=False
    )
    return result.stdout.strip()


def _atomic_write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _dependency_versions():
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
    }


def _prepare_run(config_path, run_id, result_root):
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise FunctionalRunError(f"invalid run ID: {run_id}")
    contract = load_functional_contract(config_path)
    git_status = _git_output("status", "--porcelain")
    if contract["training"]["require_clean_git"] and git_status:
        raise FunctionalRunError("training requires a clean Git worktree")
    physical_gpu = contract["training"]["physical_gpu_index"]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(physical_gpu):
        raise FunctionalRunError(
            f"CUDA_VISIBLE_DEVICES must be {physical_gpu}, got {visible!r}"
        )
    try:
        gpu_usage = ensure_gpus_have_free_memory(
            [physical_gpu], contract["training"]["gpu_minimum_free_memory_mb"]
        )
    except RunnerError as exc:
        raise FunctionalRunError(str(exc)) from exc
    if not torch.cuda.is_available():
        raise FunctionalRunError("CUDA is unavailable after GPU preflight")
    result_root = Path(result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    run_directory = result_root / run_id
    try:
        run_directory.mkdir()
    except FileExistsError as exc:
        raise FunctionalRunError(f"refusing to reuse run directory: {run_directory}") from exc
    return contract, git_status, gpu_usage, run_directory


def _build_loaders(contract, pair, batch_size, workers):
    datasets = build_base_datasets(contract, batch_size)
    data_config = attach_carla_transforms(
        datasets["train"], batch_size, workers, pair["teacher"]
    )
    attach_carla_transforms(datasets["validation"], batch_size, workers, pair["teacher"])
    geometry = front_mask_geometry(data_config["input_size"])
    class_config = contract["class_config"]
    ignore_index = contract["ignore_index"]
    generator = torch.Generator()
    generator.manual_seed(contract["training"]["seed"])
    loaders = {}
    wrapped = {}
    for split in ("train", "validation"):
        wrapped[split] = FunctionalFrameDataset(
            datasets[split], class_config, ignore_index, geometry
        )
        loaders[split] = DataLoader(
            wrapped[split],
            batch_size=batch_size,
            shuffle=split == "train",
            generator=generator if split == "train" else None,
            num_workers=workers,
            pin_memory=False,
            drop_last=split == "train",
            persistent_workers=workers > 0,
        )
    return loaders, wrapped, data_config, geometry


def _train_epochs(contract, pair, loaders, criterion, device, class_names, run_manifest):
    training = contract["training"]
    student = pair["student"]
    seg_model = pair["seg_model"]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(student.rgb_backbone.parameters()),
                "lr": training["backbone_learning_rate"],
            },
            {
                "params": [
                    parameter
                    for name, parameter in seg_model.named_parameters()
                    if not name.startswith("backbone.")
                ],
                "lr": training["learning_rate"],
            },
        ],
        weight_decay=training["weight_decay"],
    )
    accumulation = 1
    epochs = []
    best_epoch = None
    best_validation_miou = float("-inf")
    best_states = None
    for epoch_index in range(training["epochs"]):
        backbone_trainable = epoch_index >= training.get("backbone_warmup_epochs", 0)
        for parameter in student.rgb_backbone.parameters():
            parameter.requires_grad_(backbone_trainable)
        try:
            train_metrics = run_functional_epoch(
                pair,
                loaders["train"],
                criterion,
                device,
                class_names,
                optimizer=optimizer,
                distill_coefficient=contract["coefficient"],
                accumulation=accumulation,
            )
        except torch.cuda.OutOfMemoryError:
            if accumulation != 1 or epoch_index != 0:
                raise
            torch.cuda.empty_cache()
            accumulation = 2
            run_manifest["preregistered_oom_fallback"] = {
                "applied": True,
                "batch_size": training["batch_size"] // 2,
                "gradient_accumulation": 2,
                "note": "v8.1 §5 preregistered infra fallback",
            }
            loaders, _, _, _ = _build_loaders(
                contract, pair, training["batch_size"] // 2, training["num_workers"]
            )
            for parameter in student.rgb_backbone.parameters():
                parameter.requires_grad_(backbone_trainable)
            train_metrics = run_functional_epoch(
                pair,
                loaders["train"],
                criterion,
                device,
                class_names,
                optimizer=optimizer,
                distill_coefficient=contract["coefficient"],
                accumulation=accumulation,
            )
        with torch.no_grad():
            validation_metrics = run_functional_epoch(
                pair, loaders["validation"], criterion, device, class_names
            )
        epoch_record = {
            "epoch": epoch_index + 1,
            "backbone_trainable": backbone_trainable,
            "optimizer_learning_rates": [
                float(group["lr"]) for group in optimizer.param_groups
            ],
            "train": train_metrics,
            "validation": validation_metrics,
        }
        epochs.append(epoch_record)
        print(
            json.dumps(
                {
                    "epoch": epoch_record["epoch"],
                    "train_loss": round(train_metrics["loss"], 6),
                    "train_task": round(train_metrics["task_loss"], 6),
                    "train_functional": round(
                        train_metrics["functional_distillation_penalty"], 6
                    ),
                    "val_mean_iou": round(validation_metrics["mean_iou"], 6),
                    "val_functional": round(
                        validation_metrics["functional_distillation_penalty"], 6
                    ),
                    "seconds": train_metrics["duration_seconds"]
                    + validation_metrics["duration_seconds"],
                }
            ),
            flush=True,
        )
        validation_miou = validation_metrics["mean_iou"]
        if validation_miou > best_validation_miou:
            best_epoch = epoch_index + 1
            best_validation_miou = validation_miou
            best_states = {
                "student": OrderedDict(
                    (key, value.detach().cpu())
                    for key, value in student.state_dict().items()
                ),
                "semantic_head": OrderedDict(
                    (key, value.detach().cpu())
                    for key, value in seg_model.state_dict().items()
                ),
            }
    return epochs, best_epoch, best_validation_miou, best_states, accumulation


def run_training(config_path, run_id, result_root="results/thesis_m2"):
    contract, git_status, gpu_usage, run_directory = _prepare_run(
        config_path, run_id, result_root
    )
    training = contract["training"]
    set_reproducible_seed(training["seed"])
    device = torch.device("cuda")
    pair = build_functional_pair(contract)
    pair["teacher"].to(device)
    pair["student"].to(device)
    pair["seg_model"].to(device)
    loaders, wrapped, data_config, geometry = _build_loaders(
        contract, pair, training["batch_size"], training["num_workers"]
    )
    class_names = [item["name"] for item in contract["class_config"]["classes"]]
    criterion = DeterministicCrossEntropyLoss(
        ignore_index=contract["ignore_index"], class_weights=contract["class_weights"]
    ).to(device)
    run_manifest = {
        "schema_version": 1,
        "status": "running",
        "run_id": run_id,
        "protocol_document": "docs/interfuser_negative_transfer_repair_protocol_v8_1.md",
        "started_at": _utc_now(),
        "config": {"path": contract["config_path"], "sha256": contract["sha256"]},
        "git": {
            "head": _git_output("rev-parse", "HEAD"),
            "status_porcelain": git_status,
        },
        "gpu_usage_at_start": gpu_usage,
        "dependencies": _dependency_versions(),
        "bound_artifacts": {
            field: {"path": str(path), "sha256": sha256_file(path)}
            for field, path in contract["bound_paths"].items()
        },
        "corpus": {
            "train_frames_retained": len(wrapped["train"]),
            "validation_frames_retained": len(wrapped["validation"]),
            "audit_manifest_sha256": sha256_file(
                REPO_ROOT / contract["raw"]["corpus_audit"]["manifest"]
            ),
            "lidar_y_axis_multiplier": contract["dataset_block"][
                "lidar_y_axis_multiplier"
            ],
        },
        "transform_contract": {
            "data_config": {key: str(value) for key, value in data_config.items()},
            "front_mask_geometry": {
                "resize_wh": list(geometry["resize_wh"]),
                "crop_size": (
                    list(geometry["crop_size"])
                    if isinstance(geometry["crop_size"], tuple)
                    else geometry["crop_size"]
                ),
            },
        },
        "preregistered_oom_fallback": {"applied": False},
    }
    _atomic_write_json(run_directory / "run_manifest.json", run_manifest)

    epochs, best_epoch, best_validation_miou, best_states, accumulation = _train_epochs(
        contract, pair, loaders, criterion, device, class_names, run_manifest
    )
    if best_states is None or best_epoch is None:
        raise FunctionalRunError("training produced no best model state")
    pair["student"].load_state_dict(best_states["student"], strict=True)
    pair["seg_model"].load_state_dict(best_states["semantic_head"], strict=True)

    invariants = assert_pair_invariants(
        pair["student"], contract["bound_paths"]["student_initialization_checkpoint"]
    )
    backbone_export_path = run_directory / "backbone_resnet50d.pth"
    torch.save(export_student_backbone(pair["student"]), backbone_export_path)
    best_checkpoint_path = run_directory / "checkpoint_best.pth"
    torch.save(
        {
            "format_version": 1,
            "epoch": best_epoch,
            "selection_metric": "validation.mean_iou",
            "selection_metric_value": best_validation_miou,
            "training_config_sha256": contract["sha256"],
            "student_state_dict": best_states["student"],
            "semantic_head_state_dict": best_states["semantic_head"],
        },
        best_checkpoint_path,
    )
    torch.cuda.synchronize(device)
    run_manifest.update(
        {
            "status": "completed",
            "pipeline_valid": True,
            "completed_at": _utc_now(),
            "epochs": epochs,
            "gradient_accumulation": accumulation,
            "best_epoch": best_epoch,
            "best_validation_mean_iou": best_validation_miou,
            "pair_invariants": invariants,
            "exports": {
                "backbone_resnet50d": {
                    "path": str(backbone_export_path),
                    "sha256": sha256_file(backbone_export_path),
                },
                "checkpoint_best": {
                    "path": str(best_checkpoint_path),
                    "sha256": sha256_file(best_checkpoint_path),
                },
            },
        }
    )
    _atomic_write_json(run_directory / "run_manifest.json", run_manifest)
    return run_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--result-root", default="results/thesis_m2")
    args = parser.parse_args(argv)
    manifest = run_training(args.config, args.run_id, args.result_root)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "best_epoch": manifest["best_epoch"],
                "best_validation_mean_iou": manifest["best_validation_mean_iou"],
                "pair_invariants": manifest["pair_invariants"],
                "oom_fallback": manifest["preregistered_oom_fallback"]["applied"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
