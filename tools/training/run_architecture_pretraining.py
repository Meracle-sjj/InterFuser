#!/usr/bin/env python3
"""
 * [INPUT]: 依赖 v9 协议、architecture_pretraining 库、runtime_resources GPU 门禁、哈希绑定配置。
 * [OUTPUT]: 对外提供 run_architecture_pretraining CLI——执行 v9 阶段一（架构内辅助语义预训练），
 *   产出 run manifest 与 checkpoint_best（全模型态，供 Stage 2 初始化）。
 * [POS]: tools/training 的 v9 预训练入口；epoch 选择以 validation 驾驶损失最低为准（辅助任务不参与选择）。
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
from tools.training.architecture_pretraining import (  # noqa: E402
    ArchitectureContractError,
    build_pretraining_pair,
    load_architecture_contract,
    run_pretraining_epoch,
)
from tools.training.functional_distillation import (  # noqa: E402
    FunctionalFrameDataset,
    attach_carla_transforms,
    build_base_datasets,
    front_mask_geometry,
)
from tools.training.semantic_pretraining import (  # noqa: E402
    DeterministicCrossEntropyLoss,
    set_reproducible_seed,
    sha256_file,
)

from torch.utils.data import DataLoader  # noqa: E402

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class PretrainingRunError(RuntimeError):
    """Raised when the v9 pretraining run violates its contract."""


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
        raise PretrainingRunError(f"invalid run ID: {run_id}")
    contract = load_architecture_contract(config_path)
    git_status = _git_output("status", "--porcelain")
    if contract["training"]["require_clean_git"] and git_status:
        raise PretrainingRunError("training requires a clean Git worktree")
    physical_gpu = contract["training"]["physical_gpu_index"]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(physical_gpu):
        raise PretrainingRunError(
            f"CUDA_VISIBLE_DEVICES must be {physical_gpu}, got {visible!r}"
        )
    try:
        gpu_usage = ensure_gpus_have_free_memory(
            [physical_gpu], contract["training"]["gpu_minimum_free_memory_mb"]
        )
    except RunnerError as exc:
        raise PretrainingRunError(str(exc)) from exc
    if not torch.cuda.is_available():
        raise PretrainingRunError("CUDA is unavailable after GPU preflight")
    result_root = Path(result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    run_directory = result_root / run_id
    try:
        run_directory.mkdir()
    except FileExistsError as exc:
        raise PretrainingRunError(f"refusing to reuse run directory: {run_directory}") from exc
    return contract, git_status, gpu_usage, run_directory


def _build_loaders(contract, pair, batch_size, workers):
    datasets = build_base_datasets(contract, batch_size)
    data_config = attach_carla_transforms(
        datasets["train"], batch_size, workers, pair["student"]
    )
    attach_carla_transforms(datasets["validation"], batch_size, workers, pair["student"])
    geometry = front_mask_geometry(data_config["input_size"])
    generator = torch.Generator()
    generator.manual_seed(contract["training"]["seed"])
    loaders, wrapped = {}, {}
    for split in ("train", "validation"):
        wrapped[split] = FunctionalFrameDataset(
            datasets[split],
            contract["class_config"],
            contract["ignore_index"],
            geometry,
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


def run_training(config_path, run_id, result_root="results/thesis_m2"):
    contract, git_status, gpu_usage, run_directory = _prepare_run(
        config_path, run_id, result_root
    )
    training = contract["training"]
    set_reproducible_seed(training["seed"])
    device = torch.device("cuda")
    pair = build_pretraining_pair(contract)
    pair["student"].to(device)
    pair["seg_model"].to(device)
    loaders, wrapped, data_config, geometry = _build_loaders(
        contract, pair, training["batch_size"], training["num_workers"]
    )
    class_names = [item["name"] for item in contract["class_config"]["classes"]]
    criterion = DeterministicCrossEntropyLoss(
        ignore_index=contract["ignore_index"], class_weights=contract["class_weights"]
    ).to(device)
    seg_parameters = [
        parameter
        for name, parameter in pair["seg_model"].named_parameters()
        if not name.startswith("backbone.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": list(pair["student"].parameters())},
            {"params": seg_parameters, "lr": training["learning_rate"]},
        ],
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    run_manifest = {
        "schema_version": 1,
        "status": "running",
        "run_id": run_id,
        "protocol_document": "docs/interfuser_architecture_aware_semantic_pretraining_protocol_v9.md",
        "started_at": _utc_now(),
        "config": {"path": contract["config_path"], "sha256": contract["sha256"]},
        "git": {"head": _git_output("rev-parse", "HEAD"), "status_porcelain": git_status},
        "gpu_usage_at_start": gpu_usage,
        "dependencies": _dependency_versions(),
        "bound_artifacts": {
            field: {"path": str(path), "sha256": sha256_file(path)}
            for field, path in contract["bound_paths"].items()
        },
        "corpus": {
            "train_frames_retained": len(wrapped["train"]),
            "validation_frames_retained": len(wrapped["validation"]),
            "lidar_y_axis_multiplier": contract["dataset_block"]["lidar_y_axis_multiplier"],
        },
        "selection_metric": "validation.driving_loss",
        "aux_coefficient": contract["aux_coefficient"],
    }
    _atomic_write_json(run_directory / "run_manifest.json", run_manifest)

    epochs = []
    best_epoch = None
    best_driving = float("inf")
    best_states = None
    for epoch_index in range(training["epochs"]):
        train_metrics = run_pretraining_epoch(
            pair, loaders["train"], device, class_names, criterion,
            optimizer=optimizer, aux_coefficient=contract["aux_coefficient"],
        )
        with torch.no_grad():
            validation_metrics = run_pretraining_epoch(
                pair, loaders["validation"], device, class_names, criterion,
                aux_coefficient=contract["aux_coefficient"],
            )
        epoch_record = {
            "epoch": epoch_index + 1,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        epochs.append(epoch_record)
        print(
            json.dumps(
                {
                    "epoch": epoch_record["epoch"],
                    "train_driving": round(train_metrics["driving_loss"], 6),
                    "train_semantic": round(train_metrics["semantic_loss"], 6),
                    "val_driving": round(validation_metrics["driving_loss"], 6),
                    "val_semantic": round(validation_metrics["semantic_loss"], 6),
                    "val_mean_iou": round(validation_metrics["mean_iou"], 6),
                    "seconds": train_metrics["duration_seconds"]
                    + validation_metrics["duration_seconds"],
                }
            ),
            flush=True,
        )
        driving = validation_metrics["driving_loss"]
        if driving < best_driving:
            best_epoch = epoch_index + 1
            best_driving = driving
            best_states = {
                "student": OrderedDict(
                    (key, value.detach().cpu())
                    for key, value in pair["student"].state_dict().items()
                ),
                "semantic_head": OrderedDict(
                    (key, value.detach().cpu())
                    for key, value in pair["seg_model"].state_dict().items()
                ),
            }
    if best_states is None or best_epoch is None:
        raise PretrainingRunError("training produced no best model state")
    pair["student"].load_state_dict(best_states["student"], strict=True)
    pair["seg_model"].load_state_dict(best_states["semantic_head"], strict=True)

    best_checkpoint_path = run_directory / "checkpoint_best.pth"
    torch.save(
        {
            "format_version": 1,
            "epoch": best_epoch,
            "selection_metric": "validation.driving_loss",
            "selection_metric_value": best_driving,
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
            "best_epoch": best_epoch,
            "best_validation_driving_loss": best_driving,
            "exports": {
                "checkpoint_best": {
                    "path": str(best_checkpoint_path),
                    "sha256": sha256_file(best_checkpoint_path),
                }
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
                "best_validation_driving_loss": manifest["best_validation_driving_loss"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
