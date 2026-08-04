#!/usr/bin/env python3
"""
[INPUT]: 依赖已哈希绑定的语义训练配置/checkpoint 与行人危险扩充 manifest 中 route-group 隔离的 validation/test holdout。
[OUTPUT]: 提供 prepare_hazard_holdout_contract、evaluate_semantic_hazard_holdout 与 CLI，输出两个专项 holdout 的 mIoU、macro-F1 及逐类混淆指标 JSON。
[POS]: tools/training 的 M2 行人危险定向评估器；在不参与 checkpoint 选择的前提下，对 B0 与数据扩充模型使用同一专项难例口径。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERFUSER_ROOT = REPO_ROOT / "interfuser"
for import_root in (REPO_ROOT, INTERFUSER_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

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
    sha256_file,
)


REPORT_SCHEMA_VERSION = 1


class HazardHoldoutEvaluationError(RuntimeError):
    """Raised when targeted holdout evidence is incomplete or incomparable."""


def _read_json(path, label):
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HazardHoldoutEvaluationError(
            f"unable to read {label} JSON {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise HazardHoldoutEvaluationError(f"{label} must be a JSON object")
    return value


def _sample_key_sha256(keys):
    digest = hashlib.sha256()
    for key in keys:
        encoded = key.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def prepare_hazard_holdout_contract(contract, hazard_manifest_path, split):
    """Project one isolated hazard holdout onto the existing validation loader."""
    if split not in {"validation", "test"}:
        raise HazardHoldoutEvaluationError("split must be validation or test")
    path = Path(hazard_manifest_path).resolve()
    manifest = _read_json(path, "hazard manifest")
    if not manifest.get("valid"):
        raise HazardHoldoutEvaluationError("hazard manifest must be valid")
    if manifest.get("dataset_root") != contract["split_manifest_loaded"].get(
        "dataset_root"
    ):
        raise HazardHoldoutEvaluationError(
            "hazard manifest and training contract dataset roots differ"
        )
    if manifest.get("cameras") != contract["data"]["cameras"]:
        raise HazardHoldoutEvaluationError(
            "hazard manifest and training cameras differ"
        )
    if manifest.get("source", {}).get("class_config_sha256") != contract[
        "class_config_sha256"
    ]:
        raise HazardHoldoutEvaluationError(
            "hazard manifest and training class config SHA-256 differ"
        )
    records = manifest.get("hazard_holdout", {}).get(split)
    if not isinstance(records, list) or not records:
        raise HazardHoldoutEvaluationError(
            f"hazard manifest has no {split} holdout records"
        )
    route_groups = [item.get("route_group") for item in records]
    if any(not isinstance(group, str) for group in route_groups):
        raise HazardHoldoutEvaluationError("holdout record has invalid route_group")
    if len(route_groups) != len(set(route_groups)):
        raise HazardHoldoutEvaluationError(
            f"hazard {split} holdout duplicates route groups"
        )
    opposite = "test" if split == "validation" else "validation"
    opposite_groups = {
        item.get("route_group")
        for item in manifest.get("hazard_holdout", {}).get(opposite, [])
    }
    if set(route_groups) & opposite_groups:
        raise HazardHoldoutEvaluationError(
            "hazard validation and test route groups overlap"
        )

    projected_records = []
    for item in records:
        projected = dict(item)
        projected["split"] = "validation"
        projected_records.append(projected)
    available_samples = sum(
        int(item["declared_frames"]) for item in projected_records
    ) * len(contract["data"]["cameras"])
    if available_samples <= 0:
        raise HazardHoldoutEvaluationError("hazard holdout has no camera samples")
    projected_contract = copy.deepcopy(contract)
    projected_contract["split_manifest_loaded"] = {
        "valid": True,
        "dataset_root": manifest["dataset_root"],
        "cameras": manifest["cameras"],
        "source": {
            "class_config_sha256": manifest["source"]["class_config_sha256"]
        },
        "sequences": projected_records,
    }
    projected_contract["data"]["max_validation_samples"] = available_samples
    projected_contract["data"][
        "expected_available_validation_samples"
    ] = available_samples
    return projected_contract, {
        "manifest": str(path),
        "manifest_sha256": sha256_file(path),
        "holdout_split": split,
        "route_groups": sorted(route_groups),
        "sequences": len(records),
        "expected_camera_samples": available_samples,
    }


def _evaluate(model, loader, criterion, class_names, device):
    model.eval()
    metrics = ConfusionMetrics(len(class_names), criterion.ignore_index)
    total_loss = 0.0
    sample_count = 0
    sample_keys = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            count = images.shape[0]
            total_loss += float(loss.detach().cpu().item()) * count
            sample_count += count
            sample_keys.extend(batch["key"])
            metrics.update(logits, labels)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = metrics.summary(class_names)
    result.update(
        {
            "loss": total_loss / sample_count,
            "samples": sample_count,
            "batches": len(loader),
            "sample_keys_sha256": _sample_key_sha256(sample_keys),
        }
    )
    return result


def evaluate_semantic_hazard_holdout(
    training_config_path,
    checkpoint_path,
    hazard_manifest_path,
    splits=("validation", "test"),
):
    """Evaluate one semantic checkpoint on immutable pedestrian hazard holdouts."""
    try:
        contract = load_training_contract(training_config_path)
    except TrainingContractError as exc:
        raise HazardHoldoutEvaluationError(str(exc)) from exc
    physical_gpu = contract["training"]["physical_gpu_index"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise HazardHoldoutEvaluationError(
            f"CUDA_VISIBLE_DEVICES must be {physical_gpu}"
        )
    try:
        ensure_gpus_available(
            [physical_gpu],
            contract["training"]["gpu_busy_memory_threshold_mb"],
        )
    except RunnerError as exc:
        raise HazardHoldoutEvaluationError(str(exc)) from exc
    if not torch.cuda.is_available():
        raise HazardHoldoutEvaluationError("CUDA is unavailable")
    checkpoint_path = Path(checkpoint_path).resolve()
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise HazardHoldoutEvaluationError(
            f"unable to load checkpoint {checkpoint_path}: {exc}"
        ) from exc
    if checkpoint.get("training_config_sha256") != contract["sha256"]:
        raise HazardHoldoutEvaluationError(
            "checkpoint and training config SHA-256 differ"
        )
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise HazardHoldoutEvaluationError("checkpoint has no model_state_dict")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model = SemanticPretrainingModel(contract).to(device)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise HazardHoldoutEvaluationError(
            "checkpoint is not strict-load compatible"
        )
    criterion = DeterministicCrossEntropyLoss(
        ignore_index=contract["training"]["ignore_index"],
        class_weights=contract["training"].get("class_weights"),
    ).to(device)
    class_names = [
        item["name"] for item in contract["class_config_loaded"]["classes"]
    ]
    evaluations = {}
    holdout_sources = {}
    for split in splits:
        projected_contract, holdout_source = prepare_hazard_holdout_contract(
            contract, hazard_manifest_path, split
        )
        dataset = SemanticFrameDataset(projected_contract, "validation")
        loader = DataLoader(
            dataset,
            batch_size=contract["training"]["batch_size"],
            shuffle=False,
            num_workers=contract["training"]["num_workers"],
            pin_memory=True,
            drop_last=False,
        )
        evaluations[split] = _evaluate(
            model, loader, criterion, class_names, device
        )
        holdout_sources[split] = holdout_source
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "valid": True,
        "errors": [],
        "sources": {
            "training_config": str(contract["path"]),
            "training_config_sha256": contract["sha256"],
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "hazard_holdouts": holdout_sources,
        },
        "contract": {
            "privileged_hazard_used_as_model_input": False,
            "checkpoint_selection_uses_hazard_holdout": False,
            "class_weights": contract["training"].get("class_weights"),
        },
        "evaluations": evaluations,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a semantic checkpoint on pedestrian hazard holdouts"
    )
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hazard-manifest", type=Path, required=True)
    parser.add_argument(
        "--splits", nargs="+", choices=("validation", "test"), default=("validation", "test")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        print(f"evaluation error: refusing to overwrite {args.output}", file=sys.stderr)
        return 2
    try:
        report = evaluate_semantic_hazard_holdout(
            args.training_config,
            args.checkpoint,
            args.hazard_manifest,
            splits=tuple(args.splits),
        )
    except (HazardHoldoutEvaluationError, TrainingContractError) as exc:
        print(f"evaluation error: {exc}", file=sys.stderr)
        return 2
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    except FileExistsError:
        print(f"evaluation error: refusing to overwrite {args.output}", file=sys.stderr)
        return 2
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
