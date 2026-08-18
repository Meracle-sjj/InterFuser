#!/usr/bin/env python3
"""
 * [INPUT]: 依赖 v9 协议、interfuser/train.py 的原生驾驶损失（逐字移植）、functional_distillation 的
 *   数据集/审计绑定/钩子机器、官方 B0 checkpoint 与 v7 语义头（哈希绑定）。
 * [OUTPUT]: 对外提供 ArchitectureContractError、load_architecture_contract、WaypointL1Loss、MVTL1Loss、
 *   combined_driving_loss、build_pretraining_pair、run_pretraining_epoch。
 * [POS]: tools/training 的 v9 架构内辅助语义预训练库；全模型可训，语义头经 layer1-4 钩子注入，
 *   驾驶损失与 interfuser/train.py 权重逐字一致。
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""
import json
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
INTERFUSER_ROOT = REPO_ROOT / "interfuser"
for import_root in (REPO_ROOT, INTERFUSER_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tools.data.audit_semantic_pretraining_data import (  # noqa: E402
    AuditError,
    load_class_config,
)
from tools.training.functional_distillation import (  # noqa: E402
    FunctionalFrameDataset,
    _resolve_repo_path,
    _verify_sha,
    attach_carla_transforms,
    build_base_datasets,
    front_mask_geometry,
)
from tools.training.semantic_pretraining import (  # noqa: E402
    ConfusionMetrics,
    DeterministicCrossEntropyLoss,
    SemanticPretrainingModel,
    sha256_file,
)

DRIVING_LOSS_WEIGHTS = {
    "traffic": 0.5,
    "waypoints": 0.2,
    "velocity": 0.05,
    "junction": 0.05,
    "red_light": 0.1,
    "stop_sign": 0.01,
}


class ArchitectureContractError(RuntimeError):
    """Raised when the v9 architecture pretraining contract is violated."""


class WaypointL1Loss:
    """Verbatim port of interfuser/train.py WaypointL1Loss (10-step decay weights)."""

    WEIGHTS = [
        0.1407441030399059,
        0.13352157985305926,
        0.12588535273178575,
        0.11775496498388233,
        0.10901991343009122,
        0.09952110967153563,
        0.08901438656870617,
        0.07708872007078788,
        0.06294267636589287,
        0.04450719328435308,
    ]

    def __init__(self):
        self.loss = torch.nn.L1Loss(reduction="none")

    def __call__(self, output, target):
        invalid_mask = target.ge(1000)
        output = output.masked_fill(invalid_mask, 0.0)
        target = target.masked_fill(invalid_mask, 0.0)
        loss = self.loss(output, target)
        loss = torch.mean(loss, (0, 2))
        loss = loss * torch.tensor(self.WEIGHTS, device=output.device)
        return torch.mean(loss)


class MVTL1Loss:
    """Verbatim port of interfuser/train.py MVTL1Loss (balanced occupancy + geometry + speed)."""

    def __init__(self, weight=1):
        self.loss = torch.nn.L1Loss()
        self.weight = weight

    def __call__(self, output, target):
        target_1_mask = target[:, :, 0].ge(0.01)
        target_0_mask = target[:, :, 0].le(0.01)
        target_prob_1 = torch.masked_select(target[:, :, 0], target_1_mask)
        output_prob_1 = torch.masked_select(output[:, :, 0], target_1_mask)
        target_prob_0 = torch.masked_select(target[:, :, 0], target_0_mask)
        output_prob_0 = torch.masked_select(output[:, :, 0], target_0_mask)
        if target_prob_1.numel() == 0:
            loss_prob_1 = torch.tensor(0.0, device=output.device, dtype=output.dtype)
        else:
            loss_prob_1 = self.loss(output_prob_1, target_prob_1)
        if target_prob_0.numel() == 0:
            loss_prob_0 = torch.tensor(0.0, device=output.device, dtype=output.dtype)
        else:
            loss_prob_0 = self.loss(output_prob_0, target_prob_0)
        loss_1 = 0.5 * loss_prob_0 + 0.5 * loss_prob_1
        output_1 = output[target_1_mask][:][:, 1:6]
        target_1 = target[target_1_mask][:][:, 1:6]
        if target_1.numel() == 0:
            loss_2 = torch.tensor(0.0, device=output.device, dtype=output.dtype)
        else:
            loss_2 = self.loss(target_1, output_1)
        output_2 = output[target_1_mask][:][:, 6]
        target_2 = target[target_1_mask][:][:, 6]
        if target_2.numel() == 0:
            loss_3 = torch.tensor(0.0, device=output.device, dtype=output.dtype)
        else:
            loss_3 = self.loss(target_2, output_2)
        return 0.5 * loss_1 * self.weight + 0.5 * loss_2, loss_3


def combined_driving_loss(outputs, targets):
    """Replicate the exact loss combination of interfuser/train.py lines 1451-1466."""
    traffic_loss, velocity_loss = MVTL1Loss(1.0)(outputs[0], targets[4])
    waypoints_loss = WaypointL1Loss()(outputs[1], targets[1])
    classification = torch.nn.CrossEntropyLoss()
    junction_loss = classification(outputs[2], targets[2].long())
    red_light_loss = classification(outputs[3], targets[3].long())
    stop_sign_loss = classification(outputs[4], targets[6].long())
    parts = {
        "traffic": traffic_loss,
        "velocity": velocity_loss,
        "waypoints": waypoints_loss,
        "junction": junction_loss,
        "red_light": red_light_loss,
        "stop_sign": stop_sign_loss,
    }
    total = (
        parts["traffic"] * DRIVING_LOSS_WEIGHTS["traffic"]
        + parts["waypoints"] * DRIVING_LOSS_WEIGHTS["waypoints"]
        + parts["velocity"] * DRIVING_LOSS_WEIGHTS["velocity"]
        + parts["junction"] * DRIVING_LOSS_WEIGHTS["junction"]
        + parts["red_light"] * DRIVING_LOSS_WEIGHTS["red_light"]
        + parts["stop_sign"] * DRIVING_LOSS_WEIGHTS["stop_sign"]
    )
    return total, parts


def load_architecture_contract(config_path):
    """Load and hash-bind the v9 architecture pretraining config."""
    config_path = _resolve_repo_path(config_path, "config")
    raw = json.loads(config_path.read_text())
    bound = {}
    for field in ("initial_checkpoint", "semantic_head_checkpoint"):
        expected = raw.get(f"{field}_sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ArchitectureContractError(f"{field}_sha256 is required")
        path = _resolve_repo_path(raw[field], field)
        _verify_sha(path, expected, field)
        bound[field] = path

    corpus = raw.get("corpus_audit", {})
    audit_path = _resolve_repo_path(corpus.get("manifest"), "corpus_audit.manifest")
    _verify_sha(audit_path, corpus.get("manifest_sha256"), "corpus_audit.manifest")
    audit = json.loads(audit_path.read_text())
    dataset_block = raw.get("dataset", {})
    if float(dataset_block.get("lidar_y_axis_multiplier")) != float(
        audit["lidar_convention"]["selected_multiplier"]
    ):
        raise ArchitectureContractError(
            "dataset.lidar_y_axis_multiplier disagrees with audited convention"
        )
    indexes = {}
    for split in ("train", "validation"):
        spec = corpus.get(f"{split}_index", {})
        index_path = _resolve_repo_path(spec.get("path"), f"corpus_audit.{split}_index")
        _verify_sha(index_path, spec.get("sha256"), f"corpus_audit.{split}_index")
        indexes[split] = {
            "path": index_path,
            "expected_retained_frames": audit["splits"][split]["expected_retained_frames"],
        }

    class_config_path = _resolve_repo_path(raw.get("class_config"), "class_config")
    _verify_sha(class_config_path, raw.get("class_config_sha256"), "class_config")
    try:
        class_config = load_class_config(class_config_path)
    except AuditError as exc:
        raise ArchitectureContractError(str(exc)) from exc

    training = raw.get("training", {})
    for field in ("epochs", "batch_size", "seed", "physical_gpu_index"):
        if not isinstance(training.get(field), int) or training[field] <= 0:
            raise ArchitectureContractError(f"training.{field} is invalid")
    for field in ("learning_rate", "weight_decay", "aux_coefficient"):
        value = training.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
            raise ArchitectureContractError(f"training.{field} is invalid")
    if training.get("gpu_resource_policy") != "shared_capacity":
        raise ArchitectureContractError("gpu_resource_policy must be shared_capacity")
    if training.get("driving_loss_weights") != DRIVING_LOSS_WEIGHTS:
        raise ArchitectureContractError("driving_loss_weights must match the author recipe")

    return {
        "config_path": str(config_path),
        "sha256": sha256_file(config_path),
        "raw": raw,
        "model_name": raw.get("model", {}).get("name", "interfuser_baseline"),
        "bound_paths": bound,
        "audit": audit,
        "indexes": indexes,
        "dataset_block": dataset_block,
        "class_config": class_config,
        "training": training,
        "class_weights": training.get("class_weights"),
        "ignore_index": int(training.get("ignore_index", 255)),
        "aux_coefficient": float(training["aux_coefficient"]),
    }


def build_pretraining_pair(contract):
    """Full-trainable student from official B0 plus hook-fed v7-initialized seg head."""
    from timm import create_model

    student = create_model(
        contract["model_name"],
        pretrained=False,
        checkpoint_path=str(contract["bound_paths"]["initial_checkpoint"]),
        freeze_num=-1,
    )
    student.train()
    bound = contract["bound_paths"]
    seg_contract_view = {
        "backbone": {
            "feature_indices": [1, 2, 3, 4],
            "pretrained_checkpoint": str(bound["initial_checkpoint"]),
            "pretrained_checkpoint_format": "interfuser_checkpoint",
            "pretrained_state_prefix": "rgb_backbone.",
        },
        "pretrained_path": str(bound["initial_checkpoint"]),
        "model": {
            "num_classes": len(contract["class_config"]["classes"]),
            "decoder_channels": contract["raw"].get("semantic_head_decoder_channels", 64),
            "dropout": contract["raw"].get("semantic_head_dropout", 0.1),
        },
    }
    seg_model = SemanticPretrainingModel(seg_contract_view)
    payload = torch.load(
        bound["semantic_head_checkpoint"], map_location="cpu", weights_only=False
    )
    state = payload.get("model_state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ArchitectureContractError("semantic head checkpoint lacks model_state_dict")
    seg_model.load_state_dict(state, strict=True)
    seg_model.train()

    captured = {}

    def _make_hook(name):
        def hook(module, inputs, output):
            captured[name] = output
        return hook

    handles = [
        getattr(student.rgb_backbone, f"layer{stage}").register_forward_hook(
            _make_hook(str(stage))
        )
        for stage in (1, 2, 3, 4)
    ]
    return {
        "student": student,
        "seg_model": seg_model,
        "captured": captured,
        "hook_handles": handles,
    }


def run_pretraining_epoch(
    pair,
    loader,
    device,
    class_names,
    criterion,
    optimizer=None,
    aux_coefficient=1.0,
):
    training = optimizer is not None
    student = pair["student"]
    seg_model = pair["seg_model"]
    captured = pair["captured"]
    student.train(training)
    seg_model.train(training)
    metrics = ConfusionMetrics(len(class_names), criterion.ignore_index)
    totals = {"loss": 0.0, "driving_loss": 0.0, "semantic_loss": 0.0}
    part_totals = {}
    sample_count = 0
    started = time.monotonic()
    for batch in loader:
        inputs = {name: value.to(device, non_blocking=False) for name, value in batch["data"].items()}
        targets = [value.to(device, non_blocking=False) for value in batch["targets"]]
        labels = batch["label"].to(device, non_blocking=False)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            captured.clear()
            outputs = student(inputs)
            driving_loss, parts = combined_driving_loss(outputs, targets)
            features = [captured[str(stage)] for stage in (1, 2, 3, 4)]
            seg_logits = seg_model.forward_features(features)
            seg_logits = F.interpolate(
                seg_logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            semantic_loss = criterion(seg_logits, labels)
            loss = driving_loss + float(aux_coefficient) * semantic_loss
            if training:
                loss.backward()
                optimizer.step()
        batch_size = labels.shape[0]
        sample_count += batch_size
        totals["loss"] += float(loss.detach().cpu().item()) * batch_size
        totals["driving_loss"] += float(driving_loss.detach().cpu().item()) * batch_size
        totals["semantic_loss"] += float(semantic_loss.detach().cpu().item()) * batch_size
        for name, value in parts.items():
            part_totals[name] = part_totals.get(name, 0.0) + float(value.detach().cpu().item()) * batch_size
        metrics.update(seg_logits, labels)
    summary = metrics.summary(class_names)
    summary.update({key: value / sample_count for key, value in totals.items()})
    summary["driving_loss_parts"] = {name: value / sample_count for name, value in part_totals.items()}
    summary["samples"] = sample_count
    summary["batches"] = len(loader)
    summary["duration_seconds"] = round(time.monotonic() - started, 3)
    return summary
