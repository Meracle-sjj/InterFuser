#!/usr/bin/env python3
"""
 * [INPUT]: 依赖 v8.1 协议、semantic_pretraining 的 TrainingContractError/sha256_file/set_reproducible_seed/
 *   DeterministicCrossEntropyLoss/ConfusionMetrics/SemanticPretrainingModel/validate_backbone_export、
 *   interfuser/timm 的 create_model/create_carla_dataset/create_carla_loader/resolve_data_config、
 *   官方 B0 与 v7 swap-init checkpoint（哈希绑定）。
 * [OUTPUT]: 对外提供 FunctionalContractError、load_functional_contract、build_label_lut、front_mask_geometry、
 *   FunctionalFrameDataset、build_base_datasets、build_functional_pair、functional_output_l2、
 *   run_functional_epoch、assert_pair_invariants、export_student_backbone。
 * [POS]: tools/training 的 v8b BCT 功能蒸馏库；被 run_functional_distillation.py 消费。与 v7 特征蒸馏的本质
 *   差别：约束从"骨干中间特征"移到"冻结下游栈的最终输出"，LiDAR 流两端同值恒定在场（协议 §2.1）。
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""
import json
import math
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
INTERFUSER_ROOT = REPO_ROOT / "interfuser"
for import_root in (REPO_ROOT, INTERFUSER_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tools.data.audit_semantic_pretraining_data import (  # noqa: E402
    AuditError,
    load_class_config,
)
from tools.training.semantic_pretraining import (  # noqa: E402
    ConfusionMetrics,
    DeterministicCrossEntropyLoss,
    SemanticPretrainingModel,
    TrainingContractError,
    set_reproducible_seed,
    sha256_file,
    validate_backbone_export,
)

MATCHED_HEAD_NAMES = (
    "traffic",
    "waypoints",
    "is_junction",
    "traffic_light_state",
    "stop_sign",
)
FUNCTIONAL_HEAD_INDICES = (0, 1, 2, 3, 4)


class FunctionalContractError(RuntimeError):
    """Raised when the v8b functional distillation contract is violated."""


def _resolve_repo_path(value, label):
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        raise FunctionalContractError(f"{label} not found: {path}")
    return path


def _verify_sha(path, expected, label):
    actual = sha256_file(path)
    if actual != expected:
        raise FunctionalContractError(
            f"{label} sha256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def load_functional_contract(config_path):
    """Load and fully hash-bind the v8b functional distillation config."""
    config_path = _resolve_repo_path(config_path, "config")
    raw = json.loads(config_path.read_text())
    contract = {"config_path": str(config_path), "raw": raw}

    if raw.get("training", {}).get("feature_distillation") is not None:
        raise FunctionalContractError(
            "feature_distillation and functional_distillation are mutually exclusive"
        )
    functional = raw.get("functional_distillation")
    if not isinstance(functional, dict):
        raise FunctionalContractError("functional_distillation block is required")
    coefficient = functional.get("coefficient")
    if not isinstance(coefficient, (int, float)) or isinstance(coefficient, bool):
        raise FunctionalContractError("functional_distillation.coefficient must be numeric")
    if not math.isfinite(float(coefficient)) or float(coefficient) <= 0.0:
        raise FunctionalContractError("functional_distillation.coefficient must be positive")
    if list(functional.get("matched_heads", [])) != list(MATCHED_HEAD_NAMES):
        raise FunctionalContractError(
            f"matched_heads must be exactly {list(MATCHED_HEAD_NAMES)}"
        )

    bound = {}
    for field in ("teacher_checkpoint", "student_initialization_checkpoint",
                  "semantic_head_checkpoint"):
        expected = functional.get(f"{field}_sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise FunctionalContractError(f"{field}_sha256 is required")
        path = _resolve_repo_path(functional[field], field)
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
        raise FunctionalContractError(
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
    split_manifest_path = _resolve_repo_path(raw.get("split_manifest"), "split_manifest")
    _verify_sha(
        split_manifest_path, raw.get("split_manifest_sha256"), "split_manifest"
    )
    _verify_sha(
        Path(audit["split_manifest"]["path"]),
        audit["split_manifest"]["sha256"],
        "audit.split_manifest",
    )

    training = raw.get("training", {})
    for field, predicate in (
        ("epochs", lambda v: isinstance(v, int) and v > 0),
        ("batch_size", lambda v: isinstance(v, int) and v > 0),
        ("learning_rate", lambda v: isinstance(v, (int, float)) and v > 0),
        ("backbone_learning_rate", lambda v: isinstance(v, (int, float)) and v > 0),
        ("seed", lambda v: isinstance(v, int)),
        ("physical_gpu_index", lambda v: isinstance(v, int)),
    ):
        if not predicate(training.get(field)):
            raise FunctionalContractError(f"training.{field} is missing or invalid")
    if training.get("gpu_resource_policy") != "shared_capacity":
        raise FunctionalContractError("gpu_resource_policy must be shared_capacity")

    try:
        class_config = load_class_config(class_config_path)
    except AuditError as exc:
        raise FunctionalContractError(str(exc)) from exc
    contract.update(
        {
            "sha256": sha256_file(config_path),
            "model_name": raw.get("model", {}).get("name", "interfuser_baseline"),
            "functional": functional,
            "coefficient": float(coefficient),
            "bound_paths": bound,
            "audit": audit,
            "indexes": indexes,
            "dataset_block": dataset_block,
            "class_config": class_config,
            "training": training,
            "class_weights": training.get("class_weights"),
            "ignore_index": int(training.get("ignore_index", 255)),
        }
    )
    return contract


def build_label_lut(class_config, ignore_index):
    """Map CARLA source tags to train ids exactly as SemanticFrameDataset does."""
    label_lut = np.full(256, ignore_index, dtype=np.uint8)
    known_lut = np.zeros(256, dtype=bool)
    for item in class_config["classes"]:
        for source_tag in item["source_tags"]:
            label_lut[source_tag] = item["train_id"]
            known_lut[source_tag] = True
    for source_tag in class_config["ignore_tags"]:
        known_lut[source_tag] = True
    return label_lut, known_lut


def front_mask_geometry(input_size):
    """Replicate create_carla_rgb_transform's PIL geometry (resize then crop)."""
    if isinstance(input_size, (tuple, list)):
        crop_size = tuple(input_size[-2:])
        number = input_size[-1]
    else:
        crop_size = input_size
        number = input_size
    table = {112: (170, 128), 128: (195, 146), 224: (341, 256), 256: (340, 288)}
    resize_wh = table.get(number)
    if resize_wh is None:
        resize_wh = (int((number + 32) / 3.0 * 4.0), number + 32)
    return {"resize_wh": resize_wh, "crop_size": crop_size}


class FunctionalFrameDataset(Dataset):
    """Wrap CarlaMVDetDataset frames with v7-faithful 10-class seg labels."""

    def __init__(self, base, class_config, ignore_index, geometry):
        self.base = base
        self.label_lut, self.known_lut = build_label_lut(class_config, ignore_index)
        self.resize_wh = geometry["resize_wh"]
        self.center_crop = transforms.CenterCrop(geometry["crop_size"])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        route_dir, frame_id = self.base.route_frames[index]
        data, targets = self.base._get_item_impl(index)
        mask_path = Path(route_dir) / "seg_front" / f"{int(frame_id):04d}.png"
        with Image.open(mask_path) as image:
            resized = image.resize(self.resize_wh, Image.Resampling.NEAREST)
            cropped = self.center_crop(resized)
        raw = np.asarray(cropped)
        if raw.ndim == 3:
            raw = raw[..., 0]
        if raw.ndim != 2 or raw.dtype.kind not in "ui":
            raise FunctionalContractError(
                f"semantic mask must be 2D integer: {mask_path}"
            )
        unknown = sorted(int(value) for value in np.unique(raw) if not self.known_lut[int(value)])
        if unknown:
            raise FunctionalContractError(f"unmapped source tags {unknown}: {mask_path}")
        labels = torch.from_numpy(self.label_lut[raw].astype(np.int64, copy=False))
        return {"data": data, "targets": targets, "label": labels}


def build_base_datasets(contract, batch_size):
    """Create the carla datasets bound to the audited semantic indexes."""
    from timm.data import create_carla_dataset

    dataset_block = contract["dataset_block"]
    datasets = {}
    for split in ("train", "validation"):
        base = create_carla_dataset(
            "carla",
            root=dataset_block["root"],
            towns=dataset_block["towns"],
            weathers=dataset_block["weathers"],
            batch_size=batch_size,
            with_lidar=True,
            multi_view=True,
            augment_prob=0.0,
            dataset_index=str(contract["indexes"][split]["path"]),
            lidar_y_axis_multiplier=dataset_block["lidar_y_axis_multiplier"],
            navigation_frame=dataset_block["navigation_frame"],
            missing_navigation_policy=dataset_block["missing_navigation_policy"],
        )
        expected = contract["indexes"][split]["expected_retained_frames"]
        if len(base) != expected:
            raise FunctionalContractError(
                f"{split} dataset retained {len(base)} frames, audit expected {expected}"
            )
        datasets[split] = base
    return datasets


def attach_carla_transforms(base, batch_size, workers, model):
    """Assign the official transform stack, mirroring scene validation."""
    from timm.data import create_carla_loader, resolve_data_config

    data_config = resolve_data_config({}, model=model, verbose=False)
    create_carla_loader(
        base,
        input_size=data_config["input_size"],
        batch_size=batch_size,
        multi_view_input_size=(3, 128, 128),
        is_training=False,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=workers,
        distributed=False,
        pin_memory=False,
        persistent_workers=workers > 0,
    )
    return data_config


def build_functional_pair(contract):
    """Build frozen teacher, eval-frozen student, and v7-initialized seg head."""
    from timm import create_model

    bound = contract["bound_paths"]
    teacher = create_model(
        contract["model_name"],
        pretrained=False,
        checkpoint_path=str(bound["teacher_checkpoint"]),
        freeze_num=-1,
    )
    teacher.eval()
    teacher.requires_grad_(False)

    student = create_model(
        contract["model_name"],
        pretrained=False,
        checkpoint_path=str(bound["student_initialization_checkpoint"]),
        freeze_num=-1,
    )
    student.eval()
    student.requires_grad_(False)
    # cuDNN RNN backward 仅在 train 模式可用；GRU 无 dropout，train/eval 前向语义一致，
    # BN 与 Transformer dropout 仍保持 eval（冻结统计、无噪声注入）。
    for module in student.modules():
        if isinstance(module, torch.nn.RNNBase):
            module.train()

    functional = contract["functional"]
    seg_contract_view = {
        "backbone": {
            "feature_indices": [1, 2, 3, 4],
            "pretrained_checkpoint": str(bound["teacher_checkpoint"]),
            "pretrained_checkpoint_format": "interfuser_checkpoint",
            "pretrained_state_prefix": "rgb_backbone.",
        },
        "pretrained_path": str(bound["teacher_checkpoint"]),
        "model": {
            "num_classes": len(contract["class_config"]["classes"]),
            "decoder_channels": functional.get("semantic_head_decoder_channels", 64),
            "dropout": functional.get("semantic_head_dropout", 0.1),
        },
    }
    seg_model = SemanticPretrainingModel(seg_contract_view)
    payload = torch.load(
        bound["semantic_head_checkpoint"], map_location="cpu", weights_only=False
    )
    state = payload.get("model_state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise FunctionalContractError("semantic head checkpoint lacks model_state_dict")
    seg_model.load_state_dict(state, strict=True)
    seg_model.eval()
    seg_model.requires_grad_(False)

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
        "teacher": teacher,
        "student": student,
        "seg_model": seg_model,
        "captured": captured,
        "hook_handles": handles,
    }


def functional_output_l2(student_outputs, teacher_outputs):
    """Sum per-head ||os-ot||^2/||ot||^2 so head scale never becomes a knob."""
    if len(student_outputs) <= max(FUNCTIONAL_HEAD_INDICES):
        raise FunctionalContractError("model outputs lack the matched heads")
    per_head = {}
    total = None
    for name, index in zip(MATCHED_HEAD_NAMES, FUNCTIONAL_HEAD_INDICES):
        student_value = student_outputs[index]
        teacher_value = teacher_outputs[index].detach()
        if student_value.shape != teacher_value.shape:
            raise FunctionalContractError(
                f"head {name} shape differs: {tuple(student_value.shape)} vs "
                f"{tuple(teacher_value.shape)}"
            )
        denominator = teacher_value.pow(2).sum().clamp_min(1e-12)
        penalty = (student_value - teacher_value).pow(2).sum() / denominator
        per_head[name] = penalty
        total = penalty if total is None else total + penalty
    return total, per_head


def run_functional_epoch(
    pair,
    loader,
    criterion,
    device,
    class_names,
    optimizer=None,
    distill_coefficient=1.0,
    accumulation=1,
):
    training = optimizer is not None
    teacher = pair["teacher"]
    student = pair["student"]
    seg_model = pair["seg_model"]
    captured = pair["captured"]
    seg_model.train(training)
    metrics = ConfusionMetrics(len(class_names), criterion.ignore_index)
    totals = {
        "loss": 0.0,
        "task_loss": 0.0,
        "functional_distillation_penalty": 0.0,
    }
    per_head_totals = {name: 0.0 for name in MATCHED_HEAD_NAMES}
    sample_count = 0
    started = time.monotonic()
    for step, batch in enumerate(loader):
        inputs = {name: value.to(device, non_blocking=False) for name, value in batch["data"].items()}
        labels = batch["label"].to(device, non_blocking=False)
        if training and step % accumulation == 0:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.no_grad():
                teacher_outputs = teacher(inputs)
            captured.clear()
            student_outputs = student(inputs)
            features = [captured[str(stage)] for stage in (1, 2, 3, 4)]
            seg_logits = seg_model.forward_features(features)
            seg_logits = F.interpolate(
                seg_logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            task_loss = criterion(seg_logits, labels)
            functional_penalty, per_head = functional_output_l2(
                student_outputs, teacher_outputs
            )
            loss = task_loss + float(distill_coefficient) * functional_penalty
            if training:
                (loss / accumulation).backward()
        if training and (step + 1) % accumulation == 0:
            optimizer.step()
        batch_size = labels.shape[0]
        sample_count += batch_size
        totals["loss"] += float(loss.detach().cpu().item()) * batch_size
        totals["task_loss"] += float(task_loss.detach().cpu().item()) * batch_size
        totals["functional_distillation_penalty"] += (
            float(functional_penalty.detach().cpu().item()) * batch_size
        )
        for name in MATCHED_HEAD_NAMES:
            per_head_totals[name] += float(per_head[name].detach().cpu().item()) * batch_size
        metrics.update(seg_logits, labels)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    summary = metrics.summary(class_names)
    summary.update({key: value / sample_count for key, value in totals.items()})
    summary["weighted_functional_distillation_penalty"] = (
        float(distill_coefficient) * totals["functional_distillation_penalty"] / sample_count
    )
    summary["functional_distillation_per_head"] = {
        name: per_head_totals[name] / sample_count for name in MATCHED_HEAD_NAMES
    }
    summary["samples"] = sample_count
    summary["batches"] = len(loader)
    summary["duration_seconds"] = round(time.monotonic() - started, 3)
    return summary


def _load_full_state_dict(checkpoint_path):
    from timm.models.helpers import load_state_dict as load_timm_state_dict

    return OrderedDict(load_timm_state_dict(str(checkpoint_path), use_ema=False))


def assert_pair_invariants(student, init_checkpoint_path):
    """Non-RGB state must stay byte-identical; RGB BN buffers must not move."""
    initial = _load_full_state_dict(init_checkpoint_path)
    current = student.state_dict()
    if set(initial) != set(current):
        raise FunctionalContractError("student state key set differs from initialization")
    # InterFuser 将同一 resnet 模块注册为 rgb_backbone 与 rgb_patch_embed.backbone，
    # state_dict 出现同张量双键（330 别名的机制本体）；以存储身份判定归属。
    rgb_storages = {
        current[key].data_ptr() for key in current if key.startswith("rgb_backbone.")
    }
    changed = [
        key
        for key in current
        if not torch.equal(current[key].detach().cpu(), initial[key])
    ]
    non_rgb_changed = [
        key for key in changed if current[key].data_ptr() not in rgb_storages
    ]
    if non_rgb_changed:
        raise FunctionalContractError(
            f"non-RGB tensors changed during training: {non_rgb_changed[:5]}"
        )
    rgb_bn_changed = [
        key
        for key in changed
        if current[key].data_ptr() in rgb_storages
        and ("running_mean" in key or "running_var" in key)
    ]
    if rgb_bn_changed:
        raise FunctionalContractError(
            f"rgb_backbone BN buffers moved despite eval discipline: {rgb_bn_changed[:5]}"
        )
    return {
        "full_model_tensors": len(current),
        "changed_tensors": len(changed),
        "changed_tensors_all_rgb": True,
        "rgb_alias_key_count": sum(
            1 for key in current if current[key].data_ptr() in rgb_storages
        ),
    }


def export_student_backbone(student):
    """Export the trained rgb_backbone in the v7-compatible timm format."""
    export = OrderedDict(
        (key, value.detach().cpu())
        for key, value in student.rgb_backbone.state_dict().items()
        if key not in {"fc.weight", "fc.bias"}
    )
    validate_backbone_export(export)
    return export
