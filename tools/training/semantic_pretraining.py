#!/usr/bin/env python3
"""
[INPUT]: 依赖 M1 类别配置与 split manifest、原始三相机 RGB/语义帧、仓库内 timm ResNet50d 和冻结的 ImageNet 权重。
[OUTPUT]: 对外提供 TrainingContractError、load_training_contract、resolve_train_sample_limit、SemanticFrameDataset、SemanticPretrainingModel、DeterministicCrossEntropyLoss、ConfusionMetrics 与骨干导出/迁移校验 API。
[POS]: tools/training 的 M2 核心领域层，把冻结数据契约转换为可训练张量、同构视觉骨干和可比较离线指标；不负责 GPU 独占或运行目录生命周期。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import hashlib
import json
import random
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERFUSER_ROOT = REPO_ROOT / "interfuser"
for import_root in (REPO_ROOT, INTERFUSER_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from timm.models.resnet import resnet50d  # noqa: E402
from tools.data.audit_semantic_pretraining_data import (  # noqa: E402
    AuditError,
    load_class_config,
)


TRAINING_CONFIG_SCHEMA_VERSION = 1
BACKBONE_EXPORT_SCHEMA_VERSION = 1
DEFAULT_IGNORE_INDEX = 255


class TrainingContractError(ValueError):
    """Raised when training inputs cannot produce comparable M2 evidence."""


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path, label):
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingContractError(f"unable to read {label} JSON {path}: {exc}") from exc


def _resolve_repo_path(value, label):
    if not isinstance(value, str) or not value:
        raise TrainingContractError(f"{label} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _positive_int(value, label):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TrainingContractError(f"{label} must be a positive integer")
    return value


def _positive_number(value, label, allow_zero=False):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TrainingContractError(f"{label} must be numeric")
    value = float(value)
    invalid = value < 0 if allow_zero else value <= 0
    if invalid:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise TrainingContractError(f"{label} must be {qualifier}")
    return value


def _validate_sha(path, expected, label):
    if not isinstance(expected, str) or len(expected) != 64:
        raise TrainingContractError(f"{label} SHA-256 must contain 64 hex characters")
    actual = sha256_file(path)
    if actual != expected:
        raise TrainingContractError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def load_training_contract(config_path):
    """Validate one immutable config and its M1/ImageNet provenance."""
    config_path = Path(config_path).resolve()
    raw = _read_json(config_path, "training config")
    if not isinstance(raw, dict):
        raise TrainingContractError("training config must be a JSON object")
    if raw.get("schema_version") != TRAINING_CONFIG_SCHEMA_VERSION:
        raise TrainingContractError(
            f"unsupported training config schema_version: {raw.get('schema_version')}"
        )
    status = raw.get("status")
    if status not in {"smoke", "pilot"}:
        raise TrainingContractError("training config status must be smoke or pilot")

    class_path = _resolve_repo_path(raw.get("class_config"), "class_config")
    split_path = _resolve_repo_path(raw.get("split_manifest"), "split_manifest")
    _validate_sha(class_path, raw.get("class_config_sha256"), "class config")
    _validate_sha(split_path, raw.get("split_manifest_sha256"), "split manifest")
    try:
        class_config = load_class_config(class_path)
    except AuditError as exc:
        raise TrainingContractError(str(exc)) from exc
    split_manifest = _read_json(split_path, "split manifest")
    if not split_manifest.get("valid"):
        raise TrainingContractError("split manifest must be valid")
    if split_manifest.get("source", {}).get("class_config_sha256") != class_config["sha256"]:
        raise TrainingContractError("split manifest and class config SHA-256 differ")

    backbone = raw.get("backbone")
    if not isinstance(backbone, dict) or backbone.get("name") != "resnet50d":
        raise TrainingContractError("backbone.name must be resnet50d")
    if backbone.get("feature_indices") != [1, 2, 3, 4]:
        raise TrainingContractError("backbone.feature_indices must be [1, 2, 3, 4]")
    pretrained_path = _resolve_repo_path(
        backbone.get("pretrained_checkpoint"), "pretrained_checkpoint"
    )
    _validate_sha(
        pretrained_path,
        backbone.get("pretrained_checkpoint_sha256"),
        "pretrained checkpoint",
    )

    model = raw.get("model")
    if not isinstance(model, dict):
        raise TrainingContractError("model must be an object")
    num_classes = _positive_int(model.get("num_classes"), "model.num_classes")
    if num_classes != len(class_config["classes"]):
        raise TrainingContractError(
            f"model.num_classes={num_classes} differs from class config size "
            f"{len(class_config['classes'])}"
        )
    decoder_channels = _positive_int(
        model.get("decoder_channels"), "model.decoder_channels"
    )
    if decoder_channels % 8:
        raise TrainingContractError("model.decoder_channels must be divisible by 8")
    dropout = _positive_number(model.get("dropout"), "model.dropout", allow_zero=True)
    if dropout >= 1:
        raise TrainingContractError("model.dropout must be less than 1")

    data = raw.get("data")
    if not isinstance(data, dict):
        raise TrainingContractError("data must be an object")
    cameras = data.get("cameras")
    if not isinstance(cameras, list) or not cameras or len(cameras) != len(set(cameras)):
        raise TrainingContractError("data.cameras must be unique and non-empty")
    if cameras != split_manifest.get("cameras"):
        raise TrainingContractError("training cameras differ from split manifest")
    for field in (
        "input_width",
        "input_height",
        "max_train_samples",
        "max_validation_samples",
    ):
        _positive_int(data.get(field), f"data.{field}")
    sample_seed = data.get("sample_seed")
    if not isinstance(sample_seed, int) or isinstance(sample_seed, bool):
        raise TrainingContractError("data.sample_seed must be an integer")
    for field in ("image_mean", "image_std"):
        values = data.get(field)
        if not isinstance(values, list) or len(values) != 3:
            raise TrainingContractError(f"data.{field} must contain three values")
        if any(not isinstance(value, (int, float)) for value in values):
            raise TrainingContractError(f"data.{field} must be numeric")
    if any(float(value) <= 0 for value in data["image_std"]):
        raise TrainingContractError("data.image_std must be positive")
    for field in ("expected_available_train_samples", "expected_available_validation_samples"):
        if field in data:
            _positive_int(data.get(field), f"data.{field}")
    learning_curve_samples = data.get("learning_curve_train_samples")
    if status == "pilot":
        if (
            not isinstance(learning_curve_samples, list)
            or not learning_curve_samples
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in learning_curve_samples
            )
        ):
            raise TrainingContractError(
                "pilot data.learning_curve_train_samples must contain positive integers"
            )
        if learning_curve_samples != sorted(set(learning_curve_samples)):
            raise TrainingContractError(
                "data.learning_curve_train_samples must be unique and ascending"
            )
        if learning_curve_samples[-1] != data["max_train_samples"]:
            raise TrainingContractError(
                "last learning-curve sample count must equal data.max_train_samples"
            )
        if data.get("validation_mode") != "full_split":
            raise TrainingContractError("pilot data.validation_mode must be full_split")
        if data.get("expected_available_validation_samples") != data["max_validation_samples"]:
            raise TrainingContractError(
                "pilot max_validation_samples must equal expected full split size"
            )
    elif learning_curve_samples is not None:
        raise TrainingContractError(
            "smoke config must not define learning_curve_train_samples"
        )

    training = raw.get("training")
    if not isinstance(training, dict):
        raise TrainingContractError("training must be an object")
    for field in ("epochs", "batch_size", "gpu_busy_memory_threshold_mb"):
        _positive_int(training.get(field), f"training.{field}")
    for field in ("num_workers", "physical_gpu_index"):
        value = training.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise TrainingContractError(f"training.{field} must be nonnegative")
    seed = training.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TrainingContractError("training.seed must be an integer")
    if training.get("optimizer") != "adamw":
        raise TrainingContractError("training.optimizer must be adamw")
    _positive_number(training.get("learning_rate"), "training.learning_rate")
    _positive_number(
        training.get("weight_decay"), "training.weight_decay", allow_zero=True
    )
    if training.get("ignore_index") != DEFAULT_IGNORE_INDEX:
        raise TrainingContractError(f"training.ignore_index must be {DEFAULT_IGNORE_INDEX}")
    for field in ("deterministic", "require_clean_git"):
        if not isinstance(training.get(field), bool):
            raise TrainingContractError(f"training.{field} must be boolean")

    normalized = dict(raw)
    normalized["path"] = config_path
    normalized["sha256"] = sha256_file(config_path)
    normalized["class_path"] = class_path
    normalized["class_config_loaded"] = class_config
    normalized["split_path"] = split_path
    normalized["split_manifest_loaded"] = split_manifest
    normalized["pretrained_path"] = pretrained_path
    return normalized


def resolve_train_sample_limit(contract, requested=None):
    """Resolve one config-approved train budget without allowing ad hoc curve points."""
    maximum = contract["data"]["max_train_samples"]
    if requested is None:
        return maximum
    if not isinstance(requested, int) or isinstance(requested, bool) or requested <= 0:
        raise TrainingContractError("requested train sample count must be positive")
    if contract["status"] == "pilot":
        allowed = contract["data"]["learning_curve_train_samples"]
        if requested not in allowed:
            raise TrainingContractError(
                f"train sample count {requested} is not in configured learning curve {allowed}"
            )
    elif requested != maximum:
        raise TrainingContractError(
            f"smoke train sample count must remain {maximum}, got {requested}"
        )
    return requested


def _resolve_rgb_path(sequence, camera, frame_id):
    directory = sequence / f"rgb_{camera}"
    matches = sorted(directory.glob(f"{frame_id}.*"))
    matches = [path for path in matches if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if len(matches) != 1:
        raise TrainingContractError(
            f"expected one RGB file for {sequence}/{camera}/{frame_id}, found {len(matches)}"
        )
    return matches[0]


class SemanticFrameDataset(Dataset):
    """Expose one RGB/semantic camera frame from a frozen split as one sample."""

    def __init__(self, contract, split, sample_limit=None):
        if split not in {"train", "validation", "test"}:
            raise TrainingContractError(f"unknown split: {split}")
        self.split = split
        self.root = Path(contract["split_manifest_loaded"]["dataset_root"])
        self.data_config = contract["data"]
        self.width = self.data_config["input_width"]
        self.height = self.data_config["input_height"]
        self.mean = torch.tensor(self.data_config["image_mean"], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(self.data_config["image_std"], dtype=torch.float32).view(3, 1, 1)
        self.ignore_index = contract["training"]["ignore_index"]
        self.label_lut = np.full(256, self.ignore_index, dtype=np.uint8)
        self.known_lut = np.zeros(256, dtype=bool)
        class_config = contract["class_config_loaded"]
        for item in class_config["classes"]:
            for source_tag in item["source_tags"]:
                self.label_lut[source_tag] = item["train_id"]
                self.known_lut[source_tag] = True
        for source_tag in class_config["ignore_tags"]:
            self.known_lut[source_tag] = True

        records = []
        cameras = self.data_config["cameras"]
        for sequence_item in contract["split_manifest_loaded"]["sequences"]:
            if sequence_item["split"] != split:
                continue
            sequence = self.root / sequence_item["path"]
            for camera in cameras:
                semantic_dir = sequence / f"seg_{camera}"
                for semantic_path in sorted(semantic_dir.glob("*.png")):
                    frame_id = semantic_path.stem
                    rgb_path = _resolve_rgb_path(sequence, camera, frame_id)
                    key = f"{sequence_item['path']}:{camera}:{frame_id}"
                    rank = hashlib.sha256(
                        f"{self.data_config['sample_seed']}:{split}:{key}".encode("utf-8")
                    ).hexdigest()
                    records.append(
                        {
                            "key": key,
                            "rank": rank,
                            "rgb_path": rgb_path,
                            "semantic_path": semantic_path,
                        }
                    )
        records.sort(key=lambda item: (item["rank"], item["key"]))
        self.available_samples = len(records)
        expected_field = f"expected_available_{split}_samples"
        expected_available = self.data_config.get(expected_field)
        if expected_available is not None and self.available_samples != expected_available:
            raise TrainingContractError(
                f"split {split} inventory={self.available_samples} differs from "
                f"configured {expected_available}"
            )
        configured_limit = self.data_config[
            "max_train_samples" if split == "train" else "max_validation_samples"
        ]
        limit = configured_limit if sample_limit is None else sample_limit
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise TrainingContractError(f"split {split} sample limit must be positive")
        if limit > configured_limit or limit > self.available_samples:
            raise TrainingContractError(
                f"split {split} sample limit {limit} exceeds configured/data availability"
            )
        self.sample_limit = limit
        self.records = records[:limit]
        if not self.records:
            raise TrainingContractError(f"split {split} produced no training samples")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        try:
            with Image.open(record["rgb_path"]) as image:
                rgb = image.convert("RGB").resize(
                    (self.width, self.height), Image.Resampling.BILINEAR
                )
                rgb_array = np.asarray(rgb, dtype=np.float32) / 255.0
            with Image.open(record["semantic_path"]) as image:
                raw = np.asarray(
                    image.resize(
                        (self.width, self.height), Image.Resampling.NEAREST
                    )
                )
        except OSError as exc:
            raise TrainingContractError(f"unable to read {record['key']}: {exc}") from exc
        if raw.ndim != 2 or raw.dtype.kind not in "ui":
            raise TrainingContractError(f"semantic mask must be 2D integer: {record['key']}")
        unknown = sorted(int(value) for value in np.unique(raw) if not self.known_lut[int(value)])
        if unknown:
            raise TrainingContractError(f"unmapped source tags {unknown}: {record['key']}")
        labels = self.label_lut[raw]
        image_tensor = torch.from_numpy(rgb_array.transpose(2, 0, 1).copy())
        image_tensor = (image_tensor - self.mean) / self.std
        label_tensor = torch.from_numpy(labels.astype(np.int64, copy=False))
        return {"image": image_tensor, "label": label_tensor, "key": record["key"]}


def _load_pretrained_backbone(backbone, checkpoint_path):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TrainingContractError("pretrained checkpoint must contain a state dict")
    filtered = OrderedDict(
        (key, value) for key, value in state.items() if key not in {"fc.weight", "fc.bias"}
    )
    incompatible = backbone.load_state_dict(filtered, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise TrainingContractError(
            "pretrained backbone mismatch: "
            f"missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
        )


class SemanticPretrainingModel(nn.Module):
    """Pair the exact InterFuser ResNet50d backbone with a disposable FPN head."""

    def __init__(self, contract):
        super().__init__()
        feature_indices = contract["backbone"]["feature_indices"]
        self.backbone = resnet50d(
            pretrained=False,
            in_chans=3,
            features_only=True,
            out_indices=feature_indices,
        )
        _load_pretrained_backbone(self.backbone, contract["pretrained_path"])
        channels = self.backbone.feature_info.channels()
        decoder_channels = contract["model"]["decoder_channels"]
        self.lateral = nn.ModuleList(
            nn.Conv2d(channel, decoder_channels, kernel_size=1) for channel in channels
        )
        self.smooth = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
                nn.GroupNorm(8, decoder_channels),
                nn.ReLU(inplace=True),
            )
            for _ in channels
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(
                decoder_channels * len(channels),
                decoder_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(8, decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(contract["model"]["dropout"]),
            nn.Conv2d(decoder_channels, contract["model"]["num_classes"], kernel_size=1),
        )

    def forward(self, images):
        input_size = images.shape[-2:]
        features = self.backbone(images)
        pyramids = [None] * len(features)
        top = self.lateral[-1](features[-1])
        pyramids[-1] = self.smooth[-1](top)
        for index in range(len(features) - 2, -1, -1):
            top = self.lateral[index](features[index]) + F.interpolate(
                top, size=features[index].shape[-2:], mode="bilinear", align_corners=False
            )
            pyramids[index] = self.smooth[index](top)
        target_size = pyramids[0].shape[-2:]
        fused = torch.cat(
            [
                level
                if level.shape[-2:] == target_size
                else F.interpolate(
                    level, size=target_size, mode="bilinear", align_corners=False
                )
                for level in pyramids
            ],
            dim=1,
        )
        logits = self.fuse(fused)
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)


class DeterministicCrossEntropyLoss(nn.Module):
    """Compute ignore-aware cross entropy without CUDA's nondeterministic NLL kernel."""

    def __init__(self, ignore_index=DEFAULT_IGNORE_INDEX):
        super().__init__()
        self.ignore_index = int(ignore_index)

    def forward(self, logits, labels):
        valid = labels != self.ignore_index
        if not torch.any(valid):
            raise TrainingContractError("semantic batch contains no supervised pixels")
        safe_labels = labels.masked_fill(~valid, 0)
        log_probabilities = F.log_softmax(logits, dim=1)
        selected = log_probabilities.gather(1, safe_labels.unsqueeze(1)).squeeze(1)
        return -selected[valid].mean()


class ConfusionMetrics:
    """Accumulate deterministic pixel metrics without storing predictions."""

    def __init__(self, num_classes, ignore_index=DEFAULT_IGNORE_INDEX):
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.confusion = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.int64
        )

    def update(self, logits, labels):
        predictions = logits.argmax(dim=1).detach().cpu().to(torch.int64)
        labels = labels.detach().cpu().to(torch.int64)
        valid = labels != self.ignore_index
        labels = labels[valid]
        predictions = predictions[valid]
        if labels.numel():
            encoded = labels * self.num_classes + predictions
            self.confusion += torch.bincount(
                encoded, minlength=self.num_classes * self.num_classes
            ).reshape(self.num_classes, self.num_classes)

    def summary(self, class_names):
        confusion = self.confusion.to(torch.float64)
        true_positive = confusion.diag()
        support = confusion.sum(dim=1)
        predicted = confusion.sum(dim=0)
        union = support + predicted - true_positive
        f1_denominator = support + predicted
        iou = torch.where(union > 0, true_positive / union, torch.nan)
        f1 = torch.where(
            f1_denominator > 0,
            2 * true_positive / f1_denominator,
            torch.nan,
        )
        total = confusion.sum()
        pixel_accuracy = true_positive.sum() / total if total else torch.tensor(float("nan"))
        per_class = []
        for index, name in enumerate(class_names):
            per_class.append(
                {
                    "train_id": index,
                    "name": name,
                    "support_pixels": int(support[index].item()),
                    "predicted_pixels": int(predicted[index].item()),
                    "iou": None if torch.isnan(iou[index]) else float(iou[index].item()),
                    "f1": None if torch.isnan(f1[index]) else float(f1[index].item()),
                }
            )
        valid_iou = iou[~torch.isnan(iou)]
        valid_f1 = f1[~torch.isnan(f1)]
        return {
            "pixel_accuracy": None
            if torch.isnan(pixel_accuracy)
            else float(pixel_accuracy.item()),
            "mean_iou": float(valid_iou.mean().item()) if len(valid_iou) else None,
            "macro_f1": float(valid_f1.mean().item()) if len(valid_f1) else None,
            "confusion_matrix": self.confusion.tolist(),
            "per_class": per_class,
        }


def make_backbone_export(model, contract):
    """Return a weights-only artifact accepted by InterFuser's MMAD loader path."""
    state_dict = OrderedDict(
        (f"backbone.{key}", value.detach().cpu())
        for key, value in model.backbone.state_dict().items()
    )
    return {
        "format_version": BACKBONE_EXPORT_SCHEMA_VERSION,
        "architecture": "resnet50d",
        "source_training_config_sha256": contract["sha256"],
        "state_dict": state_dict,
    }


def validate_backbone_export(export):
    """Prove strict compatibility with InterFuser's out_indices=[4] RGB backbone."""
    if export.get("architecture") != "resnet50d":
        raise TrainingContractError("backbone export architecture must be resnet50d")
    state = export.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise TrainingContractError("backbone export has no state_dict")
    prefix = "backbone."
    if any(not key.startswith(prefix) for key in state):
        raise TrainingContractError("backbone export keys must use backbone. prefix")
    stripped = OrderedDict((key[len(prefix) :], value) for key, value in state.items())
    target = resnet50d(
        pretrained=False,
        in_chans=3,
        features_only=True,
        out_indices=[4],
    )
    target.load_state_dict(stripped, strict=True)
    return {"strict_load": True, "parameter_tensors": len(stripped)}


def set_reproducible_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.use_deterministic_algorithms(bool(deterministic))
