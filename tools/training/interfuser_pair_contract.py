"""
[INPUT]: 依赖版本化 B0/V JSON 配置、下游 split/初始化 manifest、checkpoint 与数据索引哈希。
[OUTPUT]: 对外提供 PairRunError、VARIANTS、sha256_file 和 load_pair_run_contract，返回路径已解析且完整校验的训练契约。
[POS]: tools.training 的配对训练契约边界；只验证数据、初始化、预算和资源策略，不启动训练或写入实验产物。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import hashlib
import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
VARIANTS = ("b0", "v")
NAVIGATION_FRAMES = (
    "legacy_upstream_yaw",
    "carla0916_standard_ego",
    "carla0916_compass",
)
MISSING_NAVIGATION_POLICIES = ("fallback", "drop")
GPU_RESOURCE_POLICIES = ("exclusive", "shared_capacity")


class PairRunError(RuntimeError):
    """Raised when paired downstream training loses comparability or provenance."""


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


def _load_input_manifests(raw):
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
    return resolved, split_manifest, init_manifest


def _validate_dataset(raw, resolved, split_manifest):
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
    lidar_multiplier = dataset.get("lidar_y_axis_multiplier", -1.0)
    if isinstance(lidar_multiplier, bool) or lidar_multiplier not in (-1.0, 1.0):
        raise PairRunError("dataset.lidar_y_axis_multiplier must be -1.0 or 1.0")
    if dataset.get("navigation_frame", "carla0916_standard_ego") not in NAVIGATION_FRAMES:
        raise PairRunError(f"dataset.navigation_frame must be one of {NAVIGATION_FRAMES}")
    if dataset.get("missing_navigation_policy", "fallback") not in MISSING_NAVIGATION_POLICIES:
        raise PairRunError(
            "dataset.missing_navigation_policy must be one of "
            f"{MISSING_NAVIGATION_POLICIES}"
        )
    expected_samples = dataset.get("expected_effective_samples")
    if raw["status"] == "pilot":
        if not isinstance(expected_samples, dict):
            raise PairRunError("pilot dataset requires expected_effective_samples")
        for split in ("train", "validation"):
            _positive_int(
                expected_samples.get(split),
                f"dataset.expected_effective_samples.{split}",
            )


def _validate_variants(raw, resolved, init_manifest):
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
        initialization_name = value.get("initialization_manifest_variant", name)
        if not isinstance(initialization_name, str) or not initialization_name:
            raise PairRunError(
                f"variants.{name}.initialization_manifest_variant must be non-empty"
            )
        init_hash = init_manifest.get("variants", {}).get(initialization_name, {}).get(
            "checkpoint_sha256"
        )
        if init_hash != value.get("initial_checkpoint_sha256"):
            raise PairRunError(f"{name} checkpoint differs from initialization manifest")


def _validate_sampling_and_training(raw):
    sampling = raw.get("smoke_sampling")
    if raw["status"] == "smoke":
        if not isinstance(sampling, dict):
            raise PairRunError("smoke config requires smoke_sampling")
        _positive_int(sampling.get("seed"), "smoke_sampling.seed", allow_zero=True)
        for field in ("train_sequences", "validation_sequences"):
            _positive_int(sampling.get(field), f"smoke_sampling.{field}")
    elif sampling is not None:
        raise PairRunError("pilot/formal config must not define smoke_sampling")
    training = raw.get("training")
    if not isinstance(training, dict) or training.get("model") != "interfuser_baseline":
        raise PairRunError("training.model must be interfuser_baseline")
    zero_allowed = {"seed", "workers_per_process", "warmup_epochs", "cooldown_epochs"}
    for field in (
        "seed", "epochs", "batch_size_per_gpu", "workers_per_process",
        "warmup_epochs", "cooldown_epochs", "master_port", "timeout_seconds_per_variant",
    ):
        _positive_int(training.get(field), f"training.{field}", field in zero_allowed)
    _positive_int(training.get("log_interval", 1), "training.log_interval")
    if training.get("optimizer") != "adamw" or training.get("scheduler") != "cosine":
        raise PairRunError("training optimizer/scheduler must be adamw/cosine")
    for field in (
        "learning_rate", "backbone_learning_rate", "weight_decay", "color_jitter", "clip_grad",
    ):
        _positive_number(training.get(field), f"training.{field}", field == "color_jitter")
    scale = training.get("scale")
    if not isinstance(scale, list) or len(scale) != 2 or any(
        not isinstance(value, (int, float)) or value <= 0 for value in scale
    ):
        raise PairRunError("training.scale must contain two positive numbers")
    gpus = training.get("gpus")
    if not isinstance(gpus, list) or not gpus or len(gpus) != len(set(gpus)):
        raise PairRunError("training.gpus must contain at least one unique index")
    if any(not isinstance(value, int) or value < 0 for value in gpus):
        raise PairRunError("training.gpus must contain nonnegative integers")
    if not isinstance(training.get("require_clean_git"), bool):
        raise PairRunError("training.require_clean_git must be boolean")
    gpu_policy = training.get("gpu_resource_policy", "exclusive")
    if gpu_policy not in GPU_RESOURCE_POLICIES:
        raise PairRunError(
            f"training.gpu_resource_policy must be one of {GPU_RESOURCE_POLICIES}"
        )
    if gpu_policy == "shared_capacity":
        _positive_int(
            training.get("gpu_minimum_free_memory_mb"),
            "training.gpu_minimum_free_memory_mb",
        )
    else:
        _positive_int(
            training.get("gpu_busy_memory_threshold_mb"),
            "training.gpu_busy_memory_threshold_mb",
        )


def load_pair_run_contract(path):
    """Validate one paired training contract and every referenced artifact."""
    path = Path(path).resolve()
    raw = _read_json(path, "pair run config")
    if raw.get("schema_version") != SCHEMA_VERSION or raw.get("status") not in {
        "smoke", "pilot", "formal",
    }:
        raise PairRunError("pair run config must be smoke/pilot/formal schema v1")
    resolved, split_manifest, init_manifest = _load_input_manifests(raw)
    _validate_dataset(raw, resolved, split_manifest)
    _validate_variants(raw, resolved, init_manifest)
    _validate_sampling_and_training(raw)
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
