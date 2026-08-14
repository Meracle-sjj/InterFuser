#!/usr/bin/env python3
"""
[INPUT]: 依赖冻结的官方 InterFuser B0/B0-V checkpoint、validation index、CarlaMVDetDataset 与离线任务指标。
[OUTPUT]: 对外提供 ModalityAblationError、perturb_inputs、OutputSensitivityAccumulator、execute_modality_ablation 与 CLI，生成 RGB/LiDAR 因果消融指标和预注册依赖判定。
[POS]: tools/evaluation 的模态依赖审计器；固定模型和非目标输入，仅破坏 RGB 或 LiDAR，用配对输出变化区分视觉质量不足与融合器忽略视觉。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import argparse
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
    ensure_gpus_have_free_memory,
)


CONFIG_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
RGB_KEYS = ("rgb", "rgb_center", "rgb_left", "rgb_right")
CONDITIONS = ("normal", "rgb_mean_fill", "rgb_blur", "rgb_shuffle", "lidar_zero")
VARIANTS = ("official_b0", "official_b0_v")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


class ModalityAblationError(RuntimeError):
    """Raised when the frozen ablation contract or output is invalid."""


class _StrictIndexedDataset:
    """Bypass CarlaMVDetDataset's corrupted-sample substitution."""

    def __init__(self, base):
        object.__setattr__(self, "_base", base)

    def __len__(self):
        return len(self._base)

    def __getattr__(self, name):
        return getattr(self._base, name)

    def __setattr__(self, name, value):
        setattr(self._base, name, value)

    def __getitem__(self, index):
        return self._base._get_item_impl(index)


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
        raise ModalityAblationError(f"unable to read {label} JSON {path}: {exc}") from exc


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
        raise ModalityAblationError(f"{label} must be a non-empty path")
    path = Path(value)
    path = path if path.is_absolute() else Path(repo_root) / path
    path = path.resolve()
    if require_file and not path.is_file():
        raise ModalityAblationError(f"{label} is not a file: {path}")
    return path


def _verify_hash(path, expected, label):
    if not isinstance(expected, str) or len(expected) != 64:
        raise ModalityAblationError(f"{label} SHA-256 must have 64 characters")
    actual = sha256_file(path)
    if actual != expected:
        raise ModalityAblationError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def _positive_int(value, label, allow_zero=False):
    valid = isinstance(value, int) and not isinstance(value, bool)
    valid = valid and (value >= 0 if allow_zero else value > 0)
    if not valid:
        raise ModalityAblationError(f"{label} must be a valid integer")
    return value


def _finite_ratio(value, label):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ModalityAblationError(f"{label} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ModalityAblationError(f"{label} must be finite and positive")
    return value


def _git_output(repo_root, *args):
    result = subprocess.run(
        ["git", *args], cwd=repo_root, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise ModalityAblationError(
            f"git {' '.join(args)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def load_contract(config_path, repo_root=REPO_ROOT):
    """Validate every input before exposing the validation split to the model."""
    repo_root = Path(repo_root).resolve()
    config_path = Path(config_path).resolve()
    raw = _read_json(config_path, "modality ablation config")
    if raw.get("schema_version") != CONFIG_SCHEMA_VERSION or raw.get("status") != "frozen":
        raise ModalityAblationError("config must be frozen schema v1")
    run_id = raw.get("run_id")
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise ModalityAblationError("run_id is invalid")
    if tuple(raw.get("condition_order") or ()) != CONDITIONS:
        raise ModalityAblationError(f"condition_order must equal {CONDITIONS}")

    split_spec = raw.get("downstream_split_manifest") or {}
    split_path = _resolve_path(repo_root, split_spec.get("path"), "split manifest")
    _verify_hash(split_path, split_spec.get("sha256"), "split manifest")
    split = _read_json(split_path, "split manifest")
    if split.get("valid") is not True:
        raise ModalityAblationError("downstream split manifest is not valid")

    dataset = raw.get("dataset") or {}
    dataset_root = _resolve_path(repo_root, dataset.get("root"), "dataset.root", False)
    if not dataset_root.is_dir():
        raise ModalityAblationError("dataset.root is not a directory")
    validation_index = _resolve_path(
        repo_root, dataset.get("validation_index"), "dataset.validation_index"
    )
    validation_hash = _verify_hash(
        validation_index,
        dataset.get("validation_index_sha256"),
        "dataset.validation_index",
    )
    split_validation = (split.get("artifacts", {}).get("validation", {}) or {})
    if validation_hash != split_validation.get("sha256"):
        raise ModalityAblationError("validation index differs from split manifest")
    dataset_implementation = dataset.get("implementation")
    if dataset_implementation is not None:
        if not isinstance(dataset_implementation, dict):
            raise ModalityAblationError("dataset.implementation must be an object")
        implementation_path = _resolve_path(
            repo_root,
            dataset_implementation.get("path"),
            "dataset.implementation",
        )
        _verify_hash(
            implementation_path,
            dataset_implementation.get("sha256"),
            "dataset.implementation",
        )
    expected_frames = (split.get("summary", {}).get("validation", {}) or {}).get(
        "logical_frames"
    )
    if dataset.get("logical_frames") != expected_frames:
        raise ModalityAblationError("validation frame count differs from split manifest")
    lidar_y_axis_multiplier = dataset.get("lidar_y_axis_multiplier", -1.0)
    if isinstance(lidar_y_axis_multiplier, bool) or not isinstance(
        lidar_y_axis_multiplier, (int, float)
    ):
        raise ModalityAblationError("dataset.lidar_y_axis_multiplier must be -1.0 or 1.0")
    if float(lidar_y_axis_multiplier) not in (-1.0, 1.0):
        raise ModalityAblationError("dataset.lidar_y_axis_multiplier must be -1.0 or 1.0")
    density_gate = dataset.get("lidar_density_gate")
    if density_gate is not None:
        if not isinstance(density_gate, dict):
            raise ModalityAblationError("dataset.lidar_density_gate must be an object")
        _positive_int(
            density_gate.get("minimum_mean_nonzero_cells"),
            "dataset.lidar_density_gate.minimum_mean_nonzero_cells",
        )
        maximum_zero = density_gate.get("maximum_all_zero_fraction")
        if (
            not isinstance(maximum_zero, (int, float))
            or isinstance(maximum_zero, bool)
            or not math.isfinite(float(maximum_zero))
            or not 0.0 <= float(maximum_zero) <= 1.0
        ):
            raise ModalityAblationError(
                "dataset.lidar_density_gate.maximum_all_zero_fraction must be in [0, 1]"
            )

    model = raw.get("model") or {}
    if model.get("name") != "interfuser_baseline" or model.get("with_lidar") is not True:
        raise ModalityAblationError("model must be lidar-enabled interfuser_baseline")
    if model.get("multi_view_input_size") != [3, 128, 128]:
        raise ModalityAblationError("multi_view_input_size must be [3, 128, 128]")
    for name in ("model_definition", "resnet_definition"):
        spec = model.get(name) or {}
        path = _resolve_path(repo_root, spec.get("path"), f"model.{name}")
        _verify_hash(path, spec.get("sha256"), f"model.{name}")

    variants = raw.get("variants") or []
    if [item.get("name") for item in variants] != list(VARIANTS):
        raise ModalityAblationError(f"variants must be ordered as {VARIANTS}")
    resolved_variants = []
    for item in variants:
        checkpoint = _resolve_path(
            repo_root, item.get("checkpoint"), f"{item.get('name')} checkpoint"
        )
        _verify_hash(
            checkpoint,
            item.get("checkpoint_sha256"),
            f"{item.get('name')} checkpoint",
        )
        resolved_variants.append({**item, "checkpoint_path": checkpoint})

    thresholds = raw.get("decision_thresholds") or {}
    for field in (
        "weak_max_rgb_to_lidar_output_ratio",
        "weak_max_relative_waypoint_ade_degradation",
        "material_min_rgb_to_lidar_output_ratio",
        "material_min_relative_waypoint_ade_degradation",
    ):
        _finite_ratio(thresholds.get(field), f"decision_thresholds.{field}")
    if thresholds["weak_max_rgb_to_lidar_output_ratio"] >= thresholds[
        "material_min_rgb_to_lidar_output_ratio"
    ]:
        raise ModalityAblationError("weak output threshold must be below material threshold")
    if thresholds["weak_max_relative_waypoint_ade_degradation"] >= thresholds[
        "material_min_relative_waypoint_ade_degradation"
    ]:
        raise ModalityAblationError("weak ADE threshold must be below material threshold")

    runtime = raw.get("runtime") or {}
    for field in ("seed", "gpu", "batch_size", "workers", "log_interval_batches"):
        _positive_int(runtime.get(field), f"runtime.{field}", field in {"seed", "gpu", "workers"})
    _positive_int(runtime.get("gpu_minimum_free_memory_mb"), "runtime.gpu_minimum_free_memory_mb")
    blur_kernel = _positive_int(runtime.get("rgb_blur_kernel"), "runtime.rgb_blur_kernel")
    if blur_kernel % 2 != 1:
        raise ModalityAblationError("rgb_blur_kernel must be odd")
    if runtime.get("shuffle_batch_shift") != 1:
        raise ModalityAblationError("shuffle_batch_shift must remain 1")
    if not isinstance(runtime.get("require_clean_git"), bool):
        raise ModalityAblationError("runtime.require_clean_git must be boolean")

    result_root = _resolve_path(repo_root, raw.get("result_root"), "result_root", False)
    try:
        result_root.relative_to(repo_root)
    except ValueError as exc:
        raise ModalityAblationError("result_root escapes repository") from exc
    return {
        **raw,
        "path": config_path,
        "sha256": sha256_file(config_path),
        "repo_root": repo_root,
        "split_manifest_path": split_path,
        "split_manifest_sha256": split_spec["sha256"],
        "dataset_root_path": dataset_root,
        "validation_index_path": validation_index,
        "resolved_variants": resolved_variants,
        "result_root_path": result_root,
    }


def perturb_inputs(inputs, condition, blur_kernel=11, shuffle_batch_shift=1):
    """Return a shallow input copy with exactly one registered modality intervention."""
    if condition not in CONDITIONS:
        raise ModalityAblationError(f"unknown condition: {condition}")
    missing_rgb = [key for key in RGB_KEYS if key not in inputs]
    if missing_rgb or "lidar" not in inputs:
        raise ModalityAblationError(
            f"model inputs missing RGB={missing_rgb} or lidar={'lidar' not in inputs}"
        )
    perturbed = dict(inputs)
    if condition == "normal":
        return perturbed
    if condition == "rgb_mean_fill":
        for key in RGB_KEYS:
            perturbed[key] = inputs[key].new_zeros(inputs[key].shape)
    elif condition == "rgb_blur":
        import torch.nn.functional as functional

        padding = blur_kernel // 2
        for key in RGB_KEYS:
            perturbed[key] = functional.avg_pool2d(
                inputs[key], kernel_size=blur_kernel, stride=1, padding=padding
            )
    elif condition == "rgb_shuffle":
        for key in RGB_KEYS:
            perturbed[key] = inputs[key].roll(shifts=shuffle_batch_shift, dims=0)
    elif condition == "lidar_zero":
        perturbed["lidar"] = inputs["lidar"].new_zeros(inputs["lidar"].shape)
    return perturbed


class OutputSensitivityAccumulator:
    """Accumulate paired output changes against the normal-input forward pass."""

    CLASS_HEADS = {"junction": 2, "red_light": 3, "stop_sign": 4}

    def __init__(self):
        self.samples = 0
        self.traffic_cells = 0
        self.waypoint_points = 0
        self.traffic_probability_abs_delta = 0.0
        self.traffic_full_abs_delta = 0.0
        self.waypoint_distance_delta = 0.0
        self.waypoint_fde_delta = 0.0
        self.feature_delta_squared = 0.0
        self.feature_reference_squared = 0.0
        self.feature_cosine_distance = 0.0
        self.class_probability_abs_delta = {name: 0.0 for name in self.CLASS_HEADS}
        self.class_flips = {name: 0 for name in self.CLASS_HEADS}

    def update(self, reference, candidate):
        import torch

        if len(reference) < 6 or len(candidate) < 6:
            raise ModalityAblationError("InterFuser output must expose five heads and features")
        batch_size = reference[0].shape[0]
        if candidate[0].shape[0] != batch_size:
            raise ModalityAblationError("paired outputs have different batch sizes")
        self.samples += batch_size

        traffic_delta = candidate[0] - reference[0]
        self.traffic_cells += int(traffic_delta.shape[0] * traffic_delta.shape[1])
        self.traffic_probability_abs_delta += float(traffic_delta[:, :, 0].abs().sum())
        self.traffic_full_abs_delta += float(traffic_delta.abs().sum())

        waypoint_delta = torch.linalg.vector_norm(candidate[1] - reference[1], dim=2)
        self.waypoint_points += int(waypoint_delta.numel())
        self.waypoint_distance_delta += float(waypoint_delta.sum())
        self.waypoint_fde_delta += float(waypoint_delta[:, -1].sum())

        reference_feature = reference[5].reshape(batch_size, -1).float()
        candidate_feature = candidate[5].reshape(batch_size, -1).float()
        feature_delta = candidate_feature - reference_feature
        self.feature_delta_squared += float(feature_delta.square().sum())
        self.feature_reference_squared += float(reference_feature.square().sum())
        cosine = torch.nn.functional.cosine_similarity(
            reference_feature, candidate_feature, dim=1, eps=1e-12
        )
        self.feature_cosine_distance += float((1.0 - cosine).sum())

        for name, index in self.CLASS_HEADS.items():
            reference_probability = torch.softmax(reference[index].float(), dim=1)[:, 1]
            candidate_probability = torch.softmax(candidate[index].float(), dim=1)[:, 1]
            self.class_probability_abs_delta[name] += float(
                (candidate_probability - reference_probability).abs().sum()
            )
            self.class_flips[name] += int(
                (candidate[index].argmax(dim=1) != reference[index].argmax(dim=1)).sum()
            )

    def finalize(self):
        if self.samples <= 0 or self.traffic_cells <= 0 or self.waypoint_points <= 0:
            raise ModalityAblationError("no paired outputs were accumulated")
        denominator = max(self.feature_reference_squared, 1e-24)
        return {
            "samples": self.samples,
            "traffic_probability_mae_from_normal": self.traffic_probability_abs_delta
            / self.traffic_cells,
            "traffic_full_output_mae_from_normal": self.traffic_full_abs_delta
            / (self.traffic_cells * 7),
            "waypoint_shift_ade": self.waypoint_distance_delta / self.waypoint_points,
            "waypoint_shift_fde_horizon_10": self.waypoint_fde_delta / self.samples,
            "feature_relative_l2_global": math.sqrt(
                self.feature_delta_squared / denominator
            ),
            "feature_cosine_distance_mean": self.feature_cosine_distance / self.samples,
            "class_positive_probability_mae_from_normal": {
                name: value / self.samples
                for name, value in self.class_probability_abs_delta.items()
            },
            "class_flip_rate": {
                name: value / self.samples for name, value in self.class_flips.items()
            },
        }


def _metric_at(metrics, *path):
    value = metrics
    for field in path:
        value = value[field]
    return float(value)


def _safe_ratio(numerator, denominator):
    return None if abs(float(denominator)) <= 1e-12 else float(numerator / denominator)


def _variant_verdict(condition_results, thresholds):
    normal_ade = _metric_at(condition_results["normal"]["task_metrics"], "waypoints", "ade")
    lidar = condition_results["lidar_zero"]["sensitivity"]
    disruptive = ("rgb_mean_fill", "rgb_shuffle")
    rgb_waypoint_shift = max(
        condition_results[name]["sensitivity"]["waypoint_shift_ade"]
        for name in disruptive
    )
    rgb_traffic_shift = max(
        condition_results[name]["sensitivity"]["traffic_probability_mae_from_normal"]
        for name in disruptive
    )
    rgb_ade_degradation = max(
        (
            _metric_at(condition_results[name]["task_metrics"], "waypoints", "ade")
            - normal_ade
        )
        / max(normal_ade, 1e-12)
        for name in disruptive
    )
    waypoint_ratio = _safe_ratio(rgb_waypoint_shift, lidar["waypoint_shift_ade"])
    traffic_ratio = _safe_ratio(
        rgb_traffic_shift, lidar["traffic_probability_mae_from_normal"]
    )
    output_ratios = [value for value in (waypoint_ratio, traffic_ratio) if value is not None]
    maximum_ratio = max(output_ratios) if output_ratios else None
    weak = (
        maximum_ratio is not None
        and maximum_ratio < thresholds["weak_max_rgb_to_lidar_output_ratio"]
        and rgb_ade_degradation < thresholds["weak_max_relative_waypoint_ade_degradation"]
    )
    material = (
        maximum_ratio is not None
        and maximum_ratio >= thresholds["material_min_rgb_to_lidar_output_ratio"]
    ) or rgb_ade_degradation >= thresholds[
        "material_min_relative_waypoint_ade_degradation"
    ]
    verdict = "weak_rgb_dependency" if weak else "material_rgb_dependency" if material else "mixed_rgb_dependency"
    return {
        "verdict": verdict,
        "rgb_disruptive_conditions": list(disruptive),
        "max_rgb_waypoint_shift_ade": rgb_waypoint_shift,
        "max_rgb_traffic_probability_shift": rgb_traffic_shift,
        "max_rgb_relative_waypoint_ade_degradation": rgb_ade_degradation,
        "rgb_to_lidar_waypoint_output_ratio": waypoint_ratio,
        "rgb_to_lidar_traffic_output_ratio": traffic_ratio,
        "maximum_rgb_to_lidar_output_ratio": maximum_ratio,
        "thresholds": dict(thresholds),
    }


def _evaluate_variant(contract, variant, max_batches=None):
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
        checkpoint_path=str(variant["checkpoint_path"]),
        freeze_num=-1,
    )
    model.cuda().eval()
    data_config = resolve_data_config({}, model=model, verbose=False)
    base_dataset = create_carla_dataset(
        "carla",
        root=str(contract["dataset_root_path"]),
        towns=contract["dataset"]["towns"],
        weathers=contract["dataset"]["weathers"],
        batch_size=runtime["batch_size"],
        with_lidar=True,
        multi_view=True,
        augment_prob=0.0,
        dataset_index=str(contract["validation_index_path"]),
        lidar_y_axis_multiplier=contract["dataset"].get(
            "lidar_y_axis_multiplier", -1.0
        ),
    )
    dataset = _StrictIndexedDataset(base_dataset)
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
        raise ModalityAblationError(
            f"validation dataset length={len(dataset)} differs from contract"
        )

    metric_args = (
        contract["metrics"]["traffic_positive_target_threshold"],
        contract["metrics"]["traffic_prediction_threshold"],
        contract["metrics"]["invalid_waypoint_threshold"],
    )
    task_accumulators = {
        condition: InterfuserMetricAccumulator(*metric_args) for condition in CONDITIONS
    }
    sensitivity_accumulators = {
        condition: OutputSensitivityAccumulator()
        for condition in CONDITIONS
        if condition != "normal"
    }
    lidar_samples = 0
    lidar_all_zero = 0
    lidar_nonzero_cells = 0
    lidar_tensor_cells = 0
    started = time.monotonic()
    with torch.inference_mode():
        for batch_index, (inputs, targets) in enumerate(loader):
            inputs = {name: value.cuda(non_blocking=False) for name, value in inputs.items()}
            lidar = inputs["lidar"]
            per_sample_nonzero = (lidar != 0).reshape(lidar.shape[0], -1).sum(dim=1)
            lidar_samples += int(lidar.shape[0])
            lidar_all_zero += int((per_sample_nonzero == 0).sum())
            lidar_nonzero_cells += int(per_sample_nonzero.sum())
            lidar_tensor_cells += int(lidar.numel())
            normal_outputs = model(inputs)
            task_accumulators["normal"].update(normal_outputs, targets)
            for condition in CONDITIONS[1:]:
                perturbed = perturb_inputs(
                    inputs,
                    condition,
                    blur_kernel=runtime["rgb_blur_kernel"],
                    shuffle_batch_shift=runtime["shuffle_batch_shift"],
                )
                candidate_outputs = model(perturbed)
                task_accumulators[condition].update(candidate_outputs, targets)
                sensitivity_accumulators[condition].update(
                    normal_outputs, candidate_outputs
                )
            if batch_index % runtime["log_interval_batches"] == 0:
                print(
                    f"variant={variant['name']} batch={batch_index}/{len(loader)} "
                    f"samples={task_accumulators['normal'].samples}",
                    flush=True,
                )
            if max_batches is not None and batch_index + 1 >= max_batches:
                break

    lidar_density = {
        "samples": lidar_samples,
        "all_zero_samples": lidar_all_zero,
        "all_zero_fraction": lidar_all_zero / max(lidar_samples, 1),
        "mean_nonzero_cells_per_sample": lidar_nonzero_cells
        / max(lidar_samples, 1),
        "nonzero_fraction": lidar_nonzero_cells / max(lidar_tensor_cells, 1),
        "tensor_cells_per_sample": lidar_tensor_cells // max(lidar_samples, 1),
        "y_axis_multiplier": float(
            contract["dataset"].get("lidar_y_axis_multiplier", -1.0)
        ),
    }
    density_gate = contract["dataset"].get("lidar_density_gate")
    if density_gate is not None:
        if lidar_density["mean_nonzero_cells_per_sample"] < density_gate[
            "minimum_mean_nonzero_cells"
        ]:
            raise ModalityAblationError(
                "LiDAR density gate failed: mean nonzero cells "
                f"{lidar_density['mean_nonzero_cells_per_sample']:.3f}"
            )
        if lidar_density["all_zero_fraction"] > density_gate[
            "maximum_all_zero_fraction"
        ]:
            raise ModalityAblationError(
                "LiDAR density gate failed: all-zero fraction "
                f"{lidar_density['all_zero_fraction']:.6f}"
            )

    condition_results = {}
    for condition in CONDITIONS:
        condition_results[condition] = {
            "task_metrics": task_accumulators[condition].finalize(),
            "sensitivity": None
            if condition == "normal"
            else sensitivity_accumulators[condition].finalize(),
        }
    return {
        "variant": variant["name"],
        "checkpoint": str(variant["checkpoint_path"]),
        "checkpoint_sha256": variant["checkpoint_sha256"],
        "samples": condition_results["normal"]["task_metrics"]["samples"],
        "duration_seconds": round(time.monotonic() - started, 3),
        "lidar_input_density": lidar_density,
        "conditions": condition_results,
        "dependency": _variant_verdict(
            condition_results, contract["decision_thresholds"]
        ),
    }


def execute_modality_ablation(config_path, run_id, repo_root=REPO_ROOT):
    """Run both checkpoints on identical validation batches and write one manifest."""
    contract = load_contract(config_path, repo_root)
    if run_id != contract["run_id"]:
        raise ModalityAblationError(f"run_id must equal frozen value {contract['run_id']}")
    git_status = _git_output(contract["repo_root"], "status", "--porcelain")
    if contract["runtime"]["require_clean_git"] and git_status:
        raise ModalityAblationError("Git worktree must be clean")
    ensure_gpus_have_free_memory(
        [contract["runtime"]["gpu"]],
        contract["runtime"]["gpu_minimum_free_memory_mb"],
    )
    run_dir = contract["result_root_path"] / run_id
    if run_dir.exists():
        raise ModalityAblationError(f"refusing to overwrite run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "ablation_manifest.json"
    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "running",
        "pipeline_valid": False,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "git_head": _git_output(contract["repo_root"], "rev-parse", "HEAD"),
        "git_status": git_status,
        "config": str(contract["path"]),
        "config_sha256": contract["sha256"],
        "split_manifest": str(contract["split_manifest_path"]),
        "split_manifest_sha256": contract["split_manifest_sha256"],
        "validation_index": str(contract["validation_index_path"]),
        "validation_index_sha256": contract["dataset"]["validation_index_sha256"],
        "condition_order": list(CONDITIONS),
        "variants": [],
        "errors": [],
    }
    _write_json(manifest_path, manifest)
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(contract["runtime"]["gpu"])
        for variant in contract["resolved_variants"]:
            result = _evaluate_variant(contract, variant)
            manifest["variants"].append(result)
            manifest["updated_at"] = _utc_now()
            _write_json(manifest_path, manifest)
        manifest.update(
            {
                "status": "completed",
                "pipeline_valid": True,
                "updated_at": _utc_now(),
                "completed_at": _utc_now(),
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
        description="Audit causal RGB/LiDAR dependence of official InterFuser checkpoints"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-batches", type=int)
    args = parser.parse_args(argv)
    try:
        if args.preflight_only:
            contract = load_contract(args.config)
            print(
                json.dumps(
                    {"config_sha256": contract["sha256"], "run_id": contract["run_id"]},
                    indent=2,
                    sort_keys=True,
                )
            )
        elif args.smoke_batches is not None:
            _positive_int(args.smoke_batches, "smoke_batches")
            contract = load_contract(args.config)
            ensure_gpus_have_free_memory(
                [contract["runtime"]["gpu"]],
                contract["runtime"]["gpu_minimum_free_memory_mb"],
            )
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(contract["runtime"]["gpu"])
            result = _evaluate_variant(
                contract, contract["resolved_variants"][0], args.smoke_batches
            )
            print(
                json.dumps(
                    {
                        "smoke_only": True,
                        "variant": result["variant"],
                        "samples": result["samples"],
                        "dependency": result["dependency"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            if not args.run_id:
                raise ModalityAblationError("execution requires --run-id")
            manifest = execute_modality_ablation(args.config, args.run_id)
            print(
                json.dumps(
                    {
                        item["variant"]: item["dependency"]
                        for item in manifest["variants"]
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
    except (ModalityAblationError, MetricError, RunnerError) as exc:
        print(f"modality ablation error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"modality ablation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
