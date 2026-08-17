#!/usr/bin/env python3
"""
[INPUT]: 依赖完成且可比的 Stage 2 B0/V manifest、scene-split v2 validation index、逐帧行人真值、best checkpoint 与纯离线指标归约器。
[OUTPUT]: 对外提供 SceneValidationError、load_scene_validation_contract、build_metric_comparison、execute_scene_validation 与 CLI，生成整体/行人条件/非行人/行人 route-group 的配对 validation 指标；score_dump_dir 存在时附带 per-sample 交通分数/真值落盘。
[POS]: tools/evaluation 的 M2 H1 validation 门禁；只读已解封 validation，不接触冻结 test，并为是否继续视觉路线提供任务级证据。
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
    InterfuserTemporalAccumulator,
)
from tools.evaluation.runtime_resources import (  # noqa: E402
    RunnerError,
    _GpuMemoryMonitor,
    ensure_gpus_have_free_memory,
)


CONFIG_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
VARIANTS = ("b0", "v")
COHORTS = ("overall", "pedestrian", "nonpedestrian")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
TOWN_PATTERN = re.compile(r"town(\d+)", re.IGNORECASE)
ROUTE_PATTERN = re.compile(r"(?:^|_)route(\d+)(?:_|$)", re.IGNORECASE)
METRIC_SPECS = {
    "traffic_average_precision": ("traffic", "occupancy", "average_precision", True),
    "traffic_roc_auc": ("traffic", "occupancy", "roc_auc", True),
    "traffic_occupied_iou": ("traffic", "occupancy", "occupied_iou", True),
    "waypoint_ade": ("waypoints", "ade", False),
    "waypoint_fde_horizon_10": ("waypoints", "fde_horizon_10", False),
    "junction_macro_f1": ("junction", "macro_f1", True),
    "red_light_macro_f1": ("red_light", "macro_f1", True),
}
TEMPORAL_SPECS = {
    "traffic_delta_residual_mae": (
        "traffic_probability_delta_residual_mae",
        False,
    ),
    "waypoint_delta_residual_ade": ("waypoint_delta_residual_ade", False),
}


class SceneValidationError(RuntimeError):
    """Raised when scene validation provenance or cohort support is invalid."""


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
        raise SceneValidationError(f"unable to read {label} JSON {path}: {exc}") from exc


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
        raise SceneValidationError(f"{label} must be a non-empty path")
    path = Path(value)
    path = path if path.is_absolute() else Path(repo_root) / path
    path = path.resolve()
    if require_file and not path.is_file():
        raise SceneValidationError(f"{label} is not a file: {path}")
    return path


def _verify_hash(path, expected, label):
    if not isinstance(expected, str) or len(expected) != 64:
        raise SceneValidationError(f"{label} SHA-256 must have 64 characters")
    actual = sha256_file(path)
    if actual != expected:
        raise SceneValidationError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def _positive_int(value, label, allow_zero=False):
    valid = isinstance(value, int) and not isinstance(value, bool)
    valid = valid and (value >= 0 if allow_zero else value > 0)
    if not valid:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise SceneValidationError(f"{label} must be a {qualifier} integer")
    return value


def _finite_number(value, label):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise SceneValidationError(f"{label} must be finite and positive")
    return float(value)


def _git_output(repo_root, *args):
    result = subprocess.run(
        ["git", *args], cwd=repo_root, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise SceneValidationError(
            f"git {' '.join(args)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _load_bound_json(repo_root, spec, label):
    if not isinstance(spec, dict):
        raise SceneValidationError(f"{label} binding must be an object")
    path = _resolve_path(repo_root, spec.get("path"), label)
    digest = _verify_hash(path, spec.get("sha256"), label)
    return path, digest, _read_json(path, label)


def load_scene_validation_contract(config_path, repo_root=REPO_ROOT):
    """Validate every artifact while keeping the test index outside the contract."""
    repo_root = Path(repo_root).resolve()
    config_path = Path(config_path).resolve()
    raw = _read_json(config_path, "scene validation config")
    if raw.get("schema_version") != CONFIG_SCHEMA_VERSION or raw.get("status") != "frozen":
        raise SceneValidationError("config must be frozen schema v1")
    if any("test_index" in key for key in raw):
        raise SceneValidationError("scene validation config must not expose test_index")
    run_id = raw.get("run_id")
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise SceneValidationError("run_id is invalid")

    pair_path, pair_hash, pair = _load_bound_json(
        repo_root, raw.get("pair_run_manifest"), "pair run manifest"
    )
    training_path, training_hash, _ = _load_bound_json(
        repo_root, raw.get("pair_training_config"), "pair training config"
    )
    split_path, split_hash, split = _load_bound_json(
        repo_root, raw.get("downstream_split_manifest"), "downstream split manifest"
    )
    if pair.get("status") != "completed" or pair.get("pipeline_valid") is not True:
        raise SceneValidationError("pair run must be completed and pipeline-valid")
    comparability = pair.get("comparability") or {}
    if not comparability.get("normalized_training_args_identical") or not comparability.get(
        "only_initial_checkpoint_differs"
    ):
        raise SceneValidationError("pair run is not a single-variable comparison")
    if tuple(comparability.get("variant_order") or ()) != VARIANTS:
        raise SceneValidationError("pair run variant order must be b0 then v")
    if Path(pair.get("config", "")).resolve() != training_path or pair.get(
        "config_sha256"
    ) != training_hash:
        raise SceneValidationError("pair run differs from bound training config")
    pair_split = (pair.get("inputs") or {}).get("downstream_split_manifest") or {}
    if Path(pair_split.get("path", "")).resolve() != split_path or pair_split.get(
        "sha256"
    ) != split_hash:
        raise SceneValidationError("pair run differs from bound downstream split")
    if split.get("valid") is not True or split.get("manifest_schema_version") != 2:
        raise SceneValidationError("downstream split must be valid scene schema v2")

    dataset = raw.get("dataset") or {}
    if "test_index" in dataset:
        raise SceneValidationError("validation runner refuses dataset.test_index")
    dataset_root = _resolve_path(repo_root, dataset.get("root"), "dataset.root", False)
    if not dataset_root.is_dir():
        raise SceneValidationError("dataset.root is not a directory")
    validation_index = _resolve_path(
        repo_root, dataset.get("validation_index"), "dataset.validation_index"
    )
    validation_hash = _verify_hash(
        validation_index,
        dataset.get("validation_index_sha256"),
        "dataset.validation_index",
    )
    if validation_hash != (split.get("artifacts", {}).get("validation") or {}).get(
        "sha256"
    ):
        raise SceneValidationError("validation index differs from split manifest")
    for field in ("towns", "weathers"):
        values = dataset.get(field)
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise SceneValidationError(f"dataset.{field} must be unique and non-empty")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise SceneValidationError(f"dataset.{field} must contain integers")
    if dataset.get("lidar_y_axis_multiplier") not in (-1.0, 1.0):
        raise SceneValidationError("dataset.lidar_y_axis_multiplier must be +/-1")
    if dataset.get("navigation_frame") != "carla0916_compass":
        raise SceneValidationError("scene validation requires carla0916_compass")
    if dataset.get("missing_navigation_policy") != "drop":
        raise SceneValidationError("scene validation requires missing navigation drop")

    scene = raw.get("scene") or {}
    expected_scene = {
        "effective_frames": _positive_int(
            scene.get("effective_frames"), "scene.effective_frames"
        ),
        "pedestrian_frames": _positive_int(
            scene.get("pedestrian_frames"), "scene.pedestrian_frames"
        ),
        "pedestrian_sequences": _positive_int(
            scene.get("pedestrian_sequences"), "scene.pedestrian_sequences"
        ),
        "pedestrian_route_groups": _positive_int(
            scene.get("pedestrian_route_groups"), "scene.pedestrian_route_groups"
        ),
    }
    split_scene = (
        split.get("scene_stratification", {})
        .get("split_scene_summary", {})
        .get("validation", {})
    )
    for field, value in expected_scene.items():
        if split_scene.get(field) != value:
            raise SceneValidationError(f"scene.{field} differs from split manifest")
    if scene.get("pedestrian_measurement_field") != "is_pedestrian_present":
        raise SceneValidationError("scene pedestrian truth field is invalid")
    temporal = scene.get("temporal") or {}
    for field in (
        "overall_adjacent_pairs",
        "overall_sequences_with_pairs",
        "pedestrian_adjacent_pairs",
        "pedestrian_sequences_with_pairs",
    ):
        _positive_int(temporal.get(field), f"scene.temporal.{field}")

    pair_variants = {item.get("variant"): item for item in pair.get("variants", [])}
    variants = raw.get("variants") or {}
    if tuple(variants) != VARIANTS or set(pair_variants) != set(VARIANTS):
        raise SceneValidationError("variants must define completed b0 then v")
    resolved_variants = {}
    for name in VARIANTS:
        spec = variants[name]
        checkpoint = _resolve_path(
            repo_root, spec.get("checkpoint"), f"{name} checkpoint"
        )
        checkpoint_hash = _verify_hash(
            checkpoint, spec.get("checkpoint_sha256"), f"{name} checkpoint"
        )
        artifact = (pair_variants[name].get("artifacts") or {}).get("best_checkpoint") or {}
        if Path(artifact.get("path", "")).resolve() != checkpoint or artifact.get(
            "sha256"
        ) != checkpoint_hash:
            raise SceneValidationError(f"{name} checkpoint is not pair best checkpoint")
        resolved_variants[name] = {
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_hash,
        }

    metrics = raw.get("metrics") or {}
    for field in (
        "traffic_positive_target_threshold",
        "traffic_prediction_threshold",
        "invalid_waypoint_threshold",
    ):
        _finite_number(metrics.get(field), f"metrics.{field}")
    runtime = raw.get("runtime") or {}
    for field in ("seed", "gpu", "batch_size", "workers", "log_interval_batches"):
        _positive_int(runtime.get(field), f"runtime.{field}", field in {"seed", "gpu", "workers"})
    _positive_int(runtime.get("gpu_minimum_free_memory_mb"), "runtime GPU free memory")
    if not isinstance(runtime.get("require_clean_git"), bool):
        raise SceneValidationError("runtime.require_clean_git must be boolean")
    result_root = _resolve_path(repo_root, raw.get("result_root"), "result_root", False)
    try:
        result_root.relative_to(repo_root)
    except ValueError as exc:
        raise SceneValidationError("result_root escapes repository") from exc
    return {
        **raw,
        "path": config_path,
        "sha256": sha256_file(config_path),
        "repo_root": repo_root,
        "pair_manifest_path": pair_path,
        "pair_manifest_sha256": pair_hash,
        "pair_manifest_loaded": pair,
        "split_manifest_path": split_path,
        "split_manifest_sha256": split_hash,
        "dataset_root_path": dataset_root,
        "validation_index_path": validation_index,
        "resolved_variants": resolved_variants,
        "result_root_path": result_root,
    }


def _presence(value):
    return bool(len(value)) if isinstance(value, (list, tuple, dict, str)) else bool(value)


def _route_group(dataset_root, route_dir):
    try:
        relative = Path(route_dir).resolve().relative_to(Path(dataset_root).resolve())
    except ValueError as exc:
        raise SceneValidationError(f"route escapes dataset root: {route_dir}") from exc
    if len(relative.parts) < 2:
        raise SceneValidationError(f"route path is too shallow: {relative}")
    town = TOWN_PATTERN.search(relative.parts[0])
    route = ROUTE_PATTERN.search(relative.parts[1])
    if not town or not route:
        raise SceneValidationError(f"unable to infer route group: {relative}")
    return f"Town{int(town.group(1)):02d}:route{int(route.group(1)):03d}", str(relative)


class _SceneIndexedDataset:
    """Attach frozen scene truth while bypassing corrupted-sample substitution."""

    def __init__(self, base, dataset_root, scene_contract):
        object.__setattr__(self, "_base", base)
        metadata = []
        pedestrian_sequences = set()
        pedestrian_groups = set()
        pedestrian_frames = 0
        for route_dir, frame_id in base.route_frames:
            measurement = base._load_json(
                os.path.join(route_dir, "measurements", f"{frame_id:04d}.json")
            )
            pedestrian = _presence(
                measurement.get(scene_contract["pedestrian_measurement_field"], [])
            )
            group, sequence = _route_group(dataset_root, route_dir)
            if pedestrian:
                pedestrian_frames += 1
                pedestrian_sequences.add(sequence)
                pedestrian_groups.add(group)
            metadata.append((sequence, frame_id, pedestrian, group))
        actual = {
            "effective_frames": len(metadata),
            "pedestrian_frames": pedestrian_frames,
            "pedestrian_sequences": len(pedestrian_sequences),
            "pedestrian_route_groups": len(pedestrian_groups),
        }
        expected = {
            name: scene_contract[name]
            for name in actual
        }
        if actual != expected:
            raise SceneValidationError(
                f"loaded scene census differs: expected {expected}, got {actual}"
            )
        object.__setattr__(self, "_metadata", metadata)
        object.__setattr__(self, "pedestrian_route_groups", sorted(pedestrian_groups))

    def __len__(self):
        return len(self._base)

    def __getattr__(self, name):
        return getattr(self._base, name)

    def __setattr__(self, name, value):
        setattr(self._base, name, value)

    def __getitem__(self, index):
        inputs, targets = self._base._get_item_impl(index)
        sequence, frame_id, pedestrian, group = self._metadata[index]
        return inputs, targets, {
            "sequence_id": sequence,
            "frame_id": frame_id,
            "pedestrian": pedestrian,
            "route_group": group,
        }


def _slice_tensors(values, indices):
    return tuple(value.index_select(0, indices.to(value.device)) for value in values)


def _finalize_accumulators(task, temporal, route_task):
    metrics = {name: accumulator.finalize() for name, accumulator in task.items()}
    metrics["overall"]["temporal"] = temporal["overall"].finalize()
    metrics["pedestrian"]["temporal"] = temporal["pedestrian"].finalize()
    return metrics, {
        group: accumulator.finalize() for group, accumulator in sorted(route_task.items())
    }


def _write_score_dump(dump_dir, variant, dump, expected_samples):
    """Persist per-sample traffic scores/targets with deterministic file bytes."""
    import numpy as np

    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    scores = np.concatenate(dump["scores"]).astype(np.float32, copy=False)
    targets = np.concatenate(dump["targets"]).astype(np.int8, copy=False)
    metadata = dump["metadata"]
    if scores.shape != targets.shape or scores.ndim != 2 or scores.shape[1] != 400:
        raise SceneValidationError("score dump arrays have unexpected shapes")
    if len(metadata) != scores.shape[0] or scores.shape[0] != expected_samples:
        raise SceneValidationError("score dump sample count differs from evaluation")
    artifacts = {}
    for name, array in (("traffic_scores", scores), ("traffic_targets", targets)):
        path = dump_dir / f"{variant}_{name}.npy"
        with path.open("wb") as stream:
            np.save(stream, array)
        artifacts[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "shape": [int(dim) for dim in array.shape],
        }
    meta_path = dump_dir / f"{variant}_meta.json"
    _write_json(meta_path, {"samples": metadata})
    artifacts["meta"] = {
        "path": str(meta_path),
        "sha256": sha256_file(meta_path),
        "samples": len(metadata),
    }
    return artifacts


def _evaluate_variant(contract, variant):
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
        contract["model"],
        pretrained=False,
        checkpoint_path=str(contract["resolved_variants"][variant]["checkpoint"]),
        freeze_num=-1,
    )
    model.cuda().eval()
    data_config = resolve_data_config({}, model=model, verbose=False)
    dataset_spec = contract["dataset"]
    base = create_carla_dataset(
        "carla",
        root=str(contract["dataset_root_path"]),
        towns=dataset_spec["towns"],
        weathers=dataset_spec["weathers"],
        batch_size=runtime["batch_size"],
        with_lidar=True,
        multi_view=True,
        augment_prob=0.0,
        dataset_index=str(contract["validation_index_path"]),
        lidar_y_axis_multiplier=dataset_spec["lidar_y_axis_multiplier"],
        navigation_frame=dataset_spec["navigation_frame"],
        missing_navigation_policy=dataset_spec["missing_navigation_policy"],
    )
    dataset = _SceneIndexedDataset(base, contract["dataset_root_path"], contract["scene"])
    loader = create_carla_loader(
        dataset,
        input_size=data_config["input_size"],
        batch_size=runtime["batch_size"],
        multi_view_input_size=(3, 128, 128),
        is_training=False,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=runtime["workers"],
        distributed=False,
        pin_memory=False,
        persistent_workers=runtime["workers"] > 0,
    )
    metric_args = (
        contract["metrics"]["traffic_positive_target_threshold"],
        contract["metrics"]["traffic_prediction_threshold"],
        contract["metrics"]["invalid_waypoint_threshold"],
    )
    task = {name: InterfuserMetricAccumulator(*metric_args) for name in COHORTS}
    temporal = {
        name: InterfuserTemporalAccumulator(metric_args[2])
        for name in ("overall", "pedestrian")
    }
    route_task = {
        group: InterfuserMetricAccumulator(*metric_args)
        for group in dataset.pedestrian_route_groups
    }
    score_dump_dir = contract.get("score_dump_dir")
    dump = None
    if score_dump_dir is not None:
        dump = {"scores": [], "targets": [], "metadata": []}
    started = time.monotonic()
    with torch.inference_mode():
        for batch_index, (inputs, targets, metadata) in enumerate(loader):
            inputs = {name: value.cuda(non_blocking=False) for name, value in inputs.items()}
            outputs = model(inputs)
            if dump is not None:
                dump["scores"].append(
                    outputs[0][:, :, 0].detach().cpu().numpy().astype(np.float32)
                )
                dump["targets"].append(
                    (
                        targets[4][:, :, 0]
                        >= contract["metrics"]["traffic_positive_target_threshold"]
                    )
                    .cpu()
                    .numpy()
                    .astype(np.int8)
                )
                dump["metadata"].extend(
                    {
                        "sequence_id": sequence,
                        "frame_id": int(frame),
                        "pedestrian": bool(flag),
                        "route_group": group,
                    }
                    for sequence, frame, flag, group in zip(
                        metadata["sequence_id"],
                        metadata["frame_id"].tolist(),
                        metadata["pedestrian"].tolist(),
                        metadata["route_group"],
                    )
                )
            task["overall"].update(outputs, targets)
            temporal["overall"].update(
                outputs, targets, metadata["sequence_id"], metadata["frame_id"]
            )
            pedestrian_mask = metadata["pedestrian"].bool()
            for name, mask in (
                ("pedestrian", pedestrian_mask),
                ("nonpedestrian", ~pedestrian_mask),
            ):
                indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
                if indices.numel() == 0:
                    continue
                cohort_outputs = _slice_tensors(outputs, indices)
                cohort_targets = _slice_tensors(targets, indices)
                task[name].update(cohort_outputs, cohort_targets)
                if name == "pedestrian":
                    positions = indices.tolist()
                    temporal[name].update(
                        cohort_outputs,
                        cohort_targets,
                        [metadata["sequence_id"][index] for index in positions],
                        metadata["frame_id"].index_select(0, indices),
                    )
            for group in dataset.pedestrian_route_groups:
                mask = pedestrian_mask & torch.tensor(
                    [value == group for value in metadata["route_group"]],
                    dtype=torch.bool,
                )
                indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
                if indices.numel():
                    route_task[group].update(
                        _slice_tensors(outputs, indices),
                        _slice_tensors(targets, indices),
                    )
            if batch_index % runtime["log_interval_batches"] == 0:
                print(
                    f"variant={variant} batch={batch_index}/{len(loader)} "
                    f"samples={task['overall'].samples}",
                    flush=True,
                )
    metrics, route_metrics = _finalize_accumulators(task, temporal, route_task)
    temporal_expected = contract["scene"]["temporal"]
    for cohort in ("overall", "pedestrian"):
        prefix = "overall" if cohort == "overall" else "pedestrian"
        actual = metrics[cohort]["temporal"]
        for field, suffix in (
            ("adjacent_pairs", "adjacent_pairs"),
            ("sequences_with_pairs", "sequences_with_pairs"),
        ):
            if actual[field] != temporal_expected[f"{prefix}_{suffix}"]:
                raise SceneValidationError(
                    f"{cohort} temporal {field} differs from frozen support"
                )
    result = {
        "variant": variant,
        "checkpoint": str(contract["resolved_variants"][variant]["checkpoint"]),
        "checkpoint_sha256": contract["resolved_variants"][variant][
            "checkpoint_sha256"
        ],
        "samples": metrics["overall"]["samples"],
        "duration_seconds": round(time.monotonic() - started, 3),
        "metrics": metrics,
        "pedestrian_route_group_metrics": route_metrics,
    }
    if dump is not None:
        result["score_dump"] = _write_score_dump(
            score_dump_dir, variant, dump, metrics["overall"]["samples"]
        )
    del model, loader, dataset, base
    torch.cuda.empty_cache()
    return result


def _value_at(metrics, path):
    value = metrics
    for key in path:
        value = value[key]
    return float(value)


def _one_delta(b0_value, v_value, higher_is_better):
    delta = v_value - b0_value
    return {
        "b0": b0_value,
        "v": v_value,
        "v_minus_b0": delta,
        "relative_percent": (delta / abs(b0_value) * 100.0) if b0_value else None,
        "higher_is_better": higher_is_better,
        "v_improved": delta > 0 if higher_is_better else delta < 0,
    }


def build_metric_comparison(b0_result, v_result):
    """Build fixed-direction overall, pedestrian and route-level paired deltas."""
    comparison = {"cohorts": {}, "pedestrian_route_groups": {}}
    for cohort in ("overall", "pedestrian", "nonpedestrian"):
        b0_metrics = b0_result["metrics"][cohort]
        v_metrics = v_result["metrics"][cohort]
        cohort_result = {}
        for name, (*path, higher_is_better) in METRIC_SPECS.items():
            if "insufficient_data" in b0_metrics.get(path[0], {}) or "insufficient_data" in v_metrics.get(path[0], {}):
                continue
            cohort_result[name] = _one_delta(
                _value_at(b0_metrics, path),
                _value_at(v_metrics, path),
                higher_is_better,
            )
        if cohort != "nonpedestrian":
            for name, (field, higher_is_better) in TEMPORAL_SPECS.items():
                cohort_result[name] = _one_delta(
                    float(b0_metrics["temporal"][field]),
                    float(v_metrics["temporal"][field]),
                    higher_is_better,
                )
        comparison["cohorts"][cohort] = cohort_result

    route_groups = sorted(b0_result["pedestrian_route_group_metrics"])
    if route_groups != sorted(v_result["pedestrian_route_group_metrics"]):
        raise SceneValidationError("B0/V pedestrian route groups differ")
    core = {
        name: spec for name, spec in METRIC_SPECS.items()
        if name in {
            "traffic_average_precision",
            "traffic_roc_auc",
            "traffic_occupied_iou",
            "waypoint_ade",
            "waypoint_fde_horizon_10",
        }
    }
    for group in route_groups:
        b0_metrics = b0_result["pedestrian_route_group_metrics"][group]
        v_metrics = v_result["pedestrian_route_group_metrics"][group]
        comparison["pedestrian_route_groups"][group] = {
            name: _one_delta(
                _value_at(b0_metrics, path),
                _value_at(v_metrics, path),
                higher_is_better,
            )
            for name, (*path, higher_is_better) in core.items()
        }
    pedestrian_core = comparison["cohorts"]["pedestrian"]
    improved = sum(pedestrian_core[name]["v_improved"] for name in core)
    comparison["gate"] = {
        "pedestrian_core_metrics": list(core),
        "pedestrian_core_improved": improved,
        "pedestrian_majority_supports_v": improved >= 3,
        "test_remains_frozen": True,
    }
    return comparison


def execute_scene_validation(config_path):
    contract = load_scene_validation_contract(config_path)
    runtime = contract["runtime"]
    if runtime["require_clean_git"] and _git_output(contract["repo_root"], "status", "--porcelain"):
        raise SceneValidationError("Git worktree must be clean")
    run_dir = contract["result_root_path"] / contract["run_id"]
    if run_dir.exists():
        raise SceneValidationError(f"refusing to overwrite run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "scene_validation_manifest.json"
    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": contract["run_id"],
        "status": "running",
        "pipeline_valid": False,
        "created_at": _utc_now(),
        "git_head": _git_output(contract["repo_root"], "rev-parse", "HEAD"),
        "git_status": "",
        "config": str(contract["path"]),
        "config_sha256": contract["sha256"],
        "pair_run_manifest": str(contract["pair_manifest_path"]),
        "pair_run_manifest_sha256": contract["pair_manifest_sha256"],
        "downstream_split_manifest": str(contract["split_manifest_path"]),
        "downstream_split_manifest_sha256": contract["split_manifest_sha256"],
        "validation_index": str(contract["validation_index_path"]),
        "validation_index_sha256": contract["dataset"]["validation_index_sha256"],
        "test_accessed": False,
        "variants": [],
        "errors": [],
    }
    _write_json(manifest_path, manifest)
    gpu = runtime["gpu"]
    monitor = _GpuMemoryMonitor([gpu])
    try:
        ensure_gpus_have_free_memory([gpu], runtime["gpu_minimum_free_memory_mb"])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        monitor.start()
        for variant in VARIANTS:
            result = _evaluate_variant(contract, variant)
            manifest["variants"].append(result)
            manifest["updated_at"] = _utc_now()
            _write_json(manifest_path, manifest)
        manifest["comparison"] = build_metric_comparison(*manifest["variants"])
        manifest.update(
            {
                "status": "completed",
                "pipeline_valid": True,
                "completed_at": _utc_now(),
            }
        )
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "pipeline_valid": False,
                "completed_at": _utc_now(),
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
        )
        raise
    finally:
        manifest["gpu_peak_memory_mb"] = monitor.stop()
        manifest["updated_at"] = _utc_now()
        _write_json(manifest_path, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate Stage 2 B0/V on scene-stratified validation only"
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = execute_scene_validation(args.config)
    except (SceneValidationError, RunnerError) as exc:
        print(f"scene validation error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"scene validation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(manifest["comparison"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
