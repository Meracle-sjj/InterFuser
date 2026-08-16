#!/usr/bin/env python3
"""
[INPUT]: 依赖冻结 M1 Town+route split manifest、全量 InterFuser dataset_index、逐帧 measurements 真值与版本化下游划分配置。
[OUTPUT]: 对外提供 DownstreamSplitError、load_downstream_split_config、build_downstream_indexes 与 CLI，生成无 route-group/预训练泄漏且可按行人真值分层的 train/validation/test index 及 manifest。
[POS]: tools/data 的 M2 下游划分器；v1 保留 M1 投影语义，v2 只从语义预训练未使用的 route group 确定性扩充行人 holdout。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSIONS = (1, 2)
TOWN_PATTERN = re.compile(r"town(\d+)", re.IGNORECASE)
ROUTE_PATTERN = re.compile(r"(?:^|_)route(\d+)(?:_|$)", re.IGNORECASE)
SPLITS = ("train", "validation", "test")
HOLDOUT_SPLITS = ("validation", "test")
V1_EXPANSION_POLICY = {
    "frozen_validation_route_groups": "validation",
    "frozen_test_route_groups": "test",
    "frozen_train_route_groups": "train",
    "unassigned_route_groups": "train",
}
V2_EXPANSION_POLICY = {
    "frozen_validation_route_groups": "validation",
    "frozen_test_route_groups": "test",
    "frozen_train_route_groups": "train",
    "unassigned_pedestrian_route_groups": "deterministic_stratified_holdout_then_train",
    "remaining_unassigned_route_groups": "train",
}


class DownstreamSplitError(ValueError):
    """Raised when the full downstream index cannot preserve the frozen split."""


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
        raise DownstreamSplitError(f"unable to read {label} JSON {path}: {exc}") from exc


def _resolve_path(value, label):
    if not isinstance(value, str) or not value:
        raise DownstreamSplitError(f"{label} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _verify_hash(path, expected, label):
    if not isinstance(expected, str) or len(expected) != 64:
        raise DownstreamSplitError(f"{label} SHA-256 must contain 64 hex characters")
    actual = sha256_file(path)
    if actual != expected:
        raise DownstreamSplitError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def load_downstream_split_config(path):
    """Load and verify the full-index projection contract."""
    path = Path(path).resolve()
    raw = _read_json(path, "downstream split config")
    schema_version = raw.get("schema_version")
    if schema_version not in SCHEMA_VERSIONS:
        raise DownstreamSplitError(
            f"unsupported schema_version: {schema_version}"
        )
    if raw.get("status") != "frozen" or raw.get("split_unit") != "town_route":
        raise DownstreamSplitError("config must be frozen with split_unit=town_route")
    expected_policy = V1_EXPANSION_POLICY if schema_version == 1 else V2_EXPANSION_POLICY
    if raw.get("expansion_policy") != expected_policy:
        raise DownstreamSplitError("expansion_policy differs from the frozen policy")
    if schema_version == 2:
        raw = _validate_scene_stratification(raw)
    dataset_root = _resolve_path(raw.get("dataset_root"), "dataset_root").resolve()
    dataset_index = _resolve_path(raw.get("dataset_index"), "dataset_index").resolve()
    semantic_manifest = _resolve_path(
        raw.get("semantic_split_manifest"), "semantic_split_manifest"
    ).resolve()
    if not dataset_root.is_dir():
        raise DownstreamSplitError(f"dataset_root is not a directory: {dataset_root}")
    _verify_hash(dataset_index, raw.get("dataset_index_sha256"), "dataset index")
    _verify_hash(
        semantic_manifest,
        raw.get("semantic_split_manifest_sha256"),
        "semantic split manifest",
    )
    normalized = dict(raw)
    normalized.update(
        {
            "path": path,
            "sha256": sha256_file(path),
            "dataset_root_path": dataset_root,
            "dataset_index_path": dataset_index,
            "semantic_split_manifest_path": semantic_manifest,
        }
    )
    return normalized


def _positive_int(value, label, allow_zero=False):
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise DownstreamSplitError(f"{label} must be a {qualifier} integer")
    return value


def _validate_scene_stratification(raw):
    stratification = raw.get("scene_stratification")
    if not isinstance(stratification, dict):
        raise DownstreamSplitError("schema v2 requires scene_stratification")
    if stratification.get("selection_method") != "seeded_town_coverage_balanced_frames_v1":
        raise DownstreamSplitError("unsupported scene stratification selection_method")
    _positive_int(stratification.get("selection_seed"), "selection_seed", allow_zero=True)
    if stratification.get("pedestrian_measurement_field") != "is_pedestrian_present":
        raise DownstreamSplitError(
            "pedestrian_measurement_field must be is_pedestrian_present"
        )
    required_fields = stratification.get("required_navigation_fields")
    if (
        not isinstance(required_fields, list)
        or not required_fields
        or len(required_fields) != len(set(required_fields))
        or any(not isinstance(value, str) or not value for value in required_fields)
    ):
        raise DownstreamSplitError(
            "required_navigation_fields must be a unique non-empty string list"
        )
    targets = stratification.get("target_pedestrian_route_groups")
    if not isinstance(targets, dict) or set(targets) != set(SPLITS):
        raise DownstreamSplitError(
            "target_pedestrian_route_groups must define train/validation/test"
        )
    for split, value in targets.items():
        _positive_int(value, f"target_pedestrian_route_groups.{split}")
    expected = stratification.get("expected_scene_totals")
    required_expected = {
        "effective_frames",
        "dropped_frames",
        "pedestrian_frames",
        "pedestrian_sequences",
        "pedestrian_route_groups",
    }
    if not isinstance(expected, dict) or set(expected) != required_expected:
        raise DownstreamSplitError(
            "expected_scene_totals must define the frozen scene census"
        )
    for name, value in expected.items():
        _positive_int(value, f"expected_scene_totals.{name}", allow_zero=name == "dropped_frames")
    if sum(targets.values()) != expected["pedestrian_route_groups"]:
        raise DownstreamSplitError(
            "pedestrian route-group targets do not match expected scene total"
        )
    required_towns = stratification.get("required_pedestrian_towns_per_holdout")
    if (
        not isinstance(required_towns, list)
        or not required_towns
        or len(required_towns) != len(set(required_towns))
        or any(not re.fullmatch(r"Town\d{2}", value or "") for value in required_towns)
    ):
        raise DownstreamSplitError(
            "required_pedestrian_towns_per_holdout must contain unique TownXX values"
        )
    for field in (
        "minimum_pedestrian_frames_per_holdout",
        "minimum_pedestrian_sequences_per_holdout",
        "minimum_pedestrian_weathers_per_holdout",
    ):
        _positive_int(stratification.get(field), field)
    ratio = stratification.get("maximum_holdout_pedestrian_frame_ratio")
    if (
        isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        or not math.isfinite(float(ratio))
        or float(ratio) < 1.0
    ):
        raise DownstreamSplitError(
            "maximum_holdout_pedestrian_frame_ratio must be finite and >= 1"
        )
    normalized = dict(raw)
    normalized["scene_stratification"] = dict(stratification)
    return normalized


def _route_group(relative_path):
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts or len(path.parts) < 2:
        raise DownstreamSplitError(f"invalid dataset path: {relative_path}")
    town_match = TOWN_PATTERN.search(path.parts[0])
    route_match = ROUTE_PATTERN.search(path.parts[1])
    if not town_match or not route_match:
        raise DownstreamSplitError(f"unable to infer Town+route: {relative_path}")
    return f"Town{int(town_match.group(1)):02d}:route{int(route_match.group(1)):03d}"


def _weather(relative_path):
    match = re.search(r"_w(\d+)(?:_|$)", str(relative_path), re.IGNORECASE)
    if not match:
        raise DownstreamSplitError(f"unable to infer weather: {relative_path}")
    return int(match.group(1))


def _read_dataset_index(path):
    records = []
    seen = set()
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) == 2:
            relative_path, frame_text = fields
        elif len(fields) == 3:
            _, relative_path, frame_text = fields
        else:
            raise DownstreamSplitError(f"dataset index line {line_number} is invalid")
        try:
            frames = int(frame_text)
        except ValueError as exc:
            raise DownstreamSplitError(
                f"dataset index line {line_number} has invalid frame count"
            ) from exc
        if frames <= 0 or relative_path in seen:
            raise DownstreamSplitError(
                f"dataset index line {line_number} is duplicate or nonpositive"
            )
        seen.add(relative_path)
        records.append(
            {
                "path": relative_path,
                "frames": frames,
                "route_group": _route_group(relative_path),
                "weather": _weather(relative_path),
            }
        )
    if not records:
        raise DownstreamSplitError("dataset index contains no records")
    return records


def _split_summary(records):
    return {
        "sequences": len(records),
        "logical_frames": sum(item["frames"] for item in records),
        "route_groups": len({item["route_group"] for item in records}),
        "towns": sorted({item["route_group"].split(":", 1)[0] for item in records}),
        "weathers": sorted({item["weather"] for item in records}),
    }


def _presence(value):
    if isinstance(value, (list, tuple, dict, set, str)):
        return bool(len(value))
    return bool(value)


def _scan_scene_coverage(records, config):
    stratification = config["scene_stratification"]
    required_fields = tuple(stratification["required_navigation_fields"])
    pedestrian_field = stratification["pedestrian_measurement_field"]
    dataset_root = config["dataset_root_path"]
    coverage = {}
    effective_frames = 0
    dropped_frames = 0
    pedestrian_frames = 0
    pedestrian_sequences = 0

    for item in records:
        group = coverage.setdefault(
            item["route_group"],
            {
                "town": item["route_group"].split(":", 1)[0],
                "sequences": 0,
                "logical_frames": 0,
                "effective_frames": 0,
                "dropped_frames": 0,
                "pedestrian_frames": 0,
                "pedestrian_sequences": 0,
                "weathers": set(),
                "pedestrian_weathers": set(),
            },
        )
        group["sequences"] += 1
        group["logical_frames"] += item["frames"]
        group["weathers"].add(item["weather"])
        sequence_pedestrian_frames = 0
        measurements_dir = dataset_root / item["path"] / "measurements"
        for frame_id in range(item["frames"]):
            path = measurements_dir / f"{frame_id:04d}.json"
            try:
                measurement = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                group["dropped_frames"] += 1
                dropped_frames += 1
                continue
            if not all(field in measurement for field in required_fields):
                group["dropped_frames"] += 1
                dropped_frames += 1
                continue
            group["effective_frames"] += 1
            effective_frames += 1
            if _presence(measurement.get(pedestrian_field, [])):
                group["pedestrian_frames"] += 1
                pedestrian_frames += 1
                sequence_pedestrian_frames += 1
        if sequence_pedestrian_frames:
            group["pedestrian_sequences"] += 1
            pedestrian_sequences += 1
            group["pedestrian_weathers"].add(item["weather"])

    pedestrian_groups = {
        name: value for name, value in coverage.items() if value["pedestrian_frames"]
    }
    actual = {
        "effective_frames": effective_frames,
        "dropped_frames": dropped_frames,
        "pedestrian_frames": pedestrian_frames,
        "pedestrian_sequences": pedestrian_sequences,
        "pedestrian_route_groups": len(pedestrian_groups),
    }
    expected = stratification["expected_scene_totals"]
    if actual != expected:
        raise DownstreamSplitError(
            f"scene census differs from frozen expectation: expected {expected}, got {actual}"
        )
    serializable = {
        name: {
            **{key: value for key, value in group.items() if not isinstance(value, set)},
            "weathers": sorted(group["weathers"]),
            "pedestrian_weathers": sorted(group["pedestrian_weathers"]),
        }
        for name, group in sorted(coverage.items())
    }
    census_sha256 = hashlib.sha256(
        json.dumps(serializable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return coverage, actual, census_sha256


def _stable_rank(seed, split, group):
    return hashlib.sha256(f"{seed}:{split}:{group}".encode("utf-8")).hexdigest()


def _select_scene_group(
    candidates,
    coverage,
    selected,
    split,
    target_group_count,
    target_frame_count,
    seed,
):
    if not candidates:
        raise DownstreamSplitError(f"no pedestrian candidate remains for {split}")
    current_frames = sum(coverage[name]["pedestrian_frames"] for name in selected)
    expected_after = target_frame_count * (len(selected) + 1) / target_group_count
    mean_effective_frames = sum(
        coverage[name]["effective_frames"] for name in coverage
    ) / len(coverage)
    return min(
        candidates,
        key=lambda name: (
            abs(
                current_frames
                + coverage[name]["pedestrian_frames"]
                - expected_after
            ),
            abs(coverage[name]["effective_frames"] - mean_effective_frames),
            _stable_rank(seed, split, name),
        ),
    )


def _scene_summary(group_names, coverage):
    groups = [coverage[name] for name in sorted(group_names)]
    pedestrian_groups = [group for group in groups if group["pedestrian_frames"]]
    pedestrian_weathers = set()
    for group in pedestrian_groups:
        pedestrian_weathers.update(group["pedestrian_weathers"])
    return {
        "effective_frames": sum(group["effective_frames"] for group in groups),
        "dropped_frames": sum(group["dropped_frames"] for group in groups),
        "pedestrian_frames": sum(
            group["pedestrian_frames"] for group in pedestrian_groups
        ),
        "pedestrian_sequences": sum(
            group["pedestrian_sequences"] for group in pedestrian_groups
        ),
        "pedestrian_route_groups": len(pedestrian_groups),
        "pedestrian_towns": sorted({group["town"] for group in pedestrian_groups}),
        "pedestrian_weathers": sorted(pedestrian_weathers),
    }


def _select_scene_stratified_assignments(assignments, coverage, config):
    stratification = config["scene_stratification"]
    pedestrian_groups = {
        name for name, group in coverage.items() if group["pedestrian_frames"]
    }
    targets = stratification["target_pedestrian_route_groups"]
    selected = {
        split: {
            group
            for group in pedestrian_groups
            if assignments.get(group) == split
        }
        for split in SPLITS
    }
    for split in SPLITS:
        if len(selected[split]) > targets[split]:
            raise DownstreamSplitError(
                f"frozen {split} already exceeds its pedestrian route-group target"
            )

    candidates = pedestrian_groups - set(assignments)
    required_towns = stratification["required_pedestrian_towns_per_holdout"]
    available_candidate_towns = {
        coverage[group]["town"] for group in candidates
    }
    unavailable = sorted(set(required_towns) - available_candidate_towns - {
        coverage[group]["town"]
        for split in HOLDOUT_SPLITS
        for group in selected[split]
    })
    if unavailable:
        raise DownstreamSplitError(
            f"required pedestrian holdout towns are unavailable: {unavailable}"
        )

    total_pedestrian_frames = sum(
        coverage[group]["pedestrian_frames"] for group in pedestrian_groups
    )
    target_total_groups = sum(targets.values())
    target_frames = {
        split: total_pedestrian_frames * targets[split] / target_total_groups
        for split in SPLITS
    }
    seed = stratification["selection_seed"]

    for town in required_towns:
        for split in HOLDOUT_SPLITS:
            if any(coverage[group]["town"] == town for group in selected[split]):
                continue
            if len(selected[split]) >= targets[split]:
                raise DownstreamSplitError(
                    f"{split} cannot cover required town {town} within its group target"
                )
            town_candidates = {
                group for group in candidates if coverage[group]["town"] == town
            }
            chosen = _select_scene_group(
                town_candidates,
                coverage,
                selected[split],
                split,
                targets[split],
                target_frames[split],
                seed,
            )
            selected[split].add(chosen)
            candidates.remove(chosen)

    while any(len(selected[split]) < targets[split] for split in HOLDOUT_SPLITS):
        eligible = [
            split
            for split in HOLDOUT_SPLITS
            if len(selected[split]) < targets[split]
        ]
        split = max(
            eligible,
            key=lambda name: (
                (targets[name] - len(selected[name])) / targets[name],
                name == "validation",
            ),
        )
        chosen = _select_scene_group(
            candidates,
            coverage,
            selected[split],
            split,
            targets[split],
            target_frames[split],
            seed,
        )
        selected[split].add(chosen)
        candidates.remove(chosen)

    selected["train"] = pedestrian_groups - selected["validation"] - selected["test"]
    actual_counts = {split: len(selected[split]) for split in SPLITS}
    if actual_counts != targets:
        raise DownstreamSplitError(
            f"pedestrian group selection missed targets: expected {targets}, got {actual_counts}"
        )

    final_assignments = {
        group: split for group, split in assignments.items() if group in coverage
    }
    for split in HOLDOUT_SPLITS:
        for group in selected[split]:
            if group not in assignments:
                final_assignments[group] = split
    for group in coverage:
        final_assignments.setdefault(group, "train")

    group_sets = {
        split: {group for group, assigned in final_assignments.items() if assigned == split}
        for split in SPLITS
    }
    scene_summaries = {
        split: _scene_summary(group_sets[split], coverage) for split in SPLITS
    }
    for split in HOLDOUT_SPLITS:
        summary = scene_summaries[split]
        if set(summary["pedestrian_towns"]) != set(required_towns):
            raise DownstreamSplitError(
                f"{split} does not cover every required pedestrian Town"
            )
        for field, threshold_field in (
            ("pedestrian_frames", "minimum_pedestrian_frames_per_holdout"),
            ("pedestrian_sequences", "minimum_pedestrian_sequences_per_holdout"),
        ):
            if summary[field] < stratification[threshold_field]:
                raise DownstreamSplitError(
                    f"{split} {field}={summary[field]} is below frozen minimum"
                )
        if len(summary["pedestrian_weathers"]) < stratification[
            "minimum_pedestrian_weathers_per_holdout"
        ]:
            raise DownstreamSplitError(
                f"{split} pedestrian weather coverage is below frozen minimum"
            )
    holdout_frames = [
        scene_summaries[split]["pedestrian_frames"] for split in HOLDOUT_SPLITS
    ]
    holdout_ratio = max(holdout_frames) / min(holdout_frames)
    if holdout_ratio > stratification["maximum_holdout_pedestrian_frame_ratio"]:
        raise DownstreamSplitError(
            f"holdout pedestrian frame ratio {holdout_ratio:.6f} exceeds frozen maximum"
        )

    semantic_train_holdout_overlap = sorted(
        group
        for group, split in assignments.items()
        if split == "train" and final_assignments.get(group, "train") in HOLDOUT_SPLITS
    )
    if semantic_train_holdout_overlap:
        raise DownstreamSplitError(
            "semantic pretraining train route groups leaked into downstream holdout"
        )
    return final_assignments, scene_summaries, {
        "selection_method": stratification["selection_method"],
        "selection_seed": seed,
        "target_pedestrian_route_groups": dict(targets),
        "additional_holdout_route_groups": {
            split: sorted(selected[split] - set(assignments))
            for split in HOLDOUT_SPLITS
        },
        "final_pedestrian_route_groups": {
            split: sorted(selected[split]) for split in SPLITS
        },
        "holdout_pedestrian_frame_ratio": holdout_ratio,
        "semantic_train_holdout_overlap": semantic_train_holdout_overlap,
    }


def build_downstream_indexes(config_path):
    """Project frozen route-group assignments onto every sequence in the full index."""
    config = load_downstream_split_config(config_path)
    semantic = _read_json(config["semantic_split_manifest_path"], "semantic split")
    if not semantic.get("valid"):
        raise DownstreamSplitError("semantic split manifest must be valid")
    if Path(semantic.get("dataset_root", "")).resolve() != config[
        "dataset_root_path"
    ]:
        raise DownstreamSplitError("semantic split and downstream dataset roots differ")
    if semantic.get("source", {}).get("dataset_index_sha256") != config[
        "dataset_index_sha256"
    ]:
        raise DownstreamSplitError("semantic split and full dataset index SHA-256 differ")
    leakage = semantic.get("leakage_check", {})
    if leakage.get("route_group_overlap_count") != 0 or not leakage.get(
        "all_selected_sequences_assigned_once"
    ):
        raise DownstreamSplitError("semantic split leakage check is not valid")
    assignments = {}
    for item in semantic.get("route_groups", []):
        group = item.get("route_group")
        split = item.get("split")
        if split not in {"train", "validation", "test"} or group in assignments:
            raise DownstreamSplitError("semantic split route-group assignments are invalid")
        assignments[group] = split
    if not assignments:
        raise DownstreamSplitError("semantic split contains no route groups")

    records = _read_dataset_index(config["dataset_index_path"])
    full_groups = {item["route_group"] for item in records}
    missing_holdout = sorted(
        group
        for group, split in assignments.items()
        if split in HOLDOUT_SPLITS and group not in full_groups
    )
    if missing_holdout:
        raise DownstreamSplitError(
            f"frozen holdout route groups are absent from full index: {missing_holdout}"
        )

    scene_coverage = None
    scene_census = None
    scene_census_sha256 = None
    scene_summaries = None
    selection = None
    final_assignments = dict(assignments)
    if config["schema_version"] == 2:
        scene_coverage, scene_census, scene_census_sha256 = _scan_scene_coverage(
            records, config
        )
        final_assignments, scene_summaries, selection = (
            _select_scene_stratified_assignments(
                assignments, scene_coverage, config
            )
        )

    splits = {name: [] for name in SPLITS}
    assignment_sources = Counter()
    for item in records:
        group = item["route_group"]
        frozen_split = assignments.get(group)
        split = final_assignments.get(group, "train")
        if frozen_split is not None:
            source = "frozen_route_group"
        elif config["schema_version"] == 2 and split in HOLDOUT_SPLITS:
            source = "stratified_pedestrian_holdout"
        else:
            source = "unassigned_to_train"
        assignment_sources[source] += 1
        splits[split].append(item)

    group_sets = {
        name: {item["route_group"] for item in values}
        for name, values in splits.items()
    }
    overlap = {
        "train_validation": sorted(group_sets["train"] & group_sets["validation"]),
        "train_test": sorted(group_sets["train"] & group_sets["test"]),
        "validation_test": sorted(group_sets["validation"] & group_sets["test"]),
    }
    if any(overlap.values()) or any(not values for values in splits.values()):
        raise DownstreamSplitError("projected splits are empty or overlap by route group")
    semantic_train_groups = {
        group for group, split in assignments.items() if split == "train"
    }
    semantic_train_holdout_overlap = sorted(
        (group_sets["validation"] | group_sets["test"]) & semantic_train_groups
    )
    if semantic_train_holdout_overlap:
        raise DownstreamSplitError(
            "semantic pretraining train route groups overlap downstream holdout"
        )
    manifest = {
        "manifest_schema_version": config["schema_version"],
        "valid": True,
        "errors": [],
        "source": {
            "config": str(config["path"]),
            "config_sha256": config["sha256"],
            "dataset_root": str(config["dataset_root_path"]),
            "dataset_index": str(config["dataset_index_path"]),
            "dataset_index_sha256": config["dataset_index_sha256"],
            "semantic_split_manifest": str(config["semantic_split_manifest_path"]),
            "semantic_split_manifest_sha256": config[
                "semantic_split_manifest_sha256"
            ],
        },
        "policy": dict(config["expansion_policy"]),
        "summary": {name: _split_summary(values) for name, values in splits.items()},
        "assignment_sources": dict(sorted(assignment_sources.items())),
        "leakage_check": {
            "route_group_overlaps": overlap,
            "all_source_sequences_assigned_once": sum(map(len, splits.values()))
            == len(records),
            "all_frozen_holdout_groups_present": not missing_holdout,
            "semantic_train_holdout_overlap": semantic_train_holdout_overlap,
        },
    }
    if config["schema_version"] == 2:
        manifest["scene_stratification"] = {
            "truth_source": {
                "measurement_field": config["scene_stratification"][
                    "pedestrian_measurement_field"
                ],
                "required_navigation_fields": config["scene_stratification"][
                    "required_navigation_fields"
                ],
                "dataset_index_sha256": config["dataset_index_sha256"],
            },
            "scene_census": scene_census,
            "scene_census_sha256": scene_census_sha256,
            "split_scene_summary": scene_summaries,
            "selection": selection,
        }
    return config, splits, manifest


def _serialize_index(records):
    return "".join(
        f"{item['path']} {item['frames']}\n"
        for item in sorted(records, key=lambda item: item["path"])
    )


def write_downstream_indexes(config_path, output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise DownstreamSplitError(f"refusing to overwrite output directory: {output_dir}")
    config, splits, manifest = build_downstream_indexes(config_path)
    output_dir.mkdir(parents=True)
    index_artifacts = {}
    for name, records in splits.items():
        path = output_dir / f"{name}_dataset_index.txt"
        path.write_text(_serialize_index(records), encoding="utf-8")
        index_artifacts[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    manifest["artifacts"] = index_artifacts
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest, manifest_path, config


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build leakage-safe full InterFuser downstream indexes"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest, _, _ = write_downstream_indexes(args.config, args.output_dir)
    except DownstreamSplitError as exc:
        print(f"downstream split error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
