#!/usr/bin/env python3
"""
[INPUT]: 依赖版本化语义类别 JSON、可选 dataset_index、Pillow/NumPy，以及 sequence 中按 frame ID 对齐的 rgb_{camera} 与 seg_{camera} 文件。
[OUTPUT]: 对外提供 AuditError、load_class_config、load_dataset_index、audit_semantic_dataset 与 CLI，输出可追溯抽样、标签覆盖、结构错误和 pilot readiness。
[POS]: tools/data 的 M1 数据准入审计器，把 CARLA 原始语义事实转换为可复现统计；它不生成标签、不修改图像，也不决定训练超参数。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


CONFIG_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 2
DEFAULT_CAMERAS = ("front", "left", "right")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "thesis" / "semantic_classes_v1.json"
TOWN_PATTERN = re.compile(r"Town(\d+)(?:HD)?", re.IGNORECASE)
WEATHER_PATTERN = re.compile(r"_w(\d+)_", re.IGNORECASE)


class AuditError(ValueError):
    """Raised when the dataset root or class contract cannot be audited."""


def _read_json(path):
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"unable to read JSON {path}: {exc}") from exc


def _nonnegative_int(value, path):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AuditError(f"{path} must be a nonnegative integer")
    return value


def load_class_config(path):
    """Load and validate the one-to-one source-tag grouping contract."""
    path = Path(path)
    raw = _read_json(path)
    if not isinstance(raw, dict):
        raise AuditError("class config must be a JSON object")
    if raw.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise AuditError(
            f"unsupported class config schema_version: {raw.get('schema_version')}"
        )

    source_labels_raw = raw.get("source_labels")
    if not isinstance(source_labels_raw, dict) or not source_labels_raw:
        raise AuditError("source_labels must be a non-empty object")
    source_labels = {}
    for key, name in source_labels_raw.items():
        try:
            tag = int(key)
        except (TypeError, ValueError) as exc:
            raise AuditError(f"source label key is not an integer: {key}") from exc
        if not isinstance(name, str) or not name:
            raise AuditError(f"source_labels[{key}] must have a non-empty name")
        if tag in source_labels:
            raise AuditError(f"duplicate source label after integer parsing: {tag}")
        source_labels[tag] = name

    ignore_tags = raw.get("ignore_source_tags")
    if not isinstance(ignore_tags, list):
        raise AuditError("ignore_source_tags must be a list")
    ignore_tags = {_nonnegative_int(tag, "ignore_source_tags[]") for tag in ignore_tags}

    classes_raw = raw.get("classes")
    if not isinstance(classes_raw, list) or not classes_raw:
        raise AuditError("classes must be a non-empty list")

    classes = []
    train_ids = set()
    names = set()
    assigned_tags = set()
    for index, item in enumerate(classes_raw):
        path_prefix = f"classes[{index}]"
        if not isinstance(item, dict):
            raise AuditError(f"{path_prefix} must be an object")
        train_id = _nonnegative_int(item.get("train_id"), f"{path_prefix}.train_id")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise AuditError(f"{path_prefix}.name must be non-empty")
        if train_id in train_ids:
            raise AuditError(f"duplicate train_id: {train_id}")
        if name in names:
            raise AuditError(f"duplicate class name: {name}")

        tags = item.get("source_tags")
        if not isinstance(tags, list) or not tags:
            raise AuditError(f"{path_prefix}.source_tags must be non-empty")
        normalized_tags = []
        for tag in tags:
            tag = _nonnegative_int(tag, f"{path_prefix}.source_tags[]")
            if tag not in source_labels:
                raise AuditError(f"unknown source tag in {path_prefix}: {tag}")
            if tag in ignore_tags:
                raise AuditError(f"ignored source tag is also mapped: {tag}")
            if tag in assigned_tags:
                raise AuditError(f"source tag is mapped more than once: {tag}")
            normalized_tags.append(tag)
            assigned_tags.add(tag)

        normalized = dict(item)
        normalized["train_id"] = train_id
        normalized["name"] = name
        normalized["source_tags"] = tuple(normalized_tags)
        for field in (
            "minimum_pixels_per_mask",
            "minimum_qualified_masks",
            "minimum_sequences",
        ):
            normalized[field] = _nonnegative_int(
                item.get(field), f"{path_prefix}.{field}"
            )
        if normalized["minimum_pixels_per_mask"] == 0:
            raise AuditError(f"{path_prefix}.minimum_pixels_per_mask must be positive")
        classes.append(normalized)
        train_ids.add(train_id)
        names.add(name)

    expected_train_ids = set(range(len(classes)))
    if train_ids != expected_train_ids:
        raise AuditError(
            "train_id values must be contiguous from 0 to " f"{len(classes) - 1}"
        )

    known_tags = set(source_labels)
    covered_tags = assigned_tags | ignore_tags
    if covered_tags != known_tags:
        missing = sorted(known_tags - covered_tags)
        extra = sorted(covered_tags - known_tags)
        raise AuditError(f"source label coverage mismatch: missing={missing} extra={extra}")

    readiness_raw = raw.get("dataset_readiness")
    if not isinstance(readiness_raw, dict):
        raise AuditError("dataset_readiness must be an object")
    readiness = {}
    for field in ("minimum_sequences", "minimum_towns", "minimum_logical_frames"):
        readiness[field] = _nonnegative_int(
            readiness_raw.get(field), f"dataset_readiness.{field}"
        )

    return {
        "path": path,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_labels": source_labels,
        "ignore_tags": ignore_tags,
        "classes": sorted(classes, key=lambda item: item["train_id"]),
        "dataset_readiness": readiness,
    }


def load_dataset_index(root, index_path):
    """Load sequence paths and declared frame counts without scanning the dataset."""
    root = Path(root).resolve()
    index_path = Path(index_path)
    if not index_path.is_absolute():
        index_path = root / index_path
    try:
        lines = index_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AuditError(f"unable to read dataset index {index_path}: {exc}") from exc

    entries = []
    seen = set()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            relative_text, count_text = line.rsplit(maxsplit=1)
            declared_frames = int(count_text)
        except (TypeError, ValueError) as exc:
            raise AuditError(f"dataset index line {line_number} is invalid") from exc
        relative = Path(relative_text)
        if relative.is_absolute() or declared_frames <= 0:
            raise AuditError(f"dataset index line {line_number} is invalid")
        sequence = (root / relative).resolve()
        try:
            sequence.relative_to(root)
        except ValueError as exc:
            raise AuditError(
                f"dataset index line {line_number} escapes dataset root"
            ) from exc
        if sequence in seen:
            raise AuditError(f"dataset index contains duplicate sequence: {relative}")
        if not sequence.is_dir():
            raise AuditError(f"indexed sequence is not a directory: {relative}")
        seen.add(sequence)
        weather_match = WEATHER_PATTERN.search(relative.as_posix())
        entries.append(
            {
                "path": sequence,
                "relative_path": relative.as_posix(),
                "declared_frames": declared_frames,
                "town": _town_from_route(relative.as_posix()),
                "weather": int(weather_match.group(1)) if weather_match else None,
            }
        )
    if not entries:
        raise AuditError(f"dataset index is empty: {index_path}")
    return {
        "path": index_path.resolve(),
        "sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "entries": entries,
    }


def _sample_index_entries(entries, per_stratum, seed):
    if per_stratum is None:
        return list(entries)
    if (
        not isinstance(per_stratum, int)
        or isinstance(per_stratum, bool)
        or per_stratum <= 0
    ):
        raise AuditError("sample_per_town_weather must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise AuditError("sample_seed must be an integer")
    groups = defaultdict(list)
    for entry in entries:
        if entry["town"] is None or entry["weather"] is None:
            raise AuditError(
                "Town/weather sampling requires every indexed path to encode both values"
            )
        groups[(entry["town"], entry["weather"])].append(entry)
    rng = random.Random(seed)
    selected = []
    for key in sorted(groups):
        candidates = sorted(groups[key], key=lambda item: item["relative_path"])
        count = min(per_stratum, len(candidates))
        selected.extend(rng.sample(candidates, count))
    return sorted(
        selected,
        key=lambda item: (item["town"], item["weather"], item["relative_path"]),
    )


def _route_dirs(root, cameras):
    routes = set()
    for camera in cameras:
        for directory in root.rglob(f"seg_{camera}"):
            if directory.is_dir():
                routes.add(directory.parent)
    return sorted(routes)


def _frame_paths(directory, suffixes):
    if not directory.is_dir():
        return {}
    suffixes = {suffix.lower() for suffix in suffixes}
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    }


def _town_from_route(route):
    match = TOWN_PATTERN.search(str(route))
    return f"Town{match.group(1)}" if match else None


def audit_semantic_dataset(
    root,
    config_path=DEFAULT_CONFIG,
    cameras=DEFAULT_CAMERAS,
    index_path=None,
    sample_per_town_weather=None,
    sample_seed=0,
):
    """Return deterministic coverage and readiness facts for a dataset root."""
    root = Path(root)
    if not root.is_dir():
        raise AuditError(f"dataset root is not a directory: {root}")
    cameras = tuple(cameras)
    if not cameras or len(set(cameras)) != len(cameras):
        raise AuditError("cameras must be a non-empty list without duplicates")
    if any(not isinstance(camera, str) or not camera for camera in cameras):
        raise AuditError("camera names must be non-empty strings")

    if sample_per_town_weather is not None and index_path is None:
        raise AuditError("Town/weather sampling requires index_path")

    config = load_class_config(config_path)
    index = load_dataset_index(root, index_path) if index_path is not None else None
    indexed_entries = (
        _sample_index_entries(index["entries"], sample_per_town_weather, sample_seed)
        if index
        else []
    )
    routes = [item["path"] for item in indexed_entries] if index else _route_dirs(root, cameras)
    if not routes:
        raise AuditError(f"no seg_<camera> directories found under {root}")
    indexed_by_path = {item["path"]: item for item in indexed_entries}

    raw_pixels = Counter()
    raw_masks = Counter()
    class_pixels = Counter()
    class_masks = Counter()
    class_qualified_masks = Counter()
    class_sequences = defaultdict(set)
    camera_masks = Counter()
    errors = []
    logical_frames = 0
    towns = set()
    known_tags = set(config["source_labels"])

    for route in routes:
        indexed_entry = indexed_by_path.get(route)
        route_key = (
            indexed_entry["relative_path"]
            if indexed_entry
            else str(route.relative_to(root))
        )
        town = _town_from_route(route_key)
        if town is None:
            errors.append(f"{route_key}: unable to infer Town from route path")
        else:
            towns.add(town)

        route_frame_sets = []
        route_qualified = set()
        for camera in cameras:
            seg_paths = _frame_paths(route / f"seg_{camera}", {".png"})
            rgb_paths = _frame_paths(
                route / f"rgb_{camera}", {".jpg", ".jpeg", ".png"}
            )
            seg_frames = set(seg_paths)
            rgb_frames = set(rgb_paths)
            route_frame_sets.append(seg_frames)
            if not seg_paths:
                errors.append(f"{route_key}: missing seg_{camera} PNG files")
            if not rgb_paths:
                errors.append(f"{route_key}: missing rgb_{camera} image files")
            if seg_frames - rgb_frames:
                errors.append(
                    f"{route_key}: seg_{camera} frames without RGB: "
                    + ", ".join(sorted(seg_frames - rgb_frames))
                )
            if rgb_frames - seg_frames:
                errors.append(
                    f"{route_key}: rgb_{camera} frames without mask: "
                    + ", ".join(sorted(rgb_frames - seg_frames))
                )

            for frame_id, path in sorted(seg_paths.items()):
                try:
                    labels = np.asarray(Image.open(path))
                except OSError as exc:
                    errors.append(f"{route_key}/{path.name}: unable to read mask: {exc}")
                    continue
                if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
                    errors.append(
                        f"{route_key}/{path.name}: semantic mask must be 2D integer"
                    )
                    continue
                values, counts = np.unique(labels, return_counts=True)
                per_mask = {
                    int(value): int(count)
                    for value, count in zip(values.tolist(), counts.tolist())
                }
                unknown = sorted(set(per_mask) - known_tags)
                if unknown:
                    errors.append(
                        f"{route_key}/{path.name}: unmapped source tags {unknown}"
                    )
                for tag, count in per_mask.items():
                    raw_pixels[tag] += count
                    raw_masks[tag] += 1

                for item in config["classes"]:
                    train_id = item["train_id"]
                    pixels = sum(per_mask.get(tag, 0) for tag in item["source_tags"])
                    class_pixels[train_id] += pixels
                    if pixels:
                        class_masks[train_id] += 1
                    if pixels >= item["minimum_pixels_per_mask"]:
                        class_qualified_masks[train_id] += 1
                        route_qualified.add(train_id)
                camera_masks[camera] += 1

        if route_frame_sets:
            first = route_frame_sets[0]
            for camera, frames in zip(cameras[1:], route_frame_sets[1:]):
                if frames != first:
                    errors.append(
                        f"{route_key}: seg_{camera} frame IDs differ from seg_{cameras[0]}"
                    )
            route_logical_frames = len(set().union(*route_frame_sets))
            logical_frames += route_logical_frames
            if (
                indexed_entry
                and route_logical_frames != indexed_entry["declared_frames"]
            ):
                errors.append(
                    f"{route_key}: index declares {indexed_entry['declared_frames']} "
                    f"frames but audit found {route_logical_frames}"
                )
        for train_id in route_qualified:
            class_sequences[train_id].add(route_key)

    total_pixels = sum(raw_pixels.values())
    raw_stats = []
    for tag, name in sorted(config["source_labels"].items()):
        pixels = raw_pixels[tag]
        raw_stats.append(
            {
                "source_tag": tag,
                "name": name,
                "pixels": pixels,
                "pixel_share": pixels / total_pixels if total_pixels else 0.0,
                "masks_with_any": raw_masks[tag],
            }
        )

    class_stats = []
    readiness_failures = []
    for item in config["classes"]:
        train_id = item["train_id"]
        qualified = class_qualified_masks[train_id]
        sequence_count = len(class_sequences[train_id])
        class_stats.append(
            {
                "train_id": train_id,
                "name": item["name"],
                "source_tags": list(item["source_tags"]),
                "pixels": class_pixels[train_id],
                "pixel_share": (
                    class_pixels[train_id] / total_pixels if total_pixels else 0.0
                ),
                "masks_with_any": class_masks[train_id],
                "qualified_masks": qualified,
                "sequences_with_qualified_mask": sequence_count,
                "minimum_pixels_per_mask": item["minimum_pixels_per_mask"],
                "required_qualified_masks": item["minimum_qualified_masks"],
                "required_sequences": item["minimum_sequences"],
            }
        )
        if qualified < item["minimum_qualified_masks"]:
            readiness_failures.append(
                f"class {item['name']}: qualified_masks={qualified} "
                f"< {item['minimum_qualified_masks']}"
            )
        if sequence_count < item["minimum_sequences"]:
            readiness_failures.append(
                f"class {item['name']}: sequences={sequence_count} "
                f"< {item['minimum_sequences']}"
            )

    global_readiness = config["dataset_readiness"]
    observed = {
        "sequences": len(routes),
        "towns": len(towns),
        "logical_frames": logical_frames,
    }
    for observed_name, required_name in (
        ("sequences", "minimum_sequences"),
        ("towns", "minimum_towns"),
        ("logical_frames", "minimum_logical_frames"),
    ):
        if observed[observed_name] < global_readiness[required_name]:
            readiness_failures.append(
                f"dataset {observed_name}={observed[observed_name]} "
                f"< {global_readiness[required_name]}"
            )
    if errors:
        readiness_failures.append(f"structural audit has {len(errors)} error(s)")

    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "dataset_root": str(root.resolve()),
        "class_config": str(config["path"]),
        "class_config_sha256": config["sha256"],
        "cameras": list(cameras),
        "sequence_selection": {
            "source": "dataset_index" if index else "filesystem_discovery",
            "dataset_index": str(index["path"]) if index else None,
            "dataset_index_sha256": index["sha256"] if index else None,
            "available_sequence_count": len(index["entries"]) if index else len(routes),
            "sample_per_town_weather": sample_per_town_weather,
            "sample_seed": sample_seed if sample_per_town_weather is not None else None,
            "selected_sequences": [
                {
                    "path": item["relative_path"],
                    "declared_frames": item["declared_frames"],
                    "town": item["town"],
                    "weather": item["weather"],
                }
                for item in indexed_entries
            ],
        },
        "sequence_count": len(routes),
        "towns": sorted(towns),
        "logical_frame_count": logical_frames,
        "semantic_mask_count": sum(camera_masks.values()),
        "semantic_masks_by_camera": dict(sorted(camera_masks.items())),
        "total_pixels": total_pixels,
        "raw_labels": raw_stats,
        "classes": class_stats,
        "valid": not errors,
        "errors": errors,
        "readiness": {
            "ready": not readiness_failures,
            "failures": readiness_failures,
            "requirements": global_readiness,
        },
    }


def _parse_cameras(value):
    cameras = tuple(item.strip() for item in value.split(",") if item.strip())
    if not cameras:
        raise argparse.ArgumentTypeError("at least one camera is required")
    return cameras


def _positive_int_arg(value):
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit CARLA semantic data for traffic-domain pretraining"
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-index", type=Path)
    parser.add_argument("--sample-per-town-weather", type=_positive_int_arg)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument(
        "--cameras", type=_parse_cameras, default=DEFAULT_CAMERAS, metavar="LIST"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args(argv)

    try:
        summary = audit_semantic_dataset(
            args.dataset_root,
            config_path=args.config,
            cameras=args.cameras,
            index_path=args.dataset_index,
            sample_per_town_weather=args.sample_per_town_weather,
            sample_seed=args.sample_seed,
        )
    except AuditError as exc:
        print(f"audit error: {exc}", file=sys.stderr)
        return 2

    serialized = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    if not summary["valid"]:
        return 2
    if args.require_ready and not summary["readiness"]["ready"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
