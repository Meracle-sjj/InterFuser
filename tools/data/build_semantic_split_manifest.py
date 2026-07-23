#!/usr/bin/env python3
"""
[INPUT]: 依赖 M1 pilot 审计 JSON、语义类别与 split 配置，以及 pilot 选中 sequence 的三相机 RGB/语义文件。
[OUTPUT]: 对外提供 SplitError、load_split_config、build_split_manifest 与 CLI，生成按 Town+route 分组、无 sequence 泄漏且带内容哈希的确定性 split manifest。
[POS]: tools/data 的 M1 数据划分准入器；复用审计器类别契约，把已通过 pilot 的样本冻结为训练/验证/测试集合，不修改源数据。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import argparse
import hashlib
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data.audit_semantic_pretraining_data import (  # noqa: E402
    AuditError,
    load_class_config,
)


SPLIT_CONFIG_SCHEMA_VERSION = 1
SPLIT_MANIFEST_SCHEMA_VERSION = 1
DEFAULT_CAMERAS = ("front", "left", "right")
DEFAULT_CLASS_CONFIG = REPO_ROOT / "configs" / "thesis" / "semantic_classes_v1.json"
DEFAULT_SPLIT_CONFIG = REPO_ROOT / "configs" / "thesis" / "semantic_split_v1.json"
ROUTE_PATTERN = re.compile(r"(?:^|_)route(\d+)(?:_|$)", re.IGNORECASE)


class SplitError(ValueError):
    """Raised when the split contract or pilot provenance is unusable."""


def _read_json(path, label):
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SplitError(f"unable to read {label} JSON {path}: {exc}") from exc


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split_config(path):
    """Load the deterministic route-group assignment policy."""
    path = Path(path)
    raw = _read_json(path, "split config")
    if not isinstance(raw, dict):
        raise SplitError("split config must be a JSON object")
    if raw.get("schema_version") != SPLIT_CONFIG_SCHEMA_VERSION:
        raise SplitError(
            f"unsupported split config schema_version: {raw.get('schema_version')}"
        )
    seed = raw.get("assignment_seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise SplitError("assignment_seed must be an integer")
    if raw.get("split_unit") != "town_route":
        raise SplitError("split_unit must be town_route")
    splits = raw.get("splits")
    if not isinstance(splits, dict) or set(splits) != {
        "train",
        "validation",
        "test",
    }:
        raise SplitError("splits must define train, validation, and test")
    normalized_splits = {}
    for name, ratio in splits.items():
        if not isinstance(ratio, (int, float)) or isinstance(ratio, bool) or ratio <= 0:
            raise SplitError(f"splits.{name} must be positive")
        normalized_splits[name] = float(ratio)
    if abs(sum(normalized_splits.values()) - 1.0) > 1e-9:
        raise SplitError("split ratios must sum to 1")
    excluded = raw.get("excluded_classes")
    if not isinstance(excluded, list) or any(
        not isinstance(name, str) or not name for name in excluded
    ):
        raise SplitError("excluded_classes must be a list of class names")
    normalized = dict(raw)
    normalized["path"] = path.resolve()
    normalized["sha256"] = _sha256_file(path)
    normalized["splits"] = normalized_splits
    normalized["excluded_classes"] = tuple(excluded)
    for field in (
        "minimum_towns_per_split",
        "minimum_sequences_per_class_per_split",
    ):
        value = raw.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise SplitError(f"{field} must be a positive integer")
    deviation = raw.get("maximum_sequence_ratio_deviation")
    if (
        not isinstance(deviation, (int, float))
        or isinstance(deviation, bool)
        or not 0 <= deviation < 1
    ):
        raise SplitError("maximum_sequence_ratio_deviation must be in [0, 1)")
    normalized["maximum_sequence_ratio_deviation"] = float(deviation)
    return normalized


def _load_pilot_report(path):
    path = Path(path)
    raw = _read_json(path, "pilot report")
    if not raw.get("valid") or not raw.get("readiness", {}).get("ready"):
        raise SplitError("pilot report must be valid and ready")
    selection = raw.get("sequence_selection")
    if not isinstance(selection, dict) or not selection.get("selected_sequences"):
        raise SplitError("pilot report has no selected_sequences")
    return raw, path.resolve(), _sha256_file(path)


def _frame_paths(directory, suffixes):
    if not directory.is_dir():
        return {}
    suffixes = {suffix.lower() for suffix in suffixes}
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    }


def _update_content_digest(digest, relative_path, content):
    encoded = relative_path.as_posix().encode("utf-8")
    digest.update(len(encoded).to_bytes(4, "big"))
    digest.update(encoded)
    digest.update(len(content).to_bytes(8, "big"))
    digest.update(content)


def _route_id(relative_path):
    parts = Path(relative_path).parts
    if len(parts) < 3:
        raise SplitError(f"sequence path has no collection component: {relative_path}")
    match = ROUTE_PATTERN.search(parts[1])
    if not match:
        raise SplitError(f"unable to infer route ID from {relative_path}")
    return int(match.group(1))


def _scan_sequence(root, item, cameras, classes):
    relative_path = item.get("path")
    town = item.get("town")
    weather = item.get("weather")
    declared_frames = item.get("declared_frames")
    if not isinstance(relative_path, str) or not relative_path:
        raise SplitError("selected sequence path must be non-empty")
    if not isinstance(town, str) or not town:
        raise SplitError(f"selected sequence has invalid Town: {relative_path}")
    if not isinstance(weather, int) or isinstance(weather, bool):
        raise SplitError(f"selected sequence has invalid weather: {relative_path}")
    if not isinstance(declared_frames, int) or declared_frames <= 0:
        raise SplitError(f"selected sequence has invalid frame count: {relative_path}")
    sequence = (root / relative_path).resolve()
    try:
        sequence.relative_to(root)
    except ValueError as exc:
        raise SplitError(f"selected sequence escapes dataset root: {relative_path}") from exc
    if not sequence.is_dir():
        raise SplitError(f"selected sequence is not a directory: {relative_path}")

    rgb_digest = hashlib.sha256()
    semantic_digest = hashlib.sha256()
    class_pixels = Counter()
    class_qualified_masks = Counter()
    review_candidates = {}
    camera_stats = {}
    errors = []
    reference_frames = None

    for camera in cameras:
        rgb_paths = _frame_paths(sequence / f"rgb_{camera}", {".jpg", ".jpeg", ".png"})
        seg_paths = _frame_paths(sequence / f"seg_{camera}", {".png"})
        rgb_frames = set(rgb_paths)
        seg_frames = set(seg_paths)
        if rgb_frames != seg_frames:
            missing_rgb = sorted(seg_frames - rgb_frames)
            missing_seg = sorted(rgb_frames - seg_frames)
            if missing_rgb:
                errors.append(f"{relative_path}/seg_{camera}: frames without RGB {missing_rgb}")
            if missing_seg:
                errors.append(f"{relative_path}/rgb_{camera}: frames without semantic mask {missing_seg}")
        if reference_frames is None:
            reference_frames = seg_frames
        elif seg_frames != reference_frames:
            errors.append(
                f"{relative_path}: seg_{camera} frame IDs differ from seg_{cameras[0]}"
            )
        for frame_id in sorted(rgb_frames & seg_frames):
            rgb_path = rgb_paths[frame_id]
            seg_path = seg_paths[frame_id]
            try:
                rgb_content = rgb_path.read_bytes()
                seg_content = seg_path.read_bytes()
                with Image.open(io.BytesIO(rgb_content)) as image:
                    rgb_size = image.size
                with Image.open(io.BytesIO(seg_content)) as image:
                    labels = np.asarray(image)
            except (OSError, ValueError) as exc:
                errors.append(f"{relative_path}/{camera}/{frame_id}: unreadable image: {exc}")
                continue
            _update_content_digest(
                rgb_digest, rgb_path.relative_to(sequence), rgb_content
            )
            _update_content_digest(
                semantic_digest, seg_path.relative_to(sequence), seg_content
            )
            if labels.ndim != 2 or labels.dtype.kind not in "ui":
                errors.append(
                    f"{relative_path}/{seg_path.name}: semantic mask must be 2D integer"
                )
                continue
            if rgb_size != (labels.shape[1], labels.shape[0]):
                errors.append(
                    f"{relative_path}/{camera}/{frame_id}: RGB/mask dimensions differ"
                )
                continue
            values, counts = np.unique(labels, return_counts=True)
            per_tag = {
                int(value): int(count)
                for value, count in zip(values.tolist(), counts.tolist())
            }
            for class_item in classes:
                name = class_item["name"]
                pixels = sum(per_tag.get(tag, 0) for tag in class_item["source_tags"])
                class_pixels[name] += pixels
                if pixels >= class_item["minimum_pixels_per_mask"]:
                    class_qualified_masks[name] += 1
                    candidate = {
                        "sequence_path": relative_path,
                        "camera": camera,
                        "frame_id": frame_id,
                        "class_pixels": pixels,
                    }
                    current = review_candidates.get(name)
                    rank = (-pixels, camera, frame_id, relative_path)
                    if current is None or rank < current[0]:
                        review_candidates[name] = (rank, candidate)
        camera_stats[camera] = {
            "rgb_frames": len(rgb_frames),
            "semantic_frames": len(seg_frames),
        }

    logical_frames = len(reference_frames or set())
    if logical_frames != declared_frames:
        errors.append(
            f"{relative_path}: pilot declares {declared_frames} frames but split scan found {logical_frames}"
        )
    route_id = _route_id(relative_path)
    parts = Path(relative_path).parts
    return {
        "path": relative_path,
        "collection_id": parts[1],
        "sequence_id": parts[-1],
        "town": town,
        "route_id": route_id,
        "weather": weather,
        "declared_frames": declared_frames,
        "camera_frames": camera_stats,
        "class_pixels": dict(sorted(class_pixels.items())),
        "qualified_masks_by_class": dict(sorted(class_qualified_masks.items())),
        "content_sha256": {
            "rgb": rgb_digest.hexdigest(),
            "semantic": semantic_digest.hexdigest(),
        },
        "review_candidates": {
            name: value[1] for name, value in sorted(review_candidates.items())
        },
    }, errors


def _group_vector(records, core_classes):
    vector = Counter(
        groups=1,
        sequences=len(records),
        frames=sum(item["declared_frames"] for item in records),
    )
    for name in core_classes:
        vector[f"class:{name}"] = sum(
            bool(item["qualified_masks_by_class"].get(name)) for item in records
        )
    return vector


def _assign_route_groups(records, core_classes, config):
    groups = defaultdict(list)
    for item in records:
        key = f"{item['town']}:route{item['route_id']:03d}"
        groups[key].append(item)
    vectors = {key: _group_vector(items, core_classes) for key, items in groups.items()}
    totals = Counter(
        groups=len(groups),
        sequences=len(records),
        frames=sum(item["declared_frames"] for item in records),
    )
    for name in core_classes:
        totals[f"class:{name}"] = sum(
            bool(item["qualified_masks_by_class"].get(name)) for item in records
        )
    seed = config["assignment_seed"]
    splits = config["splits"]
    state = {name: Counter() for name in splits}
    assignment = {}
    reasons = {}

    for split in sorted(splits, key=lambda name: (splits[name], name)):
        missing = set(core_classes)
        while missing:
            candidates = []
            for group, vector in vectors.items():
                if group in assignment:
                    continue
                covered = [name for name in missing if vector[f"class:{name}"]]
                if not covered:
                    continue
                rarity = sum(
                    vector[f"class:{name}"] / totals[f"class:{name}"]
                    for name in covered
                )
                total_cover = sum(
                    bool(vector[f"class:{name}"]) for name in core_classes
                )
                tie = hashlib.sha256(
                    f"{seed}:{split}:{group}".encode("utf-8")
                ).hexdigest()
                candidates.append((-len(covered), -rarity, -total_cover, tie, group))
            if not candidates:
                raise SplitError(
                    f"cannot cover core classes in {split}: {sorted(missing)}"
                )
            group = min(candidates)[-1]
            assignment[group] = split
            reasons[group] = "core_class_coverage_seed"
            state[split].update(vectors[group])
            missing = {
                name for name in missing if not state[split][f"class:{name}"]
            }

    rarity = {
        group: sum(
            vector[f"class:{name}"] / totals[f"class:{name}"]
            for name in core_classes
        )
        for group, vector in vectors.items()
    }
    remaining = sorted(
        (group for group in groups if group not in assignment),
        key=lambda group: (
            -rarity[group],
            hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest(),
            group,
        ),
    )
    weights = {key: 4.0 if key.startswith("class:") else 1.0 for key in totals}
    weights["frames"] = 2.0
    for group in remaining:
        choices = []
        for split in splits:
            score = 0.0
            for key, total in totals.items():
                if not total:
                    continue
                for candidate, target_ratio in splits.items():
                    value = state[candidate][key]
                    if candidate == split:
                        value += vectors[group][key]
                    score += weights[key] * ((value / total) - target_ratio) ** 2
            tie = hashlib.sha256(
                f"{seed}:{group}:{split}".encode("utf-8")
            ).hexdigest()
            choices.append((score, tie, split))
        split = min(choices)[-1]
        assignment[group] = split
        reasons[group] = "deterministic_distribution_balance"
        state[split].update(vectors[group])
    return groups, assignment, reasons, state, totals


def build_split_manifest(
    dataset_root,
    pilot_report_path,
    class_config_path=DEFAULT_CLASS_CONFIG,
    split_config_path=DEFAULT_SPLIT_CONFIG,
    cameras=DEFAULT_CAMERAS,
):
    """Scan the frozen pilot selection and return a deterministic split manifest."""
    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise SplitError(f"dataset root is not a directory: {root}")
    cameras = tuple(cameras)
    if not cameras or len(set(cameras)) != len(cameras):
        raise SplitError("cameras must be a non-empty list without duplicates")
    split_config = load_split_config(split_config_path)
    try:
        class_config = load_class_config(class_config_path)
    except AuditError as exc:
        raise SplitError(str(exc)) from exc
    pilot, pilot_path, pilot_sha256 = _load_pilot_report(pilot_report_path)
    pilot_root = Path(pilot.get("dataset_root", "")).resolve()
    if pilot_root != root:
        raise SplitError(
            f"pilot dataset root {pilot_root} differs from requested root {root}"
        )
    if pilot.get("class_config_sha256") != class_config["sha256"]:
        raise SplitError("pilot report and class config SHA-256 differ")
    pilot_cameras = tuple(pilot.get("cameras", ()))
    if pilot_cameras != cameras:
        raise SplitError(
            f"pilot cameras {pilot_cameras} differ from requested cameras {cameras}"
        )
    excluded = set(split_config["excluded_classes"])
    known_names = {item["name"] for item in class_config["classes"]}
    unknown_excluded = sorted(excluded - known_names)
    if unknown_excluded:
        raise SplitError(f"excluded_classes are unknown: {unknown_excluded}")
    core_classes = [
        item["name"]
        for item in class_config["classes"]
        if item["name"] not in excluded
    ]
    records = []
    errors = []
    global_candidates = {}
    for selected in pilot["sequence_selection"]["selected_sequences"]:
        record, sequence_errors = _scan_sequence(
            root, selected, cameras, class_config["classes"]
        )
        records.append(record)
        errors.extend(sequence_errors)
        for name, candidate in record.pop("review_candidates").items():
            rank = (
                -candidate["class_pixels"],
                candidate["sequence_path"],
                candidate["camera"],
                candidate["frame_id"],
            )
            current = global_candidates.get(name)
            if current is None or rank < current[0]:
                global_candidates[name] = (rank, candidate)

    sequence_paths = [item["path"] for item in records]
    unique_sequence_paths = set(sequence_paths)
    duplicate_sequence_count = len(sequence_paths) - len(unique_sequence_paths)
    if duplicate_sequence_count:
        errors.append(
            f"pilot selection contains {duplicate_sequence_count} duplicate sequence(s)"
        )

    groups, assignment, reasons, state, totals = _assign_route_groups(
        records, core_classes, split_config
    )
    for record in records:
        group = f"{record['town']}:route{record['route_id']:03d}"
        record["route_group"] = group
        record["split"] = assignment[group]
    split_summary = {}
    for split, target_ratio in split_config["splits"].items():
        selected_records = [item for item in records if item["split"] == split]
        towns = sorted({item["town"] for item in selected_records})
        sequence_ratio = len(selected_records) / len(records)
        class_sequences = {
            name: state[split][f"class:{name}"] for name in core_classes
        }
        if len(towns) < split_config["minimum_towns_per_split"]:
            errors.append(
                f"split {split}: towns={len(towns)} < {split_config['minimum_towns_per_split']}"
            )
        for name, count in class_sequences.items():
            minimum = split_config["minimum_sequences_per_class_per_split"]
            if count < minimum:
                errors.append(f"split {split}: class {name} sequences={count} < {minimum}")
        deviation = abs(sequence_ratio - target_ratio)
        if deviation > split_config["maximum_sequence_ratio_deviation"]:
            errors.append(
                f"split {split}: sequence ratio deviation {deviation:.6f} exceeds "
                f"{split_config['maximum_sequence_ratio_deviation']:.6f}"
            )
        split_summary[split] = {
            "target_ratio": target_ratio,
            "route_groups": state[split]["groups"],
            "sequences": len(selected_records),
            "sequence_ratio": sequence_ratio,
            "logical_frames": state[split]["frames"],
            "towns": towns,
            "weather_strata": sorted(
                {f"{item['town']}:weather{item['weather']}" for item in selected_records}
            ),
            "sequences_with_qualified_class": class_sequences,
        }
    route_groups = [
        {
            "route_group": group,
            "split": assignment[group],
            "assignment_reason": reasons[group],
            "sequence_paths": sorted(item["path"] for item in groups[group]),
        }
        for group in sorted(groups)
    ]
    records.sort(key=lambda item: item["path"])
    return {
        "split_manifest_schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "valid": not errors,
        "errors": errors,
        "dataset_root": str(root),
        "cameras": list(cameras),
        "source": {
            "pilot_report": str(pilot_path),
            "pilot_report_sha256": pilot_sha256,
            "dataset_index": pilot["sequence_selection"].get("dataset_index"),
            "dataset_index_sha256": pilot["sequence_selection"].get(
                "dataset_index_sha256"
            ),
            "class_config": str(class_config["path"]),
            "class_config_sha256": class_config["sha256"],
            "split_config": str(split_config["path"]),
            "split_config_sha256": split_config["sha256"],
        },
        "policy": {
            "assignment_seed": split_config["assignment_seed"],
            "split_unit": split_config["split_unit"],
            "target_ratios": split_config["splits"],
            "core_classes": core_classes,
            "minimum_towns_per_split": split_config["minimum_towns_per_split"],
            "minimum_sequences_per_class_per_split": split_config[
                "minimum_sequences_per_class_per_split"
            ],
            "maximum_sequence_ratio_deviation": split_config[
                "maximum_sequence_ratio_deviation"
            ],
        },
        "summary": {
            "route_groups": totals["groups"],
            "sequences": totals["sequences"],
            "logical_frames": totals["frames"],
            "splits": split_summary,
        },
        "leakage_check": {
            "sequence_overlap_count": duplicate_sequence_count,
            "route_group_overlap_count": 0,
            "all_selected_sequences_assigned_once": not duplicate_sequence_count,
        },
        "review_candidates": {
            name: global_candidates[name][1]
            for name in core_classes
            if name in global_candidates
        },
        "route_groups": route_groups,
        "sequences": records,
    }


def _parse_cameras(value):
    cameras = tuple(item.strip() for item in value.split(",") if item.strip())
    if not cameras:
        raise argparse.ArgumentTypeError("at least one camera is required")
    return cameras


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a leakage-safe semantic pretraining split manifest"
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("pilot_report", type=Path)
    parser.add_argument("--class-config", type=Path, default=DEFAULT_CLASS_CONFIG)
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument(
        "--cameras", type=_parse_cameras, default=DEFAULT_CAMERAS, metavar="LIST"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        print(f"split error: refusing to overwrite {args.output}", file=sys.stderr)
        return 2
    try:
        manifest = build_split_manifest(
            args.dataset_root,
            args.pilot_report,
            class_config_path=args.class_config,
            split_config_path=args.split_config,
            cameras=args.cameras,
        )
    except SplitError as exc:
        print(f"split error: {exc}", file=sys.stderr)
        return 2
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    except FileExistsError:
        print(f"split error: refusing to overwrite {args.output}", file=sys.stderr)
        return 2
    print(serialized, end="")
    return 0 if manifest["valid"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
