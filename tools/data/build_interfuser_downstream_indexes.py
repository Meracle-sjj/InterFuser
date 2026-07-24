#!/usr/bin/env python3
"""
[INPUT]: 依赖冻结 M1 Town+route split manifest、全量 InterFuser dataset_index 与版本化下游划分配置。
[OUTPUT]: 对外提供 DownstreamSplitError、load_downstream_split_config、build_downstream_indexes 与 CLI，生成无 route-group 泄漏的 train/validation/test index 及 manifest。
[POS]: tools/data 的 M2 下游划分扩展器；将 M1 小样本的原子 route-group 归属投影到全量索引，未见 route group 只进入 train。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
MANIFEST_VERSION = 1
TOWN_PATTERN = re.compile(r"town(\d+)", re.IGNORECASE)
ROUTE_PATTERN = re.compile(r"(?:^|_)route(\d+)(?:_|$)", re.IGNORECASE)


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
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise DownstreamSplitError(
            f"unsupported schema_version: {raw.get('schema_version')}"
        )
    if raw.get("status") != "frozen" or raw.get("split_unit") != "town_route":
        raise DownstreamSplitError("config must be frozen with split_unit=town_route")
    expected_policy = {
        "frozen_validation_route_groups": "validation",
        "frozen_test_route_groups": "test",
        "frozen_train_route_groups": "train",
        "unassigned_route_groups": "train",
    }
    if raw.get("expansion_policy") != expected_policy:
        raise DownstreamSplitError("expansion_policy differs from the frozen policy")
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
    splits = {name: [] for name in ("train", "validation", "test")}
    assignment_sources = Counter()
    for item in records:
        frozen_split = assignments.get(item["route_group"])
        split = frozen_split if frozen_split in {"validation", "test"} else "train"
        assignment_sources[
            "frozen_route_group" if frozen_split is not None else "unassigned_to_train"
        ] += 1
        splits[split].append(item)

    full_groups = {item["route_group"] for item in records}
    missing_holdout = sorted(
        group
        for group, split in assignments.items()
        if split in {"validation", "test"} and group not in full_groups
    )
    if missing_holdout:
        raise DownstreamSplitError(
            f"frozen holdout route groups are absent from full index: {missing_holdout}"
        )
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
    return config, splits, {
        "manifest_schema_version": MANIFEST_VERSION,
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
        },
    }


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
