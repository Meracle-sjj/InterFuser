#!/usr/bin/env python3
"""
[INPUT]: 依赖冻结 M1 split、行人碰撞威胁可见性审计、语义类别配置与原始三相机 RGB/mask sequence。
[OUTPUT]: 提供 build_pedestrian_hazard_training_manifest 与 CLI，产生仅向 train 增加行人威胁 sequence、保持原 validation/test 不变并预留 route-group 隔离专项 holdout 的内容哈希 manifest。
[POS]: tools/data 的 M1.1 行人危险数据准入器；将特权碰撞真值只用于选样与分层，不把 birdview/测量字段泄漏到模型输入。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data.audit_semantic_pretraining_data import (  # noqa: E402
    AuditError,
    load_class_config,
)
from tools.data.build_semantic_split_manifest import (  # noqa: E402
    DEFAULT_CAMERAS,
    SplitError,
    scan_semantic_sequence,
)


MANIFEST_SCHEMA_VERSION = 1
DEFAULT_CLASS_CONFIG = REPO_ROOT / "configs" / "thesis" / "semantic_classes_v1.json"


class PedestrianHazardManifestError(ValueError):
    """Raised when hazard augmentation would break provenance or route isolation."""


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path, label):
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PedestrianHazardManifestError(
            f"unable to read {label} JSON {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise PedestrianHazardManifestError(f"{label} must be a JSON object")
    return value, path.resolve(), _sha256(path)


def _group_assignment(base):
    assignment = {}
    for item in base.get("route_groups", []):
        group = item.get("route_group")
        split = item.get("split")
        if not isinstance(group, str) or split not in {
            "train",
            "validation",
            "test",
        }:
            raise PedestrianHazardManifestError(
                "base route_groups contains an invalid assignment"
            )
        if group in assignment:
            raise PedestrianHazardManifestError(
                f"base route_groups duplicates {group}"
            )
        assignment[group] = split
    return assignment


def _candidate_rank(item, seed):
    camera_stats = item.get("camera_stats", {})
    qualified_masks = sum(
        int(value.get("qualified_masks", 0))
        for value in camera_stats.values()
        if isinstance(value, dict)
    )
    pixels = sum(
        int(value.get("pixels", 0))
        for value in camera_stats.values()
        if isinstance(value, dict)
    )
    sequence = item["sequence"]
    tie = hashlib.sha256(f"{seed}:{sequence}".encode("utf-8")).hexdigest()
    return (
        -int(item.get("hazard_visible_frames", 0)),
        -qualified_masks,
        -pixels,
        tie,
        sequence,
    )


def _select_one_per_group(candidates, seed):
    selected = {}
    for group, items in candidates.items():
        selected[group] = min(items, key=lambda item: _candidate_rank(item, seed))
    return selected


def _assign_unseen_groups(groups, seed, holdout_ratio):
    by_town = defaultdict(list)
    for group in groups:
        town, separator, _ = group.partition(":")
        if not separator:
            raise PedestrianHazardManifestError(f"invalid Town+route group: {group}")
        by_town[town].append(group)
    assignment = {}
    for town, town_groups in sorted(by_town.items()):
        if len(town_groups) < 3:
            raise PedestrianHazardManifestError(
                f"unseen hazard groups for {town} must contain at least three groups"
            )
        ordered = sorted(
            town_groups,
            key=lambda group: (
                hashlib.sha256(f"{seed}:{town}:{group}".encode("utf-8")).hexdigest(),
                group,
            ),
        )
        validation_count = max(1, round(len(ordered) * holdout_ratio))
        test_count = max(1, round(len(ordered) * holdout_ratio))
        if validation_count + test_count >= len(ordered):
            validation_count = 1
            test_count = 1
        for group in ordered[:validation_count]:
            assignment[group] = "hazard_validation"
        for group in ordered[validation_count : validation_count + test_count]:
            assignment[group] = "hazard_test"
        for group in ordered[validation_count + test_count :]:
            assignment[group] = "augmentation_train"
    return assignment


def _scan_selected(root, selected, cameras, classes, split):
    item = {
        "path": selected["sequence"],
        "town": selected["town"],
        "weather": selected["weather"],
        "declared_frames": selected["semantic_logical_frames"],
    }
    try:
        record, errors = scan_semantic_sequence(root, item, cameras, classes)
    except SplitError as exc:
        raise PedestrianHazardManifestError(str(exc)) from exc
    if errors:
        raise PedestrianHazardManifestError(
            f"semantic scan failed for {selected['sequence']}: {errors}"
        )
    record.pop("review_candidates", None)
    record["route_group"] = selected["town_route_group"]
    record["split"] = split
    record["hazard_provenance"] = {
        "has_privileged_collision_hazard": True,
        "hazard_frames": selected["hazard_frames"],
        "hazard_visible_frames": selected["hazard_visible_frames"],
        "hazard_frame_visibility_ratio": selected["hazard_frame_visibility_ratio"],
        "trajectory_visibility_signature": selected[
            "trajectory_visibility_signature"
        ],
    }
    return record


def _split_summary(sequences, cameras):
    summary = {}
    for split in ("train", "validation", "test"):
        selected = [item for item in sequences if item["split"] == split]
        logical_frames = sum(item["declared_frames"] for item in selected)
        summary[split] = {
            "sequences": len(selected),
            "route_groups": len({item["route_group"] for item in selected}),
            "logical_frames": logical_frames,
            "camera_samples": logical_frames * len(cameras),
            "towns": sorted({item["town"] for item in selected}),
            "pedestrian_hazard_additions": sum(
                "hazard_provenance" in item for item in selected
            ),
        }
    return summary


def build_pedestrian_hazard_training_manifest(
    base_manifest_path,
    hazard_audit_path,
    class_config_path=DEFAULT_CLASS_CONFIG,
    selection_seed=20260804,
    holdout_ratio=0.15,
    cameras=DEFAULT_CAMERAS,
):
    """Build a train-only hazard augmentation plus isolated targeted holdouts."""
    if not isinstance(selection_seed, int) or isinstance(selection_seed, bool):
        raise PedestrianHazardManifestError("selection_seed must be an integer")
    if not 0 < holdout_ratio < 0.5:
        raise PedestrianHazardManifestError("holdout_ratio must be in (0, 0.5)")
    cameras = tuple(cameras)
    base, base_path, base_sha256 = _read_json(base_manifest_path, "base manifest")
    audit, audit_path, audit_sha256 = _read_json(hazard_audit_path, "hazard audit")
    if not base.get("valid"):
        raise PedestrianHazardManifestError("base manifest must be valid")
    if not audit.get("valid"):
        raise PedestrianHazardManifestError("hazard audit must be valid")
    if tuple(base.get("cameras", ())) != cameras:
        raise PedestrianHazardManifestError("base manifest cameras differ")
    if tuple(audit.get("cameras", ())) != cameras:
        raise PedestrianHazardManifestError("hazard audit cameras differ")
    root = Path(base.get("dataset_root", "")).resolve()
    if root != Path(audit.get("dataset_root", "")).resolve() or not root.is_dir():
        raise PedestrianHazardManifestError(
            "base and hazard audit must reference the same existing dataset root"
        )
    try:
        class_config = load_class_config(class_config_path)
    except AuditError as exc:
        raise PedestrianHazardManifestError(str(exc)) from exc
    if base.get("source", {}).get("class_config_sha256") != class_config["sha256"]:
        raise PedestrianHazardManifestError(
            "base manifest and class config SHA-256 differ"
        )
    if not audit.get("summary", {}).get(
        "all_sequences_have_privileged_collision_hazard"
    ) or not audit.get("summary", {}).get("all_sequences_have_visible_hazard_frame"):
        raise PedestrianHazardManifestError(
            "hazard audit must gate every sequence on collision hazard and visibility"
        )

    base_sequences = [dict(item) for item in base.get("sequences", [])]
    if not base_sequences:
        raise PedestrianHazardManifestError("base manifest has no sequences")
    existing_paths = {item["path"] for item in base_sequences}
    base_assignment = _group_assignment(base)
    candidates = defaultdict(list)
    for item in audit.get("sequences", []):
        sequence = item.get("sequence")
        group = item.get("town_route_group")
        if (
            item.get("valid")
            and item.get("has_privileged_collision_hazard")
            and item.get("has_visible_hazard_frame")
            and isinstance(sequence, str)
            and sequence not in existing_paths
            and isinstance(group, str)
        ):
            candidates[group].append(item)
    if not candidates:
        raise PedestrianHazardManifestError(
            "hazard audit provides no non-base candidate sequences"
        )
    selected_by_group = _select_one_per_group(candidates, selection_seed)
    unseen_groups = sorted(set(selected_by_group) - set(base_assignment))
    unseen_assignment = _assign_unseen_groups(
        unseen_groups, selection_seed, holdout_ratio
    )

    target_by_group = {}
    for group in selected_by_group:
        split = base_assignment.get(group)
        if split == "train":
            target_by_group[group] = "augmentation_train"
        elif split == "validation":
            target_by_group[group] = "hazard_validation"
        elif split == "test":
            target_by_group[group] = "hazard_test"
        else:
            target_by_group[group] = unseen_assignment[group]

    scanned = {}
    for group, selected in sorted(selected_by_group.items()):
        target = target_by_group[group]
        split = "train" if target == "augmentation_train" else target
        scanned[group] = _scan_selected(
            root, selected, cameras, class_config["classes"], split
        )
    train_additions = [
        scanned[group]
        for group in sorted(scanned)
        if target_by_group[group] == "augmentation_train"
    ]
    holdout_validation = [
        scanned[group]
        for group in sorted(scanned)
        if target_by_group[group] == "hazard_validation"
    ]
    holdout_test = [
        scanned[group]
        for group in sorted(scanned)
        if target_by_group[group] == "hazard_test"
    ]
    if not train_additions or not holdout_validation or not holdout_test:
        raise PedestrianHazardManifestError(
            "augmentation must produce train, hazard_validation, and hazard_test records"
        )
    train_groups = {item["route_group"] for item in train_additions}
    holdout_groups = {
        item["route_group"] for item in holdout_validation + holdout_test
    }
    if train_groups & holdout_groups:
        raise PedestrianHazardManifestError(
            "hazard train and holdout route groups overlap"
        )

    sequences = sorted(base_sequences + train_additions, key=lambda item: item["path"])
    route_groups = {item["route_group"]: dict(item) for item in base["route_groups"]}
    for item in train_additions:
        group = item["route_group"]
        if group in route_groups:
            paths = set(route_groups[group].get("sequence_paths", []))
            paths.add(item["path"])
            route_groups[group]["sequence_paths"] = sorted(paths)
        else:
            route_groups[group] = {
                "route_group": group,
                "split": "train",
                "assignment_reason": "pedestrian_hazard_augmentation_train",
                "sequence_paths": [item["path"]],
            }
    split_summary = _split_summary(sequences, cameras)
    base_split_summary = _split_summary(base_sequences, cameras)
    errors = []
    for split in ("validation", "test"):
        if split_summary[split] != base_split_summary[split]:
            errors.append(f"base {split} split changed during train-only augmentation")

    return {
        "split_manifest_schema_version": base.get(
            "split_manifest_schema_version", MANIFEST_SCHEMA_VERSION
        ),
        "augmentation_schema_version": MANIFEST_SCHEMA_VERSION,
        "valid": not errors,
        "errors": errors,
        "dataset_root": str(root),
        "cameras": list(cameras),
        "source": {
            **base["source"],
            "base_split_manifest": str(base_path),
            "base_split_manifest_sha256": base_sha256,
            "pedestrian_hazard_audit": str(audit_path),
            "pedestrian_hazard_audit_sha256": audit_sha256,
        },
        "policy": {
            **base.get("policy", {}),
            "augmentation": {
                "selection_seed": selection_seed,
                "selection_unit": "one_sequence_per_town_route_group",
                "selection_priority": [
                    "hazard_visible_frames_desc",
                    "qualified_pedestrian_masks_desc",
                    "pedestrian_pixels_desc",
                    "seeded_sha256_tie_break",
                ],
                "unseen_group_holdout_ratio_per_split": holdout_ratio,
                "base_validation_test_immutable": True,
                "privileged_fields_used_only_for_selection": [
                    "is_pedestrian_present",
                    "semantic_source_tag_12",
                ],
                "model_inputs": ["rgb_front", "rgb_left", "rgb_right"],
            },
        },
        "summary": {
            "route_groups": len(route_groups),
            "sequences": len(sequences),
            "logical_frames": sum(item["declared_frames"] for item in sequences),
            "splits": split_summary,
            "augmentation": {
                "candidate_route_groups": len(selected_by_group),
                "train_sequences_added": len(train_additions),
                "train_logical_frames_added": sum(
                    item["declared_frames"] for item in train_additions
                ),
                "train_camera_samples_added": sum(
                    item["declared_frames"] for item in train_additions
                )
                * len(cameras),
                "hazard_validation_route_groups": len(holdout_validation),
                "hazard_test_route_groups": len(holdout_test),
            },
        },
        "leakage_check": {
            "sequence_overlap_count": 0,
            "route_group_overlap_count": 0,
            "hazard_train_holdout_route_group_overlap_count": len(
                train_groups & holdout_groups
            ),
            "base_validation_test_unchanged": not errors,
        },
        "review_candidates": base.get("review_candidates", {}),
        "route_groups": [route_groups[group] for group in sorted(route_groups)],
        "sequences": sequences,
        "hazard_holdout": {
            "validation": holdout_validation,
            "test": holdout_test,
        },
    }


def _parse_cameras(value):
    cameras = tuple(item.strip() for item in value.split(",") if item.strip())
    if not cameras:
        raise argparse.ArgumentTypeError("at least one camera is required")
    return cameras


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build train-only pedestrian hazard semantic augmentation"
    )
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--hazard-audit", type=Path, required=True)
    parser.add_argument("--class-config", type=Path, default=DEFAULT_CLASS_CONFIG)
    parser.add_argument("--selection-seed", type=int, default=20260804)
    parser.add_argument("--holdout-ratio", type=float, default=0.15)
    parser.add_argument(
        "--cameras", type=_parse_cameras, default=DEFAULT_CAMERAS, metavar="LIST"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        print(f"manifest error: refusing to overwrite {args.output}", file=sys.stderr)
        return 2
    try:
        manifest = build_pedestrian_hazard_training_manifest(
            args.base_manifest,
            args.hazard_audit,
            class_config_path=args.class_config,
            selection_seed=args.selection_seed,
            holdout_ratio=args.holdout_ratio,
            cameras=args.cameras,
        )
    except PedestrianHazardManifestError as exc:
        print(f"manifest error: {exc}", file=sys.stderr)
        return 2
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    except FileExistsError:
        print(f"manifest error: refusing to overwrite {args.output}", file=sys.stderr)
        return 2
    print(serialized, end="")
    return 0 if manifest["valid"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
