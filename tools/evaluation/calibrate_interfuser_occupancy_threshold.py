#!/usr/bin/env python3
"""
[INPUT]: 依赖 direct scene probe 的 per-sample 分数落盘（traffic_scores/traffic_targets/meta）、冻结划分种子与约束、binary_score_metrics/_threshold_curve 的冻结指标口径。
[OUTPUT]: 对外提供 CalibrationDiagnosticError、split_route_groups、sweep_occupied_iou、run_calibration_diagnostic 与 CLI，生成校准/留出子集与完整 validation 的对称阈值对照 manifest。
[POS]: tools/evaluation 的 v6 纯校准诊断；只消费分数落盘，不加载模型、不读取 test、不产出任何 Stage 2 准入。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPO_ROOT, REPO_ROOT / "interfuser"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tools.evaluation.interfuser_offline_metrics import (  # noqa: E402
    _threshold_curve,
    binary_score_metrics,
)
from tools.evaluation.run_interfuser_scene_validation import sha256_file  # noqa: E402


MANIFEST_SCHEMA_VERSION = 1
VARIANTS = ("b0", "candidate")
FIXED_THRESHOLD = 0.5
FRAME_RATIO_MIN = 0.75
FRAME_RATIO_MAX = 4.0 / 3.0
MAX_SPLIT_RETRIES = 1000


class CalibrationDiagnosticError(RuntimeError):
    """Raised when calibration dump provenance or split constraints are invalid."""


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _read_json(path, label):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationDiagnosticError(f"unable to read {label} JSON {path}: {exc}") from exc


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_variant_dump(dump_dir, variant):
    """Load one variant score dump and verify internal shape alignment."""
    dump_dir = Path(dump_dir)
    scores_path = dump_dir / f"{variant}_traffic_scores.npy"
    targets_path = dump_dir / f"{variant}_traffic_targets.npy"
    meta_path = dump_dir / f"{variant}_meta.json"
    for path, label in (
        (scores_path, "traffic scores"),
        (targets_path, "traffic targets"),
        (meta_path, "metadata"),
    ):
        if not path.is_file():
            raise CalibrationDiagnosticError(f"{variant} {label} missing: {path}")
    scores = np.load(scores_path).astype(np.float64, copy=False)
    targets = np.load(targets_path).astype(np.int64, copy=False)
    metadata = _read_json(meta_path, f"{variant} metadata").get("samples")
    if scores.shape != targets.shape or scores.ndim != 2 or scores.shape[1] != 400:
        raise CalibrationDiagnosticError(f"{variant} dump arrays have unexpected shapes")
    if not isinstance(metadata, list) or len(metadata) != scores.shape[0]:
        raise CalibrationDiagnosticError(f"{variant} metadata count differs from arrays")
    return {
        "scores": scores,
        "targets": targets,
        "metadata": metadata,
        "artifacts": {
            "traffic_scores_sha256": sha256_file(scores_path),
            "traffic_targets_sha256": sha256_file(targets_path),
            "meta_sha256": sha256_file(meta_path),
        },
    }


def _town(route_group):
    return route_group.split(":", 1)[0]


def _check_split_constraints(assignment, frame_counts, pedestrian_groups, towns):
    calibration, evaluation = assignment["calibration"], assignment["evaluation"]
    if {_town(group) for group in calibration} != towns:
        return False
    if {_town(group) for group in evaluation} != towns:
        return False
    half = len(pedestrian_groups) // 2
    if sum(group in calibration for group in pedestrian_groups) != half:
        return False
    frames_a = sum(frame_counts[group] for group in calibration)
    frames_b = sum(frame_counts[group] for group in evaluation)
    if min(frames_a, frames_b) == 0:
        return False
    ratio = frames_a / frames_b
    return FRAME_RATIO_MIN <= ratio <= FRAME_RATIO_MAX


def split_route_groups(metadata, seed):
    """Snake-draft route groups into calibration/evaluation under frozen constraints."""
    frame_counts = {}
    for sample in metadata:
        group = sample["route_group"]
        frame_counts[group] = frame_counts.get(group, 0) + 1
    groups = sorted(frame_counts)
    towns = {_town(group) for group in groups}
    pedestrian_groups = sorted({sample["route_group"] for sample in metadata if sample["pedestrian"]})
    if len(pedestrian_groups) % 2:
        raise CalibrationDiagnosticError("pedestrian route group count must be even")
    for attempt in range(MAX_SPLIT_RETRIES):
        attempt_seed = seed + attempt
        rng = np.random.RandomState(attempt_seed)
        order = list(groups)
        rng.shuffle(order)
        calibration, evaluation = [], []
        for index, group in enumerate(order):
            (calibration if index % 4 in (0, 3) else evaluation).append(group)
        assignment = {
            "calibration": sorted(calibration),
            "evaluation": sorted(evaluation),
        }
        if _check_split_constraints(assignment, frame_counts, pedestrian_groups, towns):
            return {
                **assignment,
                "seed": attempt_seed,
                "retries": attempt,
                "frame_counts": {group: frame_counts[group] for group in groups},
                "pedestrian_route_groups": pedestrian_groups,
            }
    raise CalibrationDiagnosticError("no seed satisfies frozen split constraints")


def sweep_occupied_iou(targets, scores):
    """Exact occupied-IoU curve over distinct-score thresholds, frozen metric口径."""
    targets = np.asarray(targets).reshape(-1).astype(np.int64, copy=False)
    scores = np.asarray(scores).reshape(-1).astype(np.float64, copy=False)
    if targets.shape != scores.shape or targets.size == 0:
        raise CalibrationDiagnosticError("sweep inputs must be non-empty and aligned")
    positives = int(targets.sum())
    if positives == 0 or positives == targets.size:
        raise CalibrationDiagnosticError("sweep requires both target classes")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    true_positives, false_positives = _threshold_curve(targets, scores)
    distinct_ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    thresholds = sorted_scores[distinct_ends]
    false_negatives = positives - true_positives
    iou = true_positives / (true_positives + false_positives + false_negatives)
    best_index = int(np.argmax(iou))
    return {
        "thresholds": thresholds,
        "occupied_iou": iou,
        "best_threshold": float(thresholds[best_index]),
        "best_occupied_iou": float(iou[best_index]),
    }


def _subset_mask(metadata, groups, pedestrian):
    return np.array(
        [
            sample["route_group"] in groups and sample["pedestrian"] == pedestrian
            for sample in metadata
        ],
        dtype=bool,
    )


def _point_metrics(targets, scores, threshold):
    result = binary_score_metrics(targets.reshape(-1), scores.reshape(-1), threshold)
    return {
        "threshold": float(threshold),
        "occupied_iou": result["occupied_iou"],
        "average_precision": result["average_precision"],
        "roc_auc": result["roc_auc"],
    }


def run_calibration_diagnostic(scores_dir, output_path, seed):
    dumps = {variant: load_variant_dump(scores_dir, variant) for variant in VARIANTS}
    reference = dumps["b0"]["metadata"]
    for variant in VARIANTS[1:]:
        other = dumps[variant]["metadata"]
        if [(s["sequence_id"], s["frame_id"]) for s in other] != [
            (s["sequence_id"], s["frame_id"]) for s in reference
        ]:
            raise CalibrationDiagnosticError(f"{variant} sample order differs from b0")
        if [s["pedestrian"] for s in other] != [s["pedestrian"] for s in reference]:
            raise CalibrationDiagnosticError(f"{variant} pedestrian flags differ from b0")

    split = split_route_groups(reference, seed)
    nonpedestrian = np.array(
        [not sample["pedestrian"] for sample in reference], dtype=bool
    )
    masks = {
        "calibration": _subset_mask(reference, set(split["calibration"]), False),
        "evaluation": _subset_mask(reference, set(split["evaluation"]), False),
        "full_validation": nonpedestrian,
    }
    union = masks["calibration"] | masks["evaluation"]
    if not np.array_equal(union, nonpedestrian) or (
        masks["calibration"] & masks["evaluation"]
    ).any():
        raise CalibrationDiagnosticError("route group split must partition nonpedestrian samples")

    models = {}
    for variant in VARIANTS:
        scores = dumps[variant]["scores"]
        targets = dumps[variant]["targets"]
        sweep = sweep_occupied_iou(
            targets[masks["calibration"]], scores[masks["calibration"]]
        )
        t_star = sweep["best_threshold"]
        entry = {"t_star": t_star, "t_star_occupied_iou_on_calibration": sweep["best_occupied_iou"]}
        for subset, mask in masks.items():
            entry[subset] = {
                "fixed_threshold": _point_metrics(
                    targets[mask], scores[mask], FIXED_THRESHOLD
                ),
                "calibrated_threshold": _point_metrics(
                    targets[mask], scores[mask], t_star
                ),
            }
        models[variant] = entry

    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol": "docs/interfuser_negative_transfer_repair_protocol_v6.md",
        "created_at": _utc_now(),
        "scores_dir": str(Path(scores_dir).resolve()),
        "input_artifacts": {
            variant: dumps[variant]["artifacts"] for variant in VARIANTS
        },
        "fixed_threshold": FIXED_THRESHOLD,
        "split": split,
        "cohort": "nonpedestrian",
        "models": models,
        "comparison": {
            "t_star_gap": models["candidate"]["t_star"] - models["b0"]["t_star"],
            "evaluation_occupied_iou_delta_at_fixed": models["candidate"]["evaluation"][
                "fixed_threshold"
            ]["occupied_iou"]
            - models["b0"]["evaluation"]["fixed_threshold"]["occupied_iou"],
            "evaluation_occupied_iou_delta_at_calibrated": models["candidate"]["evaluation"][
                "calibrated_threshold"
            ]["occupied_iou"]
            - models["b0"]["evaluation"]["calibrated_threshold"]["occupied_iou"],
            "evaluation_average_precision_delta": models["candidate"]["evaluation"][
                "fixed_threshold"
            ]["average_precision"]
            - models["b0"]["evaluation"]["fixed_threshold"]["average_precision"],
            "evaluation_roc_auc_delta": models["candidate"]["evaluation"]["fixed_threshold"][
                "roc_auc"
            ]
            - models["b0"]["evaluation"]["fixed_threshold"]["roc_auc"],
        },
        "stage2_admission": None,
        "test_accessed": False,
    }
    _write_json(output_path, manifest)
    manifest["output_sha256"] = sha256_file(output_path)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Pure calibration diagnostic over dumped occupancy scores"
    )
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args(argv)
    try:
        manifest = run_calibration_diagnostic(args.scores_dir, args.output, args.seed)
    except CalibrationDiagnosticError as exc:
        print(f"calibration diagnostic error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest["comparison"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
