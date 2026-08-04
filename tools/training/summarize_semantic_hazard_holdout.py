#!/usr/bin/env python3
"""
[INPUT]: 依赖 baseline/augmented 两份 pipeline-valid 行人危险 holdout 评估 JSON，要求样本 key、route group、类别权重与特权输入边界完全一致。
[OUTPUT]: 提供 summarize_semantic_hazard_holdout 与 CLI，输出 augmented-baseline 的 mIoU、macro-F1、loss 和逐类 IoU/F1 配对差值 JSON。
[POS]: tools/training 的 M2 行人危险实验归约器；在生成论文表格前阻止 holdout 或 loss 口径漂移。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPORT_SCHEMA_VERSION = 1


class HazardHoldoutSummaryError(ValueError):
    """Raised when two targeted evaluation reports are not strictly paired."""


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_report(path, label):
    path = Path(path).resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HazardHoldoutSummaryError(
            f"unable to read {label} report {path}: {exc}"
        ) from exc
    if not isinstance(value, dict) or not value.get("valid"):
        raise HazardHoldoutSummaryError(f"{label} report must be pipeline-valid")
    return value, path, _sha256(path)


def _class_map(metrics, label, split):
    items = metrics.get("per_class")
    if not isinstance(items, list) or not items:
        raise HazardHoldoutSummaryError(
            f"{label} {split} has no per_class metrics"
        )
    by_name = {}
    for item in items:
        name = item.get("name") if isinstance(item, dict) else None
        if not isinstance(name, str) or name in by_name:
            raise HazardHoldoutSummaryError(
                f"{label} {split} has invalid or duplicate class names"
            )
        by_name[name] = item
    return by_name


def summarize_semantic_hazard_holdout(baseline_path, augmented_path):
    """Return strict paired deltas for two immutable hazard evaluations."""
    baseline, baseline_path, baseline_sha256 = _read_report(
        baseline_path, "baseline"
    )
    augmented, augmented_path, augmented_sha256 = _read_report(
        augmented_path, "augmented"
    )
    if baseline.get("contract") != augmented.get("contract"):
        raise HazardHoldoutSummaryError(
            "baseline and augmented evaluation contracts differ"
        )
    baseline_splits = baseline.get("evaluations", {})
    augmented_splits = augmented.get("evaluations", {})
    if set(baseline_splits) != {"validation", "test"} or set(
        augmented_splits
    ) != {"validation", "test"}:
        raise HazardHoldoutSummaryError(
            "both reports must contain validation and test evaluations"
        )

    split_reports = {}
    for split in ("validation", "test"):
        baseline_source = baseline["sources"]["hazard_holdouts"][split]
        augmented_source = augmented["sources"]["hazard_holdouts"][split]
        for field in (
            "manifest_sha256",
            "holdout_split",
            "route_groups",
            "sequences",
            "expected_camera_samples",
        ):
            if baseline_source.get(field) != augmented_source.get(field):
                raise HazardHoldoutSummaryError(
                    f"{split} holdout differs in {field}"
                )
        baseline_metrics = baseline_splits[split]
        augmented_metrics = augmented_splits[split]
        for field in ("samples", "sample_keys_sha256"):
            if baseline_metrics.get(field) != augmented_metrics.get(field):
                raise HazardHoldoutSummaryError(
                    f"{split} evaluated samples differ in {field}"
                )
        baseline_classes = _class_map(baseline_metrics, "baseline", split)
        augmented_classes = _class_map(augmented_metrics, "augmented", split)
        if set(baseline_classes) != set(augmented_classes):
            raise HazardHoldoutSummaryError(f"{split} class sets differ")
        per_class = []
        for name in baseline_classes:
            baseline_item = baseline_classes[name]
            augmented_item = augmented_classes[name]
            for field in ("train_id", "support_pixels"):
                if baseline_item.get(field) != augmented_item.get(field):
                    raise HazardHoldoutSummaryError(
                        f"{split} class {name} differs in {field}"
                    )
            per_class.append(
                {
                    "name": name,
                    "train_id": baseline_item["train_id"],
                    "support_pixels": baseline_item["support_pixels"],
                    "baseline": {
                        "iou": baseline_item["iou"],
                        "f1": baseline_item["f1"],
                    },
                    "augmented": {
                        "iou": augmented_item["iou"],
                        "f1": augmented_item["f1"],
                    },
                    "augmented_minus_baseline": {
                        "iou": augmented_item["iou"] - baseline_item["iou"],
                        "f1": augmented_item["f1"] - baseline_item["f1"],
                    },
                }
            )
        split_reports[split] = {
            "holdout": baseline_source,
            "samples": baseline_metrics["samples"],
            "sample_keys_sha256": baseline_metrics["sample_keys_sha256"],
            "baseline": {
                field: baseline_metrics[field]
                for field in ("mean_iou", "macro_f1", "loss")
            },
            "augmented": {
                field: augmented_metrics[field]
                for field in ("mean_iou", "macro_f1", "loss")
            },
            "augmented_minus_baseline": {
                field: augmented_metrics[field] - baseline_metrics[field]
                for field in ("mean_iou", "macro_f1", "loss")
            },
            "classes_improved_iou": sum(
                item["augmented_minus_baseline"]["iou"] > 0
                for item in per_class
            ),
            "classes_tied_iou": sum(
                item["augmented_minus_baseline"]["iou"] == 0
                for item in per_class
            ),
            "classes_worse_iou": sum(
                item["augmented_minus_baseline"]["iou"] < 0
                for item in per_class
            ),
            "per_class": per_class,
        }
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "valid": True,
        "errors": [],
        "contract": {
            "delta_direction": "augmented minus baseline",
            "strict_same_holdout_samples": True,
            "privileged_hazard_used_as_model_input": False,
            "checkpoint_selection_uses_hazard_holdout": False,
        },
        "sources": {
            "baseline_report": str(baseline_path),
            "baseline_report_sha256": baseline_sha256,
            "baseline_checkpoint_sha256": baseline["sources"][
                "checkpoint_sha256"
            ],
            "augmented_report": str(augmented_path),
            "augmented_report_sha256": augmented_sha256,
            "augmented_checkpoint_sha256": augmented["sources"][
                "checkpoint_sha256"
            ],
        },
        "splits": split_reports,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Summarize paired pedestrian hazard holdout evaluations"
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--augmented", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        print(f"summary error: refusing to overwrite {args.output}", file=sys.stderr)
        return 2
    try:
        report = summarize_semantic_hazard_holdout(
            args.baseline, args.augmented
        )
    except HazardHoldoutSummaryError as exc:
        print(f"summary error: {exc}", file=sys.stderr)
        return 2
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    except FileExistsError:
        print(f"summary error: refusing to overwrite {args.output}", file=sys.stderr)
        return 2
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
