#!/usr/bin/env python3
"""
[INPUT]: 依赖同一 pilot 配置产生的多个 pipeline-valid 训练 manifest，以及 manifest 引用的 checkpoint/骨干产物。
[OUTPUT]: 对外提供 LearningCurveError、summarize_learning_curve 与 CLI，验证嵌套训练样本、完整一致 validation、provenance 和产物哈希后生成确定性学习曲线 JSON。
[POS]: tools/training 的 M2 pilot 汇总门禁；只归约可比 run，不训练模型，也不允许缺失点或临时样本预算进入数据量判断。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import argparse
import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.training.semantic_pretraining import sha256_file  # noqa: E402


CURVE_SUMMARY_SCHEMA_VERSION = 1


class LearningCurveError(ValueError):
    """Raised when training runs do not form one comparable learning curve."""


def _read_json(path, label):
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LearningCurveError(f"unable to read {label} JSON {path}: {exc}") from exc


def _require_finite_metrics(metrics, label):
    for field in ("loss", "pixel_accuracy", "mean_iou", "macro_f1"):
        value = metrics.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise LearningCurveError(f"{label} {field} must be finite")
    per_class = metrics.get("per_class")
    if not isinstance(per_class, list) or not per_class:
        raise LearningCurveError(f"{label} has no per_class metrics")
    for item in per_class:
        for field in ("iou", "f1"):
            value = item.get(field)
            if value is not None and (
                not isinstance(value, (int, float)) or not math.isfinite(float(value))
            ):
                raise LearningCurveError(
                    f"{label} class {item.get('name')} {field} must be finite or null"
                )


def _validate_artifact(manifest, field, hash_field):
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise LearningCurveError(f"run {manifest.get('run_id')} has no artifacts")
    path = Path(artifacts.get(field, ""))
    expected = artifacts.get(hash_field)
    if not path.is_file():
        raise LearningCurveError(f"run {manifest.get('run_id')} missing artifact {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise LearningCurveError(
            f"run {manifest.get('run_id')} artifact hash mismatch for {field}"
        )
    return str(path), actual


def summarize_learning_curve(manifest_paths):
    """Return a deterministic summary after enforcing the full pilot matrix."""
    paths = [Path(path).resolve() for path in manifest_paths]
    if not paths:
        raise LearningCurveError("at least one run manifest is required")
    manifests = []
    for path in paths:
        manifest = _read_json(path, "run manifest")
        manifests.append((path, manifest, sha256_file(path)))
    first = manifests[0][1]
    config_path = Path(first.get("config", ""))
    config = _read_json(config_path, "pilot config")
    config_sha256 = sha256_file(config_path)
    if config.get("status") != "pilot":
        raise LearningCurveError("learning curve requires a pilot config")
    expected_train_samples = config.get("data", {}).get(
        "learning_curve_train_samples"
    )
    expected_validation_samples = config.get("data", {}).get(
        "expected_available_validation_samples"
    )
    if not isinstance(expected_train_samples, list) or not expected_train_samples:
        raise LearningCurveError("pilot config has no learning-curve sample counts")
    if len(manifests) != len(expected_train_samples):
        raise LearningCurveError(
            f"expected {len(expected_train_samples)} manifests, got {len(manifests)}"
        )

    provenance_fields = (
        "git_head",
        "config_sha256",
        "class_config_sha256",
        "split_manifest_sha256",
        "pretrained_source",
        "pretrained_checkpoint_sha256",
    )
    expected_provenance = {field: first.get(field) for field in provenance_fields}
    if expected_provenance["config_sha256"] != config_sha256:
        raise LearningCurveError("run manifest and pilot config SHA-256 differ")
    run_ids = set()
    points = []
    train_key_sets = {}
    validation_keys = None
    for path, manifest, manifest_sha256 in manifests:
        run_id = manifest.get("run_id")
        if not isinstance(run_id, str) or not run_id or run_id in run_ids:
            raise LearningCurveError(f"invalid or duplicate run ID: {run_id}")
        run_ids.add(run_id)
        if manifest.get("status") != "completed" or not manifest.get("pipeline_valid"):
            raise LearningCurveError(f"run {run_id} is not pipeline valid")
        if manifest.get("experiment_status") != "pilot":
            raise LearningCurveError(f"run {run_id} is not a pilot run")
        for field, expected in expected_provenance.items():
            if manifest.get(field) != expected:
                raise LearningCurveError(f"run {run_id} provenance differs at {field}")
        data = manifest.get("data", {})
        train_samples = data.get("train_samples")
        validation_samples = data.get("validation_samples")
        if train_samples != data.get("train_sample_limit_requested"):
            raise LearningCurveError(f"run {run_id} train sample limit was not honored")
        if validation_samples != expected_validation_samples:
            raise LearningCurveError(
                f"run {run_id} validation samples={validation_samples}, "
                f"expected {expected_validation_samples}"
            )
        train_keys = data.get("train_sample_keys")
        current_validation_keys = data.get("validation_sample_keys")
        if not isinstance(train_keys, list) or len(train_keys) != train_samples:
            raise LearningCurveError(f"run {run_id} train sample keys are incomplete")
        if (
            not isinstance(current_validation_keys, list)
            or len(current_validation_keys) != validation_samples
        ):
            raise LearningCurveError(f"run {run_id} validation sample keys are incomplete")
        if len(set(train_keys)) != len(train_keys):
            raise LearningCurveError(f"run {run_id} has duplicate train sample keys")
        if validation_keys is None:
            validation_keys = current_validation_keys
        elif current_validation_keys != validation_keys:
            raise LearningCurveError("validation sample keys differ across runs")
        epochs = manifest.get("epochs")
        if not isinstance(epochs, list) or len(epochs) != config["training"]["epochs"]:
            raise LearningCurveError(f"run {run_id} epoch count differs from config")
        final_epoch = epochs[-1]
        _require_finite_metrics(final_epoch.get("train", {}), f"run {run_id} train")
        _require_finite_metrics(
            final_epoch.get("validation", {}), f"run {run_id} validation"
        )
        checkpoint, checkpoint_sha256 = _validate_artifact(
            manifest, "checkpoint", "checkpoint_sha256"
        )
        backbone, backbone_sha256 = _validate_artifact(
            manifest, "backbone_export", "backbone_export_sha256"
        )
        train_key_sets[train_samples] = set(train_keys)
        points.append(
            {
                "run_id": run_id,
                "run_manifest": str(path),
                "run_manifest_sha256": manifest_sha256,
                "train_samples": train_samples,
                "validation_samples": validation_samples,
                "train": final_epoch["train"],
                "validation": final_epoch["validation"],
                "checkpoint": checkpoint,
                "checkpoint_sha256": checkpoint_sha256,
                "backbone_export": backbone,
                "backbone_export_sha256": backbone_sha256,
                "resources": manifest.get("resources"),
            }
        )
    observed_train_samples = sorted(point["train_samples"] for point in points)
    if observed_train_samples != expected_train_samples:
        raise LearningCurveError(
            f"train sample matrix differs: {observed_train_samples} != {expected_train_samples}"
        )
    for smaller, larger in zip(expected_train_samples, expected_train_samples[1:]):
        if not train_key_sets[smaller] < train_key_sets[larger]:
            raise LearningCurveError(
                f"train samples are not strictly nested: {smaller} is not inside {larger}"
            )
    points.sort(key=lambda item: item["train_samples"])
    return {
        "learning_curve_summary_schema_version": CURVE_SUMMARY_SCHEMA_VERSION,
        "valid": True,
        "config": str(config_path),
        "config_sha256": config_sha256,
        "contract": expected_provenance,
        "expected_train_samples": expected_train_samples,
        "validation_samples": expected_validation_samples,
        "nested_train_samples": True,
        "identical_validation_samples": True,
        "points": points,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate and summarize an M2 semantic learning curve"
    )
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        print(f"summary error: refusing to overwrite {args.output}", file=sys.stderr)
        return 2
    try:
        summary = summarize_learning_curve(args.manifests)
    except LearningCurveError as exc:
        print(f"summary error: {exc}", file=sys.stderr)
        return 2
    serialized = json.dumps(summary, indent=2, sort_keys=True) + "\n"
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
