#!/usr/bin/env python3
"""
[INPUT]: 依赖正式 B0/V 各自完整 D7 run_manifest、baseline_eval_config 快照，以及两份配置共同绑定的 pipeline-valid 冻结 test/formal manifest。
[OUTPUT]: 对外提供 PairedSummaryError、normalize_d7_config_for_pair、build_paired_summary、write_paired_summary 与 CLI，生成 V-B0 的 21 attempt/route/seed 配对差值和预注册 H1 判定。
[POS]: tools/evaluation 的 M2 H1 最终纯离线归约器；复用 M0 单组汇总门禁，只放行 checkpoint/provenance 差异，不启动 CARLA 或读取训练数据。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import argparse
import copy
import hashlib
import json
import math
import statistics
from pathlib import Path

from tools.evaluation.summarize_thesis_baseline import (
    METRIC_KEYS,
    SummaryError,
    build_summary,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_SCHEMA_VERSION = 1
ROUTE_ORDER = (18, 6, 12, 30, 36, 39, 0)
SEEDS = (0, 1, 2)
VARIANTS = ("b0", "v")
OFFLINE_DIRECTIONS = {
    "traffic_average_precision": "higher",
    "traffic_roc_auc": "higher",
    "traffic_occupied_iou": "higher",
    "waypoint_ade": "lower",
    "waypoint_fde_horizon_10": "lower",
}


class PairedSummaryError(ValueError):
    """Raised when B0/V evidence cannot support one paired H1 conclusion."""


def _read_json(path, label):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PairedSummaryError(f"cannot read {label} JSON {path}: {exc}") from exc


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_file(repo_root, value, expected_sha256, label):
    if not isinstance(value, str) or not value:
        raise PairedSummaryError(f"{label} path must be non-empty")
    path = Path(value)
    path = path if path.is_absolute() else Path(repo_root) / path
    path = path.resolve()
    if not path.is_file():
        raise PairedSummaryError(f"{label} is not a file: {path}")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise PairedSummaryError(f"{label} SHA-256 is invalid")
    actual = _sha256(path)
    if actual != expected_sha256:
        raise PairedSummaryError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return path


def _load_variant(manifest_path, variant):
    manifest_path = Path(manifest_path).resolve()
    manifest = _read_json(manifest_path, f"{variant} D7 manifest")
    config_path = manifest_path.parent / "baseline_eval_config.json"
    config = _read_json(config_path, f"{variant} D7 config snapshot")
    try:
        summary = build_summary([manifest_path])
    except SummaryError as exc:
        raise PairedSummaryError(f"{variant} D7 is invalid: {exc}") from exc
    expected_order = [
        (route_id, seed) for route_id in ROUTE_ORDER for seed in SEEDS
    ]
    actual_order = [
        (attempt.get("route_id"), attempt.get("traffic_manager_seed"))
        for attempt in manifest.get("attempts", [])
    ]
    if actual_order != expected_order:
        raise PairedSummaryError(
            f"{variant} D7 attempt order differs from frozen route/seed order"
        )
    if config.get("route_sets", {}).get("development_d7") is None or set(
        config["route_sets"]["development_d7"]
    ) != set(ROUTE_ORDER):
        raise PairedSummaryError(f"{variant} D7 route set differs from frozen D7")
    if config.get("random_seeds") != list(SEEDS):
        raise PairedSummaryError(f"{variant} D7 seeds differ from frozen seeds")
    return {
        "variant": variant,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "manifest_sha256": _sha256(manifest_path),
        "config_path": config_path.resolve(),
        "config": config,
        "config_sha256": _sha256(config_path),
        "summary": summary,
    }


def _comparison_contract(record, repo_root):
    variant = record["variant"]
    config = record["config"]
    comparison = config.get("comparison") or {}
    if comparison.get("schema_version") != 1:
        raise PairedSummaryError(f"{variant} comparison schema must be v1")
    if comparison.get("variant") != variant:
        raise PairedSummaryError(f"{variant} comparison variant identity differs")
    checkpoint = config.get("checkpoint") or {}
    if checkpoint.get("architecture") != "interfuser_baseline":
        raise PairedSummaryError(f"{variant} checkpoint architecture differs")
    checkpoint_path = _resolve_file(
        repo_root,
        checkpoint.get("path"),
        checkpoint.get("sha256"),
        f"{variant} checkpoint",
    )
    planned_checkpoint = (record["manifest"].get("run_plan") or {}).get(
        "checkpoint_path"
    )
    if (
        not isinstance(planned_checkpoint, str)
        or Path(planned_checkpoint).resolve() != checkpoint_path
    ):
        raise PairedSummaryError(
            f"{variant} D7 run plan checkpoint differs from its config snapshot"
        )
    if comparison.get("formal_checkpoint_sha256") != checkpoint.get("sha256"):
        raise PairedSummaryError(
            f"{variant} checkpoint differs from formal checkpoint provenance"
        )
    formal_manifest = _resolve_file(
        repo_root,
        comparison.get("formal_training_manifest"),
        comparison.get("formal_training_manifest_sha256"),
        "formal training manifest",
    )
    test_manifest = _resolve_file(
        repo_root,
        comparison.get("visual_test_manifest"),
        comparison.get("visual_test_manifest_sha256"),
        "visual test manifest",
    )
    return {
        "variant": variant,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint["sha256"],
        "formal_manifest_path": formal_manifest,
        "formal_manifest_sha256": comparison["formal_training_manifest_sha256"],
        "test_manifest_path": test_manifest,
        "test_manifest_sha256": comparison["visual_test_manifest_sha256"],
    }


def normalize_d7_config_for_pair(config):
    """Mask only the variant fields allowed to differ in a D7 pair."""
    normalized = copy.deepcopy(config)
    checkpoint = normalized.get("checkpoint") or {}
    for field in ("path", "sha256", "epoch", "best_metric"):
        if field in checkpoint:
            checkpoint[field] = "<VARIANT_CHECKPOINT>"
    comparison = normalized.get("comparison") or {}
    for field in ("variant", "formal_checkpoint_sha256"):
        if field in comparison:
            comparison[field] = "<VARIANT_PROVENANCE>"
    return normalized


def _validate_pair_contract(b0, v, repo_root):
    b0_provenance = _comparison_contract(b0, repo_root)
    v_provenance = _comparison_contract(v, repo_root)
    if b0_provenance["checkpoint_sha256"] == v_provenance["checkpoint_sha256"]:
        raise PairedSummaryError("B0/V D7 checkpoints must differ")
    for field in (
        "formal_manifest_path",
        "formal_manifest_sha256",
        "test_manifest_path",
        "test_manifest_sha256",
    ):
        if b0_provenance[field] != v_provenance[field]:
            raise PairedSummaryError(f"B0/V provenance differs in {field}")
    if normalize_d7_config_for_pair(b0["config"]) != normalize_d7_config_for_pair(
        v["config"]
    ):
        raise PairedSummaryError(
            "B0/V D7 configs differ outside checkpoint/provenance fields"
        )
    for field in (
        "route_set",
        "route_ids",
        "random_seeds",
        "runtime",
        "environment",
        "input_sha256",
    ):
        if b0["summary"]["contract"].get(field) != v["summary"]["contract"].get(field):
            raise PairedSummaryError(f"B0/V summarized contract differs in {field}")

    formal_manifest = _read_json(
        b0_provenance["formal_manifest_path"], "formal training manifest"
    )
    if (
        formal_manifest.get("status") != "completed"
        or formal_manifest.get("pipeline_valid") is not True
    ):
        raise PairedSummaryError("formal training manifest is not pipeline valid")
    formal_variants = formal_manifest.get("variants") or []
    if [item.get("variant") for item in formal_variants] != list(VARIANTS):
        raise PairedSummaryError("formal training variants must be B0 then V")
    test_manifest = _read_json(
        b0_provenance["test_manifest_path"], "visual test manifest"
    )
    if (
        test_manifest.get("status") != "completed"
        or test_manifest.get("pipeline_valid") is not True
    ):
        raise PairedSummaryError("visual test manifest is not pipeline valid")
    if (
        test_manifest.get("formal_training_manifest_sha256")
        != b0_provenance["formal_manifest_sha256"]
    ):
        raise PairedSummaryError("visual test does not bind the formal training manifest")
    test_variant_list = test_manifest.get("variants") or []
    if [item.get("variant") for item in test_variant_list] != list(VARIANTS):
        raise PairedSummaryError("visual test variants must be B0 then V")
    test_variants = {item["variant"]: item for item in test_variant_list}
    formal_by_variant = {item["variant"]: item for item in formal_variants}
    for provenance in (b0_provenance, v_provenance):
        variant = provenance["variant"]
        formal_best = (
            (formal_by_variant[variant].get("artifacts") or {}).get("best_checkpoint")
            or {}
        )
        if formal_best.get("sha256") != provenance["checkpoint_sha256"]:
            raise PairedSummaryError(
                f"{variant} D7 checkpoint differs from formal best artifact"
            )
        test_variant = test_variants[variant]
        worker = test_variant.get("worker_result") or {}
        if test_variant.get("pipeline_valid") is not True:
            raise PairedSummaryError(f"visual test {variant} variant is invalid")
        if worker.get("checkpoint_sha256") != provenance["checkpoint_sha256"]:
            raise PairedSummaryError(
                f"{variant} D7 checkpoint differs from frozen visual test"
            )
    return b0_provenance, v_provenance, test_manifest


def _attempt_lookup(summary):
    lookup = {}
    for route in summary["per_route"]:
        route_id = route["route_id"]
        for seed in route["seeds"]:
            lookup[(route_id, seed["seed"])] = seed
    return lookup


def _mean(values):
    return statistics.fmean(values)


def _paired_d7(b0_summary, v_summary):
    b0_attempts = _attempt_lookup(b0_summary)
    v_attempts = _attempt_lookup(v_summary)
    per_attempt = []
    for route_id in ROUTE_ORDER:
        for seed in SEEDS:
            b0 = b0_attempts[(route_id, seed)]
            v = v_attempts[(route_id, seed)]
            per_attempt.append(
                {
                    "route_id": route_id,
                    "seed": seed,
                    "b0": b0["metrics"],
                    "v": v["metrics"],
                    "v_minus_b0": {
                        metric: v["metrics"][metric] - b0["metrics"][metric]
                        for metric in METRIC_KEYS
                    },
                    "status": {"b0": b0["status"], "v": v["status"]},
                }
            )

    per_route = []
    for route_id in ROUTE_ORDER:
        items = [item for item in per_attempt if item["route_id"] == route_id]
        per_route.append(
            {
                "route_id": route_id,
                "v_minus_b0_seed_values": {
                    metric: [item["v_minus_b0"][metric] for item in items]
                    for metric in METRIC_KEYS
                },
                "v_minus_b0_mean": {
                    metric: _mean(item["v_minus_b0"][metric] for item in items)
                    for metric in METRIC_KEYS
                },
            }
        )

    per_seed = []
    for seed in SEEDS:
        items = [item for item in per_attempt if item["seed"] == seed]
        per_seed.append(
            {
                "seed": seed,
                "v_minus_b0_macro_mean": {
                    metric: _mean(item["v_minus_b0"][metric] for item in items)
                    for metric in METRIC_KEYS
                },
            }
        )

    aggregate = {}
    for metric in METRIC_KEYS:
        pair_values = [item["v_minus_b0"][metric] for item in per_attempt]
        seed_values = [item["v_minus_b0_macro_mean"][metric] for item in per_seed]
        route_values = [item["v_minus_b0_mean"][metric] for item in per_route]
        aggregate[metric] = {
            "macro_mean_v_minus_b0": _mean(route_values),
            "population_stddev_across_21_pairs": statistics.pstdev(pair_values),
            "population_stddev_across_seed_macro_means": statistics.pstdev(
                seed_values
            ),
            "minimum_pair_delta": min(pair_values),
            "maximum_pair_delta": max(pair_values),
            "improved_pairs": sum(value > 0 for value in pair_values),
            "tied_pairs": sum(value == 0 for value in pair_values),
            "worse_pairs": sum(value < 0 for value in pair_values),
        }
    return {
        "per_attempt": per_attempt,
        "per_route": per_route,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }


def _offline_decision(test_manifest):
    deltas = test_manifest.get("v_minus_b0") or {}
    directional = {}
    for metric, direction in OFFLINE_DIRECTIONS.items():
        value = deltas.get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise PairedSummaryError(f"visual test delta {metric} is missing or non-finite")
        value = float(value)
        improved = value > 0 if direction == "higher" else value < 0
        directional[metric] = {
            "v_minus_b0": value,
            "better_direction": direction,
            "improved": improved,
            "tied": value == 0,
        }
    improved_count = sum(item["improved"] for item in directional.values())
    return {
        "directional_metrics": directional,
        "improved_count": improved_count,
        "metric_count": len(OFFLINE_DIRECTIONS),
        "majority_supports_v": improved_count >= 3,
    }


def _h1_decision(driving_score_delta, offline):
    offline_support = offline["majority_supports_v"]
    if driving_score_delta > 0 and offline_support:
        classification = "supported"
    elif driving_score_delta < 0 and not offline_support:
        classification = "not_supported"
    else:
        classification = "mixed_or_insufficient"
    return {
        "classification": classification,
        "d7_macro_driving_score_v_minus_b0": driving_score_delta,
        "offline_majority_supports_v": offline_support,
        "noninferiority_claim_allowed": False,
        "scope": "descriptive M2 v1 evidence on the frozen D7 and test contracts",
    }


def _count_delta(b0, v):
    return {
        key: int(v.get(key, 0)) - int(b0.get(key, 0))
        for key in sorted(set(b0) | set(v))
    }


def build_paired_summary(b0_manifest, v_manifest, repo_root=REPO_ROOT):
    """Build the frozen offline+D7 H1 comparison from complete raw manifests."""
    repo_root = Path(repo_root).resolve()
    b0 = _load_variant(b0_manifest, "b0")
    v = _load_variant(v_manifest, "v")
    b0_provenance, v_provenance, test_manifest = _validate_pair_contract(
        b0, v, repo_root
    )
    paired = _paired_d7(b0["summary"], v["summary"])
    offline = _offline_decision(test_manifest)
    driving_score_delta = paired["aggregate"]["driving_score"][
        "macro_mean_v_minus_b0"
    ]
    return {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "valid": True,
        "contract": {
            "variant_order": list(VARIANTS),
            "route_order": list(ROUTE_ORDER),
            "seeds": list(SEEDS),
            "attempts_per_variant": len(ROUTE_ORDER) * len(SEEDS),
            "delta_direction": "V minus B0",
            "only_checkpoint_and_provenance_differ": True,
        },
        "sources": {
            "formal_training_manifest": {
                "path": str(b0_provenance["formal_manifest_path"]),
                "sha256": b0_provenance["formal_manifest_sha256"],
            },
            "visual_test_manifest": {
                "path": str(b0_provenance["test_manifest_path"]),
                "sha256": b0_provenance["test_manifest_sha256"],
            },
            "b0": {
                "manifest": str(b0["manifest_path"]),
                "manifest_sha256": b0["manifest_sha256"],
                "config": str(b0["config_path"]),
                "config_sha256": b0["config_sha256"],
                "checkpoint_sha256": b0_provenance["checkpoint_sha256"],
            },
            "v": {
                "manifest": str(v["manifest_path"]),
                "manifest_sha256": v["manifest_sha256"],
                "config": str(v["config_path"]),
                "config_sha256": v["config_sha256"],
                "checkpoint_sha256": v_provenance["checkpoint_sha256"],
            },
        },
        "variant_summaries": {"b0": b0["summary"], "v": v["summary"]},
        "paired_d7": paired,
        "offline_test": {
            "v_minus_b0": test_manifest["v_minus_b0"],
            "decision_metrics": offline,
        },
        "failure_analysis": {
            "leaderboard_status_counts": {
                "b0": b0["summary"]["leaderboard_status_counts"],
                "v": v["summary"]["leaderboard_status_counts"],
            },
            "infraction_counts": {
                "b0": b0["summary"]["infraction_counts"],
                "v": v["summary"]["infraction_counts"],
                "v_minus_b0": _count_delta(
                    b0["summary"]["infraction_counts"],
                    v["summary"]["infraction_counts"],
                ),
            },
        },
        "h1": _h1_decision(driving_score_delta, offline),
    }


def write_paired_summary(summary, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8") as target:
            json.dump(summary, target, ensure_ascii=False, indent=2, sort_keys=True)
            target.write("\n")
    except FileExistsError as exc:
        raise PairedSummaryError(
            f"refusing to overwrite existing output: {output_path}"
        ) from exc


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build the frozen paired B0/V D7 and offline H1 summary."
    )
    parser.add_argument("--b0-manifest", required=True, type=Path)
    parser.add_argument("--v-manifest", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        summary = build_paired_summary(
            args.b0_manifest, args.v_manifest, repo_root=args.repo_root
        )
        write_paired_summary(summary, args.output)
    except PairedSummaryError as exc:
        parser.error(str(exc))
    print(json.dumps({"valid": True, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
