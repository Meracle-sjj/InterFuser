#!/usr/bin/env python3
"""
[INPUT]: 依赖一个或多个完整 run_manifest.json 及其同目录 baseline_eval_config.json，消费冻结路线、随机种子、指标与输入哈希。
[OUTPUT]: 对外提供 SummaryError、build_summary、write_summary 与 CLI，生成确定性三种子路线宏平均、种子波动、失败类型和资源证据 JSON。
[POS]: tools/evaluation 的 M0 纯离线汇总器，位于 runner 之后、论文实验记录之前；默认拒绝缺失、重复、pipeline-invalid 与未授权输入漂移。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import argparse
import copy
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path


SUMMARY_SCHEMA_VERSION = 1
METRIC_KEYS = {
    "driving_score": "score_composed",
    "route_completion": "score_route",
    "infraction_score": "score_penalty",
}
CONTRACT_INPUTS = (
    "routes",
    "scenarios",
    "agent",
    "agent_config",
    "controller",
    "model_definition",
    "leaderboard_evaluator",
    "leaderboard_route_scenario",
    "scenario_runner_route_scenario",
)


class SummaryError(ValueError):
    """Raised when run manifests cannot form one comparable complete matrix."""


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SummaryError(f"cannot read JSON {path}: {exc}") from exc


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_hashes(config):
    hashes = {}
    inputs = config.get("inputs", {})
    for name in CONTRACT_INPUTS:
        item = inputs.get(name)
        if item is not None:
            hashes[name] = item.get("sha256")
    return hashes


def _contract(manifest, config):
    plan = manifest.get("run_plan") or {}
    route_set = plan.get("route_set")
    route_ids = config.get("route_sets", {}).get(route_set)
    if not route_set or route_ids is None:
        raise SummaryError("manifest route_set is absent from its config snapshot")
    return {
        "route_set": route_set,
        "route_ids": sorted(route_ids),
        "random_seeds": sorted(config.get("random_seeds", [])),
        "checkpoint": config.get("checkpoint"),
        "runtime": plan.get("runtime"),
        "environment": plan.get("environment"),
        "input_sha256": _input_hashes(config),
    }


def _validate_summary_counts(path, manifest):
    attempts = manifest.get("attempts")
    summary = manifest.get("summary") or {}
    if not isinstance(attempts, list):
        raise SummaryError(f"{path}: attempts must be a list")
    expected = {
        "recorded_attempts": len(attempts),
        "pipeline_valid_attempts": sum(
            attempt.get("pipeline_valid") is True for attempt in attempts
        ),
        "pipeline_invalid_attempts": sum(
            attempt.get("pipeline_valid") is False for attempt in attempts
        ),
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise SummaryError(f"{path}: summary {key} does not match attempts")
    if summary.get("recorded_attempts") != summary.get("planned_attempts"):
        raise SummaryError(f"{path}: run is incomplete")
    if summary.get("pipeline_invalid_attempts") != 0:
        raise SummaryError(f"{path}: run contains pipeline-invalid attempts")


def _compare_contracts(reference, candidate, allowed_input_drift, path):
    for key in (
        "route_set",
        "route_ids",
        "random_seeds",
        "checkpoint",
        "runtime",
        "environment",
    ):
        if candidate[key] != reference[key]:
            raise SummaryError(f"{path}: experiment contract drift in {key}")

    reference_hashes = reference["input_sha256"]
    candidate_hashes = candidate["input_sha256"]
    if set(reference_hashes) != set(candidate_hashes):
        raise SummaryError(f"{path}: configured input set drift")
    for name in sorted(reference_hashes):
        if (
            candidate_hashes[name] != reference_hashes[name]
            and name not in allowed_input_drift
        ):
            raise SummaryError(
                f"{path}: input hash drift in {name}; explicit approval required"
            )


def _mean(values):
    return statistics.fmean(values)


def _attempt_metrics(attempt, path):
    if attempt.get("pipeline_valid") is not True:
        raise SummaryError(f"{path}: attempt is not pipeline valid")
    result = attempt.get("leaderboard_result") or {}
    if result.get("valid") is not True:
        raise SummaryError(f"{path}: attempt has no valid leaderboard result")
    scores = result.get("scores") or {}
    metrics = {}
    for output_name, score_name in METRIC_KEYS.items():
        value = scores.get(score_name)
        if not isinstance(value, (int, float)):
            raise SummaryError(f"{path}: attempt is missing numeric {score_name}")
        metrics[output_name] = float(value)
    return metrics


def build_summary(manifest_paths, allowed_input_drift=()):
    paths = [Path(path).resolve() for path in manifest_paths]
    if not paths:
        raise SummaryError("at least one run manifest is required")

    allowed_input_drift = frozenset(allowed_input_drift)
    unknown = allowed_input_drift.difference(CONTRACT_INPUTS)
    if unknown:
        raise SummaryError(f"unknown allowed input drift: {', '.join(sorted(unknown))}")

    records = []
    reference_contract = None
    attempts_by_key = {}
    status_counts = Counter()
    infraction_counts = Counter()
    provenance = []
    attempt_durations = []
    max_port_release_wait_seconds = 0.0
    max_gpu_release_wait_seconds = 0.0
    gpu_peak_memory_mb = Counter()

    for path in paths:
        manifest = _read_json(path)
        config_path = path.parent / "baseline_eval_config.json"
        config = _read_json(config_path)
        _validate_summary_counts(path, manifest)
        contract = _contract(manifest, config)
        if reference_contract is None:
            reference_contract = contract
        else:
            _compare_contracts(reference_contract, contract, allowed_input_drift, path)

        plan = manifest["run_plan"]
        config_snapshot_sha256 = _sha256(config_path)
        if plan.get("config_sha256") != config_snapshot_sha256:
            raise SummaryError(f"{path}: config snapshot hash does not match run plan")
        records.append(
            {
                "run_id": plan.get("run_id"),
                "manifest": str(path),
                "manifest_sha256": _sha256(path),
                "config_snapshot": str(config_path.resolve()),
                "config_snapshot_sha256": config_snapshot_sha256,
            }
        )
        provenance.append(
            {
                "run_id": plan.get("run_id"),
                "git_head": plan.get("git_head"),
                "code_anchor": plan.get("code_anchor"),
                "runner_sha256": plan.get("runner_sha256"),
                "config_sha256": plan.get("config_sha256"),
                "input_sha256": contract["input_sha256"],
            }
        )

        for attempt in manifest["attempts"]:
            route_id = attempt.get("route_id")
            seed = attempt.get("traffic_manager_seed")
            key = (route_id, seed)
            if key in attempts_by_key:
                raise SummaryError(f"duplicate route/seed attempt: {key}")
            metrics = _attempt_metrics(attempt, path)
            result = attempt["leaderboard_result"]
            attempts_by_key[key] = {
                "run_id": plan.get("run_id"),
                "attempt_id": attempt.get("attempt_id"),
                "metrics": metrics,
                "status": result.get("status"),
            }
            status_counts[result.get("status") or "unknown"] += 1
            infraction_counts.update(result.get("infraction_counts") or {})
            attempt_durations.append(float(attempt.get("duration_seconds") or 0.0))
            max_port_release_wait_seconds = max(
                max_port_release_wait_seconds,
                float(attempt.get("port_release_wait_seconds") or 0.0),
            )
            max_gpu_release_wait_seconds = max(
                max_gpu_release_wait_seconds,
                float(attempt.get("gpu_release_wait_seconds") or 0.0),
            )
            for gpu, value in (attempt.get("gpu_peak_memory_mb") or {}).items():
                gpu_peak_memory_mb[gpu] = max(gpu_peak_memory_mb[gpu], value)
            if attempt.get("cleanup_error") is not None:
                raise SummaryError(f"{path}: attempt has cleanup_error")

    expected_keys = {
        (route_id, seed)
        for route_id in reference_contract["route_ids"]
        for seed in reference_contract["random_seeds"]
    }
    actual_keys = set(attempts_by_key)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise SummaryError(f"route/seed matrix mismatch; missing={missing}, extra={extra}")

    per_route = []
    for route_id in reference_contract["route_ids"]:
        seeds = []
        for seed in reference_contract["random_seeds"]:
            attempt = attempts_by_key[(route_id, seed)]
            seeds.append({"seed": seed, **attempt})
        per_route.append(
            {
                "route_id": route_id,
                "seeds": seeds,
                "mean": {
                    metric: _mean(item["metrics"][metric] for item in seeds)
                    for metric in METRIC_KEYS
                },
            }
        )

    per_seed = []
    for seed in reference_contract["random_seeds"]:
        per_seed.append(
            {
                "seed": seed,
                "macro_mean": {
                    metric: _mean(
                        attempts_by_key[(route_id, seed)]["metrics"][metric]
                        for route_id in reference_contract["route_ids"]
                    )
                    for metric in METRIC_KEYS
                },
            }
        )

    aggregate = {}
    for metric in METRIC_KEYS:
        seed_values = [item["macro_mean"][metric] for item in per_seed]
        aggregate[metric] = {
            "mean": _mean(item["mean"][metric] for item in per_route),
            "population_stddev_across_seed_macro_means": statistics.pstdev(seed_values),
        }

    input_variants = {}
    for name in CONTRACT_INPUTS:
        values = sorted(
            {
                item["input_sha256"].get(name)
                for item in provenance
                if item["input_sha256"].get(name) is not None
            }
        )
        if len(values) > 1:
            input_variants[name] = values

    output_contract = copy.deepcopy(reference_contract)
    for name in input_variants:
        output_contract["input_sha256"].pop(name, None)

    return {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "valid": True,
        "aggregation": {
            "seed_reduction": "mean_per_route",
            "route_reduction": "macro_mean",
            "stddev": "population stddev of per-seed route macro means",
        },
        "contract": output_contract,
        "allowed_input_drift": sorted(allowed_input_drift),
        "input_hash_variants": input_variants,
        "source_runs": sorted(records, key=lambda item: item["run_id"]),
        "runtime_provenance": sorted(provenance, key=lambda item: item["run_id"]),
        "attempt_count": len(attempts_by_key),
        "pipeline_valid_attempts": len(attempts_by_key),
        "pipeline_invalid_attempts": 0,
        "per_route": per_route,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "leaderboard_status_counts": dict(sorted(status_counts.items())),
        "infraction_counts": dict(sorted(infraction_counts.items())),
        "resources": {
            "total_attempt_duration_seconds": math.fsum(attempt_durations),
            "max_port_release_wait_seconds": max_port_release_wait_seconds,
            "max_gpu_release_wait_seconds": max_gpu_release_wait_seconds,
            "gpu_peak_memory_mb": dict(sorted(gpu_peak_memory_mb.items())),
            "cleanup_errors": 0,
        },
    }


def write_summary(summary, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8") as target:
            json.dump(summary, target, ensure_ascii=False, indent=2, sort_keys=True)
            target.write("\n")
    except FileExistsError as exc:
        raise SummaryError(f"refusing to overwrite existing output: {output_path}") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Deterministically summarize complete thesis baseline manifests."
    )
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-input-drift",
        action="append",
        default=[],
        choices=CONTRACT_INPUTS,
        help="Explicitly approved input whose hash differs between source runs.",
    )
    args = parser.parse_args(argv)
    try:
        summary = build_summary(args.manifests, args.allow_input_drift)
        write_summary(summary, args.output)
    except SummaryError as exc:
        parser.error(str(exc))
    print(json.dumps({"valid": True, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
