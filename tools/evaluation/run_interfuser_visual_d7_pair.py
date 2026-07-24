#!/usr/bin/env python3
"""
[INPUT]: 依赖 pipeline-valid 冻结 visual test、仅 checkpoint/provenance 不同的 B0/V baseline_eval 配置，以及 M0 D7 runner/runtime_resources。
[OUTPUT]: 对外提供 D7PairError、load_d7_pair_contract、build_d7_pair_plans、execute_d7_pair 与 CLI，按 B0→V 串行执行并固化父级 manifest。
[POS]: tools/evaluation 的 M2 H1 D7 父级编排器；只组合已验证的单组 runner，不复制 CARLA 生命周期逻辑，显式 --execute 才启动外部进程。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import argparse
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from tools.evaluation.run_thesis_baseline import build_run_plan, execute_run_plan
from tools.evaluation.runtime_resources import RunnerError
from tools.evaluation.summarize_interfuser_visual_d7 import (
    ROUTE_ORDER,
    SEEDS,
    VARIANTS,
    normalize_d7_config_for_pair,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


class D7PairError(RuntimeError):
    """Raised when a paired D7 run would lose provenance or idempotence."""


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _read_json(path, label):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise D7PairError(f"cannot read {label} JSON {path}: {exc}") from exc


def _write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_file(repo_root, value, expected_sha256, label):
    if not isinstance(value, str) or not value:
        raise D7PairError(f"{label} path must be non-empty")
    path = Path(value)
    path = path if path.is_absolute() else Path(repo_root) / path
    path = path.resolve()
    if not path.is_file():
        raise D7PairError(f"{label} is not a file: {path}")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise D7PairError(f"{label} SHA-256 is invalid")
    actual = _sha256(path)
    if actual != expected_sha256:
        raise D7PairError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return path


def _resolved_result_root(repo_root, value):
    if not isinstance(value, str) or not value:
        raise D7PairError("result_root must be a non-empty path")
    path = Path(value)
    path = path if path.is_absolute() else Path(repo_root) / path
    path = path.resolve()
    try:
        path.relative_to(Path(repo_root).resolve())
    except ValueError as exc:
        raise D7PairError("result_root escapes repository") from exc
    return path


def _test_checkpoint_hashes(test_manifest):
    if (
        test_manifest.get("status") != "completed"
        or test_manifest.get("pipeline_valid") is not True
    ):
        raise D7PairError("visual test must be completed and pipeline valid")
    variants = test_manifest.get("variants") or []
    if [item.get("variant") for item in variants] != list(VARIANTS):
        raise D7PairError("visual test variants must be B0 then V")
    hashes = {}
    for item in variants:
        variant = item["variant"]
        worker = item.get("worker_result") or {}
        value = worker.get("checkpoint_sha256")
        if item.get("pipeline_valid") is not True or not isinstance(value, str):
            raise D7PairError(f"visual test {variant} checkpoint is invalid")
        hashes[variant] = value
    if hashes["b0"] == hashes["v"]:
        raise D7PairError("visual test B0/V checkpoints must differ")
    return hashes


def load_d7_pair_contract(config_path, repo_root=REPO_ROOT):
    """Validate frozen test and both future child configs without starting CARLA."""
    repo_root = Path(repo_root).resolve()
    config_path = Path(config_path).resolve()
    raw = _read_json(config_path, "visual D7 pair config")
    if raw.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise D7PairError("visual D7 pair config schema must be v1")
    if raw.get("status") != "preregistered":
        raise D7PairError("visual D7 pair config must be preregistered")
    pair_run_id = raw.get("pair_run_id")
    if not isinstance(pair_run_id, str) or not RUN_ID_PATTERN.fullmatch(pair_run_id):
        raise D7PairError("pair_run_id is invalid")
    result_root = _resolved_result_root(repo_root, raw.get("result_root"))
    test_manifest_path = _resolve_file(
        repo_root,
        raw.get("visual_test_manifest"),
        raw.get("visual_test_manifest_sha256"),
        "visual test manifest",
    )
    test_manifest = _read_json(test_manifest_path, "visual test manifest")
    checkpoint_hashes = _test_checkpoint_hashes(test_manifest)

    variants = raw.get("variants") or {}
    if tuple(variants) != VARIANTS:
        raise D7PairError("D7 pair variants must be ordered B0 then V")
    resolved_variants = {}
    child_run_ids = []
    for variant in VARIANTS:
        spec = variants[variant]
        run_id = spec.get("run_id")
        if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
            raise D7PairError(f"{variant} child run_id is invalid")
        child_run_ids.append(run_id)
        child_config_path = _resolve_file(
            repo_root,
            spec.get("config"),
            spec.get("config_sha256"),
            f"{variant} D7 config",
        )
        child_config = _read_json(child_config_path, f"{variant} D7 config")
        comparison = child_config.get("comparison") or {}
        if (
            comparison.get("schema_version") != 1
            or comparison.get("variant") != variant
        ):
            raise D7PairError(f"{variant} comparison identity is invalid")
        if comparison.get("visual_test_manifest_sha256") != raw.get(
            "visual_test_manifest_sha256"
        ):
            raise D7PairError(f"{variant} config binds a different visual test")
        comparison_test_path = Path(comparison.get("visual_test_manifest", ""))
        comparison_test_path = (
            comparison_test_path
            if comparison_test_path.is_absolute()
            else repo_root / comparison_test_path
        ).resolve()
        if comparison_test_path != test_manifest_path:
            raise D7PairError(f"{variant} config visual test path differs")
        checkpoint = child_config.get("checkpoint") or {}
        if checkpoint.get("architecture") != "interfuser_baseline":
            raise D7PairError(f"{variant} checkpoint architecture differs")
        if checkpoint.get("sha256") != checkpoint_hashes[variant]:
            raise D7PairError(
                f"{variant} D7 checkpoint differs from pipeline-valid visual test"
            )
        if comparison.get("formal_checkpoint_sha256") != checkpoint_hashes[variant]:
            raise D7PairError(f"{variant} formal checkpoint provenance differs")
        if child_config.get("random_seeds") != list(SEEDS):
            raise D7PairError(f"{variant} D7 seeds differ from frozen seeds")
        route_set = child_config.get("route_sets", {}).get("development_d7")
        if not isinstance(route_set, list) or set(route_set) != set(ROUTE_ORDER):
            raise D7PairError(f"{variant} D7 route set differs from frozen D7")
        child_result_root = _resolved_result_root(
            repo_root, child_config.get("result_root")
        )
        if child_result_root != result_root:
            raise D7PairError(f"{variant} result_root differs from pair result_root")
        resolved_variants[variant] = {
            "run_id": run_id,
            "config_path": child_config_path,
            "config_sha256": spec["config_sha256"],
            "config": child_config,
            "checkpoint_sha256": checkpoint_hashes[variant],
        }
    if len(set(child_run_ids + [pair_run_id])) != 3:
        raise D7PairError("pair and child run IDs must be distinct")
    if normalize_d7_config_for_pair(
        resolved_variants["b0"]["config"]
    ) != normalize_d7_config_for_pair(resolved_variants["v"]["config"]):
        raise D7PairError(
            "B0/V D7 configs differ outside checkpoint/provenance fields"
        )
    return {
        "path": config_path,
        "sha256": _sha256(config_path),
        "raw": raw,
        "repo_root": repo_root,
        "pair_run_id": pair_run_id,
        "result_root": result_root,
        "pair_directory": result_root / pair_run_id,
        "test_manifest_path": test_manifest_path,
        "test_manifest_sha256": raw["visual_test_manifest_sha256"],
        "variants": resolved_variants,
    }


def build_d7_pair_plans(contract):
    """Build both existing M0 run plans and verify their frozen ordering."""
    plans = {}
    expected_order = [
        (route_id, seed) for route_id in ROUTE_ORDER for seed in SEEDS
    ]
    for variant in VARIANTS:
        spec = contract["variants"][variant]
        try:
            plan = build_run_plan(
                config_path=spec["config_path"],
                repo_root=contract["repo_root"],
                run_id=spec["run_id"],
                route_set="development_d7",
                route_ids=list(ROUTE_ORDER),
                seeds=list(SEEDS),
            )
        except RunnerError as exc:
            raise D7PairError(f"{variant} D7 preflight failed: {exc}") from exc
        actual_order = [
            (attempt["route_id"], attempt["traffic_manager_seed"])
            for attempt in plan["attempts"]
        ]
        if actual_order != expected_order:
            raise D7PairError(f"{variant} run plan order differs from frozen order")
        configured_checkpoint = Path(spec["config"]["checkpoint"]["path"])
        configured_checkpoint = (
            configured_checkpoint
            if configured_checkpoint.is_absolute()
            else contract["repo_root"] / configured_checkpoint
        ).resolve()
        if Path(plan["checkpoint_path"]).resolve() != configured_checkpoint:
            raise D7PairError(f"{variant} run plan checkpoint differs")
        if Path(plan["result_root"]).resolve() != contract["result_root"]:
            raise D7PairError(f"{variant} run plan result root differs")
        plans[variant] = plan
    run_directories = {
        Path(plan["run_directory"]).resolve() for plan in plans.values()
    }
    if len(run_directories) != 2 or contract["pair_directory"] in run_directories:
        raise D7PairError("pair and child result directories must be distinct")
    return plans


def _validate_child_result(manifest, variant):
    summary = manifest.get("summary") or {}
    if summary.get("planned_attempts") != 21:
        raise D7PairError(f"{variant} D7 planned attempts must be 21")
    if summary.get("recorded_attempts") != 21:
        raise D7PairError(f"{variant} D7 recorded attempts must be 21")
    if summary.get("pipeline_valid_attempts") != 21:
        raise D7PairError(f"{variant} D7 must contain 21 pipeline-valid attempts")
    if summary.get("pipeline_invalid_attempts") != 0:
        raise D7PairError(f"{variant} D7 contains pipeline-invalid attempts")


def execute_d7_pair(config_path, pair_run_id, resume=False, repo_root=REPO_ROOT):
    """Execute one idempotent B0→V parent run using the existing child runner."""
    contract = load_d7_pair_contract(config_path, repo_root=repo_root)
    if pair_run_id != contract["pair_run_id"]:
        raise D7PairError(
            f"pair_run_id must match preregistered value: {contract['pair_run_id']}"
        )
    plans = build_d7_pair_plans(contract)
    pair_dir = contract["pair_directory"]
    manifest_path = pair_dir / "pair_manifest.json"
    if pair_dir.exists() and not resume:
        raise D7PairError(f"pair run directory already exists: {pair_dir}")
    if not pair_dir.exists():
        for plan in plans.values():
            if Path(plan["run_directory"]).exists():
                raise D7PairError(
                    f"child run directory exists without pair resume: {plan['run_directory']}"
                )
        pair_dir.mkdir(parents=True)
        shutil.copy2(contract["path"], pair_dir / "visual_d7_pair_config.json")
        manifest = {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "pair_run_id": pair_run_id,
            "status": "running",
            "pipeline_valid": False,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "pid": os.getpid(),
            "config": str(contract["path"]),
            "config_sha256": contract["sha256"],
            "visual_test_manifest": str(contract["test_manifest_path"]),
            "visual_test_manifest_sha256": contract["test_manifest_sha256"],
            "variant_order": list(VARIANTS),
            "route_order": list(ROUTE_ORDER),
            "seeds": list(SEEDS),
            "variants": [],
            "errors": [],
        }
        _write_json_atomic(manifest_path, manifest)
    else:
        manifest = _read_json(manifest_path, "visual D7 pair manifest")
        if manifest.get("config_sha256") != contract["sha256"]:
            raise D7PairError("resume pair config hash differs")
        if manifest.get("variant_order") != list(VARIANTS):
            raise D7PairError("resume variant order differs")
        if (
            manifest.get("status") == "completed"
            and manifest.get("pipeline_valid") is True
        ):
            return manifest
        previous_variants = manifest.get("variants") or []
        if [item.get("variant") for item in previous_variants] not in (
            [],
            ["b0"],
            ["b0", "v"],
        ):
            raise D7PairError("resume manifest variant prefix is invalid")
        manifest.setdefault("resume_history", []).append(
            {
                "status": manifest.get("status"),
                "errors": manifest.get("errors") or [],
                "completed_at": manifest.get("completed_at"),
                "resumed_at": _utc_now(),
            }
        )
        manifest.update(
            {
                "status": "running",
                "pipeline_valid": False,
                "updated_at": _utc_now(),
                "pid": os.getpid(),
                "errors": [],
                "completed_at": None,
            }
        )
        _write_json_atomic(manifest_path, manifest)

    completed = {
        item.get("variant"): item for item in manifest.get("variants", [])
    }
    try:
        for variant in VARIANTS:
            plan = plans[variant]
            child_dir = Path(plan["run_directory"])
            child_manifest = execute_run_plan(
                plan,
                repo_root=contract["repo_root"],
                resume=child_dir.exists(),
            )
            _validate_child_result(child_manifest, variant)
            child_manifest_path = child_dir / "run_manifest.json"
            completed[variant] = {
                "variant": variant,
                "run_id": plan["run_id"],
                "config_sha256": plan["config_sha256"],
                "checkpoint_sha256": contract["variants"][variant][
                    "checkpoint_sha256"
                ],
                "manifest": str(child_manifest_path),
                "manifest_sha256": _sha256(child_manifest_path),
                "pipeline_valid": True,
                "recorded_attempts": 21,
            }
            manifest["variants"] = [
                completed[name] for name in VARIANTS if name in completed
            ]
            manifest["updated_at"] = _utc_now()
            _write_json_atomic(manifest_path, manifest)
        manifest.update(
            {
                "status": "completed",
                "pipeline_valid": True,
                "updated_at": _utc_now(),
                "completed_at": _utc_now(),
                "comparability": {
                    "only_checkpoint_and_provenance_differ": True,
                    "attempt_order_identical": True,
                    "total_pipeline_valid_attempts": 42,
                },
            }
        )
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "pipeline_valid": False,
                "updated_at": _utc_now(),
                "completed_at": _utc_now(),
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
        )
        _write_json_atomic(manifest_path, manifest)
        raise
    _write_json_atomic(manifest_path, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Preflight or execute one serial B0/V D7 pair."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--pair-run-id")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.execute or args.resume:
            if not args.pair_run_id:
                raise D7PairError("execution requires --pair-run-id")
            manifest = execute_d7_pair(
                args.config,
                args.pair_run_id,
                resume=args.resume,
                repo_root=args.repo_root,
            )
            output = {
                "pair_run_id": manifest["pair_run_id"],
                "status": manifest["status"],
                "pipeline_valid": manifest["pipeline_valid"],
            }
        else:
            contract = load_d7_pair_contract(args.config, repo_root=args.repo_root)
            plans = build_d7_pair_plans(contract)
            output = {
                "pair_run_id": contract["pair_run_id"],
                "config_sha256": contract["sha256"],
                "variant_order": list(VARIANTS),
                "attempts_per_variant": {
                    variant: len(plans[variant]["attempts"])
                    for variant in VARIANTS
                },
            }
    except (D7PairError, RunnerError) as exc:
        print(f"visual D7 pair error: {exc}")
        return 2
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
