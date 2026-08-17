#!/usr/bin/env python3
"""
[INPUT]: 依赖 validation-only scene contract、pipeline-valid 的B0/candidate初始化manifest、严格checkpoint哈希与冻结保持门禁。
[OUTPUT]: 对外提供 DirectSceneProbeError、load_direct_scene_probe_contract、execute_direct_scene_probe 与 CLI，在相同validation上生成零微调B0/candidate整体/行人/非行人配对证据。
[POS]: tools/evaluation 的 M2 H1 修复候选回接门禁；不经过Stage 2即先判断新RGB骨干是否保留官方普通驾驶表征，显式拒绝test。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import argparse
import copy
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERFUSER_ROOT = REPO_ROOT / "interfuser"
for import_root in (REPO_ROOT, INTERFUSER_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tools.evaluation.run_interfuser_scene_validation import (  # noqa: E402
    _evaluate_variant,
    build_metric_comparison,
    load_scene_validation_contract,
    sha256_file,
)
from tools.evaluation.run_interfuser_weight_interpolation_probe import (  # noqa: E402
    evaluate_candidate_gate,
)
from tools.evaluation.runtime_resources import (  # noqa: E402
    RunnerError,
    _GpuMemoryMonitor,
    ensure_gpus_have_free_memory,
)


CONFIG_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
VARIANTS = ("b0", "candidate")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


class DirectSceneProbeError(RuntimeError):
    """Raised when direct candidate provenance or validation execution is invalid."""


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _read_json(path, label):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DirectSceneProbeError(f"unable to read {label} JSON {path}: {exc}") from exc


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _resolve_path(repo_root, value, label, require_file=True):
    if not isinstance(value, str) or not value:
        raise DirectSceneProbeError(f"{label} must be a non-empty path")
    path = Path(value)
    path = path if path.is_absolute() else Path(repo_root) / path
    path = path.resolve()
    if require_file and not path.is_file():
        raise DirectSceneProbeError(f"{label} is not a file: {path}")
    return path


def _verify_hash(path, expected, label):
    if not isinstance(expected, str) or len(expected) != 64:
        raise DirectSceneProbeError(f"{label} SHA-256 must have 64 characters")
    actual = sha256_file(path)
    if actual != expected:
        raise DirectSceneProbeError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def _git_output(repo_root, *args):
    result = subprocess.run(
        ["git", *args], cwd=repo_root, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise DirectSceneProbeError(
            f"git {' '.join(args)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _positive_int(value, label, allow_zero=False):
    valid = isinstance(value, int) and not isinstance(value, bool)
    valid = valid and (value >= 0 if allow_zero else value > 0)
    if not valid:
        raise DirectSceneProbeError(f"{label} must be a valid integer")
    return value


def load_direct_scene_probe_contract(config_path, repo_root=REPO_ROOT):
    repo_root = Path(repo_root).resolve()
    config_path = Path(config_path).resolve()
    raw = _read_json(config_path, "direct scene probe config")
    if raw.get("schema_version") != CONFIG_SCHEMA_VERSION or raw.get("status") != "frozen":
        raise DirectSceneProbeError("config must be frozen schema v1")
    run_id = raw.get("run_id")
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise DirectSceneProbeError("run_id is invalid")
    source = raw.get("source_scene_validation_config") or {}
    source_path = _resolve_path(
        repo_root, source.get("path"), "source scene validation config"
    )
    source_hash = _verify_hash(
        source_path, source.get("sha256"), "source scene validation config"
    )
    scene_contract = load_scene_validation_contract(source_path, repo_root)
    initialization = raw.get("initialization_manifest") or {}
    initialization_path = _resolve_path(
        repo_root, initialization.get("path"), "initialization manifest"
    )
    initialization_hash = _verify_hash(
        initialization_path,
        initialization.get("sha256"),
        "initialization manifest",
    )
    initialization_value = _read_json(initialization_path, "initialization manifest")
    if initialization_value.get("pipeline_valid") is not True:
        raise DirectSceneProbeError("initialization manifest must be pipeline-valid")
    variants = raw.get("variants") or {}
    if tuple(variants) != VARIANTS:
        raise DirectSceneProbeError("variants must define b0 then candidate")
    resolved_variants = {}
    for name in VARIANTS:
        spec = variants[name]
        checkpoint = _resolve_path(
            repo_root, spec.get("checkpoint"), f"{name} checkpoint"
        )
        checkpoint_hash = _verify_hash(
            checkpoint, spec.get("checkpoint_sha256"), f"{name} checkpoint"
        )
        manifest_name = spec.get("initialization_manifest_variant")
        manifest_variant = (initialization_value.get("variants") or {}).get(
            manifest_name, {}
        )
        if Path(manifest_variant.get("checkpoint", "")).resolve() != checkpoint or (
            manifest_variant.get("checkpoint_sha256") != checkpoint_hash
        ):
            raise DirectSceneProbeError(f"{name} differs from initialization manifest")
        resolved_variants[name] = {
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_hash,
        }
    acceptance = raw.get("acceptance") or {}
    required_acceptance = {
        "minimum_pedestrian_core_improvements",
        "maximum_nonpedestrian_core_degradation_percent",
        "maximum_overall_traffic_degradation_percent",
        "maximum_pedestrian_temporal_degradation_percent",
    }
    if set(acceptance) != required_acceptance:
        raise DirectSceneProbeError("acceptance fields differ from retention contract")
    runtime = raw.get("runtime") or {}
    for field in ("seed", "gpu", "batch_size", "workers", "log_interval_batches"):
        _positive_int(runtime.get(field), field, field in {"seed", "gpu", "workers"})
    _positive_int(runtime.get("gpu_minimum_free_memory_mb"), "GPU free memory")
    if not isinstance(runtime.get("require_clean_git"), bool):
        raise DirectSceneProbeError("runtime.require_clean_git must be boolean")
    result_root = _resolve_path(repo_root, raw.get("result_root"), "result_root", False)
    try:
        result_root.relative_to(repo_root)
    except ValueError as exc:
        raise DirectSceneProbeError("result_root escapes repository") from exc
    return {
        **raw,
        "path": config_path,
        "sha256": sha256_file(config_path),
        "repo_root": repo_root,
        "source_config_path": source_path,
        "source_config_sha256": source_hash,
        "scene_contract": scene_contract,
        "initialization_manifest_path": initialization_path,
        "initialization_manifest_sha256": initialization_hash,
        "resolved_variants": resolved_variants,
        "result_root_path": result_root,
    }


def execute_direct_scene_probe(config_path):
    contract = load_direct_scene_probe_contract(config_path)
    runtime = contract["runtime"]
    if runtime["require_clean_git"] and _git_output(
        contract["repo_root"], "status", "--porcelain"
    ):
        raise DirectSceneProbeError("Git worktree must be clean")
    run_dir = contract["result_root_path"] / contract["run_id"]
    if run_dir.exists():
        raise DirectSceneProbeError(f"refusing to overwrite run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "direct_scene_probe_manifest.json"
    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": contract["run_id"],
        "status": "running",
        "pipeline_valid": False,
        "created_at": _utc_now(),
        "git_head": _git_output(contract["repo_root"], "rev-parse", "HEAD"),
        "git_status": "",
        "config": str(contract["path"]),
        "config_sha256": contract["sha256"],
        "initialization_manifest": str(contract["initialization_manifest_path"]),
        "initialization_manifest_sha256": contract[
            "initialization_manifest_sha256"
        ],
        "test_accessed": False,
        "variants": [],
        "errors": [],
    }
    _write_json(manifest_path, manifest)
    gpu = runtime["gpu"]
    monitor = _GpuMemoryMonitor([gpu])
    try:
        ensure_gpus_have_free_memory([gpu], runtime["gpu_minimum_free_memory_mb"])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        monitor.start()
        evaluation_contract = copy.deepcopy(contract["scene_contract"])
        evaluation_contract["runtime"] = dict(runtime)
        evaluation_contract["resolved_variants"] = contract["resolved_variants"]
        for variant in VARIANTS:
            result = _evaluate_variant(evaluation_contract, variant)
            manifest["variants"].append(result)
            manifest["updated_at"] = _utc_now()
            _write_json(manifest_path, manifest)
        comparison = build_metric_comparison(*manifest["variants"])
        gate = evaluate_candidate_gate(comparison, contract["acceptance"])
        manifest["comparison"] = comparison
        manifest["gate"] = gate
        manifest.update(
            {
                "status": "completed",
                "pipeline_valid": True,
                "stage2_admitted": gate["passed"],
                "completed_at": _utc_now(),
            }
        )
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "pipeline_valid": False,
                "completed_at": _utc_now(),
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
        )
        raise
    finally:
        manifest["gpu_peak_memory_mb"] = monitor.stop()
        manifest["updated_at"] = _utc_now()
        _write_json(manifest_path, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare zero-finetune B0 and repaired RGB candidate on validation"
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = execute_direct_scene_probe(args.config)
    except (DirectSceneProbeError, RunnerError) as exc:
        print(f"direct scene probe error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"direct scene probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(manifest["gate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
