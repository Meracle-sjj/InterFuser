#!/usr/bin/env python3
"""
[INPUT]: 依赖 pipeline-valid 的 Stage 2 scene validation、其哈希绑定 B0/V best checkpoint、冻结插值 alpha 与 validation-only 分层指标执行器。
[OUTPUT]: 对外提供 InterpolationProbeError、interpolate_state_dicts、evaluate_candidate_gate、execute_interpolation_probe 与 CLI，生成全模型浮点权重插值候选及整体/行人/非行人配对证据。
[POS]: tools/evaluation 的 M2 H1 负迁移修复筛选器；以零额外训练的权重空间折中测试稳定性-可塑性边界，失败后才准入教师保持或 L2-SP。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import argparse
import copy
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


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
from tools.evaluation.runtime_resources import (  # noqa: E402
    RunnerError,
    _GpuMemoryMonitor,
    ensure_gpus_have_free_memory,
)
from tools.training.interfuser_visual_swap_pair import state_dict_sha256  # noqa: E402


CONFIG_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
CORE_METRICS = (
    "traffic_average_precision",
    "traffic_roc_auc",
    "traffic_occupied_iou",
    "waypoint_ade",
    "waypoint_fde_horizon_10",
)
OVERALL_TRAFFIC_METRICS = (
    "traffic_average_precision",
    "traffic_roc_auc",
    "traffic_occupied_iou",
)
PEDESTRIAN_TEMPORAL_METRICS = (
    "traffic_delta_residual_mae",
    "waypoint_delta_residual_ade",
)
RGB_PREFIXES = ("rgb_backbone.", "rgb_patch_embed.backbone.")


class InterpolationProbeError(RuntimeError):
    """Raised when interpolation provenance, tensors or retention gates are invalid."""


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _read_json(path, label):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InterpolationProbeError(f"unable to read {label} JSON {path}: {exc}") from exc


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
        raise InterpolationProbeError(f"{label} must be a non-empty path")
    path = Path(value)
    path = path if path.is_absolute() else Path(repo_root) / path
    path = path.resolve()
    if require_file and not path.is_file():
        raise InterpolationProbeError(f"{label} is not a file: {path}")
    return path


def _verify_hash(path, expected, label):
    if not isinstance(expected, str) or len(expected) != 64:
        raise InterpolationProbeError(f"{label} SHA-256 must have 64 characters")
    actual = sha256_file(path)
    if actual != expected:
        raise InterpolationProbeError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def _git_output(repo_root, *args):
    result = subprocess.run(
        ["git", *args], cwd=repo_root, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise InterpolationProbeError(
            f"git {' '.join(args)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _positive_number(value, label, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InterpolationProbeError(f"{label} must be numeric")
    value = float(value)
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise InterpolationProbeError(f"{label} must be finite and positive")
    return value


def _positive_int(value, label, allow_zero=False):
    valid = isinstance(value, int) and not isinstance(value, bool)
    valid = valid and (value >= 0 if allow_zero else value > 0)
    if not valid:
        raise InterpolationProbeError(f"{label} must be a valid integer")
    return value


def load_interpolation_probe_contract(config_path, repo_root=REPO_ROOT):
    repo_root = Path(repo_root).resolve()
    config_path = Path(config_path).resolve()
    raw = _read_json(config_path, "interpolation probe config")
    if raw.get("schema_version") != CONFIG_SCHEMA_VERSION or raw.get("status") != "frozen":
        raise InterpolationProbeError("config must be frozen schema v1")
    run_id = raw.get("run_id")
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise InterpolationProbeError("run_id is invalid")
    source_config = raw.get("source_scene_validation_config") or {}
    source_config_path = _resolve_path(
        repo_root, source_config.get("path"), "source scene validation config"
    )
    source_config_hash = _verify_hash(
        source_config_path,
        source_config.get("sha256"),
        "source scene validation config",
    )
    scene_contract = load_scene_validation_contract(source_config_path, repo_root)
    source_manifest = raw.get("source_scene_validation_manifest") or {}
    source_manifest_path = _resolve_path(
        repo_root, source_manifest.get("path"), "source scene validation manifest"
    )
    source_manifest_hash = _verify_hash(
        source_manifest_path,
        source_manifest.get("sha256"),
        "source scene validation manifest",
    )
    source = _read_json(source_manifest_path, "source scene validation manifest")
    if source.get("status") != "completed" or source.get("pipeline_valid") is not True:
        raise InterpolationProbeError("source scene validation must be completed")
    if source.get("test_accessed") is not False:
        raise InterpolationProbeError("source scene validation accessed test")
    if Path(source.get("config", "")).resolve() != source_config_path or source.get(
        "config_sha256"
    ) != source_config_hash:
        raise InterpolationProbeError("source manifest differs from bound config")
    source_variants = {item.get("variant"): item for item in source.get("variants", [])}
    if set(source_variants) != {"b0", "v"}:
        raise InterpolationProbeError("source validation must contain b0 and v")
    for name in ("b0", "v"):
        resolved = scene_contract["resolved_variants"][name]
        if source_variants[name].get("checkpoint_sha256") != resolved[
            "checkpoint_sha256"
        ]:
            raise InterpolationProbeError(f"source {name} checkpoint differs")

    interpolation = raw.get("interpolation") or {}
    if interpolation.get("scope") not in {
        "all_floating_tensors",
        "rgb_floating_tensors",
    }:
        raise InterpolationProbeError("interpolation scope is unsupported")
    if interpolation.get("nonfloating_policy") != "copy_b0":
        raise InterpolationProbeError("nonfloating policy must be copy_b0")
    alphas = interpolation.get("alphas")
    if not isinstance(alphas, list) or not alphas or alphas != sorted(set(alphas)):
        raise InterpolationProbeError("alphas must be unique and sorted")
    if any(
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not 0.0 < float(alpha) < 1.0
        for alpha in alphas
    ):
        raise InterpolationProbeError("every alpha must be in (0, 1)")

    acceptance = raw.get("acceptance") or {}
    _positive_int(
        acceptance.get("minimum_pedestrian_core_improvements"),
        "minimum pedestrian core improvements",
    )
    if acceptance["minimum_pedestrian_core_improvements"] > len(CORE_METRICS):
        raise InterpolationProbeError("pedestrian improvement target exceeds core metrics")
    for field in (
        "maximum_nonpedestrian_core_degradation_percent",
        "maximum_overall_traffic_degradation_percent",
        "maximum_pedestrian_temporal_degradation_percent",
    ):
        _positive_number(acceptance.get(field), field, allow_zero=True)
    runtime = raw.get("runtime") or {}
    for field in ("seed", "gpu", "batch_size", "workers", "log_interval_batches"):
        _positive_int(runtime.get(field), field, field in {"seed", "gpu", "workers"})
    _positive_int(runtime.get("gpu_minimum_free_memory_mb"), "GPU free memory")
    if not isinstance(runtime.get("require_clean_git"), bool):
        raise InterpolationProbeError("runtime.require_clean_git must be boolean")
    result_root = _resolve_path(repo_root, raw.get("result_root"), "result_root", False)
    try:
        result_root.relative_to(repo_root)
    except ValueError as exc:
        raise InterpolationProbeError("result_root escapes repository") from exc
    return {
        **raw,
        "path": config_path,
        "sha256": sha256_file(config_path),
        "repo_root": repo_root,
        "source_config_path": source_config_path,
        "source_config_sha256": source_config_hash,
        "source_manifest_path": source_manifest_path,
        "source_manifest_sha256": source_manifest_hash,
        "source_manifest_loaded": source,
        "scene_contract": scene_contract,
        "result_root_path": result_root,
    }


def interpolate_state_dicts(
    b0_state, v_state, alpha, scope="all_floating_tensors"
):
    """Interpolate selected floating tensors and retain the remaining B0 state."""
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise InterpolationProbeError("alpha must be in (0, 1)")
    if scope not in {"all_floating_tensors", "rgb_floating_tensors"}:
        raise InterpolationProbeError("interpolation scope is unsupported")
    if tuple(b0_state) != tuple(v_state):
        raise InterpolationProbeError("source state schemas have different keys/order")
    output = {}
    floating = 0
    floating_copied = 0
    nonfloating = 0
    nonfloating_differences = 0
    for name, b0_tensor in b0_state.items():
        v_tensor = v_state[name]
        if (
            not torch.is_tensor(b0_tensor)
            or not torch.is_tensor(v_tensor)
            or b0_tensor.shape != v_tensor.shape
            or b0_tensor.dtype != v_tensor.dtype
        ):
            raise InterpolationProbeError(f"source tensor schema differs: {name}")
        selected = scope == "all_floating_tensors" or name.startswith(RGB_PREFIXES)
        if (b0_tensor.is_floating_point() or b0_tensor.is_complex()) and selected:
            output[name] = torch.lerp(b0_tensor, v_tensor, alpha).detach().clone()
            floating += 1
        elif b0_tensor.is_floating_point() or b0_tensor.is_complex():
            output[name] = b0_tensor.detach().clone()
            floating_copied += 1
        else:
            output[name] = b0_tensor.detach().clone()
            nonfloating += 1
            nonfloating_differences += int(not torch.equal(b0_tensor, v_tensor))
    return output, {
        "floating_tensors_interpolated": floating,
        "floating_tensors_copied_from_b0": floating_copied,
        "nonfloating_tensors_copied_from_b0": nonfloating,
        "nonfloating_source_differences": nonfloating_differences,
    }


def _degradation_percent(metric):
    relative = metric.get("relative_percent")
    if relative is None:
        return float("inf")
    return max(0.0, -relative if metric["higher_is_better"] else relative)


def evaluate_candidate_gate(comparison, acceptance):
    cohorts = comparison["cohorts"]
    pedestrian_improvements = sum(
        cohorts["pedestrian"][name]["v_improved"] for name in CORE_METRICS
    )
    nonpedestrian_degradation = max(
        _degradation_percent(cohorts["nonpedestrian"][name])
        for name in CORE_METRICS
    )
    overall_traffic_degradation = max(
        _degradation_percent(cohorts["overall"][name])
        for name in OVERALL_TRAFFIC_METRICS
    )
    pedestrian_temporal_degradation = max(
        _degradation_percent(cohorts["pedestrian"][name])
        for name in PEDESTRIAN_TEMPORAL_METRICS
    )
    checks = {
        "pedestrian_core": pedestrian_improvements
        >= acceptance["minimum_pedestrian_core_improvements"],
        "nonpedestrian_retention": nonpedestrian_degradation
        <= acceptance["maximum_nonpedestrian_core_degradation_percent"],
        "overall_traffic_retention": overall_traffic_degradation
        <= acceptance["maximum_overall_traffic_degradation_percent"],
        "pedestrian_temporal_retention": pedestrian_temporal_degradation
        <= acceptance["maximum_pedestrian_temporal_degradation_percent"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "pedestrian_core_improvements": pedestrian_improvements,
        "maximum_nonpedestrian_core_degradation_percent": nonpedestrian_degradation,
        "maximum_overall_traffic_degradation_percent": overall_traffic_degradation,
        "maximum_pedestrian_temporal_degradation_percent": pedestrian_temporal_degradation,
    }


def _candidate_name(alpha):
    return f"alpha_{int(round(float(alpha) * 100)):03d}"


def _load_checkpoint(path, label):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise InterpolationProbeError(f"{label} checkpoint has no state_dict")
    if payload.get("arch") != "interfuser_baseline":
        raise InterpolationProbeError(f"{label} checkpoint architecture differs")
    return payload


def _candidate_rank(candidate):
    gate = candidate["gate"]
    return (
        int(gate["passed"]),
        gate["pedestrian_core_improvements"],
        -gate["maximum_nonpedestrian_core_degradation_percent"],
        -gate["maximum_overall_traffic_degradation_percent"],
        -abs(candidate["alpha"] - 0.5),
    )


def execute_interpolation_probe(config_path):
    contract = load_interpolation_probe_contract(config_path)
    runtime = contract["runtime"]
    if runtime["require_clean_git"] and _git_output(
        contract["repo_root"], "status", "--porcelain"
    ):
        raise InterpolationProbeError("Git worktree must be clean")
    run_dir = contract["result_root_path"] / contract["run_id"]
    if run_dir.exists():
        raise InterpolationProbeError(f"refusing to overwrite run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "interpolation_probe_manifest.json"
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
        "source_scene_validation_manifest": str(contract["source_manifest_path"]),
        "source_scene_validation_manifest_sha256": contract[
            "source_manifest_sha256"
        ],
        "test_accessed": False,
        "candidates": [],
        "errors": [],
    }
    _write_json(manifest_path, manifest)
    source_contract = contract["scene_contract"]
    b0_payload = _load_checkpoint(
        source_contract["resolved_variants"]["b0"]["checkpoint"], "b0"
    )
    v_payload = _load_checkpoint(
        source_contract["resolved_variants"]["v"]["checkpoint"], "v"
    )
    b0_result = {
        item["variant"]: item for item in contract["source_manifest_loaded"]["variants"]
    }["b0"]
    gpu = runtime["gpu"]
    monitor = _GpuMemoryMonitor([gpu])
    try:
        ensure_gpus_have_free_memory([gpu], runtime["gpu_minimum_free_memory_mb"])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        monitor.start()
        for alpha in contract["interpolation"]["alphas"]:
            name = _candidate_name(alpha)
            state, tensor_summary = interpolate_state_dicts(
                b0_payload["state_dict"],
                v_payload["state_dict"],
                alpha,
                contract["interpolation"]["scope"],
            )
            checkpoint_path = run_dir / f"{name}.pth"
            torch.save(
                {
                    "arch": "interfuser_baseline",
                    "state_dict": state,
                    "interpolation": {
                        "alpha": float(alpha),
                        "scope": contract["interpolation"]["scope"],
                        "nonfloating_policy": "copy_b0",
                    },
                },
                checkpoint_path,
            )
            candidate_contract = copy.deepcopy(source_contract)
            candidate_contract["runtime"] = dict(runtime)
            candidate_contract["resolved_variants"][name] = {
                "checkpoint": checkpoint_path,
                "checkpoint_sha256": sha256_file(checkpoint_path),
            }
            result = _evaluate_variant(candidate_contract, name)
            comparison = build_metric_comparison(b0_result, result)
            gate = evaluate_candidate_gate(comparison, contract["acceptance"])
            candidate = {
                "name": name,
                "alpha": float(alpha),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "state_sha256": state_dict_sha256(state),
                "tensor_summary": tensor_summary,
                "evaluation": result,
                "comparison_to_b0": comparison,
                "gate": gate,
            }
            manifest["candidates"].append(candidate)
            manifest["updated_at"] = _utc_now()
            _write_json(manifest_path, manifest)
            del state
        ranked = sorted(manifest["candidates"], key=_candidate_rank, reverse=True)
        passed = [candidate for candidate in ranked if candidate["gate"]["passed"]]
        manifest["selection"] = {
            "candidate_order": [candidate["name"] for candidate in ranked],
            "passing_candidates": [candidate["name"] for candidate in passed],
            "recommended_candidate": passed[0]["name"] if passed else None,
            "full_retraining_admitted": bool(passed),
        }
        manifest.update(
            {
                "status": "completed",
                "pipeline_valid": True,
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
        description="Probe B0/V weight interpolation against scene retention gates"
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = execute_interpolation_probe(args.config)
    except (InterpolationProbeError, RunnerError) as exc:
        print(f"interpolation probe error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"interpolation probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(manifest["selection"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
