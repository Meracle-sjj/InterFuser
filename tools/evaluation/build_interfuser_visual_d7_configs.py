#!/usr/bin/env python3
"""
[INPUT]: 依赖预注册 D7 build 配置、M0 baseline 模板、completed+pipeline-valid formal/test manifest 与两组 test 实际使用的 best checkpoint。
[OUTPUT]: 对外提供 D7ConfigBuildError、load_d7_build_contract、build_d7_configs 与 CLI，只在 test 有效后一次性生成 B0/V child config 和 pair config。
[POS]: tools/evaluation 的 M2 H1 D7 配置冻结器；位于离线 test 与 D7 parent runner 之间，消除手工复制 checkpoint/provenance 的漂移风险。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import argparse
import copy
import hashlib
import json
import math
import re
from pathlib import Path

from tools.evaluation.summarize_interfuser_visual_d7 import (
    ROUTE_ORDER,
    SEEDS,
    VARIANTS,
    normalize_d7_config_for_pair,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_SCHEMA_VERSION = 1
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


class D7ConfigBuildError(RuntimeError):
    """Raised when final D7 configs cannot be frozen without provenance drift."""


def _read_json(path, label):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise D7ConfigBuildError(f"cannot read {label} JSON {path}: {exc}") from exc


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _resolve_file(repo_root, value, expected_sha256, label):
    if not isinstance(value, str) or not value:
        raise D7ConfigBuildError(f"{label} path must be non-empty")
    path = Path(value)
    path = path if path.is_absolute() else Path(repo_root) / path
    path = path.resolve()
    if not path.is_file():
        raise D7ConfigBuildError(f"{label} is not a file: {path}")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise D7ConfigBuildError(f"{label} SHA-256 is invalid")
    actual = _sha256(path)
    if actual != expected_sha256:
        raise D7ConfigBuildError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return path


def _resolve_inside_repo(repo_root, value, label):
    if not isinstance(value, str) or not value:
        raise D7ConfigBuildError(f"{label} must be a non-empty path")
    path = Path(value)
    path = path if path.is_absolute() else Path(repo_root) / path
    path = path.resolve()
    try:
        path.relative_to(Path(repo_root).resolve())
    except ValueError as exc:
        raise D7ConfigBuildError(f"{label} escapes repository") from exc
    return path


def _relative(repo_root, path):
    try:
        return str(Path(path).resolve().relative_to(Path(repo_root).resolve()))
    except ValueError:
        return str(Path(path).resolve())


def _valid_run_id(value, label):
    if not isinstance(value, str) or not RUN_ID_PATTERN.fullmatch(value):
        raise D7ConfigBuildError(f"{label} is invalid")
    return value


def _serialize(value):
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def load_d7_build_contract(config_path, repo_root=REPO_ROOT):
    """Validate every static D7 identity while allowing test output to be absent."""
    repo_root = Path(repo_root).resolve()
    config_path = Path(config_path).resolve()
    raw = _read_json(config_path, "D7 build config")
    if raw.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise D7ConfigBuildError("D7 build config schema must be v1")
    if raw.get("status") != "preregistered":
        raise D7ConfigBuildError("D7 build config must be preregistered")

    baseline_path = _resolve_file(
        repo_root,
        raw.get("baseline_template"),
        raw.get("baseline_template_sha256"),
        "baseline template",
    )
    baseline = _read_json(baseline_path, "baseline template")
    if baseline.get("schema_version") != 1 or baseline.get("status") != "frozen":
        raise D7ConfigBuildError("baseline template must be frozen schema v1")
    runtime = baseline.get("runtime") or {}
    expected_runtime = {
        "agent_cuda_visible_device": 6,
        "carla_graphics_adapter": 7,
        "carla_port": 2155,
        "traffic_manager_port": 2255,
    }
    for field, expected in expected_runtime.items():
        if runtime.get(field) != expected:
            raise D7ConfigBuildError(f"baseline runtime {field} differs")
    if set(baseline.get("route_sets", {}).get("development_d7", [])) != set(
        ROUTE_ORDER
    ):
        raise D7ConfigBuildError("baseline development_d7 differs")
    if baseline.get("random_seeds") != list(SEEDS):
        raise D7ConfigBuildError("baseline seeds differ")

    formal_config_path = _resolve_file(
        repo_root,
        raw.get("formal_training_config"),
        raw.get("formal_training_config_sha256"),
        "formal training config",
    )
    formal_config = _read_json(formal_config_path, "formal training config")
    if formal_config.get("status") != "formal":
        raise D7ConfigBuildError("formal training config status differs")

    test_config_path = _resolve_file(
        repo_root,
        raw.get("visual_test_config"),
        raw.get("visual_test_config_sha256"),
        "visual test config",
    )
    test_config = _read_json(test_config_path, "visual test config")
    if test_config.get("run_id") != raw.get("visual_test_run_id"):
        raise D7ConfigBuildError("visual test run ID differs")
    if test_config.get("training_run_id") != raw.get("formal_training_run_id"):
        raise D7ConfigBuildError("visual test binds a different formal run")
    if test_config.get("training_config_sha256") != raw.get(
        "formal_training_config_sha256"
    ):
        raise D7ConfigBuildError("visual test binds a different formal config")

    if raw.get("route_order") != list(ROUTE_ORDER):
        raise D7ConfigBuildError("route_order differs from frozen D7 order")
    if raw.get("seeds") != list(SEEDS):
        raise D7ConfigBuildError("seeds differ from frozen D7 seeds")
    result_root = _resolve_inside_repo(repo_root, raw.get("result_root"), "result_root")
    test_manifest_path = _resolve_inside_repo(
        repo_root, raw.get("visual_test_manifest"), "visual_test_manifest"
    )
    expected_test_manifest = (
        result_root / raw["visual_test_run_id"] / "test_manifest.json"
    )
    if test_manifest_path != expected_test_manifest:
        raise D7ConfigBuildError("visual test manifest path differs from frozen run ID")

    run_ids = raw.get("run_ids") or {}
    if tuple(run_ids) != ("pair", "b0", "v"):
        raise D7ConfigBuildError("run_ids must be ordered pair, b0, v")
    for name, value in run_ids.items():
        _valid_run_id(value, f"run_ids.{name}")
    if len(set(run_ids.values())) != 3:
        raise D7ConfigBuildError("D7 run IDs must be distinct")

    outputs = raw.get("output_configs") or {}
    if tuple(outputs) != ("b0", "v", "pair"):
        raise D7ConfigBuildError("output_configs must be ordered b0, v, pair")
    output_paths = {
        name: _resolve_inside_repo(repo_root, value, f"output_configs.{name}")
        for name, value in outputs.items()
    }
    if len(set(output_paths.values())) != 3:
        raise D7ConfigBuildError("D7 output config paths must be distinct")

    return {
        "path": config_path,
        "sha256": _sha256(config_path),
        "raw": raw,
        "repo_root": repo_root,
        "baseline_path": baseline_path,
        "baseline": baseline,
        "formal_config_path": formal_config_path,
        "formal_config": formal_config,
        "test_config_path": test_config_path,
        "test_config": test_config,
        "test_manifest_path": test_manifest_path,
        "result_root": result_root,
        "output_paths": output_paths,
    }


def _validated_dynamic_inputs(contract):
    raw = contract["raw"]
    test_manifest_path = contract["test_manifest_path"]
    if not test_manifest_path.is_file():
        raise D7ConfigBuildError(
            "visual test manifest is absent; D7 configs cannot be generated"
        )
    test_manifest = _read_json(test_manifest_path, "visual test manifest")
    if (
        test_manifest.get("run_id") != raw["visual_test_run_id"]
        or test_manifest.get("status") != "completed"
        or test_manifest.get("pipeline_valid") is not True
    ):
        raise D7ConfigBuildError("visual test is not completed and pipeline valid")
    if test_manifest.get("config_sha256") != raw["visual_test_config_sha256"]:
        raise D7ConfigBuildError("visual test config hash differs")

    formal_manifest_value = test_manifest.get("formal_training_manifest")
    formal_manifest_sha256 = test_manifest.get("formal_training_manifest_sha256")
    formal_manifest_path = _resolve_file(
        contract["repo_root"],
        formal_manifest_value,
        formal_manifest_sha256,
        "formal training manifest",
    )
    expected_formal_manifest = (
        contract["result_root"]
        / raw["formal_training_run_id"]
        / "run_manifest.json"
    )
    if formal_manifest_path != expected_formal_manifest:
        raise D7ConfigBuildError("formal manifest path differs from frozen run ID")
    formal_manifest = _read_json(formal_manifest_path, "formal training manifest")
    if (
        formal_manifest.get("run_id") != raw["formal_training_run_id"]
        or formal_manifest.get("status") != "completed"
        or formal_manifest.get("pipeline_valid") is not True
    ):
        raise D7ConfigBuildError("formal training is not completed and pipeline valid")
    if formal_manifest.get("config_sha256") != raw["formal_training_config_sha256"]:
        raise D7ConfigBuildError("formal training config hash differs")

    formal_variants = formal_manifest.get("variants") or []
    test_variants = test_manifest.get("variants") or []
    if [item.get("variant") for item in formal_variants] != list(VARIANTS):
        raise D7ConfigBuildError("formal variants must be B0 then V")
    if [item.get("variant") for item in test_variants] != list(VARIANTS):
        raise D7ConfigBuildError("test variants must be B0 then V")
    test_by_variant = {item["variant"]: item for item in test_variants}
    validated = {}
    for item in formal_variants:
        variant = item["variant"]
        if (
            item.get("pipeline_valid") is not True
            or item.get("state_tensors") != 1132
            or item.get("errors")
        ):
            raise D7ConfigBuildError(f"formal {variant} variant is invalid")
        best = ((item.get("artifacts") or {}).get("best_checkpoint") or {})
        checkpoint_path = _resolve_file(
            contract["repo_root"],
            best.get("path"),
            best.get("sha256"),
            f"formal {variant} best checkpoint",
        )
        test_item = test_by_variant[variant]
        worker = test_item.get("worker_result") or {}
        if test_item.get("pipeline_valid") is not True:
            raise D7ConfigBuildError(f"visual test {variant} variant is invalid")
        if worker.get("checkpoint_sha256") != best.get("sha256"):
            raise D7ConfigBuildError(
                f"visual test {variant} checkpoint differs from formal best"
            )
        best_metric = item.get("best_metric")
        if not isinstance(best_metric, (int, float)) or not math.isfinite(best_metric):
            raise D7ConfigBuildError(f"formal {variant} best metric is invalid")
        validated[variant] = {
            "path": checkpoint_path,
            "sha256": best["sha256"],
            "epoch": item.get("best_epoch"),
            "best_metric": float(best_metric),
        }
    return {
        "test_manifest": test_manifest,
        "test_manifest_sha256": _sha256(test_manifest_path),
        "formal_manifest": formal_manifest,
        "formal_manifest_path": formal_manifest_path,
        "formal_manifest_sha256": formal_manifest_sha256,
        "variants": validated,
    }


def _child_config(contract, dynamic, variant):
    config = copy.deepcopy(contract["baseline"])
    config["result_root"] = contract["raw"]["result_root"]
    checkpoint = dynamic["variants"][variant]
    config["checkpoint"] = {
        "path": str(checkpoint["path"]),
        "sha256": checkpoint["sha256"],
        "architecture": "interfuser_baseline",
        "epoch": checkpoint["epoch"],
        "best_metric": checkpoint["best_metric"],
        "training_run_id": contract["raw"]["formal_training_run_id"],
        "training_config_sha256": contract["raw"][
            "formal_training_config_sha256"
        ],
    }
    config["comparison"] = {
        "schema_version": 1,
        "variant": variant,
        "formal_checkpoint_sha256": checkpoint["sha256"],
        "formal_training_manifest": _relative(
            contract["repo_root"], dynamic["formal_manifest_path"]
        ),
        "formal_training_manifest_sha256": dynamic["formal_manifest_sha256"],
        "visual_test_manifest": _relative(
            contract["repo_root"], contract["test_manifest_path"]
        ),
        "visual_test_manifest_sha256": dynamic["test_manifest_sha256"],
        "build_contract": _relative(contract["repo_root"], contract["path"]),
        "build_contract_sha256": contract["sha256"],
    }
    return config


def _write_payloads(payloads):
    for path in payloads:
        if path.exists():
            raise D7ConfigBuildError(f"refusing to overwrite output config: {path}")
    temporary_paths = {}
    committed = []
    try:
        for path, payload in payloads.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            if temporary.exists():
                raise D7ConfigBuildError(f"stale temporary output exists: {temporary}")
            temporary.write_bytes(payload)
            temporary_paths[path] = temporary
        for path, temporary in temporary_paths.items():
            temporary.replace(path)
            committed.append(path)
    except Exception:
        for temporary in temporary_paths.values():
            temporary.unlink(missing_ok=True)
        for path in committed:
            path.unlink(missing_ok=True)
        raise


def build_d7_configs(config_path, repo_root=REPO_ROOT):
    """Generate all final configs only after formal and visual test are valid."""
    contract = load_d7_build_contract(config_path, repo_root=repo_root)
    dynamic = _validated_dynamic_inputs(contract)
    child_configs = {
        variant: _child_config(contract, dynamic, variant) for variant in VARIANTS
    }
    if normalize_d7_config_for_pair(
        child_configs["b0"]
    ) != normalize_d7_config_for_pair(child_configs["v"]):
        raise D7ConfigBuildError(
            "generated B0/V configs differ outside checkpoint/provenance fields"
        )
    child_payloads = {
        variant: _serialize(child_configs[variant]) for variant in VARIANTS
    }
    child_hashes = {
        variant: _sha256_bytes(child_payloads[variant]) for variant in VARIANTS
    }
    raw = contract["raw"]
    pair_config = {
        "schema_version": 1,
        "status": "preregistered",
        "pair_run_id": raw["run_ids"]["pair"],
        "visual_test_manifest": raw["visual_test_manifest"],
        "visual_test_manifest_sha256": dynamic["test_manifest_sha256"],
        "result_root": raw["result_root"],
        "build_contract": _relative(contract["repo_root"], contract["path"]),
        "build_contract_sha256": contract["sha256"],
        "variants": {
            variant: {
                "run_id": raw["run_ids"][variant],
                "config": raw["output_configs"][variant],
                "config_sha256": child_hashes[variant],
            }
            for variant in VARIANTS
        },
    }
    pair_payload = _serialize(pair_config)
    payloads = {
        contract["output_paths"]["b0"]: child_payloads["b0"],
        contract["output_paths"]["v"]: child_payloads["v"],
        contract["output_paths"]["pair"]: pair_payload,
    }
    _write_payloads(payloads)
    return {
        "build_contract_sha256": contract["sha256"],
        "formal_manifest_sha256": dynamic["formal_manifest_sha256"],
        "visual_test_manifest_sha256": dynamic["test_manifest_sha256"],
        "outputs": {
            "b0": {
                "path": str(contract["output_paths"]["b0"]),
                "sha256": child_hashes["b0"],
            },
            "v": {
                "path": str(contract["output_paths"]["v"]),
                "sha256": child_hashes["v"],
            },
            "pair": {
                "path": str(contract["output_paths"]["pair"]),
                "sha256": _sha256_bytes(pair_payload),
            },
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Preflight or build frozen B0/V D7 configs after visual test."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.build:
            output = build_d7_configs(args.config, repo_root=args.repo_root)
        else:
            contract = load_d7_build_contract(args.config, repo_root=args.repo_root)
            output = {
                "config_sha256": contract["sha256"],
                "visual_test_manifest": str(contract["test_manifest_path"]),
                "test_result_required_for_build": True,
            }
    except D7ConfigBuildError as exc:
        print(f"visual D7 config build error: {exc}")
        return 2
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
