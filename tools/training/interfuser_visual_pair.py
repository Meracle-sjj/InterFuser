#!/usr/bin/env python3
"""
[INPUT]: 依赖冻结 InterFuser/ResNet 定义、ImageNet ResNet50d checkpoint、M2 best 骨干导出与配对初始化配置。
[OUTPUT]: 对外提供 VisualPairError、load_visual_pair_contract、state_dict_sha256、prepare_visual_initialization_pair 与 CLI，生成 strict-loadable B0/V 全模型初始 checkpoint 及不变量证据。
[POS]: tools/training 的 M2→InterFuser 迁移边界；复用下游原生 --initial-checkpoint，不在模型本体中常驻论文实验分支。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERFUSER_ROOT = REPO_ROOT / "interfuser"
for import_root in (REPO_ROOT, INTERFUSER_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


SCHEMA_VERSION = 1
MANIFEST_VERSION = 1
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
RGB_PREFIXES = ("rgb_backbone.", "rgb_patch_embed.backbone.")


class VisualPairError(RuntimeError):
    """Raised when B0 and V cannot differ only by the frozen RGB backbone."""


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path, label):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VisualPairError(f"unable to read {label} JSON {path}: {exc}") from exc


def _resolve_repo_path(value, label):
    if not isinstance(value, str) or not value:
        raise VisualPairError(f"{label} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _verify_hash(path, expected, label):
    if not isinstance(expected, str) or len(expected) != 64:
        raise VisualPairError(f"{label} SHA-256 must contain 64 hex characters")
    actual = sha256_file(path)
    if actual != expected:
        raise VisualPairError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def load_visual_pair_contract(path):
    """Validate all code and initialization provenance before model construction."""
    path = Path(path).resolve()
    raw = _read_json(path, "visual initialization config")
    if raw.get("schema_version") != SCHEMA_VERSION or raw.get("status") != "frozen":
        raise VisualPairError("visual initialization config must be frozen schema v1")
    model = raw.get("model")
    if not isinstance(model, dict) or model.get("name") != "interfuser_baseline":
        raise VisualPairError("model.name must be interfuser_baseline")
    resolved = {}
    for field in ("model_definition", "resnet_definition"):
        resolved[field] = _resolve_repo_path(model.get(field), f"model.{field}")
        _verify_hash(
            resolved[field], model.get(f"{field}_sha256"), f"model.{field}"
        )
    for variant in ("b0_rgb_initialization", "v_rgb_initialization"):
        value = raw.get(variant)
        if not isinstance(value, dict):
            raise VisualPairError(f"{variant} must be an object")
        resolved[variant] = _resolve_repo_path(value.get("checkpoint"), variant)
        _verify_hash(
            resolved[variant], value.get("checkpoint_sha256"), variant
        )
    v_config = raw["v_rgb_initialization"]
    if v_config.get("architecture") != "resnet50d":
        raise VisualPairError("V architecture must be resnet50d")
    source_hash = v_config.get("source_training_config_sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise VisualPairError("V source training config SHA-256 is invalid")
    invariant = raw.get("pair_invariant")
    expected_invariant = {
        "only_changed_module": "rgb_backbone",
        "expected_unique_rgb_tensors": 330,
        "expected_full_model_rgb_alias_tensors": 660,
        "expected_imagenet_dtype_normalized_buffers": 55,
        "strict_full_model_checkpoint_load": True,
    }
    if invariant != expected_invariant:
        raise VisualPairError("pair_invariant differs from the frozen contract")
    seed = raw.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise VisualPairError("seed must be an integer")
    if not isinstance(raw.get("require_clean_git"), bool):
        raise VisualPairError("require_clean_git must be boolean")
    result_root = _resolve_repo_path(raw.get("result_root"), "result_root").resolve()
    try:
        result_root.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise VisualPairError("result_root escapes the repository") from exc
    normalized = dict(raw)
    normalized.update(
        {
            "path": path,
            "sha256": sha256_file(path),
            "resolved": resolved,
            "result_root_path": result_root,
        }
    )
    return normalized


def state_dict_sha256(state_dict, include=None):
    """Hash tensor values with stable key, dtype and shape framing."""
    digest = hashlib.sha256()
    selected = sorted(
        [
            (key, value)
            for key, value in state_dict.items()
            if include is None or include(key)
        ],
        key=lambda item: item[0],
    )
    for key, tensor in selected:
        if not torch.is_tensor(tensor):
            raise VisualPairError(f"state value is not a tensor: {key}")
        value = tensor.detach().cpu().contiguous()
        key_bytes = key.encode("utf-8")
        dtype_bytes = str(value.dtype).encode("ascii")
        shape_bytes = ",".join(map(str, value.shape)).encode("ascii")
        data = value.numpy().tobytes()
        for item in (key_bytes, dtype_bytes, shape_bytes, data):
            digest.update(len(item).to_bytes(8, "big"))
            digest.update(item)
    return digest.hexdigest()


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _create_interfuser_model(name):
    from timm.models import create_model

    return create_model(name)


def _git_output(*args):
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise VisualPairError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _load_imagenet_rgb_state(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise VisualPairError("ImageNet checkpoint must contain a state dict")
    return {
        key: value
        for key, value in state.items()
        if key not in {"fc.weight", "fc.bias"}
    }


def _load_v_rgb_state(path, contract):
    export = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(export, dict) or not isinstance(export.get("state_dict"), dict):
        raise VisualPairError("V checkpoint must contain an exported state_dict")
    if export.get("format_version") != 1 or export.get("architecture") != "resnet50d":
        raise VisualPairError("V checkpoint metadata differs from the transfer contract")
    if export.get("source_training_config_sha256") != contract[
        "v_rgb_initialization"
    ]["source_training_config_sha256"]:
        raise VisualPairError("V source training config SHA-256 differs")
    state = export["state_dict"]
    if any(not key.startswith("backbone.") for key in state):
        raise VisualPairError("V export contains a non-backbone key")
    return {key.removeprefix("backbone."): value for key, value in state.items()}


def _assert_tensor_state_equal(actual, expected, label):
    if list(actual) != list(expected):
        raise VisualPairError(f"{label} keys differ")
    mismatches = [
        key
        for key in actual
        if actual[key].shape != expected[key].shape
        or actual[key].dtype != expected[key].dtype
        or not torch.equal(actual[key].cpu(), expected[key].cpu())
    ]
    if mismatches:
        raise VisualPairError(f"{label} tensor mismatch: {mismatches[:5]}")


def _assert_imagenet_load_equivalent(actual, source):
    """Match PyTorch strict-load semantics while exposing BN counter dtype casts."""
    if list(actual) != list(source):
        raise VisualPairError("B0 ImageNet RGB keys differ")
    normalized = []
    mismatches = []
    for key in actual:
        target = actual[key].detach().cpu()
        value = source[key].detach().cpu()
        if target.shape != value.shape:
            mismatches.append(key)
            continue
        if target.dtype != value.dtype:
            if not key.endswith("num_batches_tracked"):
                mismatches.append(key)
                continue
            normalized.append(key)
            value = value.to(dtype=target.dtype)
        if not torch.equal(target, value):
            mismatches.append(key)
    if mismatches:
        raise VisualPairError(f"B0 ImageNet RGB tensor mismatch: {mismatches[:5]}")
    return normalized


def _save_initial_checkpoint(path, model_name, variant, state_dict, provenance):
    torch.save(
        {
            "format_version": 1,
            "arch": model_name,
            "variant": variant,
            "state_dict": state_dict,
            "initialization_provenance": provenance,
        },
        path,
    )


def prepare_visual_initialization_pair(config_path, run_id, result_root=None):
    """Create and prove a B0/V checkpoint pair with one changed module."""
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise VisualPairError("run_id must use letters, digits, dot, dash or underscore")
    contract = load_visual_pair_contract(config_path)
    git_status = _git_output("status", "--porcelain")
    if contract["require_clean_git"] and git_status:
        raise VisualPairError("Git worktree must be clean")
    git_head = _git_output("rev-parse", "HEAD")
    root = contract["result_root_path"] if result_root is None else Path(result_root)
    root = root.resolve()
    try:
        root.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise VisualPairError("result root escapes the repository") from exc
    run_dir = root / run_id
    if run_dir.exists():
        raise VisualPairError(f"refusing to overwrite run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "initialization_manifest.json"
    manifest = {
        "manifest_schema_version": MANIFEST_VERSION,
        "run_id": run_id,
        "status": "running",
        "pipeline_valid": False,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "git_head": git_head,
        "git_status": git_status,
        "config": str(contract["path"]),
        "config_sha256": contract["sha256"],
        "errors": [],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    try:
        _set_seed(contract["seed"])
        model = _create_interfuser_model(contract["model"]["name"])
        b0_rgb = {
            key: value.detach().cpu().clone()
            for key, value in model.rgb_backbone.state_dict().items()
        }
        imagenet_rgb = _load_imagenet_rgb_state(
            contract["resolved"]["b0_rgb_initialization"]
        )
        normalized_imagenet_buffers = _assert_imagenet_load_equivalent(
            b0_rgb, imagenet_rgb
        )
        if len(b0_rgb) != contract["pair_invariant"]["expected_unique_rgb_tensors"]:
            raise VisualPairError("B0 RGB tensor count differs from the contract")
        if len(normalized_imagenet_buffers) != contract["pair_invariant"].get(
            "expected_imagenet_dtype_normalized_buffers", 0
        ):
            raise VisualPairError(
                "ImageNet dtype-normalized buffer count differs from the contract"
            )

        b0_path = run_dir / "b0_initial_checkpoint.pth"
        b0_provenance = {
            "source": contract["b0_rgb_initialization"]["source"],
            "source_checkpoint_sha256": contract["b0_rgb_initialization"][
                "checkpoint_sha256"
            ],
        }
        _save_initial_checkpoint(
            b0_path, contract["model"]["name"], "b0", model.state_dict(), b0_provenance
        )

        v_rgb = _load_v_rgb_state(
            contract["resolved"]["v_rgb_initialization"], contract
        )
        if len(v_rgb) != contract["pair_invariant"]["expected_unique_rgb_tensors"]:
            raise VisualPairError("V RGB tensor count differs from the contract")
        model.rgb_backbone.load_state_dict(v_rgb, strict=True)
        _assert_tensor_state_equal(
            model.rgb_backbone.state_dict(), v_rgb, "V strict RGB load"
        )
        changed_unique = sum(not torch.equal(b0_rgb[key], v_rgb[key]) for key in b0_rgb)
        if changed_unique != len(b0_rgb):
            raise VisualPairError("V did not change every frozen RGB tensor")

        v_path = run_dir / "v_initial_checkpoint.pth"
        v_provenance = {
            "source": contract["v_rgb_initialization"]["source"],
            "source_checkpoint_sha256": contract["v_rgb_initialization"][
                "checkpoint_sha256"
            ],
            "source_training_config_sha256": contract["v_rgb_initialization"][
                "source_training_config_sha256"
            ],
        }
        _save_initial_checkpoint(
            v_path, contract["model"]["name"], "v", model.state_dict(), v_provenance
        )

        b0_payload = torch.load(b0_path, map_location="cpu", weights_only=True)
        v_payload = torch.load(v_path, map_location="cpu", weights_only=True)
        b0_state = b0_payload["state_dict"]
        v_state = v_payload["state_dict"]
        model.load_state_dict(b0_state, strict=True)
        model.load_state_dict(v_state, strict=True)
        changed_keys = [
            key for key in b0_state if not torch.equal(b0_state[key], v_state[key])
        ]
        if len(changed_keys) != contract["pair_invariant"][
            "expected_full_model_rgb_alias_tensors"
        ] or any(not key.startswith(RGB_PREFIXES) for key in changed_keys):
            raise VisualPairError("full checkpoints differ outside the RGB aliases")
        non_rgb = lambda key: not key.startswith(RGB_PREFIXES)
        b0_non_rgb_hash = state_dict_sha256(b0_state, include=non_rgb)
        v_non_rgb_hash = state_dict_sha256(v_state, include=non_rgb)
        if b0_non_rgb_hash != v_non_rgb_hash:
            raise VisualPairError("non-RGB initialization tensors differ")

        manifest.update(
            {
                "status": "completed",
                "pipeline_valid": True,
                "updated_at": _utc_now(),
                "completed_at": _utc_now(),
                "invariants": {
                    "full_model_tensors": len(b0_state),
                    "unique_rgb_tensors": len(b0_rgb),
                    "changed_unique_rgb_tensors": changed_unique,
                    "changed_full_model_alias_tensors": len(changed_keys),
                    "imagenet_dtype_normalized_buffers": len(
                        normalized_imagenet_buffers
                    ),
                    "imagenet_dtype_normalized_buffer_names": normalized_imagenet_buffers,
                    "non_rgb_state_sha256": b0_non_rgb_hash,
                    "strict_full_checkpoint_load": True,
                },
                "variants": {
                    "b0": {
                        "checkpoint": str(b0_path),
                        "checkpoint_sha256": sha256_file(b0_path),
                        "full_state_sha256": state_dict_sha256(b0_state),
                        "rgb_state_sha256": state_dict_sha256(b0_rgb),
                        "provenance": b0_provenance,
                    },
                    "v": {
                        "checkpoint": str(v_path),
                        "checkpoint_sha256": sha256_file(v_path),
                        "full_state_sha256": state_dict_sha256(v_state),
                        "rgb_state_sha256": state_dict_sha256(v_rgb),
                        "provenance": v_provenance,
                    },
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
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare provenance-locked InterFuser B0/V initial checkpoints"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--result-root", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = prepare_visual_initialization_pair(
            args.config, args.run_id, result_root=args.result_root
        )
    except VisualPairError as exc:
        print(f"visual pair error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"visual pair failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
