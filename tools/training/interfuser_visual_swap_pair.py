#!/usr/bin/env python3
"""
[INPUT]: 依赖冻结 InterFuser checkpoint、同构交通域 ResNet50d 骨干导出与 swap 初始化配置。
[OUTPUT]: 对外提供 load_visual_swap_contract、prepare_visual_swap_pair 与 CLI，兼容 checkpoint 原始架构别名并生成可命名的 strict-loadable 对照/视觉替换 checkpoint 及逐张量不变量证据。
[POS]: tools/training 的固定底座视觉替换边界；保留底座非视觉能力，只以 RGB backbone 区分对照与实验分支。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import argparse
import hashlib
import json
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


SCHEMA_VERSION = 1
MANIFEST_VERSION = 1
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
RGB_PREFIXES = ("rgb_backbone.", "rgb_patch_embed.backbone.")


class VisualSwapError(RuntimeError):
    """Raised when the fixed-base M0-FT/M0-V contract cannot be proven."""


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state_dict, include=None):
    """Hash tensor values with stable key, dtype and shape framing."""
    digest = hashlib.sha256()
    selected = sorted(
        (
            (key, value)
            for key, value in state_dict.items()
            if include is None or include(key)
        ),
        key=lambda item: item[0],
    )
    for key, tensor in selected:
        if not torch.is_tensor(tensor):
            raise VisualSwapError(f"state value is not a tensor: {key}")
        value = tensor.detach().cpu().contiguous()
        fields = (
            key.encode("utf-8"),
            str(value.dtype).encode("ascii"),
            ",".join(map(str, value.shape)).encode("ascii"),
            value.numpy().tobytes(),
        )
        for field in fields:
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
    return digest.hexdigest()


def _read_json(path, label):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VisualSwapError(f"unable to read {label} JSON {path}: {exc}") from exc


def _resolve_repo_path(value, label):
    if not isinstance(value, str) or not value:
        raise VisualSwapError(f"{label} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _verify_hash(path, expected, label):
    if not isinstance(expected, str) or len(expected) != 64:
        raise VisualSwapError(f"{label} SHA-256 must contain 64 hex characters")
    actual = sha256_file(path)
    if actual != expected:
        raise VisualSwapError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def _git_output(*args):
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise VisualSwapError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def load_visual_swap_contract(path):
    """Validate M0, visual backbone and model code provenance before loading tensors."""
    path = Path(path).resolve()
    raw = _read_json(path, "visual swap config")
    if raw.get("schema_version") != SCHEMA_VERSION or raw.get("status") != "frozen":
        raise VisualSwapError("visual swap config must be frozen schema v1")

    model = raw.get("model")
    if not isinstance(model, dict) or model.get("name") != "interfuser_baseline":
        raise VisualSwapError("model.name must be interfuser_baseline")
    resolved = {}
    for field in ("model_definition", "resnet_definition"):
        resolved[field] = _resolve_repo_path(model.get(field), f"model.{field}")
        _verify_hash(resolved[field], model.get(f"{field}_sha256"), f"model.{field}")

    base = raw.get("base_checkpoint")
    if not isinstance(base, dict):
        raise VisualSwapError("base_checkpoint must be an object")
    if base.get("architecture") != model["name"] or not isinstance(base.get("epoch"), int):
        raise VisualSwapError("base checkpoint architecture/epoch contract is invalid")
    checkpoint_architecture = base.get(
        "checkpoint_architecture", base["architecture"]
    )
    if not isinstance(checkpoint_architecture, str) or not checkpoint_architecture:
        raise VisualSwapError(
            "base checkpoint checkpoint_architecture must be a non-empty string"
        )
    resolved["base_checkpoint"] = _resolve_repo_path(
        base.get("path"), "base_checkpoint.path"
    )
    _verify_hash(
        resolved["base_checkpoint"], base.get("sha256"), "base_checkpoint"
    )

    visual = raw.get("visual_rgb_initialization")
    if not isinstance(visual, dict) or visual.get("architecture") != "resnet50d":
        raise VisualSwapError("visual_rgb_initialization must be a ResNet50d export")
    source_config_hash = visual.get("source_training_config_sha256")
    if not isinstance(source_config_hash, str) or len(source_config_hash) != 64:
        raise VisualSwapError("visual source training config SHA-256 is invalid")
    resolved["visual_rgb_initialization"] = _resolve_repo_path(
        visual.get("checkpoint"), "visual_rgb_initialization.checkpoint"
    )
    _verify_hash(
        resolved["visual_rgb_initialization"],
        visual.get("checkpoint_sha256"),
        "visual_rgb_initialization",
    )

    required_invariant = {
        "only_changed_module": "rgb_backbone",
        "expected_full_model_tensors": 1132,
        "expected_unique_rgb_tensors": 330,
        "strict_base_checkpoint_load": True,
        "strict_full_model_checkpoint_load": True,
    }
    pair_invariant = raw.get("pair_invariant")
    if not isinstance(pair_invariant, dict) or any(
        pair_invariant.get(key) != value for key, value in required_invariant.items()
    ):
        raise VisualSwapError("pair_invariant differs from the frozen contract")
    allowed_invariant_keys = set(required_invariant) | {
        "expected_full_model_rgb_alias_tensors",
        "expected_changed_unique_rgb_tensors",
    }
    if set(pair_invariant) - allowed_invariant_keys:
        raise VisualSwapError("pair_invariant contains unsupported fields")
    expected_changed_unique = pair_invariant.get(
        "expected_changed_unique_rgb_tensors",
        pair_invariant["expected_unique_rgb_tensors"],
    )
    if (
        not isinstance(expected_changed_unique, int)
        or isinstance(expected_changed_unique, bool)
        or not 0 < expected_changed_unique <= pair_invariant["expected_unique_rgb_tensors"]
        or pair_invariant.get("expected_full_model_rgb_alias_tensors")
        != expected_changed_unique * len(RGB_PREFIXES)
    ):
        raise VisualSwapError("changed RGB alias invariant is inconsistent")
    if not isinstance(raw.get("seed"), int) or isinstance(raw.get("seed"), bool):
        raise VisualSwapError("seed must be an integer")
    if not isinstance(raw.get("require_clean_git"), bool):
        raise VisualSwapError("require_clean_git must be boolean")

    variant_names = raw.get(
        "variant_names", {"base": "m0_ft", "visual": "m0_v"}
    )
    if not isinstance(variant_names, dict) or set(variant_names) != {
        "base",
        "visual",
    }:
        raise VisualSwapError("variant_names must contain exactly base and visual")
    if any(
        not isinstance(value, str) or not RUN_ID_PATTERN.fullmatch(value)
        for value in variant_names.values()
    ):
        raise VisualSwapError("variant names must use safe run-id characters")
    if variant_names["base"] == variant_names["visual"]:
        raise VisualSwapError("base and visual variant names must differ")

    result_root = _resolve_repo_path(raw.get("result_root"), "result_root").resolve()
    try:
        result_root.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise VisualSwapError("result_root escapes the repository") from exc

    normalized = dict(raw)
    normalized["pair_invariant"] = {
        **pair_invariant,
        "expected_changed_unique_rgb_tensors": expected_changed_unique,
    }
    normalized.update(
        {
            "path": path,
            "sha256": sha256_file(path),
            "resolved": resolved,
            "result_root_path": result_root,
            "checkpoint_architecture": checkpoint_architecture,
            "variant_names": variant_names,
        }
    )
    return normalized


def _create_interfuser_model(name):
    from timm.models import create_model

    return create_model(name)


def _load_base_state(path, contract):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise VisualSwapError("base checkpoint must contain a state_dict")
    base_contract = contract["base_checkpoint"]
    checkpoint_architecture = contract.get(
        "checkpoint_architecture",
        base_contract.get("checkpoint_architecture", base_contract["architecture"]),
    )
    if payload.get("arch") != checkpoint_architecture:
        raise VisualSwapError("base checkpoint architecture differs from the contract")
    if payload.get("epoch") != base_contract["epoch"]:
        raise VisualSwapError("base checkpoint epoch differs from the contract")
    state = payload["state_dict"]
    if any(not torch.is_tensor(value) for value in state.values()):
        raise VisualSwapError("base checkpoint state_dict contains non-tensor values")
    return state


def _load_visual_rgb_state(path, contract):
    export = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(export, dict) or not isinstance(export.get("state_dict"), dict):
        raise VisualSwapError("visual checkpoint must contain an exported state_dict")
    visual = contract["visual_rgb_initialization"]
    if export.get("format_version") != 1 or export.get("architecture") != "resnet50d":
        raise VisualSwapError("visual checkpoint metadata differs from the contract")
    if export.get("source_training_config_sha256") != visual[
        "source_training_config_sha256"
    ]:
        raise VisualSwapError("visual source training config SHA-256 differs")
    state = export["state_dict"]
    if any(not key.startswith("backbone.") for key in state):
        raise VisualSwapError("visual export contains a non-backbone key")
    return {key.removeprefix("backbone."): value for key, value in state.items()}


def _clone_state(state):
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _save_checkpoint(path, arch, variant, state_dict, provenance):
    torch.save(
        {
            "format_version": 1,
            "arch": arch,
            "variant": variant,
            "state_dict": state_dict,
            "initialization_provenance": provenance,
        },
        path,
    )


def prepare_visual_swap_pair(config_path, run_id, result_root=None):
    """Create fixed-base checkpoints whose only tensor difference is RGB state."""
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise VisualSwapError("run_id must use letters, digits, dot, dash or underscore")
    contract = load_visual_swap_contract(config_path)
    git_status = _git_output("status", "--porcelain")
    if contract["require_clean_git"] and git_status:
        raise VisualSwapError("Git worktree must be clean")
    git_head = _git_output("rev-parse", "HEAD")

    root = contract["result_root_path"] if result_root is None else Path(result_root)
    root = root.resolve()
    try:
        root.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise VisualSwapError("result root escapes the repository") from exc
    run_dir = root / run_id
    if run_dir.exists():
        raise VisualSwapError(f"refusing to overwrite run directory: {run_dir}")
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
        torch.manual_seed(contract["seed"])
        model = _create_interfuser_model(contract["model"]["name"])
        base_source = _load_base_state(
            contract["resolved"]["base_checkpoint"], contract
        )
        if len(base_source) != contract["pair_invariant"]["expected_full_model_tensors"]:
            raise VisualSwapError("base checkpoint tensor count differs from the contract")
        model.load_state_dict(base_source, strict=True)
        m0_ft_state = _clone_state(model.state_dict())
        if state_dict_sha256(m0_ft_state) != state_dict_sha256(base_source):
            raise VisualSwapError("M0-FT state differs from the frozen base checkpoint")

        m0_rgb = _clone_state(model.rgb_backbone.state_dict())
        visual_rgb = _load_visual_rgb_state(
            contract["resolved"]["visual_rgb_initialization"], contract
        )
        expected_rgb = contract["pair_invariant"]["expected_unique_rgb_tensors"]
        if len(m0_rgb) != expected_rgb or len(visual_rgb) != expected_rgb:
            raise VisualSwapError("RGB tensor count differs from the contract")
        model.rgb_backbone.load_state_dict(visual_rgb, strict=True)
        loaded_rgb = _clone_state(model.rgb_backbone.state_dict())
        if any(not torch.equal(loaded_rgb[key], visual_rgb[key]) for key in loaded_rgb):
            raise VisualSwapError("visual RGB strict load changed tensor values")
        m0_v_state = _clone_state(model.state_dict())

        changed_keys = [
            key
            for key in m0_ft_state
            if not torch.equal(m0_ft_state[key], m0_v_state[key])
        ]
        expected_aliases = contract["pair_invariant"][
            "expected_full_model_rgb_alias_tensors"
        ]
        if len(changed_keys) != expected_aliases or any(
            not key.startswith(RGB_PREFIXES) for key in changed_keys
        ):
            raise VisualSwapError("full checkpoints differ outside the RGB aliases")
        changed_unique = sum(
            not torch.equal(m0_rgb[key], visual_rgb[key]) for key in m0_rgb
        )
        expected_changed_unique = contract["pair_invariant"].get(
            "expected_changed_unique_rgb_tensors", expected_rgb
        )
        if changed_unique != expected_changed_unique:
            raise VisualSwapError("visual swap changed an unexpected RGB tensor count")

        non_rgb = lambda key: not key.startswith(RGB_PREFIXES)
        m0_ft_non_rgb_hash = state_dict_sha256(m0_ft_state, include=non_rgb)
        m0_v_non_rgb_hash = state_dict_sha256(m0_v_state, include=non_rgb)
        if m0_ft_non_rgb_hash != m0_v_non_rgb_hash:
            raise VisualSwapError("M0 non-RGB tensors changed during visual replacement")

        base = contract["base_checkpoint"]
        visual = contract["visual_rgb_initialization"]
        common_provenance = {
            "base_checkpoint": str(contract["resolved"]["base_checkpoint"]),
            "base_checkpoint_sha256": base["sha256"],
            "base_epoch": base["epoch"],
        }
        variant_names = contract.get(
            "variant_names", {"base": "m0_ft", "visual": "m0_v"}
        )
        base_variant = variant_names["base"]
        visual_variant = variant_names["visual"]
        m0_ft_path = run_dir / f"{base_variant}_initial_checkpoint.pth"
        m0_v_path = run_dir / f"{visual_variant}_initial_checkpoint.pth"
        _save_checkpoint(
            m0_ft_path,
            contract["model"]["name"],
            base_variant,
            m0_ft_state,
            {**common_provenance, "rgb_action": "preserve_base"},
        )
        _save_checkpoint(
            m0_v_path,
            contract["model"]["name"],
            visual_variant,
            m0_v_state,
            {
                **common_provenance,
                "rgb_action": "strict_replace",
                "visual_source": visual["source"],
                "visual_checkpoint_sha256": visual["checkpoint_sha256"],
                "visual_source_training_config_sha256": visual[
                    "source_training_config_sha256"
                ],
            },
        )

        for state in (m0_ft_state, m0_v_state):
            model.load_state_dict(state, strict=True)

        manifest.update(
            {
                "status": "completed",
                "pipeline_valid": True,
                "updated_at": _utc_now(),
                "completed_at": _utc_now(),
                "invariants": {
                    "full_model_tensors": len(m0_ft_state),
                    "unique_rgb_tensors": len(m0_rgb),
                    "changed_unique_rgb_tensors": changed_unique,
                    "changed_full_model_alias_tensors": len(changed_keys),
                    "base_state_preserved": True,
                    "non_rgb_state_sha256": m0_ft_non_rgb_hash,
                    "strict_base_checkpoint_load": True,
                    "strict_full_checkpoint_load": True,
                },
                "variants": {
                    base_variant: {
                        "checkpoint": str(m0_ft_path),
                        "checkpoint_sha256": sha256_file(m0_ft_path),
                        "full_state_sha256": state_dict_sha256(m0_ft_state),
                        "rgb_state_sha256": state_dict_sha256(m0_rgb),
                    },
                    visual_variant: {
                        "checkpoint": str(m0_v_path),
                        "checkpoint_sha256": sha256_file(m0_v_path),
                        "full_state_sha256": state_dict_sha256(m0_v_state),
                        "rgb_state_sha256": state_dict_sha256(visual_rgb),
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
        description="Prepare fixed-base InterFuser M0-FT/M0-V initial checkpoints"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--result-root", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = prepare_visual_swap_pair(
            args.config, args.run_id, result_root=args.result_root
        )
    except VisualSwapError as exc:
        print(f"visual swap error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"visual swap failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
