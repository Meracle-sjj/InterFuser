#!/usr/bin/env python3
"""
[INPUT]: 依赖 v7 蒸馏训练配置（数据/划分/模型字段复用）、官方 B0 教师骨干、v7/v5 学生骨干导出（均哈希绑定）、SemanticFrameDataset 冻结划分。
[OUTPUT]: 对外提供 LinearStitchDiagnosticError、deterministic_frame_split、StageFitAccumulator、StageEvalAccumulator、solve_ridge、evaluate_stitch_decision、run_linear_stitch_diagnostic 与 CLI，生成双臂逐 stage 线性缝合对照 manifest。
[POS]: tools/evaluation 的 v8a 纯测量诊断；只拟合线性映射并报告误差消减率，不加载下游、不读取 test、不产出任何 Stage 2 准入。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPO_ROOT, REPO_ROOT / "interfuser"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from tools.evaluation.runtime_resources import (  # noqa: E402
    ensure_gpus_have_free_memory,
)
from tools.training.semantic_pretraining import (  # noqa: E402
    TrainingContractError,
    load_training_contract,
    make_frozen_feature_teacher,
    sha256_file,
    SemanticFrameDataset,
)


MANIFEST_SCHEMA_VERSION = 1
CONFIG_SCHEMA_VERSION = 1
DEFAULT_RIDGE_SCALE = 1e-3
DEFAULT_DECISION_THRESHOLD = 0.5


class LinearStitchDiagnosticError(RuntimeError):
    """Raised when diagnostic provenance, split, or tensor contracts are invalid."""


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _read_json(path, label):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LinearStitchDiagnosticError(f"unable to read {label} JSON {path}: {exc}") from exc


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _git_head():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _git_status():
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def deterministic_frame_split(keys, seed):
    """Split frame keys into fit/eval halves by sha256 rank of f"{seed}:{key}"."""
    fit, evaluation = [], []
    for key in sorted(keys):
        digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
        (fit if digest[0] < 0x80 else evaluation).append(key)
    if not fit or not evaluation:
        raise LinearStitchDiagnosticError("deterministic split produced an empty half")
    return fit, evaluation


class StageFitAccumulator:
    """Accumulate ridge sufficient statistics G=Σxx^T and H=Σxy^T on fit-half vectors."""

    def __init__(self, channels, device):
        self.channels = channels
        self.gram_xx = torch.zeros(channels, channels, dtype=torch.float64, device=device)
        self.gram_xy = torch.zeros(channels, channels, dtype=torch.float64, device=device)
        self.vectors = 0

    def update(self, student, teacher):
        if student.shape != teacher.shape:
            raise LinearStitchDiagnosticError("student/teacher feature shapes differ")
        x = student.permute(0, 2, 3, 1).reshape(-1, self.channels).to(torch.float64)
        y = teacher.permute(0, 2, 3, 1).reshape(-1, self.channels).to(torch.float64)
        self.gram_xx += x.T @ x
        self.gram_xy += x.T @ y
        self.vectors += x.shape[0]


def solve_ridge(gram_xx, gram_xy, ridge_scale=DEFAULT_RIDGE_SCALE):
    """Closed-form ridge W=(G+λI)^{-1}H with λ=ridge_scale·tr(G)/C (scale-free)."""
    channels = gram_xx.shape[0]
    lam = ridge_scale * torch.diagonal(gram_xx).sum() / channels
    identity = torch.eye(channels, dtype=gram_xx.dtype, device=gram_xx.device)
    weight = torch.linalg.solve(gram_xx + lam * identity, gram_xy)
    return weight, float(lam)


class StageEvalAccumulator:
    """Accumulate identity/stitched squared errors and cosine sums on eval-half vectors."""

    def __init__(self, device):
        self.device = device
        self.identity_error = torch.zeros((), dtype=torch.float64, device=device)
        self.stitched_error = torch.zeros((), dtype=torch.float64, device=device)
        self.teacher_energy = torch.zeros((), dtype=torch.float64, device=device)
        self.cos_identity_sum = torch.zeros((), dtype=torch.float64, device=device)
        self.cos_stitched_sum = torch.zeros((), dtype=torch.float64, device=device)
        self.vectors = 0

    def update(self, student, teacher, weight):
        if student.shape != teacher.shape:
            raise LinearStitchDiagnosticError("student/teacher feature shapes differ")
        channels = student.shape[1]
        x = student.permute(0, 2, 3, 1).reshape(-1, channels).to(torch.float64)
        y = teacher.permute(0, 2, 3, 1).reshape(-1, channels).to(torch.float64)
        mapped = x @ weight
        self.identity_error += (x - y).pow(2).sum()
        self.stitched_error += (mapped - y).pow(2).sum()
        self.teacher_energy += y.pow(2).sum()
        eps = 1e-12
        self.cos_identity_sum += (
            (x * y).sum(dim=1) / (x.norm(dim=1) * y.norm(dim=1)).clamp_min(eps)
        ).sum()
        self.cos_stitched_sum += (
            (mapped * y).sum(dim=1) / (mapped.norm(dim=1) * y.norm(dim=1)).clamp_min(eps)
        ).sum()
        self.vectors += x.shape[0]

    def result(self):
        if self.vectors == 0:
            raise LinearStitchDiagnosticError("eval half produced no vectors")
        identity_relative = float(self.identity_error / self.teacher_energy.clamp_min(1e-12))
        stitched_relative = float(self.stitched_error / self.teacher_energy.clamp_min(1e-12))
        return {
            "vectors": self.vectors,
            "identity_relative_error": identity_relative,
            "stitched_relative_error": stitched_relative,
            "error_reduction_ratio": 1.0 - stitched_relative / max(identity_relative, 1e-12),
            "cosine_identity_mean": float(self.cos_identity_sum / self.vectors),
            "cosine_stitched_mean": float(self.cos_stitched_sum / self.vectors),
        }


def evaluate_stitch_decision(arm_results, primary_arm, threshold=DEFAULT_DECISION_THRESHOLD):
    """Preregistered rule: primary arm deepest-stage reduction >= threshold => stitch viable."""
    if primary_arm not in arm_results:
        raise LinearStitchDiagnosticError(f"primary arm {primary_arm} missing from results")
    final_stage = arm_results[primary_arm]["stages"][-1]
    reduction = final_stage["error_reduction_ratio"]
    return {
        "primary_arm": primary_arm,
        "final_stage": final_stage["stage_index"],
        "error_reduction_ratio": reduction,
        "threshold": threshold,
        "linear_stitch_viable": bool(reduction >= threshold),
    }


def load_diagnostic_config(config_path):
    """Validate the v8a config and hash-bind every input artifact."""
    config_path = Path(config_path).resolve()
    raw = _read_json(config_path, "diagnostic config")
    if raw.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise LinearStitchDiagnosticError(
            f"unsupported diagnostic config schema_version: {raw.get('schema_version')}"
        )
    training_config = (REPO_ROOT / raw["training_config"]).resolve()
    if sha256_file(training_config) != raw.get("training_config_sha256"):
        raise LinearStitchDiagnosticError("training config SHA-256 mismatch")
    arms = raw.get("arms")
    if not isinstance(arms, dict) or len(arms) < 1:
        raise LinearStitchDiagnosticError("diagnostic config must define at least one arm")
    resolved_arms = {}
    for name, arm in arms.items():
        checkpoint = (REPO_ROOT / arm["checkpoint"]).resolve()
        if sha256_file(checkpoint) != arm.get("checkpoint_sha256"):
            raise LinearStitchDiagnosticError(f"arm {name} checkpoint SHA-256 mismatch")
        resolved_arms[name] = {"checkpoint": str(checkpoint), "checkpoint_sha256": arm["checkpoint_sha256"]}
    if raw.get("primary_arm") not in resolved_arms:
        raise LinearStitchDiagnosticError("primary_arm must name a configured arm")
    return {
        "sha256": sha256_file(config_path),
        "path": str(config_path),
        "training_config": str(training_config),
        "training_config_sha256": raw["training_config_sha256"],
        "split_seed": int(raw["split_seed"]),
        "ridge_scale": float(raw.get("ridge_scale", DEFAULT_RIDGE_SCALE)),
        "decision_threshold": float(raw.get("decision_threshold", DEFAULT_DECISION_THRESHOLD)),
        "primary_arm": raw["primary_arm"],
        "arms": resolved_arms,
        "physical_gpu_index": int(raw.get("physical_gpu_index", 1)),
        "gpu_minimum_free_memory_mb": int(raw.get("gpu_minimum_free_memory_mb", 20000)),
        "protocol_document": raw.get("protocol_document"),
    }


def _load_student_backbone(export_path, feature_indices):
    """Strict-load a semantic backbone export (backbone.-prefixed state dict)."""
    from timm.models.resnet import resnet50d

    payload = torch.load(export_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("architecture") != "resnet50d":
        raise LinearStitchDiagnosticError(f"student export architecture mismatch: {export_path}")
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise LinearStitchDiagnosticError(f"student export has no state_dict: {export_path}")
    prefix = "backbone."
    if any(not key.startswith(prefix) for key in state):
        raise LinearStitchDiagnosticError("student export keys must use backbone. prefix")
    stripped = OrderedDict((key[len(prefix):], value) for key, value in state.items())
    backbone = resnet50d(
        pretrained=False, in_chans=3, features_only=True, out_indices=feature_indices
    )
    backbone.load_state_dict(stripped, strict=True)
    backbone.eval()
    backbone.requires_grad_(False)
    return backbone


def _forward_stage_features(model, images):
    features = model(images)
    if not isinstance(features, (list, tuple)) or not features:
        raise LinearStitchDiagnosticError("backbone must return a non-empty stage list")
    return features


def run_linear_stitch_diagnostic(config_path, run_id, result_root):
    """Run the v8a two-pass linear stitch diagnostic and write its manifest."""
    config = load_diagnostic_config(config_path)
    run_directory = Path(result_root) / run_id
    if run_directory.exists():
        raise LinearStitchDiagnosticError(f"refusing to reuse run directory: {run_directory}")
    run_directory.mkdir(parents=True)
    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "running",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "config": config["path"],
        "config_sha256": config["sha256"],
        "protocol_document": config["protocol_document"],
        "git_head": _git_head(),
        "git_status": _git_status(),
        "input_artifacts": {
            "training_config_sha256": config["training_config_sha256"],
            "arms": {name: arm["checkpoint_sha256"] for name, arm in config["arms"].items()},
        },
        "split": {"seed": config["split_seed"]},
        "arms": {},
        "decision": None,
        "boundaries": {"test_accessed": False, "stage2_admitted": False},
        "errors": [],
    }
    manifest_path = run_directory / "linear_stitch_manifest.json"
    _write_json(manifest_path, manifest)
    try:
        ensure_gpus_have_free_memory(
            [config["physical_gpu_index"]], config["gpu_minimum_free_memory_mb"]
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config["physical_gpu_index"])
        contract = load_training_contract(config["training_config"])
        dataset = SemanticFrameDataset(contract, "validation")
        keys = [item["key"] for item in dataset.records]
        fit_keys, eval_keys = deterministic_frame_split(keys, config["split_seed"])
        fit_set, eval_set = set(fit_keys), set(eval_keys)
        manifest["split"].update(
            {
                "validation_frames": len(keys),
                "fit_frames": len(fit_keys),
                "eval_frames": len(eval_keys),
                "fit_keys_sha256": hashlib.sha256(
                    "\n".join(fit_keys).encode("utf-8")
                ).hexdigest(),
                "eval_keys_sha256": hashlib.sha256(
                    "\n".join(eval_keys).encode("utf-8")
                ).hexdigest(),
            }
        )
        device = torch.device("cuda:0")
        teacher = make_frozen_feature_teacher(contract).to(device)
        feature_indices = contract["backbone"]["feature_indices"]
        loader = DataLoader(
            dataset,
            batch_size=contract["training"]["batch_size"],
            shuffle=False,
            num_workers=contract["training"]["num_workers"],
            drop_last=False,
        )
        for arm_name, arm in config["arms"].items():
            student = _load_student_backbone(arm["checkpoint"], feature_indices).to(device)
            # Pass 1: fit-half sufficient statistics.
            fit_accumulators = None
            with torch.no_grad():
                for batch in loader:
                    membership = [
                        1.0 if key in fit_set else 0.0 for key in batch["key"]
                    ]
                    if not any(membership):
                        continue
                    mask = torch.tensor(membership, dtype=torch.bool, device=device)
                    images = batch["image"].to(device, non_blocking=True)
                    teacher_features = _forward_stage_features(teacher, images)
                    student_features = _forward_stage_features(student, images)
                    if fit_accumulators is None:
                        fit_accumulators = [
                            StageFitAccumulator(stage.shape[1], device)
                            for stage in student_features
                        ]
                    for accumulator, s_stage, t_stage in zip(
                        fit_accumulators, student_features, teacher_features
                    ):
                        accumulator.update(s_stage[mask], t_stage[mask])
            weights = []
            lambdas = []
            for accumulator in fit_accumulators:
                if accumulator.vectors == 0:
                    raise LinearStitchDiagnosticError(f"arm {arm_name} fit half is empty")
                weight, lam = solve_ridge(
                    accumulator.gram_xx, accumulator.gram_xy, config["ridge_scale"]
                )
                weights.append(weight)
                lambdas.append(lam)
            # Pass 2: eval-half error accumulation.
            eval_accumulators = [
                StageEvalAccumulator(device) for _ in fit_accumulators
            ]
            with torch.no_grad():
                for batch in loader:
                    membership = [
                        1.0 if key in eval_set else 0.0 for key in batch["key"]
                    ]
                    if not any(membership):
                        continue
                    mask = torch.tensor(membership, dtype=torch.bool, device=device)
                    images = batch["image"].to(device, non_blocking=True)
                    teacher_features = _forward_stage_features(teacher, images)
                    student_features = _forward_stage_features(student, images)
                    for accumulator, s_stage, t_stage, weight in zip(
                        eval_accumulators, student_features, teacher_features, weights
                    ):
                        accumulator.update(s_stage[mask], t_stage[mask], weight)
            stage_results = []
            for position, (accumulator, fit_acc, lam) in enumerate(
                zip(eval_accumulators, fit_accumulators, lambdas)
            ):
                result = accumulator.result()
                result["stage_index"] = feature_indices[position]
                result["channels"] = fit_acc.channels
                result["fit_vectors"] = fit_acc.vectors
                result["ridge_lambda"] = lam
                stage_results.append(result)
            manifest["arms"][arm_name] = {"stages": stage_results}
            del student
            torch.cuda.empty_cache()
        manifest["decision"] = evaluate_stitch_decision(
            manifest["arms"], config["primary_arm"], config["decision_threshold"]
        )
        manifest["status"] = "completed"
        manifest["pipeline_valid"] = True
    except (LinearStitchDiagnosticError, TrainingContractError, RuntimeError) as exc:
        manifest["status"] = "failed"
        manifest["pipeline_valid"] = False
        manifest["errors"].append(str(exc))
    manifest["completed_at"] = _utc_now()
    manifest["updated_at"] = manifest["completed_at"]
    _write_json(manifest_path, manifest)
    if manifest["status"] != "completed":
        raise LinearStitchDiagnosticError("; ".join(manifest["errors"]))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="v8a linear stitchability diagnostic")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--result-root", default="results/thesis_m2")
    args = parser.parse_args(argv)
    manifest = run_linear_stitch_diagnostic(args.config, args.run_id, args.result_root)
    print(json.dumps(manifest["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
