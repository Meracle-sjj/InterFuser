#!/usr/bin/env python3
"""
 * [INPUT]: 依赖 v8.1 协议 §4 语义准入、v7 训练配置（复用其验证管线保证可比）、
 *   v8b run 的 backbone_resnet50d.pth 与 checkpoint_best.pth、semantic_pretraining 评估件。
 * [OUTPUT]: 对外提供 evaluate_functional_export_semantics CLI——在 v7 同一 160×120 validation
 *   （1,725 相机样本）上评估 v8b 导出骨干 + v8b 解码器，输出 mIoU/逐类 IoU 与准入判定 JSON。
 * [POS]: tools/training 的 v8b 语义准入门禁入口；只读数据与 run 产物。
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""
import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
INTERFUSER_ROOT = REPO_ROOT / "interfuser"
for import_root in (REPO_ROOT, INTERFUSER_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tools.training.functional_distillation import load_functional_contract  # noqa: E402
from tools.training.semantic_pretraining import (  # noqa: E402
    ConfusionMetrics,
    SemanticFrameDataset,
    SemanticPretrainingModel,
    TrainingContractError,
    load_training_contract,
    sha256_file,
)
from torch.utils.data import DataLoader  # noqa: E402

GATES = {
    "minimum_validation_mean_iou": 0.46324,
    "minimum_pedestrian_iou": 0.08,
    "minimum_traffic_light_iou": 0.15,
}


def build_model(v7_contract, export_path, checkpoint_path):
    model = SemanticPretrainingModel(v7_contract)
    export = torch.load(export_path, map_location="cpu", weights_only=False)
    if export.get("architecture") != "resnet50d":
        raise TrainingContractError("export architecture must be resnet50d")
    export_state = export["state_dict"]
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    head_state = payload["semantic_head_state_dict"]
    combined = OrderedDict()
    for key, value in head_state.items():
        if key.startswith("backbone."):
            continue
        combined[key] = value
    for key, value in export_state.items():
        combined[key] = value.clone()
    model.load_state_dict(combined, strict=True)
    return model, export


def evaluate(v7_config, v8b_config, export_path, checkpoint_path, output_path):
    v7_contract = load_training_contract(v7_config)
    v8b_contract = load_functional_contract(v8b_config)
    model, export = build_model(v7_contract, export_path, checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    dataset = SemanticFrameDataset(v7_contract, "validation")
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    class_names = [item["name"] for item in v7_contract["class_config_loaded"]["classes"]]
    metrics = ConfusionMetrics(len(class_names), v7_contract["training"]["ignore_index"])
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            metrics.update(logits, labels)
    summary = metrics.summary(class_names)
    per_class = {item["name"]: item["iou"] for item in summary["per_class"]}
    checks = {
        "validation_mean_iou": summary["mean_iou"],
        "pedestrian_iou": per_class.get("pedestrian"),
        "traffic_light_iou": per_class.get("traffic_light"),
    }
    verdict = {
        "schema_version": 1,
        "protocol_document": "docs/interfuser_negative_transfer_repair_protocol_v8_1.md",
        "v7_config": {"path": str(v7_config), "sha256": sha256_file(REPO_ROOT / v7_config)},
        "v8b_config": {"path": str(v8b_config), "sha256": v8b_contract["sha256"]},
        "export": {"path": str(export_path), "sha256": sha256_file(export_path)},
        "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
        "validation_samples": len(dataset),
        "metrics": summary,
        "gates": GATES,
        "checks": checks,
        "semantic_admission_passed": (
            checks["validation_mean_iou"] >= GATES["minimum_validation_mean_iou"]
            and checks["pedestrian_iou"] >= GATES["minimum_pedestrian_iou"]
            and checks["traffic_light_iou"] >= GATES["minimum_traffic_light_iou"]
            and export.get("source_training_config_sha256") == v8b_contract["sha256"]
        ),
    }
    Path(output_path).write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    return verdict


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-config", required=True)
    parser.add_argument("--v8b-config", required=True)
    parser.add_argument("--export", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    verdict = evaluate(
        args.v7_config, args.v8b_config, args.export, args.checkpoint, args.output
    )
    print(
        json.dumps(
            {
                "mIoU": verdict["checks"]["validation_mean_iou"],
                "pedestrian_iou": verdict["checks"]["pedestrian_iou"],
                "traffic_light_iou": verdict["checks"]["traffic_light_iou"],
                "passed": verdict["semantic_admission_passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
