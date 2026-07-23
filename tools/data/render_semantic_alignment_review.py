#!/usr/bin/env python3
"""
[INPUT]: 依赖有效 M1 split manifest、语义类别配置、Pillow/NumPy，以及 manifest 冻结的 RGB/语义候选帧。
[OUTPUT]: 对外提供 ReviewError、render_alignment_review 与 CLI，为每个核心类别生成 RGB、全语义着色和类别高亮三联图及待人工判定 JSON。
[POS]: tools/data 的 M1 人工复核证据渲染器；只把 split 候选转为可见证据，不替代人工对齐判断，也不改写训练图像。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data.audit_semantic_pretraining_data import (  # noqa: E402
    AuditError,
    load_class_config,
)


REVIEW_REPORT_SCHEMA_VERSION = 1
DEFAULT_CLASS_CONFIG = REPO_ROOT / "configs" / "thesis" / "semantic_classes_v1.json"
PALETTE = np.array(
    [
        [30, 30, 30],
        [128, 64, 128],
        [244, 35, 232],
        [255, 255, 255],
        [0, 80, 220],
        [220, 20, 60],
        [255, 140, 0],
        [250, 220, 0],
        [0, 180, 120],
        [110, 110, 110],
    ],
    dtype=np.uint8,
)


class ReviewError(ValueError):
    """Raised when frozen review candidates cannot be rendered."""


def _read_json(path, label):
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewError(f"unable to read {label} JSON {path}: {exc}") from exc


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _render_sheet(rgb, labels, class_item):
    height, width = labels.shape
    mapped = np.zeros((height, width), dtype=np.uint8)
    for train_id, source_tags in class_item["all_class_tags"]:
        mapped[np.isin(labels, source_tags)] = train_id
    semantic = PALETTE[mapped]
    target = np.isin(labels, class_item["source_tags"])
    highlighted = rgb.astype(np.float32)
    tint = np.array([255, 40, 40], dtype=np.float32)
    highlighted[target] = highlighted[target] * 0.35 + tint * 0.65
    highlighted = np.clip(highlighted, 0, 255).astype(np.uint8)
    header = 30
    canvas = Image.new("RGB", (width * 3, height + header), "white")
    canvas.paste(Image.fromarray(rgb), (0, header))
    canvas.paste(Image.fromarray(semantic), (width, header))
    canvas.paste(Image.fromarray(highlighted), (width * 2, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "RGB", fill="black")
    draw.text((width + 8, 8), "Semantic classes", fill="black")
    draw.text((width * 2 + 8, 8), f"Highlight: {class_item['name']}", fill="black")
    return canvas, int(target.sum())


def render_alignment_review(
    split_manifest_path,
    output_dir,
    class_config_path=DEFAULT_CLASS_CONFIG,
):
    """Render one deterministic evidence sheet for every core class."""
    manifest_path = Path(split_manifest_path).resolve()
    manifest = _read_json(manifest_path, "split manifest")
    if not manifest.get("valid"):
        raise ReviewError("split manifest must be valid")
    try:
        class_config = load_class_config(class_config_path)
    except AuditError as exc:
        raise ReviewError(str(exc)) from exc
    if manifest.get("source", {}).get("class_config_sha256") != class_config["sha256"]:
        raise ReviewError("split manifest and class config SHA-256 differ")
    root = Path(manifest.get("dataset_root", ""))
    if not root.is_dir():
        raise ReviewError(f"dataset root is not a directory: {root}")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ReviewError(f"refusing to use non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    excluded = set(class_config_item["name"] for class_config_item in class_config["classes"])
    core_classes = manifest.get("policy", {}).get("core_classes")
    if not isinstance(core_classes, list) or not core_classes:
        raise ReviewError("split manifest has no core_classes")
    if set(core_classes) - excluded:
        raise ReviewError("split manifest references unknown core classes")
    all_class_tags = [
        (item["train_id"], item["source_tags"]) for item in class_config["classes"]
    ]
    if max(train_id for train_id, _ in all_class_tags) >= len(PALETTE):
        raise ReviewError("semantic class config exceeds the review palette")
    by_name = {item["name"]: dict(item) for item in class_config["classes"]}
    sequence_splits = {
        item["path"]: item["split"] for item in manifest.get("sequences", [])
    }
    items = []
    for index, name in enumerate(core_classes, 1):
        candidate = manifest.get("review_candidates", {}).get(name)
        if not isinstance(candidate, dict):
            raise ReviewError(f"split manifest has no review candidate for {name}")
        sequence_path = candidate["sequence_path"]
        camera = candidate["camera"]
        frame_id = candidate["frame_id"]
        rgb_path = root / sequence_path / f"rgb_{camera}" / f"{frame_id}.jpg"
        if not rgb_path.is_file():
            alternatives = list((root / sequence_path / f"rgb_{camera}").glob(f"{frame_id}.*"))
            if len(alternatives) != 1:
                raise ReviewError(f"unable to resolve RGB frame for {name}")
            rgb_path = alternatives[0]
        mask_path = root / sequence_path / f"seg_{camera}" / f"{frame_id}.png"
        try:
            with Image.open(rgb_path) as image:
                rgb = np.asarray(image.convert("RGB"))
            with Image.open(mask_path) as image:
                labels = np.asarray(image)
        except OSError as exc:
            raise ReviewError(f"unable to read review frame for {name}: {exc}") from exc
        if labels.ndim != 2 or rgb.shape[:2] != labels.shape:
            raise ReviewError(f"RGB/mask alignment is structurally invalid for {name}")
        class_item = by_name[name]
        class_item["all_class_tags"] = all_class_tags
        sheet, observed_pixels = _render_sheet(rgb, labels, class_item)
        if observed_pixels != candidate["class_pixels"]:
            raise ReviewError(
                f"candidate pixel count drift for {name}: {observed_pixels} != {candidate['class_pixels']}"
            )
        render_path = output_dir / f"{index:02d}_{name}.png"
        sheet.save(render_path)
        items.append(
            {
                "class": name,
                "train_id": class_item["train_id"],
                "source_tags": list(class_item["source_tags"]),
                "sequence_path": sequence_path,
                "split": sequence_splits[sequence_path],
                "camera": camera,
                "frame_id": frame_id,
                "class_pixels": observed_pixels,
                "rgb_path": str(rgb_path),
                "rgb_sha256": _sha256_file(rgb_path),
                "semantic_path": str(mask_path),
                "semantic_sha256": _sha256_file(mask_path),
                "render_path": str(render_path.resolve()),
                "render_sha256": _sha256_file(render_path),
                "review": {"status": "pending", "note": None},
            }
        )
    return {
        "review_report_schema_version": REVIEW_REPORT_SCHEMA_VERSION,
        "status": "pending_manual_review",
        "split_manifest": str(manifest_path),
        "split_manifest_sha256": _sha256_file(manifest_path),
        "class_config": str(class_config["path"]),
        "class_config_sha256": class_config["sha256"],
        "items": items,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Render semantic RGB/mask alignment review evidence"
    )
    parser.add_argument("split_manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--class-config", type=Path, default=DEFAULT_CLASS_CONFIG)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.report.exists():
        print(f"review error: refusing to overwrite {args.report}", file=sys.stderr)
        return 2
    try:
        report = render_alignment_review(
            args.split_manifest,
            args.output_dir,
            class_config_path=args.class_config,
        )
    except ReviewError as exc:
        print(f"review error: {exc}", file=sys.stderr)
        return 2
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.report.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    except FileExistsError:
        print(f"review error: refusing to overwrite {args.report}", file=sys.stderr)
        return 2
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
