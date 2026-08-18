#!/usr/bin/env python3
"""
 * [INPUT]: 依赖 v8.1 协议 §3、语义 split manifest（哈希绑定）、/data1/shijj/interfuser_data 作者格式存储、
 *   interfuser/timm 的 lidar_to_histogram_features（坐标契约参照）。
 * [OUTPUT]: 对外提供 audit_functional_distillation_corpus CLI——产出 train/validation 帧索引 txt 与审计
 *   manifest（覆盖率、导航缺失清单哈希与 expected_retained、LiDAR 坐标乘子判定与参考带对照）；
 *   文件洞或 LiDAR 缺失/非有限即非零退出（协议 §3 致命项）。
 * [POS]: tools/training 的 v8b 功能蒸馏 P0 预检入口；只读数据、只写 results/thesis_m1 下新目录。
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""
import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INTERFUSER_ROOT = REPO_ROOT / "interfuser"
for import_root in (REPO_ROOT, INTERFUSER_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import numpy as np  # noqa: E402

TOWN_PATTERN = re.compile(r"town(\d+)", re.IGNORECASE)
WEATHER_PATTERN = re.compile(r"_w(\d+)_")
REQUIRED_MEASUREMENT_FIELDS = (
    "command",
    "x_command",
    "y_command",
    "future_waypoints",
    "theta",
    "gps_x",
    "gps_y",
    "speed",
)
NAVIGATION_DROP_FIELDS = ("command", "x_command", "y_command", "future_waypoints", "theta")


class CorpusAuditError(RuntimeError):
    """Raised when the semantic corpus cannot support full-model forward."""


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_intersection(sequence_dir):
    groups = {}
    for sub, suffix in (
        ("rgb_front", ".jpg"),
        ("rgb_left", ".jpg"),
        ("rgb_right", ".jpg"),
        ("seg_front", ".png"),
        ("lidar", ".npy"),
        ("measurements", ".json"),
    ):
        directory = sequence_dir / sub
        if not directory.is_dir():
            raise CorpusAuditError(f"missing directory {directory}")
        groups[sub] = {path.stem for path in directory.glob(f"*{suffix}")}
    shared = groups["rgb_front"]
    for stems in groups.values():
        shared = shared & stems
    if not shared:
        raise CorpusAuditError(f"sequence has no complete frames: {sequence_dir}")
    numbers = sorted(int(stem) for stem in shared)
    if numbers != list(range(len(numbers))):
        raise CorpusAuditError(f"frame id hole inside prefix: {sequence_dir}")
    return [f"{number:04d}" for number in numbers]


def _measurement_problem(sequence_dir, frame_id):
    path = sequence_dir / "measurements" / f"{frame_id}.json"
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return f"measurements unreadable: {exc}"
    missing = [field for field in REQUIRED_MEASUREMENT_FIELDS if field not in payload]
    if missing:
        return f"measurements missing {missing}"
    waypoints = payload.get("future_waypoints")
    if not isinstance(waypoints, list) or not waypoints:
        return "future_waypoints empty"
    theta = payload.get("theta")
    if theta is None or not math.isfinite(float(theta)):
        return "theta non-finite"
    return None


def _lidar_nonzero_cells(points, multiplier):
    from interfuser.timm.data.carla_dataset import lidar_to_histogram_features

    adjusted = points.copy()
    adjusted[:, 1] *= multiplier
    histogram = lidar_to_histogram_features(adjusted, crop=256)
    return int(np.count_nonzero(histogram))


def audit_corpus(split_manifest_path, output_dir, sample_limit=200, reference_band=(5000.0, 20000.0)):
    split_manifest_path = Path(split_manifest_path)
    manifest = json.loads(split_manifest_path.read_text())
    dataset_root = Path(manifest["dataset_root"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    per_split = {}
    all_train_pairs = []
    for split in ("train", "validation"):
        sequences = [item for item in manifest["sequences"] if item["split"] == split]
        index_lines = []
        missing_frames = []
        town_values, weather_values = set(), set()
        for item in sequences:
            relative = item["path"]
            sequence_dir = dataset_root / relative
            town_match = TOWN_PATTERN.search(relative)
            weather_match = WEATHER_PATTERN.search(relative)
            if not town_match or not weather_match:
                raise CorpusAuditError(f"sequence path not index-parseable: {relative}")
            town_values.add(int(town_match.group(1)))
            weather_values.add(int(weather_match.group(1)))
            frames = _frame_intersection(sequence_dir)
            for frame_id in frames:
                problem = _measurement_problem(sequence_dir, frame_id)
                if problem is not None:
                    missing_frames.append(f"{relative}:{frame_id} {problem}")
                elif split == "train":
                    all_train_pairs.append((sequence_dir, frame_id))
            index_lines.append(f"{relative} {len(frames)}")
        index_path = output_dir / f"{split}_index.txt"
        index_path.write_text("\n".join(index_lines) + "\n")
        total_frames = sum(int(line.rsplit(" ", 1)[1]) for line in index_lines)
        per_split[split] = {
            "sequences": len(sequences),
            "frames": total_frames,
            "expected_retained_frames": total_frames - len(missing_frames),
            "navigation_dropped_frames": len(missing_frames),
            "navigation_dropped_examples": missing_frames[:5],
            "navigation_dropped_sha256": hashlib.sha256(
                "\n".join(sorted(missing_frames)).encode("utf-8")
            ).hexdigest(),
            "index_path": str(index_path),
            "index_sha256": sha256_file(index_path),
            "towns": sorted(town_values),
            "weathers": sorted(weather_values),
        }

    # 契约说明：缺导航字段的帧由 CarlaMVDetDataset(missing_navigation_policy="drop") 在索引期
    # 逐帧剔除（与 scene validation 同机制），非协议 §3 致命项；文件洞与 LiDAR 问题才停机。
    rng = np.random.default_rng(20260818)
    chosen = [
        all_train_pairs[index]
        for index in rng.choice(
            len(all_train_pairs), size=min(sample_limit, len(all_train_pairs)), replace=False
        )
    ]
    multiplier_stats = {1.0: [], -1.0: []}
    min_points = None
    for sequence_dir, frame_id in chosen:
        points = np.load(sequence_dir / "lidar" / f"{frame_id}.npy")[..., :3]
        if points.shape[0] == 0 or not math.isfinite(
            float(points[:, :3].astype(np.float64).sum())
        ):
            raise CorpusAuditError(
                f"lidar empty or non-finite: {sequence_dir.name}/{frame_id}"
            )
        count = points.shape[0]
        min_points = count if min_points is None else min(min_points, count)
        for multiplier in (1.0, -1.0):
            multiplier_stats[multiplier].append(_lidar_nonzero_cells(points, multiplier))
    verdict = {}
    chosen_multiplier = None
    for multiplier, values in multiplier_stats.items():
        mean = float(np.mean(values))
        inside = reference_band[0] <= mean <= reference_band[1]
        verdict[str(multiplier)] = {
            "mean_nonzero_cells": mean,
            "inside_reference_band": inside,
        }
        if inside and chosen_multiplier is None:
            chosen_multiplier = multiplier
    ambiguous = sum(1 for value in verdict.values() if value["inside_reference_band"])
    if ambiguous != 1:
        raise CorpusAuditError(
            f"lidar multiplier selection ambiguous ({ambiguous} candidates): {verdict}"
        )

    report = {
        "schema_version": 1,
        "protocol_document": "docs/interfuser_negative_transfer_repair_protocol_v8_1.md",
        "split_manifest": {
            "path": str(split_manifest_path),
            "sha256": sha256_file(split_manifest_path),
        },
        "dataset_root": str(dataset_root),
        "splits": per_split,
        "lidar_convention": {
            "sample_size": len(chosen),
            "reference_band_nonzero_cells": list(reference_band),
            "per_multiplier": verdict,
            "selected_multiplier": chosen_multiplier,
            "min_points_per_frame": min_points,
        },
        "required_measurement_fields": list(REQUIRED_MEASUREMENT_FIELDS),
        "navigation_drop_fields": list(NAVIGATION_DROP_FIELDS),
    }
    report_path = output_dir / "audit_manifest.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-limit", type=int, default=200)
    args = parser.parse_args(argv)
    report = audit_corpus(args.split_manifest, args.output_dir, args.sample_limit)
    print(
        json.dumps(
            {
                "train": {
                    key: report["splits"]["train"][key]
                    for key in (
                        "sequences",
                        "frames",
                        "expected_retained_frames",
                        "navigation_dropped_frames",
                        "towns",
                        "weathers",
                    )
                },
                "validation": {
                    key: report["splits"]["validation"][key]
                    for key in ("sequences", "frames", "expected_retained_frames")
                },
                "lidar_multiplier": report["lidar_convention"]["selected_multiplier"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
