#!/usr/bin/env python3
"""Export deterministic virtual stop-boundary labels without modifying RGB."""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np


EXPORT_SCHEMA_VERSION = 1
TRAFFIC_ELEMENT_SCHEMA_VERSION = 2
TRAFFIC_ELEMENT_VIEW_SCHEMA_VERSION = 3
LABEL_TYPE = "leaderboard_virtual_stop_boundary"


class ExportError(ValueError):
    """Raised when source data cannot produce a trustworthy export."""


def _load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportError(f"unable to read JSON {path}: {exc}") from exc


def _world_to_ego(point, ego):
    try:
        dx = float(point["x"]) - float(ego["location"]["x"])
        dy = float(point["y"]) - float(ego["location"]["y"])
        dz = float(point["z"]) - float(ego["location"]["z"])
        yaw = math.radians(float(ego["rotation"]["yaw"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ExportError("invalid ego transform or boundary endpoint") from exc
    return {
        "forward": dx * math.cos(yaw) + dy * math.sin(yaw),
        "right": -dx * math.sin(yaw) + dy * math.cos(yaw),
        "up": dz,
    }


def _source_pairs(root, limit):
    phase_paths = sorted(Path(root).rglob("traffic_elements/*.json"))
    if not phase_paths:
        raise ExportError(f"no traffic_elements records under {root}")
    if limit is not None:
        phase_paths = phase_paths[:limit]
    for phase_path in phase_paths:
        route_run = phase_path.parent.parent
        view_path = route_run / "traffic_element_views" / phase_path.name
        if not view_path.is_file():
            raise ExportError(f"missing evidence record for {phase_path}: {view_path}")
        yield route_run, phase_path, view_path


def _target_view(camera_record, target_id, view_path):
    matches = [
        target
        for target in camera_record.get("stop_targets", [])
        if isinstance(target, dict) and target.get("target_id") == target_id
    ]
    if len(matches) != 1:
        raise ExportError(
            f"target {target_id!r} must appear exactly once in {view_path}"
        )
    return matches[0]


def _validate_segment(segment, width, height, path):
    array = np.asarray(segment, dtype=np.float64)
    if array.shape != (2, 2) or not np.isfinite(array).all():
        raise ExportError(f"invalid projected boundary segment in {path}")
    if not (
        np.all(array[:, 0] >= 0.0)
        and np.all(array[:, 0] < width)
        and np.all(array[:, 1] >= 0.0)
        and np.all(array[:, 1] < height)
    ):
        raise ExportError(
            f"projected boundary segment is outside image bounds in {path}"
        )
    return array


def _collect_export(root, camera, limit, primary_only, line_thickness):
    entries_and_masks = []
    summary = {
        "source_views": 0,
        "phase_targets": 0,
        "valid_targets": 0,
        "labels_exported": 0,
        "skipped_non_primary": 0,
        "skipped_camera_unknown": 0,
        "skipped_unprojected": 0,
    }

    for route_run, phase_path, view_path in _source_pairs(root, limit):
        summary["source_views"] += 1
        phase = _load_json(phase_path)
        view = _load_json(view_path)
        if phase.get("schema_version") != TRAFFIC_ELEMENT_SCHEMA_VERSION:
            raise ExportError(f"unsupported traffic-element schema in {phase_path}")
        if view.get("schema_version") != TRAFFIC_ELEMENT_VIEW_SCHEMA_VERSION:
            raise ExportError(f"unsupported evidence schema in {view_path}")
        if (
            phase.get("frame_id") != phase_path.stem
            or view.get("frame_id") != phase_path.stem
        ):
            raise ExportError(f"frame ID mismatch for {phase_path}")

        camera_record = view.get("cameras", {}).get(camera)
        if not isinstance(camera_record, dict):
            raise ExportError(f"camera {camera!r} is absent from {view_path}")
        width = camera_record.get("width")
        height = camera_record.get("height")
        if (
            not isinstance(width, int)
            or width <= 0
            or not isinstance(height, int)
            or height <= 0
        ):
            raise ExportError(f"invalid camera dimensions in {view_path}")

        rgb_path = route_run / f"rgb_{camera}" / f"{phase_path.stem}.jpg"
        rgb_shape = None
        route_relative = route_run.relative_to(root)
        for target_index, target in enumerate(phase.get("stop_targets", [])):
            if not isinstance(target, dict):
                raise ExportError(f"invalid stop target in {phase_path}")
            summary["phase_targets"] += 1
            if target.get("status") != "valid":
                continue
            summary["valid_targets"] += 1
            if primary_only and target.get("primary_for_ego") is not True:
                summary["skipped_non_primary"] += 1
                continue

            target_id = target.get("target_id")
            target_view = _target_view(camera_record, target_id, view_path)
            if target_view.get("status") != "available":
                summary["skipped_camera_unknown"] += 1
                continue
            projection = target_view.get("boundary", {})
            if projection.get("projection_status") != "projected":
                summary["skipped_unprojected"] += 1
                continue
            segment = _validate_segment(
                projection.get("image_segment"), width, height, view_path
            )

            if rgb_shape is None:
                image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ExportError(f"unable to read source RGB: {rgb_path}")
                rgb_shape = image.shape[:2]
                if rgb_shape != (height, width):
                    raise ExportError(
                        f"RGB dimensions {rgb_shape[::-1]} do not match evidence "
                        f"dimensions {(width, height)} for {rgb_path}"
                    )

            boundary = target.get("leaderboard_infraction_boundary", {})
            endpoints = [boundary.get("left_endpoint"), boundary.get("right_endpoint")]
            if any(not isinstance(point, dict) for point in endpoints):
                raise ExportError(f"missing world boundary endpoints in {phase_path}")
            ego_segment = [
                _world_to_ego(point, phase.get("ego", {})) for point in endpoints
            ]
            stop_pose = target.get("recommended_ego_stop_pose", {}).get("location")
            if not isinstance(stop_pose, dict):
                raise ExportError(f"missing recommended stop pose in {phase_path}")

            mask = np.zeros((height, width), dtype=np.uint8)
            pixels = np.rint(segment).astype(np.int32)
            cv2.line(
                mask,
                tuple(pixels[0]),
                tuple(pixels[1]),
                255,
                line_thickness,
                cv2.LINE_8,
            )
            mask_relative = (
                Path("masks")
                / route_relative
                / camera
                / f"{phase_path.stem}__target_{target_index:02d}.png"
            )
            entry = {
                "schema_version": EXPORT_SCHEMA_VERSION,
                "label_type": LABEL_TYPE,
                "geometry_source": target.get("geometry_source"),
                "frame_id": phase_path.stem,
                "camera": camera,
                "target_id": target_id,
                "primary_for_ego": target.get("primary_for_ego") is True,
                "source_rgb": rgb_path.relative_to(root).as_posix(),
                "source_traffic_elements": phase_path.relative_to(root).as_posix(),
                "source_traffic_element_view": view_path.relative_to(root).as_posix(),
                "mask_path": mask_relative.as_posix(),
                "image_size": {"width": width, "height": height},
                "image_segment": segment.tolist(),
                "ego_bev_segment_m": ego_segment,
                "recommended_stop_pose_ego_m": _world_to_ego(
                    stop_pose, phase.get("ego", {})
                ),
                "signed_route_distance_m": target.get("signed_route_distance_m"),
                "relative_heading_degrees": target.get("relative_heading_degrees"),
                "ego_before_boundary": target.get("ego_before_boundary"),
                "owner_traffic_light_actor_ids": target.get(
                    "owner_traffic_light_actor_ids", []
                ),
                "state_by_actor_id": target.get("state_by_actor_id", {}),
                "painted_line_status": target_view.get("painted_line", {}).get(
                    "status", "unknown"
                ),
            }
            entries_and_masks.append((entry, mask_relative, mask))
            summary["labels_exported"] += 1

    if not entries_and_masks:
        raise ExportError("no projected virtual stop-boundary labels found")
    return entries_and_masks, summary


def export_stop_boundary_labels(
    root,
    output_dir,
    camera="front",
    limit=None,
    primary_only=False,
    line_thickness=3,
):
    """Export binary masks and a JSONL manifest from schema-v2/v3 records."""
    root = Path(root)
    output_dir = Path(output_dir)
    if not root.is_dir():
        raise ExportError(f"dataset root is not a directory: {root}")
    if camera not in {"front", "left", "right"}:
        raise ExportError(f"unsupported camera: {camera}")
    if limit is not None and limit < 0:
        raise ExportError("limit must be nonnegative")
    if not isinstance(line_thickness, int) or line_thickness <= 0:
        raise ExportError("line thickness must be a positive integer")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ExportError(f"output directory must be empty: {output_dir}")

    entries_and_masks, summary = _collect_export(
        root.resolve(), camera, limit, primary_only, line_thickness
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for entry, mask_relative, mask in entries_and_masks:
        mask_path = output_dir / mask_relative
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(mask_path), mask):
            raise ExportError(f"unable to write mask: {mask_path}")
        entries.append(entry)

    manifest_path = output_dir / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries),
        encoding="utf-8",
    )
    result = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "label_type": LABEL_TYPE,
        "dataset_root": str(root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "camera": camera,
        "primary_only": bool(primary_only),
        "line_thickness_pixels": line_thickness,
        "manifest": str(manifest_path.resolve()),
        **summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--camera", choices=("front", "left", "right"), default="front"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--primary-only", action="store_true")
    parser.add_argument("--line-thickness", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        result = export_stop_boundary_labels(
            args.root,
            args.output_dir,
            camera=args.camera,
            limit=args.limit,
            primary_only=args.primary_only,
            line_thickness=args.line_thickness,
        )
    except ExportError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
