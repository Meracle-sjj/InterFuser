#!/usr/bin/env python3
"""Validate and summarize traffic-element evidence schema v3 records."""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TEAM_CODE = REPO_ROOT / "leaderboard" / "team_code"
if str(TEAM_CODE) not in sys.path:
    sys.path.insert(0, str(TEAM_CODE))

from traffic_element_labels import SCHEMA_VERSION as TRAFFIC_SCHEMA_VERSION
from traffic_element_projection import EVIDENCE, IMAGE_SCHEMA_VERSION


FORBIDDEN_KEY_FRAGMENT = "stop_sign"
FORBIDDEN_VALUE_PREFIX = "traffic.stop"
FORBIDDEN_PROVENANCE = "trigger_volume_route_entry_approximation"
CAMERAS = ("front", "left", "right")


class AuditError(ValueError):
    pass


def _is_finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _forbidden_occurrences(value, path="record"):
    errors = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if FORBIDDEN_KEY_FRAGMENT in str(key).lower():
                errors.append(child_path)
            errors.extend(_forbidden_occurrences(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_forbidden_occurrences(child, f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        if (
            lowered.startswith(FORBIDDEN_VALUE_PREFIX)
            or lowered == FORBIDDEN_PROVENANCE
        ):
            errors.append(path)
    return errors


def _route_dirs(root):
    return sorted(
        path.parent
        for path in Path(root).glob("**/traffic_elements")
        if path.is_dir()
    )


def _frame_ids(directory, suffix):
    return {path.stem for path in directory.glob(f"*.{suffix}")}


def _frame_alignment_errors(route):
    phase = _frame_ids(route / "traffic_elements", "json")
    sources = {
        "view": _frame_ids(route / "traffic_element_views", "json"),
        "lidar": _frame_ids(route / "lidar", "npy"),
    }
    for camera in CAMERAS:
        sources[f"rgb_{camera}"] = _frame_ids(route / f"rgb_{camera}", "jpg")

    errors = []
    for name, frames in sources.items():
        if phase - frames:
            errors.append(f"missing {name} frames: " + ", ".join(sorted(phase - frames)))
        if frames - phase:
            errors.append(
                f"{name} frames without traffic_elements: "
                + ", ".join(sorted(frames - phase))
            )
    return errors


def _point_errors(point, path, width, height):
    if not isinstance(point, list) or len(point) != 2:
        return [f"{path} must be a finite image point"]
    if not all(_is_finite_number(value) for value in point):
        return [f"{path} must be a finite image point"]
    if not (0.0 <= point[0] < width and 0.0 <= point[1] < height):
        return [f"{path} must be inside image bounds"]
    return []


def _points_errors(points, path, width, height, minimum=0):
    if not isinstance(points, list) or len(points) < minimum:
        return [f"{path} must contain at least {minimum} image points"]
    errors = []
    for index, point in enumerate(points):
        errors.extend(_point_errors(point, f"{path}[{index}]", width, height))
    return errors


def _box_errors(item, path, width, height):
    visibility = item.get("visibility")
    if visibility not in {"visible", "not_visible", "unknown"}:
        return [f"{path}.visibility is invalid"]
    box = item.get("bbox_xyxy")
    if visibility != "visible":
        return [] if box is None else [f"{path}.bbox_xyxy must be null"]
    if not isinstance(box, list) or len(box) != 4 or not all(
        _is_finite_number(value) for value in box
    ):
        return [f"{path}.bbox_xyxy must be four finite values"]
    x1, y1, x2, y2 = box
    errors = []
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        errors.append(f"{path}.bbox_xyxy must have positive area inside image")
    if not isinstance(item.get("semantic_pixel_count"), int) or item[
        "semantic_pixel_count"
    ] < EVIDENCE["minimum_semantic_pixels"]:
        errors.append(
            f"{path} must have at least {EVIDENCE['minimum_semantic_pixels']} semantic pixels"
        )
    return errors


def _projection_errors(projection, path, width, height, segment=False):
    if not isinstance(projection, dict):
        return [f"{path} must be an object"]
    status = projection.get("projection_status")
    if status not in {"projected", "outside_image", "behind_camera", "unknown"}:
        return [f"{path}.projection_status is invalid"]
    errors = []
    if segment and status == "projected":
        errors.extend(
            _points_errors(
                projection.get("image_segment"),
                f"{path}.image_segment",
                width,
                height,
                minimum=2,
            )
        )
    if not segment and status == "projected":
        errors.extend(
            _point_errors(
                projection.get("image_point"),
                f"{path}.image_point",
                width,
                height,
            )
        )
    return errors


def _painted_line_errors(painted, path, width, height):
    if not isinstance(painted, dict):
        return [f"{path} must be an object"]
    status = painted.get("status")
    if status not in {"unknown", "candidate", "verified"}:
        return [f"{path}.status is invalid"]
    if status == "unknown":
        if painted.get("image_segment") is not None:
            return [f"{path}.image_segment must be null when unknown"]
        return []
    errors = _points_errors(
        painted.get("image_segment"),
        f"{path}.image_segment",
        width,
        height,
        minimum=2,
    )
    for field in ("score", "angle_error_degrees", "median_depth_residual_m"):
        if not _is_finite_number(painted.get(field)):
            errors.append(f"{path}.{field} must be finite")
    if status == "verified" and painted.get("review_source") != "manual_manifest":
        errors.append(f"{path}.verified requires review_source manual_manifest")
    return errors


def _camera_target_errors(view, phase_target, path, width, height):
    if not isinstance(view, dict):
        return [f"{path} must be an object"]
    errors = []
    if view.get("target_id") != phase_target.get("target_id"):
        errors.append(f"{path}.target_id does not match phase target")
    status = view.get("status")
    reason = view.get("unknown_reason")
    if phase_target.get("status") == "unknown":
        if status != "unknown" or reason != "geometry_unknown":
            errors.append(f"{path} must preserve geometry_unknown")
        return errors
    if status == "unknown":
        if reason not in {"sensor_unavailable", "projection_error"}:
            errors.append(f"{path}.unknown_reason must describe sensor evidence")
        return errors
    if status != "available" or reason is not None:
        errors.append(f"{path}.status must be available or sensor unknown")
        return errors

    errors.extend(
        _projection_errors(view.get("boundary"), f"{path}.boundary", width, height, True)
    )
    errors.extend(
        _projection_errors(
            view.get("recommended_stop_pose"),
            f"{path}.recommended_stop_pose",
            width,
            height,
        )
    )
    corridor = view.get("corridor")
    if not isinstance(corridor, dict):
        errors.append(f"{path}.corridor must be an object")
    else:
        projection_status = corridor.get("projection_status")
        if projection_status not in {"projected", "outside_image", "behind_camera", "unknown"}:
            errors.append(f"{path}.corridor.projection_status is invalid")
        if projection_status == "projected":
            errors.extend(
                _points_errors(
                    corridor.get("image_polyline"),
                    f"{path}.corridor.image_polyline",
                    width,
                    height,
                    minimum=1,
                )
            )
            errors.extend(
                _points_errors(
                    corridor.get("image_envelope"),
                    f"{path}.corridor.image_envelope",
                    width,
                    height,
                    minimum=3,
                )
            )
        finite = corridor.get("finite_depth_sample_count")
        supported = corridor.get("depth_supported_sample_count")
        if not isinstance(finite, int) or finite < 0:
            errors.append(f"{path}.corridor.finite_depth_sample_count must be non-negative")
        if not isinstance(supported, int) or supported < 0:
            errors.append(f"{path}.corridor.depth_supported_sample_count must be non-negative")
        elif isinstance(finite, int) and supported > finite:
            errors.append(f"{path}.corridor depth support exceeds finite samples")
    errors.extend(
        _painted_line_errors(view.get("painted_line"), f"{path}.painted_line", width, height)
    )
    return errors


def _camera_errors(camera, path, phase_lights, phase_targets):
    if not isinstance(camera, dict):
        return [f"{path} must be an object"]
    width, height = camera.get("width"), camera.get("height")
    if not isinstance(width, int) or width <= 0 or not isinstance(height, int) or height <= 0:
        return [f"{path} dimensions must be positive integers"]
    errors = []
    lights = camera.get("traffic_lights")
    targets = camera.get("stop_targets")
    if not isinstance(lights, list):
        errors.append(f"{path}.traffic_lights must be a list")
        lights = []
    if not isinstance(targets, list):
        errors.append(f"{path}.stop_targets must be a list")
        targets = []
    if not isinstance(camera.get("errors"), list):
        errors.append(f"{path}.errors must be a list")

    light_ids = set(phase_lights)
    seen_lights = set()
    for index, light in enumerate(lights):
        item_path = f"{path}.traffic_lights[{index}]"
        if not isinstance(light, dict):
            errors.append(f"{item_path} must be an object")
            continue
        actor_id = light.get("actor_id")
        if actor_id not in light_ids:
            errors.append(f"{item_path} actor {actor_id} missing from phase record")
        seen_lights.add(actor_id)
        errors.extend(_box_errors(light, item_path, width, height))
    if seen_lights != light_ids:
        errors.append(f"{path}.traffic_lights must cover every phase actor")

    by_id = {}
    for target in targets:
        if isinstance(target, dict):
            by_id.setdefault(target.get("target_id"), []).append(target)
    for target_id, phase_target in phase_targets.items():
        matches = by_id.get(target_id, [])
        if len(matches) != 1:
            errors.append(f"{path} target {target_id} must appear exactly once")
            continue
        errors.extend(
            _camera_target_errors(
                matches[0],
                phase_target,
                f"{path}.stop_targets[{target_id}]",
                width,
                height,
            )
        )
    if set(by_id) - set(phase_targets):
        errors.append(f"{path}.stop_targets contains unknown target IDs")
    return errors


def _matrix_valid(value):
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(row, list) and len(row) == 4 for row in value)
        and all(_is_finite_number(item) for row in value for item in row)
    )


def _xyz_list_valid(value):
    return isinstance(value, list) and all(
        isinstance(point, list)
        and len(point) == 3
        and all(_is_finite_number(item) for item in point)
        for point in value
    )


def _lidar_target_errors(view, phase_target, path):
    if not isinstance(view, dict):
        return [f"{path} must be an object"]
    errors = []
    if view.get("target_id") != phase_target.get("target_id"):
        errors.append(f"{path}.target_id does not match phase target")
    if phase_target.get("status") == "unknown":
        if view.get("status") != "unknown" or view.get("unknown_reason") != "geometry_unknown":
            errors.append(f"{path} must preserve geometry_unknown")
        return errors
    if view.get("status") == "unknown":
        if view.get("unknown_reason") not in {"sensor_unavailable", "projection_error"}:
            errors.append(f"{path}.unknown_reason must describe sensor evidence")
        return errors
    if view.get("status") != "available":
        return [f"{path}.status must be available or unknown"]
    for field in ("sensor_to_ego", "ego_to_world"):
        if not _matrix_valid(view.get(field)):
            errors.append(f"{path}.{field} must be a 4x4 finite matrix")
    if not _xyz_list_valid(view.get("corridor_centerline_xyz")):
        errors.append(f"{path}.corridor_centerline_xyz must contain finite xyz points")
    for field in ("in_corridor_point_count", "road_surface_point_count"):
        value = view.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"{path}.{field} must be a non-negative integer")
    if (
        isinstance(view.get("in_corridor_point_count"), int)
        and isinstance(view.get("road_surface_point_count"), int)
        and view["road_surface_point_count"] > view["in_corridor_point_count"]
    ):
        errors.append(f"{path}.road_surface_point_count exceeds corridor count")
    return errors


def _record_errors(record, phase, frame_id):
    if not isinstance(record, dict):
        return ["record must be an object"]
    errors = []
    if record.get("schema_version") != IMAGE_SCHEMA_VERSION:
        errors.append("unsupported evidence schema_version")
    if record.get("source_traffic_element_schema_version") != TRAFFIC_SCHEMA_VERSION:
        errors.append("source traffic-element schema_version mismatch")
    if record.get("frame_id") != frame_id:
        errors.append("frame_id must match filename")
    if record.get("association") != EVIDENCE:
        errors.append("association metadata mismatch")
    if record.get("errors"):
        errors.append("record contains unexplained evidence errors")

    phase_lights = {
        item.get("actor_id"): item
        for item in phase.get("traffic_lights", [])
        if isinstance(item, dict)
    }
    phase_targets = {
        item.get("target_id"): item
        for item in phase.get("stop_targets", [])
        if isinstance(item, dict)
    }
    cameras = record.get("cameras")
    if not isinstance(cameras, dict):
        errors.append("cameras must be an object")
    else:
        for camera_name in CAMERAS:
            errors.extend(
                _camera_errors(
                    cameras.get(camera_name),
                    f"cameras.{camera_name}",
                    phase_lights,
                    phase_targets,
                )
            )

    lidar = record.get("lidar")
    if not isinstance(lidar, dict):
        errors.append("lidar must be an object")
    else:
        lidar_targets = lidar.get("targets")
        if not isinstance(lidar_targets, list):
            errors.append("lidar.targets must be a list")
        else:
            by_id = {}
            for target in lidar_targets:
                if isinstance(target, dict):
                    by_id.setdefault(target.get("target_id"), []).append(target)
            for target_id, phase_target in phase_targets.items():
                matches = by_id.get(target_id, [])
                if len(matches) != 1:
                    errors.append(f"lidar target {target_id} must appear exactly once")
                    continue
                errors.extend(
                    _lidar_target_errors(
                        matches[0],
                        phase_target,
                        f"lidar.targets[{target_id}]",
                    )
                )
            if set(by_id) - set(phase_targets):
                errors.append("lidar contains unknown target IDs")
        if not isinstance(lidar.get("errors"), list):
            errors.append("lidar.errors must be a list")
    forbidden = _forbidden_occurrences(record)
    if forbidden:
        errors.append("forbidden STOP label at " + ", ".join(forbidden))
    return errors


def audit_traffic_element_views(root):
    routes = _route_dirs(root)
    if not routes:
        raise AuditError("no traffic element routes found")
    invalid = []
    summary = Counter()
    unknown_reasons = Counter()
    per_camera = defaultdict(Counter)
    corridor_counts = []
    surface_counts = []

    for route in routes:
        invalid.extend(f"{route}: {error}" for error in _frame_alignment_errors(route))
        for phase_path in sorted((route / "traffic_elements").glob("*.json")):
            frame_id = phase_path.stem
            view_path = route / "traffic_element_views" / f"{frame_id}.json"
            if not view_path.exists():
                continue
            try:
                phase = json.loads(phase_path.read_text(encoding="utf-8"))
                record = json.loads(view_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                invalid.append(f"{view_path}: invalid JSON: {exc}")
                continue
            if phase.get("schema_version") != TRAFFIC_SCHEMA_VERSION:
                invalid.append(f"{phase_path}: unsupported traffic-element schema_version")
                continue
            forbidden_phase = _forbidden_occurrences(phase)
            if forbidden_phase:
                invalid.append(
                    f"{phase_path}: forbidden STOP label at "
                    + ", ".join(forbidden_phase)
                )
                continue
            errors = _record_errors(record, phase, frame_id)
            if errors:
                invalid.append(f"{view_path}: " + "; ".join(errors))
                continue

            summary["frames"] += 1
            summary["unknown_stop_targets"] += sum(
                target.get("status") == "unknown"
                for target in phase["stop_targets"]
            )
            frame_visible = False
            frame_hard_negative = False
            frame_errors = bool(record.get("errors"))
            for camera_name, camera in record["cameras"].items():
                frame_errors = frame_errors or bool(camera.get("errors"))
                for light in camera["traffic_lights"]:
                    if light["visibility"] == "visible":
                        frame_visible = True
                        per_camera[camera_name]["visible_traffic_lights"] += 1
                        summary["semantic_confirmed_traffic_lights"] += 1
                        if not light.get("relevant_to_ego", False):
                            frame_hard_negative = True
                for target in camera["stop_targets"]:
                    if target["status"] == "unknown":
                        reason = target["unknown_reason"]
                        unknown_reasons[reason] += 1
                        if reason == "geometry_unknown":
                            summary["geometry_unknown_camera_evidence"] += 1
                        else:
                            summary["sensor_unknown_camera_evidence"] += 1
                        continue
                    if target["boundary"]["projection_status"] == "projected":
                        summary["projected_stop_boundaries"] += 1
                        per_camera[camera_name]["projected_stop_boundaries"] += 1
                    if target["corridor"]["projection_status"] == "projected":
                        summary["projected_stop_corridors"] += 1
                        per_camera[camera_name]["projected_stop_corridors"] += 1
                    painted_status = target["painted_line"]["status"]
                    if painted_status == "candidate":
                        summary["painted_line_candidates"] += 1
                    elif painted_status == "verified":
                        summary["verified_painted_lines"] += 1
            if frame_visible:
                summary["visible_traffic_light_frames"] += 1
            if frame_hard_negative:
                summary["hard_negative_frames"] += 1

            frame_errors = frame_errors or bool(record["lidar"].get("errors"))
            for target in record["lidar"]["targets"]:
                if target["status"] == "available":
                    summary["lidar_available_targets"] += 1
                    corridor_counts.append(target["in_corridor_point_count"])
                    surface_counts.append(target["road_surface_point_count"])
                else:
                    summary["lidar_unknown_targets"] += 1
            if frame_errors:
                summary["error_frames"] += 1

    if invalid:
        raise AuditError("\n".join(invalid))

    return {
        "frames": summary["frames"],
        "invalid_frames": 0,
        "visible_traffic_light_frames": summary["visible_traffic_light_frames"],
        "semantic_confirmed_traffic_lights": summary[
            "semantic_confirmed_traffic_lights"
        ],
        "projected_stop_boundaries": summary["projected_stop_boundaries"],
        "projected_stop_corridors": summary["projected_stop_corridors"],
        "painted_line_candidates": summary["painted_line_candidates"],
        "verified_painted_lines": summary["verified_painted_lines"],
        "unknown_stop_targets": summary["unknown_stop_targets"],
        "geometry_unknown_camera_evidence": summary[
            "geometry_unknown_camera_evidence"
        ],
        "sensor_unknown_camera_evidence": summary[
            "sensor_unknown_camera_evidence"
        ],
        "unknown_camera_evidence": (
            summary["geometry_unknown_camera_evidence"]
            + summary["sensor_unknown_camera_evidence"]
        ),
        "camera_unknown_reasons": dict(sorted(unknown_reasons.items())),
        "lidar_available_targets": summary["lidar_available_targets"],
        "lidar_unknown_targets": summary["lidar_unknown_targets"],
        "in_corridor_point_counts": corridor_counts,
        "road_surface_point_counts": surface_counts,
        "hard_negative_frames": summary["hard_negative_frames"],
        "error_frames": summary["error_frames"],
        "per_camera": {
            name: dict(sorted(counts.items()))
            for name, counts in sorted(per_camera.items())
        },
        "forbidden_stop_occurrences": 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args(argv)
    try:
        result = audit_traffic_element_views(args.root)
    except AuditError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
