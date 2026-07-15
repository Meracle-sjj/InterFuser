#!/usr/bin/env python3
"""Validate and summarize traffic-element image labels."""

import argparse
import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TEAM_CODE = REPO_ROOT / "leaderboard" / "team_code"
if str(TEAM_CODE) not in sys.path:
    sys.path.insert(0, str(TEAM_CODE))

from traffic_element_projection import ASSOCIATION, IMAGE_SCHEMA_VERSION


class AuditError(ValueError):
    pass


def _route_dirs(root):
    root = Path(root)
    if root.name == "traffic_element_views":
        return [root.parent]
    routes = {path.parent for path in root.rglob("traffic_element_views")}
    routes.update(path.parent for path in root.rglob("rgb_front"))
    return sorted(routes)


def _frame_alignment_errors(route):
    rgb_ids = {path.stem for path in (route / "rgb_front").glob("*.jpg")}
    phase1_ids = {
        path.stem for path in (route / "traffic_elements").glob("*.json")
    }
    view_ids = {
        path.stem for path in (route / "traffic_element_views").glob("*.json")
    }
    errors = []
    missing_views = sorted((rgb_ids | phase1_ids) - view_ids)
    missing_phase1 = sorted((rgb_ids | view_ids) - phase1_ids)
    missing_rgb = sorted((phase1_ids | view_ids) - rgb_ids)
    if missing_views:
        errors.append(f"{route}: missing view frames: {', '.join(missing_views)}")
    if missing_phase1:
        errors.append(
            f"{route}: missing traffic_elements frames: {', '.join(missing_phase1)}"
        )
    if missing_rgb:
        errors.append(f"{route}: missing rgb_front frames: {', '.join(missing_rgb)}")
    return errors, sorted(view_ids)


def _is_finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _validate_box(item, path, width, height):
    errors = []
    visibility = item.get("visibility")
    if visibility not in {"visible", "not_visible", "unknown"}:
        return [f"{path}.visibility is invalid"]

    box = item.get("bbox_xyxy")
    pixel_count = item.get("semantic_pixel_count")
    if not isinstance(pixel_count, int) or isinstance(pixel_count, bool):
        errors.append(f"{path}.semantic_pixel_count must be an integer")
        return errors
    if pixel_count < 0:
        errors.append(f"{path}.semantic_pixel_count must be nonnegative")

    if visibility != "visible":
        if box is not None:
            errors.append(f"{path}.bbox_xyxy must be null when not visible")
        return errors

    if not isinstance(box, list) or len(box) != 4:
        errors.append(f"{path}.bbox_xyxy must contain four coordinates")
        return errors
    if not all(_is_finite_number(value) for value in box):
        errors.append(f"{path}.bbox_xyxy coordinates must be finite")
        return errors
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        errors.append(f"{path}.bbox_xyxy must have positive area inside the image")
    if pixel_count < ASSOCIATION["minimum_semantic_pixels"]:
        errors.append(f"{path} requires at least 3 semantic pixels")
    if item.get("association_source") != "semantic_depth_confirmed":
        errors.append(f"{path}.association_source must confirm semantic depth")
    if not _is_finite_number(item.get("median_depth_residual_m")):
        errors.append(f"{path}.median_depth_residual_m must be finite")
    return errors


def _validate_line(line, path, actor_ids, width, height):
    errors = []
    if not isinstance(line, dict):
        return [f"{path} must be an object"]
    owner_id = line.get("owner_actor_id")
    if owner_id not in actor_ids:
        errors.append(f"{path} actor {owner_id} missing from Phase 1 labels")
    owner_type = line.get("owner_type")
    expected_source = {
        "traffic_light": "carla_stop_waypoint",
        "stop_sign": "trigger_volume_route_entry_approximation",
    }.get(owner_type)
    if expected_source is None or line.get("geometry_source") != expected_source:
        errors.append(f"{path} geometry source does not match owner type")
    if not _is_finite_number(line.get("longitudinal_distance")):
        errors.append(f"{path}.longitudinal_distance must be finite")
    if not isinstance(line.get("ego_before_line"), bool):
        errors.append(f"{path}.ego_before_line must be boolean")

    status = line.get("projection_status")
    if status not in {"projected", "outside_image", "behind_camera"}:
        errors.append(f"{path}.projection_status is invalid")
        return errors
    segment = line.get("image_segment")
    if status != "projected":
        if segment is not None:
            errors.append(f"{path}.image_segment must be null unless projected")
        return errors
    if not isinstance(segment, list) or len(segment) != 2:
        errors.append(f"{path}.image_segment must contain two image points")
        return errors
    for point_index, point in enumerate(segment):
        if (
            not isinstance(point, list)
            or len(point) != 2
            or not all(_is_finite_number(value) for value in point)
        ):
            errors.append(f"{path}.image_segment[{point_index}] must be a finite image point")
            continue
        if not (0 <= point[0] <= width - 1 and 0 <= point[1] <= height - 1):
            errors.append(f"{path}.image_segment[{point_index}] is outside the image")
    return errors


def _validate_camera(camera, path, light_ids, stop_ids):
    errors = []
    if not isinstance(camera, dict):
        return [f"{path} must be an object"]
    width = camera.get("width")
    height = camera.get("height")
    if not isinstance(width, int) or width <= 0:
        errors.append(f"{path}.width must be a positive integer")
    if not isinstance(height, int) or height <= 0:
        errors.append(f"{path}.height must be a positive integer")
    if errors:
        return errors

    for key, valid_ids in (("traffic_lights", light_ids), ("stop_signs", stop_ids)):
        items = camera.get(key)
        if not isinstance(items, list):
            errors.append(f"{path}.{key} must be a list")
            continue
        seen = set()
        for index, item in enumerate(items):
            item_path = f"{path}.{key}[{index}]"
            if not isinstance(item, dict):
                errors.append(f"{item_path} must be an object")
                continue
            actor_id = item.get("actor_id")
            if actor_id not in valid_ids:
                errors.append(f"{item_path} actor {actor_id} missing from Phase 1 labels")
            if actor_id in seen:
                errors.append(f"{item_path} duplicates actor {actor_id}")
            seen.add(actor_id)
            errors.extend(_validate_box(item, item_path, width, height))
        if seen != valid_ids:
            missing = sorted(valid_ids - seen)
            if missing:
                errors.append(f"{path}.{key} omits actors: {missing}")

    lines = camera.get("stop_lines")
    if not isinstance(lines, list):
        errors.append(f"{path}.stop_lines must be a list")
    else:
        actor_ids = light_ids | stop_ids
        for index, line in enumerate(lines):
            errors.extend(
                _validate_line(
                    line,
                    f"{path}.stop_lines[{index}]",
                    actor_ids,
                    width,
                    height,
                )
            )
    if not isinstance(camera.get("errors"), list):
        errors.append(f"{path}.errors must be a list")
    return errors


def _validate_record(record, phase1, frame_id):
    errors = []
    if not isinstance(record, dict):
        return ["record must be an object"]
    if record.get("schema_version") != IMAGE_SCHEMA_VERSION:
        errors.append("unsupported image schema_version")
    if record.get("source_traffic_element_schema_version") != phase1.get(
        "schema_version"
    ):
        errors.append("source schema version mismatch")
    if record.get("frame_id") != frame_id:
        errors.append("frame_id does not match filename")
    if record.get("association") != ASSOCIATION:
        errors.append("association metadata mismatch")
    if not isinstance(record.get("errors"), list):
        errors.append("errors must be a list")

    light_ids = {
        int(item["actor_id"])
        for item in phase1.get("traffic_lights", [])
        if isinstance(item, dict) and isinstance(item.get("actor_id"), int)
    }
    stop_ids = {
        int(item["actor_id"])
        for item in phase1.get("stop_signs", [])
        if isinstance(item, dict) and isinstance(item.get("actor_id"), int)
    }
    cameras = record.get("cameras")
    if not isinstance(cameras, dict):
        errors.append("cameras must be an object")
        return errors
    expected_cameras = {"front", "left", "right"}
    if set(cameras) != expected_cameras:
        errors.append("cameras must contain front, left, and right")
    for camera_name, camera in cameras.items():
        errors.extend(
            _validate_camera(
                camera,
                f"cameras.{camera_name}",
                light_ids,
                stop_ids,
            )
        )
    return errors


def audit_traffic_element_views(root):
    """Audit a dataset root and return coverage statistics."""
    routes = _route_dirs(root)
    if not routes:
        raise AuditError(f"no traffic element image labels under {root}")

    alignment_errors = []
    route_frames = {}
    for route in routes:
        errors, frames = _frame_alignment_errors(route)
        alignment_errors.extend(errors)
        route_frames[route] = frames
    if alignment_errors:
        raise AuditError("\n".join(alignment_errors))

    summary = {
        "frames": sum(len(frames) for frames in route_frames.values()),
        "invalid_frames": 0,
        "error_frames": 0,
        "unknown_frames": 0,
        "visible_traffic_light_frames": 0,
        "visible_stop_sign_frames": 0,
        "semantic_confirmed_traffic_lights": 0,
        "semantic_confirmed_stop_signs": 0,
        "active_traffic_light_frames": 0,
        "route_relevant_stop_sign_frames": 0,
        "projected_stop_lines": 0,
        "projected_stop_lines_before": 0,
        "projected_stop_lines_after": 0,
        "hard_negative_frames": 0,
        "unique_traffic_light_actors": 0,
        "unique_stop_sign_actors": 0,
        "per_camera": {},
    }
    invalid = []
    unique_lights = set()
    unique_stops = set()

    for route, frame_ids in route_frames.items():
        for frame_id in frame_ids:
            phase1_path = route / "traffic_elements" / f"{frame_id}.json"
            view_path = route / "traffic_element_views" / f"{frame_id}.json"
            try:
                phase1 = json.loads(phase1_path.read_text(encoding="utf-8"))
                record = json.loads(view_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                invalid.append(f"{view_path}: {exc}")
                continue

            record_errors = _validate_record(record, phase1, frame_id)
            if record_errors:
                invalid.append(f"{view_path}: {'; '.join(record_errors)}")
                continue

            frame_has_error = bool(record["errors"])
            frame_has_unknown = False
            frame_visible_light = False
            frame_visible_stop = False
            frame_active_light = False
            frame_relevant_stop = False
            for camera_name, camera in record["cameras"].items():
                camera_summary = summary["per_camera"].setdefault(
                    camera_name,
                    {
                        "visible_traffic_lights": 0,
                        "visible_stop_signs": 0,
                        "projected_stop_lines": 0,
                    },
                )
                frame_has_error = frame_has_error or bool(camera["errors"])
                for item in camera["traffic_lights"]:
                    unique_lights.add(item["actor_id"])
                    frame_has_unknown = (
                        frame_has_unknown or item["visibility"] == "unknown"
                    )
                    if item["visibility"] == "visible":
                        summary["semantic_confirmed_traffic_lights"] += 1
                        camera_summary["visible_traffic_lights"] += 1
                        frame_visible_light = True
                        frame_active_light = (
                            frame_active_light or item.get("is_active_for_ego") is True
                        )
                for item in camera["stop_signs"]:
                    unique_stops.add(item["actor_id"])
                    frame_has_unknown = (
                        frame_has_unknown or item["visibility"] == "unknown"
                    )
                    if item["visibility"] == "visible":
                        summary["semantic_confirmed_stop_signs"] += 1
                        camera_summary["visible_stop_signs"] += 1
                        frame_visible_stop = True
                        frame_relevant_stop = (
                            frame_relevant_stop
                            or item.get("affects_ego_route") is True
                        )
                for line in camera["stop_lines"]:
                    if line["projection_status"] == "projected":
                        summary["projected_stop_lines"] += 1
                        camera_summary["projected_stop_lines"] += 1
                        if line["ego_before_line"]:
                            summary["projected_stop_lines_before"] += 1
                        else:
                            summary["projected_stop_lines_after"] += 1

            summary["error_frames"] += int(frame_has_error)
            summary["unknown_frames"] += int(frame_has_unknown)
            summary["visible_traffic_light_frames"] += int(frame_visible_light)
            summary["visible_stop_sign_frames"] += int(frame_visible_stop)
            summary["active_traffic_light_frames"] += int(frame_active_light)
            summary["route_relevant_stop_sign_frames"] += int(frame_relevant_stop)
            summary["hard_negative_frames"] += int(
                not any(
                    item.get("is_active_for_ego") is True
                    for camera in record["cameras"].values()
                    for item in camera["traffic_lights"]
                )
            )

    summary["invalid_frames"] = len(invalid)
    summary["unique_traffic_light_actors"] = len(unique_lights)
    summary["unique_stop_sign_actors"] = len(unique_stops)
    if invalid:
        raise AuditError("\n".join(invalid))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit traffic-element image labels"
    )
    parser.add_argument("root")
    args = parser.parse_args(argv)
    try:
        summary = audit_traffic_element_views(args.root)
    except AuditError as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
