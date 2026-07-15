#!/usr/bin/env python3
"""Validate and summarize versioned traffic-element collection labels."""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TEAM_CODE = REPO_ROOT / "leaderboard" / "team_code"
if str(TEAM_CODE) not in sys.path:
    sys.path.insert(0, str(TEAM_CODE))

from traffic_element_labels import validate_traffic_element_record


class AuditError(ValueError):
    pass


def _is_finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _stop_line_validation_errors(line, path, expected_source):
    errors = []
    if not isinstance(line, dict):
        return [f"{path} must be an object"]

    if line.get("geometry_source") != expected_source:
        errors.append(f"{path} must use {expected_source}")
    expected_exact = expected_source == "carla_stop_waypoint"
    if line.get("is_exact_carla_stop_position") is not expected_exact:
        errors.append(
            f"{path}.is_exact_carla_stop_position must be {expected_exact}"
        )

    for field in ("road_id", "section_id", "lane_id"):
        value = line.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"{path}.{field} must be an integer")
    for field in ("s", "lane_width", "longitudinal_distance", "lateral_offset"):
        if not _is_finite_number(line.get(field)):
            errors.append(f"{path}.{field} must be a finite number")

    for field in ("center", "left_endpoint", "right_endpoint"):
        vector = line.get(field)
        if not isinstance(vector, dict):
            errors.append(f"{path}.{field} must be an object")
            continue
        for component in ("x", "y", "z"):
            if not _is_finite_number(vector.get(component)):
                errors.append(f"{path}.{field}.{component} must be a finite number")

    relative = line.get("relative_center")
    if not isinstance(relative, dict):
        errors.append(f"{path}.relative_center must be an object")
    else:
        for component in ("forward", "right", "up"):
            if not _is_finite_number(relative.get(component)):
                errors.append(
                    f"{path}.relative_center.{component} must be a finite number"
                )

    if not isinstance(line.get("ego_before_line"), bool):
        errors.append(f"{path}.ego_before_line must be boolean")
    return errors


def _label_files(root):
    root = Path(root)
    if root.name == "traffic_elements":
        return sorted(root.glob("*.json"))
    return sorted(root.rglob("traffic_elements/*.json"))


def _frame_alignment_errors(root, label_files):
    root = Path(root)
    route_dirs = {path.parent.parent for path in label_files}
    if root.name == "traffic_elements":
        route_dirs.add(root.parent)
    else:
        route_dirs.update(path.parent for path in root.rglob("rgb_front"))

    errors = []
    for route_dir in sorted(route_dirs):
        label_ids = {
            path.stem for path in (route_dir / "traffic_elements").glob("*.json")
        }
        rgb_ids = {path.stem for path in (route_dir / "rgb_front").glob("*.jpg")}
        missing_rgb = sorted(label_ids - rgb_ids)
        missing_labels = sorted(rgb_ids - label_ids)
        if missing_rgb:
            errors.append(
                f"{route_dir}: missing rgb_front frames: {', '.join(missing_rgb)}"
            )
        if missing_labels:
            errors.append(
                f"{route_dir}: missing traffic_elements frames: "
                f"{', '.join(missing_labels)}"
            )
    return errors


def _nested_validation_errors(record):
    errors = []
    for index, item in enumerate(record.get("traffic_lights", [])):
        if not isinstance(item, dict):
            errors.append(f"traffic_lights[{index}] must be an object")
            continue
        if not isinstance(item.get("actor_id"), int):
            errors.append(f"traffic_lights[{index}].actor_id must be an integer")
        stop_lines = item.get("stop_lines")
        if not isinstance(stop_lines, list):
            errors.append(f"traffic_lights[{index}].stop_lines must be a list")
        else:
            for line_index, line in enumerate(stop_lines):
                errors.extend(
                    _stop_line_validation_errors(
                        line,
                        f"traffic_lights[{index}].stop_lines[{line_index}]",
                        "carla_stop_waypoint",
                    )
                )
        state = item.get("state")
        if not isinstance(state, str) or not state:
            errors.append(
                f"traffic_lights[{index}].state must be a non-empty string"
            )
        if not isinstance(item.get("is_active_for_ego"), bool):
            errors.append(
                f"traffic_lights[{index}].is_active_for_ego must be boolean"
            )
    for index, item in enumerate(record.get("stop_signs", [])):
        if not isinstance(item, dict):
            errors.append(f"stop_signs[{index}] must be an object")
            continue
        if not isinstance(item.get("actor_id"), int):
            errors.append(f"stop_signs[{index}].actor_id must be an integer")
        stop_lines = item.get("stop_lines")
        if not isinstance(stop_lines, list):
            errors.append(f"stop_signs[{index}].stop_lines must be a list")
        else:
            for line_index, line in enumerate(stop_lines):
                errors.extend(
                    _stop_line_validation_errors(
                        line,
                        f"stop_signs[{index}].stop_lines[{line_index}]",
                        "trigger_volume_route_entry_approximation",
                    )
                )
        affects_route = item.get("affects_ego_route")
        if not isinstance(affects_route, bool):
            errors.append(
                f"stop_signs[{index}].affects_ego_route must be boolean"
            )
        elif isinstance(stop_lines, list):
            if affects_route and len(stop_lines) != 1:
                errors.append(
                    f"stop_signs[{index}].stop_lines must contain exactly one "
                    "line when affects_ego_route is true"
                )
            if not affects_route and stop_lines:
                errors.append(
                    f"stop_signs[{index}].stop_lines must be empty when "
                    "affects_ego_route is false"
                )

    active_id = record.get("active_traffic_light_id")
    if active_id is not None and (
        not isinstance(active_id, int) or isinstance(active_id, bool)
    ):
        errors.append("active_traffic_light_id must be an integer or null")
    else:
        active_items = [
            item
            for item in record.get("traffic_lights", [])
            if isinstance(item, dict) and item.get("is_active_for_ego") is True
        ]
        matching_active_items = [
            item for item in active_items if item.get("actor_id") == active_id
        ]
        if active_id is None:
            if active_items:
                errors.append(
                    "active_traffic_light_id must be set when an item is active"
                )
        elif len(active_items) != 1 or len(matching_active_items) != 1:
            errors.append(
                "active_traffic_light_id must match exactly one active item"
            )
    return errors


def audit_dataset(root):
    files = _label_files(root)
    if not files:
        raise AuditError(f"no traffic element label files under {root}")
    alignment_errors = _frame_alignment_errors(root, files)
    if alignment_errors:
        raise AuditError("\n".join(alignment_errors))

    summary = {
        "frames": len(files),
        "invalid_frames": 0,
        "extraction_error_frames": 0,
        "traffic_lights": 0,
        "traffic_light_states": {},
        "active_traffic_light_frames": 0,
        "exact_traffic_light_stop_lines": 0,
        "stop_signs": 0,
        "stop_sign_frames": 0,
        "unique_stop_sign_actors": 0,
        "route_relevant_stop_sign_frames": 0,
        "approximate_stop_sign_stop_lines": 0,
    }
    state_counts = Counter()
    stop_sign_actor_ids = set()
    invalid = []

    for path in files:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            invalid.append(f"{path}: {exc}")
            continue

        record_errors = validate_traffic_element_record(record)
        if isinstance(record, dict):
            record_errors.extend(_nested_validation_errors(record))
        if record_errors:
            invalid.append(f"{path}: {'; '.join(record_errors)}")
            continue

        if record["errors"]:
            summary["extraction_error_frames"] += 1

        lights = record["traffic_lights"]
        summary["traffic_lights"] += len(lights)
        if any(item["is_active_for_ego"] for item in lights):
            summary["active_traffic_light_frames"] += 1
        for item in lights:
            state_counts[item["state"]] += 1
            summary["exact_traffic_light_stop_lines"] += sum(
                line.get("geometry_source") == "carla_stop_waypoint"
                for line in item["stop_lines"]
            )

        stop_signs = record["stop_signs"]
        summary["stop_signs"] += len(stop_signs)
        if stop_signs:
            summary["stop_sign_frames"] += 1
            stop_sign_actor_ids.update(item["actor_id"] for item in stop_signs)
        if any(item["affects_ego_route"] for item in stop_signs):
            summary["route_relevant_stop_sign_frames"] += 1
        for item in stop_signs:
            summary["approximate_stop_sign_stop_lines"] += sum(
                line.get("geometry_source")
                == "trigger_volume_route_entry_approximation"
                for line in item["stop_lines"]
            )

    summary["invalid_frames"] = len(invalid)
    summary["traffic_light_states"] = dict(sorted(state_counts.items()))
    summary["unique_stop_sign_actors"] = len(stop_sign_actor_ids)
    if invalid:
        raise AuditError("\n".join(invalid))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit versioned traffic-element labels"
    )
    parser.add_argument("root", help="Dataset root or traffic_elements directory")
    args = parser.parse_args(argv)

    try:
        summary = audit_dataset(args.root)
    except AuditError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
