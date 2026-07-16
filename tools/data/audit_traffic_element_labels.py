#!/usr/bin/env python3
"""Validate and summarize traffic-element schema v2 collection labels."""

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

from leaderboard_stop_targets import GEOMETRY_SOURCE
from traffic_element_labels import SCHEMA_VERSION, validate_traffic_element_record


FORBIDDEN_KEY_FRAGMENT = "stop_sign"
FORBIDDEN_VALUE_PREFIX = "traffic.stop"
FORBIDDEN_PROVENANCE = "trigger_volume_route_entry_approximation"
GENERATED_JSON_DIRS = {
    "traffic_elements",
    "traffic_element_views",
    "measurements",
    "affordances",
    "3d_bbs",
    "other_actors",
}


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


def _point_errors(value, path):
    if not isinstance(value, dict):
        return [f"{path} must be an object"]
    return [
        f"{path}.{axis} must be finite"
        for axis in ("x", "y", "z")
        if not _is_finite_number(value.get(axis))
    ]


def _target_errors(target, path, map_name, light_ids):
    if not isinstance(target, dict):
        return [f"{path} must be an object"]
    errors = []
    target_id = target.get("target_id")
    if not isinstance(target_id, str) or not target_id:
        errors.append(f"{path}.target_id must be a non-empty string")
    if target.get("geometry_source") != GEOMETRY_SOURCE:
        errors.append(f"{path}.geometry_source must be {GEOMETRY_SOURCE}")

    status = target.get("status")
    reason = target.get("unknown_reason")
    if status not in {"valid", "unknown"}:
        errors.append(f"{path}.status must be valid or unknown")
    elif status == "valid" and reason is not None:
        errors.append(f"{path}.unknown_reason must be null for valid targets")
    elif status == "unknown" and not isinstance(reason, str):
        errors.append(f"{path}.unknown_reason must explain unknown targets")

    owners = target.get("owner_traffic_light_actor_ids")
    if not isinstance(owners, list) or not owners:
        errors.append(f"{path}.owner_traffic_light_actor_ids must be non-empty")
    elif any(not isinstance(item, int) or item not in light_ids for item in owners):
        errors.append(f"{path}.owner_traffic_light_actor_ids must reference lights")

    boundary = target.get("leaderboard_infraction_boundary")
    if not isinstance(boundary, dict):
        errors.append(f"{path}.leaderboard_infraction_boundary must be an object")
    else:
        if boundary.get("geometry_source") != GEOMETRY_SOURCE:
            errors.append(
                f"{path}.leaderboard_infraction_boundary.geometry_source "
                f"must be {GEOMETRY_SOURCE}"
            )
        for key in ("road_id", "section_id", "lane_id"):
            if not isinstance(boundary.get(key), int):
                errors.append(f"{path}.leaderboard_infraction_boundary.{key} must be int")
        if not _is_finite_number(boundary.get("s")):
            errors.append(f"{path}.leaderboard_infraction_boundary.s must be finite")
        for key in ("center", "left_endpoint", "right_endpoint"):
            errors.extend(_point_errors(boundary.get(key), f"{path}.{key}"))
        if (
            isinstance(target_id, str)
            and all(isinstance(boundary.get(key), int) for key in ("road_id", "section_id", "lane_id"))
            and _is_finite_number(boundary.get("s"))
        ):
            expected_id = "{}:{}:{}:{}:{:.1f}".format(
                str(map_name).split("/")[-1],
                boundary["road_id"],
                boundary["section_id"],
                boundary["lane_id"],
                boundary["s"],
            )
            if target_id != expected_id:
                errors.append(f"{path}.target_id must equal {expected_id}")

    stop_pose = target.get("recommended_ego_stop_pose")
    if not isinstance(stop_pose, dict):
        errors.append(f"{path}.recommended_ego_stop_pose must be an object")
    else:
        errors.extend(
            _point_errors(
                stop_pose.get("location"),
                f"{path}.recommended_ego_stop_pose.location",
            )
        )

    corridor = target.get("stop_evidence_corridor")
    centerline = corridor.get("centerline") if isinstance(corridor, dict) else None
    if not isinstance(centerline, list) or not centerline:
        errors.append(f"{path}.stop_evidence_corridor.centerline must be non-empty")
    else:
        for index, sample in enumerate(centerline):
            sample_path = f"{path}.stop_evidence_corridor.centerline[{index}]"
            if not isinstance(sample, dict):
                errors.append(f"{sample_path} must be an object")
                continue
            errors.extend(_point_errors(sample.get("location"), f"{sample_path}.location"))
            if not _is_finite_number(sample.get("lane_width")) or sample["lane_width"] <= 0:
                errors.append(f"{sample_path}.lane_width must be positive and finite")

    for field in (
        "signed_route_distance_m",
        "euclidean_distance_m",
        "relative_heading_degrees",
        "trigger_to_boundary_route_distance_m",
    ):
        if not _is_finite_number(target.get(field)):
            errors.append(f"{path}.{field} must be finite")
    if target.get("primary_for_ego") is True and status != "valid":
        errors.append(f"{path}.primary_for_ego requires valid geometry")
    return errors


def _record_errors(record, filename):
    errors = list(validate_traffic_element_record(record))
    if not isinstance(record, dict):
        return errors
    if record.get("schema_version") != SCHEMA_VERSION:
        return errors
    if record.get("frame_id") != filename.stem:
        errors.append("frame_id must match filename")
    if record.get("errors"):
        errors.append("record contains unexplained collection errors")

    lights = record.get("traffic_lights", [])
    light_ids = set()
    active_ids = []
    for index, light in enumerate(lights):
        path = f"traffic_lights[{index}]"
        if not isinstance(light, dict):
            errors.append(f"{path} must be an object")
            continue
        actor_id = light.get("actor_id")
        if not isinstance(actor_id, int):
            errors.append(f"{path}.actor_id must be int")
        else:
            light_ids.add(actor_id)
        if not isinstance(light.get("state"), str) or not light["state"]:
            errors.append(f"{path}.state must be a non-empty string")
        if light.get("is_active_for_ego") is True and isinstance(actor_id, int):
            active_ids.append(actor_id)
    active_id = record.get("active_traffic_light_id")
    if active_id is None:
        if active_ids:
            errors.append("active_traffic_light_id must be null without active items")
    elif active_ids != [active_id]:
        errors.append("active_traffic_light_id must match exactly one active item")

    target_ids = []
    primary_count = 0
    for index, target in enumerate(record.get("stop_targets", [])):
        path = f"stop_targets[{index}]"
        errors.extend(_target_errors(target, path, record.get("map_name"), light_ids))
        if isinstance(target, dict):
            target_ids.append(target.get("target_id"))
            primary_count += target.get("primary_for_ego") is True
    if len(target_ids) != len(set(target_ids)):
        errors.append("target_id must be unique within each frame")
    if primary_count > 1:
        errors.append("each frame may have at most one primary stop target")
    return errors


def _label_files(root):
    return sorted(Path(root).glob("**/traffic_elements/*.json"))


def _route_dirs(label_files):
    return sorted({path.parent.parent for path in label_files})


def _frame_alignment_errors(route):
    labels = {path.stem for path in (route / "traffic_elements").glob("*.json")}
    rgb = {path.stem for path in (route / "rgb_front").glob("*.jpg")}
    errors = []
    if labels - rgb:
        errors.append("missing rgb_front frames: " + ", ".join(sorted(labels - rgb)))
    if rgb - labels:
        errors.append("missing traffic_elements frames: " + ", ".join(sorted(rgb - labels)))
    return errors


def _generated_json_files(route):
    for directory in sorted(route.iterdir() if route.exists() else []):
        if not directory.is_dir():
            continue
        if directory.name not in GENERATED_JSON_DIRS and not directory.name.startswith(
            "2d_bbs_"
        ):
            continue
        yield from sorted(directory.glob("*.json"))


def _scan_forbidden(route):
    errors = []
    for path in _generated_json_files(route):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: invalid generated JSON: {exc}")
            continue
        occurrences = _forbidden_occurrences(value)
        if occurrences:
            errors.append(
                f"{path}: forbidden STOP label at " + ", ".join(occurrences)
            )
    return errors


def audit_dataset(root):
    root = Path(root)
    label_files = _label_files(root)
    if not label_files:
        raise AuditError("no traffic element label files found")

    invalid = []
    for route in _route_dirs(label_files):
        invalid.extend(f"{route}: {error}" for error in _frame_alignment_errors(route))
        invalid.extend(_scan_forbidden(route))

    traffic_states = Counter()
    unknown_reasons = Counter()
    town_counts = defaultdict(Counter)
    summary = Counter()
    signed_distances = []
    trigger_distances = []
    for path in label_files:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            invalid.append(f"{path}: invalid label JSON: {exc}")
            continue
        errors = _record_errors(record, path)
        if errors:
            invalid.append(f"{path}: " + "; ".join(errors))
            continue

        summary["frames"] += 1
        town = record["map_name"]
        town_counts[town]["frames"] += 1
        lights = record["traffic_lights"]
        summary["traffic_lights"] += len(lights)
        for light in lights:
            traffic_states[light["state"]] += 1
        if record["active_traffic_light_id"] is not None:
            summary["active_traffic_light_frames"] += 1

        has_primary = False
        for target in record["stop_targets"]:
            key = "valid_stop_targets" if target["status"] == "valid" else "unknown_stop_targets"
            summary[key] += 1
            town_counts[town][key] += 1
            if target["status"] == "unknown":
                unknown_reasons[target["unknown_reason"]] += 1
            if target["primary_for_ego"]:
                has_primary = True
            if target["ego_before_boundary"]:
                summary["before_boundary_targets"] += 1
            else:
                summary["after_boundary_targets"] += 1
            signed_distances.append(target["signed_route_distance_m"])
            trigger_distances.append(target["trigger_to_boundary_route_distance_m"])
        if has_primary:
            summary["primary_stop_target_frames"] += 1

    if invalid:
        raise AuditError("\n".join(invalid))

    return {
        "frames": summary["frames"],
        "invalid_frames": 0,
        "traffic_lights": summary["traffic_lights"],
        "traffic_light_states": dict(sorted(traffic_states.items())),
        "active_traffic_light_frames": summary["active_traffic_light_frames"],
        "valid_stop_targets": summary["valid_stop_targets"],
        "unknown_stop_targets": summary["unknown_stop_targets"],
        "unknown_reasons": dict(sorted(unknown_reasons.items())),
        "primary_stop_target_frames": summary["primary_stop_target_frames"],
        "before_boundary_targets": summary["before_boundary_targets"],
        "after_boundary_targets": summary["after_boundary_targets"],
        "signed_route_distances_m": signed_distances,
        "trigger_to_boundary_route_distances_m": trigger_distances,
        "towns": {
            town: dict(sorted(counts.items()))
            for town, counts in sorted(town_counts.items())
        },
        "forbidden_stop_occurrences": 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args(argv)
    try:
        result = audit_dataset(args.root)
    except AuditError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
