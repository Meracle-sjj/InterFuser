#!/usr/bin/env python3
"""Compare collector stop boundaries with an independent Leaderboard reproduction."""

import argparse
import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TEAM_CODE = REPO_ROOT / "leaderboard" / "team_code"
CARLA_API = REPO_ROOT / "carla" / "PythonAPI"
for dependency in (TEAM_CODE, CARLA_API, CARLA_API / "carla"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from leaderboard_stop_targets import (
    advance_to_infraction_boundary,
    boundary_from_waypoint,
    sample_trigger_lane_waypoints,
)


def _distance(first, second):
    return math.sqrt(
        sum((float(first[axis]) - float(second[axis])) ** 2 for axis in ("x", "y", "z"))
    )


def _sort_key(boundary):
    return (
        int(boundary.get("road_id", 0)),
        int(boundary.get("section_id", 0)),
        int(boundary.get("lane_id", 0)),
        float(boundary.get("s", 0.0)),
    )


def compare_boundaries(actual, expected, tolerance_m=1e-3):
    """Return deterministic geometry mismatch descriptions."""
    tolerance_m = float(tolerance_m)
    if not math.isfinite(tolerance_m) or tolerance_m < 0.0:
        raise ValueError("tolerance_m must be finite and nonnegative")
    actual = sorted(list(actual), key=_sort_key)
    expected = sorted(list(expected), key=_sort_key)
    mismatches = []
    if len(actual) != len(expected):
        mismatches.append(
            f"boundary count differs: actual={len(actual)} expected={len(expected)}"
        )
    for index, (observed, reference) in enumerate(zip(actual, expected)):
        observed_lane = _sort_key(observed)[:3]
        reference_lane = _sort_key(reference)[:3]
        if observed_lane != reference_lane:
            mismatches.append(
                f"boundary {index} lane identity differs: "
                f"actual={observed_lane} expected={reference_lane}"
            )
        for field in ("center", "left_endpoint", "right_endpoint"):
            try:
                error = _distance(observed[field], reference[field])
            except (KeyError, TypeError, ValueError) as exc:
                mismatches.append(f"boundary {index} {field} is invalid: {exc}")
                continue
            if error > tolerance_m:
                mismatches.append(
                    f"boundary {index} {field} differs by {error:.6f} m"
                )
    return mismatches


def _lane_id(waypoint):
    return (
        int(waypoint.road_id),
        int(getattr(waypoint, "section_id", 0)),
        int(waypoint.lane_id),
    )


def _is_intersection(waypoint):
    return bool(
        getattr(waypoint, "is_intersection", getattr(waypoint, "is_junction", False))
    )


def _location_like(reference, x, y, z):
    try:
        return type(reference)(x=x, y=y, z=z)
    except TypeError:
        result = type("ReferenceLocation", (), {})()
        result.x, result.y, result.z = x, y, z
        return result


def _reference_trigger_waypoints(light, world_map):
    transform = light.get_transform()
    trigger = light.trigger_volume
    center = transform.transform(trigger.location)
    yaw = math.radians(float(transform.rotation.yaw))
    offset = -0.9 * float(trigger.extent.x)
    maximum = 0.9 * float(trigger.extent.x)
    result = []
    while offset < maximum:
        waypoint = world_map.get_waypoint(
            _location_like(
                center,
                center.x + offset * math.cos(yaw),
                center.y + offset * math.sin(yaw),
                center.z,
            )
        )
        if waypoint is not None and (
            not result or _lane_id(result[-1]) != _lane_id(waypoint)
        ):
            result.append(waypoint)
        offset += 1.0
    return result


def _reference_advance(start):
    waypoint = start
    branch_seen = False
    visited = set()
    steps = 0
    while not _is_intersection(waypoint):
        identity = _lane_id(waypoint) + (round(float(getattr(waypoint, "s", 0.0)), 2),)
        if identity in visited or steps >= 400:
            return waypoint, False, branch_seen
        visited.add(identity)
        successors = list(waypoint.next(0.5) or [])
        if not successors:
            return waypoint, False, branch_seen
        branch_seen = branch_seen or len(successors) > 1
        following = successors[0]
        if _is_intersection(following):
            break
        waypoint = following
        steps += 1
    return waypoint, not branch_seen, branch_seen


def _reference_boundary(waypoint):
    center = waypoint.transform.location
    yaw = math.radians(float(waypoint.transform.rotation.yaw))
    offset = 0.4 * float(waypoint.lane_width)
    normal = (-math.sin(yaw), math.cos(yaw))

    def point(sign):
        return {
            "x": float(center.x + sign * offset * normal[0]),
            "y": float(center.y + sign * offset * normal[1]),
            "z": float(center.z),
        }

    return {
        "road_id": int(waypoint.road_id),
        "section_id": int(getattr(waypoint, "section_id", 0)),
        "lane_id": int(waypoint.lane_id),
        "s": float(getattr(waypoint, "s", 0.0)),
        "center": {"x": float(center.x), "y": float(center.y), "z": float(center.z)},
        "left_endpoint": point(1.0),
        "right_endpoint": point(-1.0),
    }


def check_town(world, tolerance_m=1e-3):
    world_map = world.get_map()
    lights = list(world.get_actors().filter("traffic.traffic_light*"))
    actual = []
    expected = []
    unknown = 0
    branches = 0
    trigger_distances = []
    for light in lights:
        actual_starts = sample_trigger_lane_waypoints(light, world_map)
        reference_starts = _reference_trigger_waypoints(light, world_map)
        for start in actual_starts:
            traversal = advance_to_infraction_boundary(start)
            if traversal["status"] != "valid":
                unknown += 1
                branches += traversal["unknown_reason"] == "waypoint_branch"
                continue
            boundary = boundary_from_waypoint(traversal["waypoint"])
            actual.append(boundary)
            trigger = start.transform.location
            trigger_distances.append(
                math.sqrt(
                    (trigger.x - boundary["center"]["x"]) ** 2
                    + (trigger.y - boundary["center"]["y"]) ** 2
                    + (trigger.z - boundary["center"]["z"]) ** 2
                )
            )
        for start in reference_starts:
            waypoint, valid, _branch = _reference_advance(start)
            if valid:
                expected.append(_reference_boundary(waypoint))
    mismatches = compare_boundaries(actual, expected, tolerance_m)
    return {
        "map": str(world_map.name).split("/")[-1],
        "traffic_light_actors": len(lights),
        "valid_boundaries": len(actual),
        "unknown_boundaries": unknown,
        "branches": branches,
        "trigger_to_boundary_distances_m": trigger_distances,
        "mismatches": mismatches,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("towns", nargs="+")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2400)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    results = []
    for town in args.towns:
        world = client.load_world(town)
        results.append(check_town(world, args.tolerance))
    report = {
        "host": args.host,
        "port": args.port,
        "towns": results,
        "mismatch_count": sum(len(item["mismatches"]) for item in results),
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 2 if report["mismatch_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
