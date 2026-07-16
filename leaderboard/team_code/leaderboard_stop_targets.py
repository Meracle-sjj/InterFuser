"""Build traffic-light stop geometry that matches Leaderboard evaluation."""

import math


GEOMETRY_SOURCE = "scenario_runner_running_red_light_test_v1"
TRIGGER_SOURCE = "carla_traffic_light_trigger_waypoint"
TRIGGER_EXTENT_RATIO = 0.9
TRIGGER_SAMPLE_STEP_M = 1.0
BOUNDARY_STEP_M = 0.5
BOUNDARY_HALF_LANE_RATIO = 0.4
MAX_BOUNDARY_STEPS = 400


def _lane_identity(waypoint):
    return (
        int(waypoint.road_id),
        int(getattr(waypoint, "section_id", 0)),
        int(waypoint.lane_id),
    )


def _waypoint_identity(waypoint):
    return _lane_identity(waypoint) + (
        round(float(getattr(waypoint, "s", 0.0)), 2),
    )


def _is_intersection(waypoint):
    return bool(
        getattr(
            waypoint,
            "is_intersection",
            getattr(waypoint, "is_junction", False),
        )
    )


def _location_like(reference, x, y, z):
    try:
        return type(reference)(x=x, y=y, z=z)
    except TypeError:
        value = type("LocationValue", (), {})()
        value.x = x
        value.y = y
        value.z = z
        return value


def sample_trigger_lane_waypoints(light, world_map):
    """Mirror RunningRedLightTest trigger sampling and lane deduplication."""
    transform = light.get_transform()
    trigger = light.trigger_volume
    center = transform.transform(trigger.location)
    yaw = math.radians(float(transform.rotation.yaw))
    current = -TRIGGER_EXTENT_RATIO * float(trigger.extent.x)
    maximum = TRIGGER_EXTENT_RATIO * float(trigger.extent.x)
    waypoints = []

    while current < maximum:
        query = _location_like(
            center,
            center.x + current * math.cos(yaw),
            center.y + current * math.sin(yaw),
            center.z,
        )
        waypoint = world_map.get_waypoint(query)
        if waypoint is not None and (
            not waypoints
            or _lane_identity(waypoints[-1]) != _lane_identity(waypoint)
        ):
            waypoints.append(waypoint)
        current += TRIGGER_SAMPLE_STEP_M
    return waypoints


def advance_to_infraction_boundary(start):
    """Advance as the evaluator does while surfacing ambiguous topology."""
    waypoint = start
    visited = set()
    branch_seen = False
    steps = 0

    while not _is_intersection(waypoint):
        identity = _waypoint_identity(waypoint)
        if identity in visited:
            return {
                "status": "unknown",
                "unknown_reason": "waypoint_loop",
                "waypoint": waypoint,
                "steps": steps,
            }
        visited.add(identity)

        if steps >= MAX_BOUNDARY_STEPS:
            return {
                "status": "unknown",
                "unknown_reason": "intersection_not_found",
                "waypoint": waypoint,
                "steps": steps,
            }

        successors = list(waypoint.next(BOUNDARY_STEP_M) or [])
        if not successors:
            return {
                "status": "unknown",
                "unknown_reason": "intersection_not_found",
                "waypoint": waypoint,
                "steps": steps,
            }
        branch_seen = branch_seen or len(successors) > 1
        following = successors[0]
        if _is_intersection(following):
            break
        waypoint = following
        steps += 1

    return {
        "status": "unknown" if branch_seen else "valid",
        "unknown_reason": "waypoint_branch" if branch_seen else None,
        "waypoint": waypoint,
        "steps": steps,
    }


def boundary_from_waypoint(waypoint):
    """Create the transverse line used by RunningRedLightTest."""
    center = waypoint.transform.location
    yaw = math.radians(float(waypoint.transform.rotation.yaw))
    offset = BOUNDARY_HALF_LANE_RATIO * float(waypoint.lane_width)
    right_x = -math.sin(yaw)
    right_y = math.cos(yaw)

    def endpoint(sign):
        return {
            "x": float(center.x + sign * offset * right_x),
            "y": float(center.y + sign * offset * right_y),
            "z": float(center.z),
        }

    return {
        "geometry_source": GEOMETRY_SOURCE,
        "road_id": int(waypoint.road_id),
        "section_id": int(getattr(waypoint, "section_id", 0)),
        "lane_id": int(waypoint.lane_id),
        "s": float(getattr(waypoint, "s", 0.0)),
        "lane_width": float(waypoint.lane_width),
        "center": {
            "x": float(center.x),
            "y": float(center.y),
            "z": float(center.z),
        },
        "left_endpoint": endpoint(1.0),
        "right_endpoint": endpoint(-1.0),
    }
