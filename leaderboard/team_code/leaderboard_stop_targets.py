"""Build traffic-light stop geometry that matches Leaderboard evaluation."""

import math


GEOMETRY_SOURCE = "scenario_runner_running_red_light_test_v1"
TRIGGER_SOURCE = "carla_traffic_light_trigger_waypoint"
TRIGGER_EXTENT_RATIO = 0.9
TRIGGER_SAMPLE_STEP_M = 1.0
BOUNDARY_STEP_M = 0.5
BOUNDARY_HALF_LANE_RATIO = 0.4
MAX_BOUNDARY_STEPS = 400
ROUTE_MIN_DISTANCE_M = -10.0
ROUTE_MAX_DISTANCE_M = 80.0
ROUTE_MATCH_DISTANCE_M = 4.0
CORRIDOR_EXTENSION_M = 3.0
CORRIDOR_STEP_M = 0.5
SAFETY_MARGIN_M = 1.0


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


def _location_distance(left, right):
    return math.sqrt(
        (float(left.x) - float(right.x)) ** 2
        + (float(left.y) - float(right.y)) ** 2
        + (float(left.z) - float(right.z)) ** 2
    )


def _route_index(route_waypoints):
    route = list(route_waypoints or [])
    if not route:
        return [], []
    cumulative = [0.0]
    for previous, current in zip(route, route[1:]):
        cumulative.append(
            cumulative[-1]
            + _location_distance(
                previous.transform.location,
                current.transform.location,
            )
        )
    return route, cumulative


def _nearest_route_index(route, location, lane=None):
    candidates = [
        (
            index,
            _location_distance(waypoint.transform.location, location),
        )
        for index, waypoint in enumerate(route)
        if lane is None or _lane_identity(waypoint) == lane
    ]
    if not candidates:
        return None
    index, distance = min(candidates, key=lambda item: item[1])
    return index if distance <= ROUTE_MATCH_DISTANCE_M else None


def _rotation_dict(rotation):
    return {
        "pitch": float(getattr(rotation, "pitch", 0.0)),
        "yaw": float(getattr(rotation, "yaw", 0.0)),
        "roll": float(getattr(rotation, "roll", 0.0)),
    }


def _location_dict(location):
    return {
        "x": float(location.x),
        "y": float(location.y),
        "z": float(location.z),
    }


def _interpolate_route_sample(route, cumulative, distance_m):
    if not route:
        raise ValueError("route must not be empty")
    distance_m = min(max(float(distance_m), cumulative[0]), cumulative[-1])
    for index in range(1, len(route)):
        if cumulative[index] + 1e-9 < distance_m:
            continue
        previous = route[index - 1]
        current = route[index]
        span = cumulative[index] - cumulative[index - 1]
        ratio = (
            0.0
            if span <= 1e-9
            else (distance_m - cumulative[index - 1]) / span
        )
        location = {
            axis: float(
                getattr(previous.transform.location, axis)
                + ratio
                * (
                    getattr(current.transform.location, axis)
                    - getattr(previous.transform.location, axis)
                )
            )
            for axis in ("x", "y", "z")
        }
        return {
            "location": location,
            "rotation": _rotation_dict(previous.transform.rotation),
            "lane_width": float(previous.lane_width),
            "route_distance_m": float(distance_m),
        }
    last = route[-1]
    return {
        "location": _location_dict(last.transform.location),
        "rotation": _rotation_dict(last.transform.rotation),
        "lane_width": float(last.lane_width),
        "route_distance_m": float(cumulative[-1]),
    }


def _sample_route_interval(route, cumulative, start_m, end_m):
    start_m = max(cumulative[0], float(start_m))
    end_m = min(cumulative[-1], float(end_m))
    if end_m < start_m:
        return []
    distances = []
    current = start_m
    while current < end_m - 1e-9:
        distances.append(current)
        current += CORRIDOR_STEP_M
    distances.append(end_m)
    return [
        _interpolate_route_sample(route, cumulative, distance)
        for distance in distances
    ]


def _heading_dot(left_yaw, right_yaw):
    difference = math.radians(float(left_yaw) - float(right_yaw))
    return math.cos(difference)


def _normalize_angle(angle):
    return (float(angle) + 180.0) % 360.0 - 180.0


def _state_name(state):
    name = getattr(state, "name", None)
    return str(name) if name else str(state).split(".")[-1]


def _trigger_waypoint_record(waypoint):
    return {
        "geometry_source": TRIGGER_SOURCE,
        "road_id": int(waypoint.road_id),
        "section_id": int(getattr(waypoint, "section_id", 0)),
        "lane_id": int(waypoint.lane_id),
        "s": float(getattr(waypoint, "s", 0.0)),
        "lane_width": float(waypoint.lane_width),
        "center": _location_dict(waypoint.transform.location),
        "rotation": _rotation_dict(waypoint.transform.rotation),
    }


def _target_key(map_name, boundary):
    return "{}:{}:{}:{}:{:.1f}".format(
        str(map_name).split("/")[-1],
        boundary["road_id"],
        boundary["section_id"],
        boundary["lane_id"],
        boundary["s"],
    )


def build_stop_targets(
    lights,
    world_map,
    route_waypoints,
    ego_transform,
    ego_bounding_box,
    map_name,
):
    """Build route-owned stop targets without using learned junction output."""
    route, cumulative = _route_index(route_waypoints)
    if not route:
        return []
    ego_index = _nearest_route_index(route, ego_transform.location)
    if ego_index is None:
        return []

    direction_mismatch = _heading_dot(
        ego_transform.rotation.yaw,
        route[ego_index].transform.rotation.yaw,
    ) <= 0.0
    vehicle_front_offset = float(ego_bounding_box.location.x) + float(
        ego_bounding_box.extent.x
    )
    grouped = {}

    for light in sorted(lights, key=lambda item: int(item.id)):
        trigger_waypoints = sample_trigger_lane_waypoints(light, world_map)
        for trigger_waypoint in trigger_waypoints:
            traversal = advance_to_infraction_boundary(trigger_waypoint)
            boundary_waypoint = traversal["waypoint"]
            lane = _lane_identity(boundary_waypoint)
            trigger_index = _nearest_route_index(
                route,
                trigger_waypoint.transform.location,
                _lane_identity(trigger_waypoint),
            )
            boundary_index = _nearest_route_index(
                route,
                boundary_waypoint.transform.location,
                lane,
            )
            if trigger_index is None or boundary_index is None:
                continue

            boundary_route_distance = cumulative[boundary_index]
            signed_distance = boundary_route_distance - cumulative[ego_index]
            if not ROUTE_MIN_DISTANCE_M <= signed_distance <= ROUTE_MAX_DISTANCE_M:
                continue

            route_heading_mismatch = _heading_dot(
                boundary_waypoint.transform.rotation.yaw,
                route[boundary_index].transform.rotation.yaw,
            ) <= 0.0
            status = traversal["status"]
            unknown_reason = traversal["unknown_reason"]
            if direction_mismatch or route_heading_mismatch:
                status = "unknown"
                unknown_reason = "direction_mismatch"

            boundary = boundary_from_waypoint(boundary_waypoint)
            stop_route_distance = boundary_route_distance - (
                vehicle_front_offset + SAFETY_MARGIN_M
            )
            stop_pose = _interpolate_route_sample(
                route,
                cumulative,
                stop_route_distance,
            )
            corridor_start = min(
                cumulative[trigger_index],
                boundary_route_distance,
            ) - CORRIDOR_EXTENSION_M
            corridor_end = max(
                cumulative[trigger_index],
                boundary_route_distance,
            ) + CORRIDOR_EXTENSION_M
            centerline = _sample_route_interval(
                route,
                cumulative,
                corridor_start,
                corridor_end,
            )
            target_id = _target_key(map_name, boundary)
            owner_id = int(light.id)
            target = grouped.get(target_id)
            if target is None:
                boundary_location = boundary_waypoint.transform.location
                target = {
                    "target_id": target_id,
                    "map_name": str(map_name).split("/")[-1],
                    "owner_traffic_light_actor_ids": [],
                    "state_by_actor_id": {},
                    "primary_for_ego": False,
                    "status": status,
                    "unknown_reason": unknown_reason,
                    "route_lane": {
                        "road_id": lane[0],
                        "section_id": lane[1],
                        "lane_id": lane[2],
                    },
                    "geometry_source": GEOMETRY_SOURCE,
                    "trigger_stop_waypoint": _trigger_waypoint_record(
                        trigger_waypoint
                    ),
                    "leaderboard_infraction_boundary": boundary,
                    "recommended_ego_stop_pose": stop_pose,
                    "vehicle_front_offset_m": vehicle_front_offset,
                    "safety_margin_m": SAFETY_MARGIN_M,
                    "stop_evidence_corridor": {
                        "sample_step_m": CORRIDOR_STEP_M,
                        "extension_m": CORRIDOR_EXTENSION_M,
                        "centerline": centerline,
                    },
                    "signed_route_distance_m": float(signed_distance),
                    "euclidean_distance_m": _location_distance(
                        ego_transform.location,
                        boundary_location,
                    ),
                    "relative_heading_degrees": _normalize_angle(
                        boundary_waypoint.transform.rotation.yaw
                        - ego_transform.rotation.yaw
                    ),
                    "ego_before_boundary": signed_distance >= 0.0,
                    "trigger_to_boundary_route_distance_m": float(
                        boundary_route_distance - cumulative[trigger_index]
                    ),
                    "search_steps": int(traversal["steps"]),
                }
                grouped[target_id] = target
            elif status == "unknown" and target["status"] == "valid":
                target["status"] = "unknown"
                target["unknown_reason"] = unknown_reason

            target["owner_traffic_light_actor_ids"].append(owner_id)
            target["state_by_actor_id"][str(owner_id)] = _state_name(light.state)

    targets = sorted(
        grouped.values(),
        key=lambda item: (
            item["signed_route_distance_m"],
            item["target_id"],
        ),
    )
    for target in targets:
        target["owner_traffic_light_actor_ids"] = sorted(
            set(target["owner_traffic_light_actor_ids"])
        )
    primary = [
        target
        for target in targets
        if target["status"] == "valid"
        and target["signed_route_distance_m"] >= 0.0
    ]
    if primary:
        min(primary, key=lambda item: item["signed_route_distance_m"])[
            "primary_for_ego"
        ] = True
    return targets
