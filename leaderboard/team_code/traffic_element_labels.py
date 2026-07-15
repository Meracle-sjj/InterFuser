"""Build auditable traffic-element labels from CARLA actors and waypoints."""

import fnmatch
import math
from collections import deque


SCHEMA_VERSION = 1
DEFAULT_MAX_DISTANCE = 80.0
DEFAULT_ROUTE_HORIZON = 30.0
DEFAULT_ROUTE_STEP = 1.0


def _location_dict(location):
    return {
        "x": float(location.x),
        "y": float(location.y),
        "z": float(location.z),
    }


def _rotation_dict(rotation):
    return {
        "pitch": float(getattr(rotation, "pitch", 0.0)),
        "yaw": float(getattr(rotation, "yaw", 0.0)),
        "roll": float(getattr(rotation, "roll", 0.0)),
    }


def _normalize_angle_degrees(angle):
    return (float(angle) + 180.0) % 360.0 - 180.0


def _state_name(state):
    name = getattr(state, "name", None)
    if name:
        return str(name)
    return str(state).split(".")[-1]


def world_to_ego(location, ego_transform):
    """Return CARLA world coordinates in ego forward/right/up axes."""
    dx = float(location.x - ego_transform.location.x)
    dy = float(location.y - ego_transform.location.y)
    dz = float(location.z - ego_transform.location.z)
    yaw = math.radians(float(ego_transform.rotation.yaw))
    return {
        "forward": dx * math.cos(yaw) + dy * math.sin(yaw),
        "right": -dx * math.sin(yaw) + dy * math.cos(yaw),
        "up": dz,
    }


def _lane_identity(waypoint):
    if waypoint is None:
        return None
    return (
        int(waypoint.road_id),
        int(getattr(waypoint, "section_id", 0)),
        int(waypoint.lane_id),
    )


def _waypoint_identity(waypoint):
    lane = _lane_identity(waypoint)
    return lane + (round(float(getattr(waypoint, "s", 0.0)), 2),)


def _waypoint_metadata(waypoint):
    if waypoint is None:
        return None
    return {
        "road_id": int(waypoint.road_id),
        "section_id": int(getattr(waypoint, "section_id", 0)),
        "lane_id": int(waypoint.lane_id),
        "s": float(getattr(waypoint, "s", 0.0)),
        "lane_width": float(getattr(waypoint, "lane_width", 0.0)),
    }


def _stop_line_from_waypoint(waypoint, ego_transform, geometry_source):
    center = waypoint.transform.location
    lane_width = float(getattr(waypoint, "lane_width", 0.0))
    half_width = lane_width / 2.0
    yaw = math.radians(float(waypoint.transform.rotation.yaw))
    right_x = -math.sin(yaw)
    right_y = math.cos(yaw)

    left_endpoint = {
        "x": float(center.x - right_x * half_width),
        "y": float(center.y - right_y * half_width),
        "z": float(center.z),
    }
    right_endpoint = {
        "x": float(center.x + right_x * half_width),
        "y": float(center.y + right_y * half_width),
        "z": float(center.z),
    }
    relative_center = world_to_ego(center, ego_transform)

    result = _waypoint_metadata(waypoint)
    result.update(
        {
            "geometry_source": geometry_source,
            "is_exact_carla_stop_position": geometry_source == "carla_stop_waypoint",
            "center": _location_dict(center),
            "left_endpoint": left_endpoint,
            "right_endpoint": right_endpoint,
            "relative_center": relative_center,
            "longitudinal_distance": relative_center["forward"],
            "lateral_offset": relative_center["right"],
            "ego_before_line": relative_center["forward"] >= 0.0,
        }
    )
    return result


def _actor_filter(actors, pattern):
    if hasattr(actors, "filter"):
        return list(actors.filter(pattern))
    return [
        actor
        for actor in actors
        if fnmatch.fnmatch(str(getattr(actor, "type_id", "")), pattern)
    ]


def _trigger_volume_label(actor, ego_transform):
    actor_transform = actor.get_transform()
    trigger = actor.trigger_volume
    center = actor_transform.transform(trigger.location)
    trigger_rotation = getattr(trigger, "rotation", None)
    yaw = float(actor_transform.rotation.yaw) + float(
        getattr(trigger_rotation, "yaw", 0.0)
    )
    return {
        "center": _location_dict(center),
        "relative_center": world_to_ego(center, ego_transform),
        "extent": {
            "x": float(trigger.extent.x),
            "y": float(trigger.extent.y),
            "z": float(trigger.extent.z),
        },
        "rotation": {
            "pitch": float(actor_transform.rotation.pitch)
            + float(getattr(trigger_rotation, "pitch", 0.0)),
            "yaw": yaw,
            "roll": float(actor_transform.rotation.roll)
            + float(getattr(trigger_rotation, "roll", 0.0)),
        },
    }


def _point_in_trigger_volume(location, trigger):
    dx = float(location.x) - trigger["center"]["x"]
    dy = float(location.y) - trigger["center"]["y"]
    yaw = math.radians(trigger["rotation"]["yaw"])
    local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
    local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)
    return (
        abs(local_x) <= trigger["extent"]["x"]
        and abs(local_y) <= trigger["extent"]["y"]
    )


def _route_entry_stop_line(
    start_waypoint,
    trigger,
    ego_transform,
    route_horizon,
    route_step,
):
    if start_waypoint is None:
        return None

    queue = deque([(start_waypoint, 0.0, None)])
    visited = set()
    while queue:
        waypoint, travelled, previous = queue.popleft()
        identity = _waypoint_identity(waypoint)
        if identity in visited:
            continue
        visited.add(identity)

        if _point_in_trigger_volume(waypoint.transform.location, trigger):
            stop_waypoint = previous or waypoint
            return _stop_line_from_waypoint(
                stop_waypoint,
                ego_transform,
                "trigger_volume_route_entry_approximation",
            )

        if travelled >= route_horizon:
            continue
        next_waypoints = waypoint.next(route_step)
        for next_waypoint in next_waypoints or []:
            queue.append((next_waypoint, travelled + route_step, waypoint))

    return None


def _traffic_light_label(light, ego_transform, ego_lane, active_light_id, errors):
    transform = light.get_transform()
    location = light.get_location()
    try:
        stop_waypoints = list(light.get_stop_waypoints())
    except Exception as exc:
        stop_waypoints = []
        errors.append(
            {
                "actor_id": int(light.id),
                "field": "stop_waypoints",
                "error": str(exc),
            }
        )
    try:
        affected_waypoints = list(light.get_affected_lane_waypoints())
    except Exception as exc:
        affected_waypoints = []
        errors.append(
            {
                "actor_id": int(light.id),
                "field": "affected_lane_waypoints",
                "error": str(exc),
            }
        )

    affected_lanes = {
        lane for lane in (_lane_identity(item) for item in affected_waypoints) if lane
    }
    stop_lanes = {
        lane for lane in (_lane_identity(item) for item in stop_waypoints) if lane
    }
    is_active = active_light_id == int(light.id)
    controls_ego_lane = is_active or (
        ego_lane is not None and ego_lane in affected_lanes.union(stop_lanes)
    )
    relative = world_to_ego(location, ego_transform)

    label = {
        "actor_id": int(light.id),
        "type_id": str(light.type_id),
        "state": _state_name(light.state),
        "location": _location_dict(location),
        "rotation": _rotation_dict(transform.rotation),
        "relative_position": relative,
        "relative_heading": _normalize_angle_degrees(
            float(transform.rotation.yaw) - float(ego_transform.rotation.yaw)
        ),
        "distance": math.sqrt(
            relative["forward"] ** 2
            + relative["right"] ** 2
            + relative["up"] ** 2
        ),
        "trigger_volume": _trigger_volume_label(light, ego_transform),
        "is_active_for_ego": is_active,
        "controls_ego_lane": controls_ego_lane,
        "relevant_to_ego": controls_ego_lane,
        "affected_lanes": [
            _waypoint_metadata(item) for item in affected_waypoints
        ],
        "stop_lines": [
            _stop_line_from_waypoint(item, ego_transform, "carla_stop_waypoint")
            for item in stop_waypoints
        ],
    }
    return label


def _stop_sign_label(
    stop_sign,
    ego_transform,
    start_waypoint,
    route_horizon,
    route_step,
):
    transform = stop_sign.get_transform()
    location = stop_sign.get_location()
    relative = world_to_ego(location, ego_transform)
    trigger = _trigger_volume_label(stop_sign, ego_transform)
    stop_line = _route_entry_stop_line(
        start_waypoint,
        trigger,
        ego_transform,
        route_horizon,
        route_step,
    )
    return {
        "actor_id": int(stop_sign.id),
        "type_id": str(stop_sign.type_id),
        "location": _location_dict(location),
        "rotation": _rotation_dict(transform.rotation),
        "relative_position": relative,
        "relative_heading": _normalize_angle_degrees(
            float(transform.rotation.yaw) - float(ego_transform.rotation.yaw)
        ),
        "distance": math.sqrt(
            relative["forward"] ** 2
            + relative["right"] ** 2
            + relative["up"] ** 2
        ),
        "trigger_volume": trigger,
        "affects_ego_route": stop_line is not None,
        "stop_lines": [stop_line] if stop_line is not None else [],
    }


def collect_traffic_element_labels(
    hero,
    world,
    max_distance=DEFAULT_MAX_DISTANCE,
    route_horizon=DEFAULT_ROUTE_HORIZON,
    route_step=DEFAULT_ROUTE_STEP,
):
    """Collect a JSON-serializable traffic-element record for one frame."""
    errors = []
    ego_transform = hero.get_transform()
    ego_location = hero.get_location()
    world_map = world.get_map()
    try:
        ego_waypoint = world_map.get_waypoint(ego_location)
    except Exception as exc:
        ego_waypoint = None
        errors.append({"field": "ego_waypoint", "error": str(exc)})
    else:
        if ego_waypoint is None:
            errors.append(
                {
                    "field": "ego_waypoint",
                    "error": "Map.get_waypoint returned None",
                }
            )

    try:
        active_light = hero.get_traffic_light()
    except Exception as exc:
        active_light = None
        errors.append({"field": "active_traffic_light", "error": str(exc)})
    active_light_id = int(active_light.id) if active_light is not None else None

    actors = world.get_actors()
    traffic_lights = []
    for light in _actor_filter(actors, "traffic.traffic_light*"):
        is_active_light = active_light_id == int(light.id)
        if (
            ego_location.distance(light.get_location()) > max_distance
            and not is_active_light
        ):
            continue
        try:
            traffic_lights.append(
                _traffic_light_label(
                    light,
                    ego_transform,
                    _lane_identity(ego_waypoint),
                    active_light_id,
                    errors,
                )
            )
        except Exception as exc:
            errors.append(
                {
                    "actor_id": int(light.id),
                    "field": "traffic_light",
                    "error": str(exc),
                }
            )

    stop_signs = []
    stop_sign_actors = (
        _actor_filter(actors, "traffic.stop*") if ego_waypoint is not None else []
    )
    for stop_sign in stop_sign_actors:
        if ego_location.distance(stop_sign.get_location()) > max_distance:
            continue
        try:
            stop_signs.append(
                _stop_sign_label(
                    stop_sign,
                    ego_transform,
                    ego_waypoint,
                    route_horizon,
                    route_step,
                )
            )
        except Exception as exc:
            errors.append(
                {
                    "actor_id": int(stop_sign.id),
                    "field": "stop_sign",
                    "error": str(exc),
                }
            )

    traffic_lights.sort(key=lambda item: item["actor_id"])
    stop_signs.sort(key=lambda item: item["actor_id"])
    return {
        "schema_version": SCHEMA_VERSION,
        "ego": {
            "actor_id": int(hero.id),
            "location": _location_dict(ego_location),
            "rotation": _rotation_dict(ego_transform.rotation),
            "lane": _waypoint_metadata(ego_waypoint),
        },
        "active_traffic_light_id": active_light_id,
        "traffic_lights": traffic_lights,
        "stop_signs": stop_signs,
        "errors": errors,
    }


def legacy_affordances_from_labels(labels):
    active_lights = [
        item for item in labels["traffic_lights"] if item["is_active_for_ego"]
    ]
    return {
        "traffic_light": active_lights[0]["state"] if active_lights else None,
        "stop_sign": any(
            item["affects_ego_route"] for item in labels["stop_signs"]
        ),
    }


def merge_legacy_affordances(base, labels):
    merged = dict(base)
    merged.update(legacy_affordances_from_labels(labels))
    return merged


def validate_traffic_element_record(record):
    if not isinstance(record, dict):
        return ["record must be an object"]

    errors = []
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    if not isinstance(record.get("traffic_lights"), list):
        errors.append("traffic_lights must be a list")
    if not isinstance(record.get("stop_signs"), list):
        errors.append("stop_signs must be a list")
    if not isinstance(record.get("errors"), list):
        errors.append("errors must be a list")
    if not isinstance(record.get("ego"), dict):
        errors.append("ego must be an object")
    return errors
