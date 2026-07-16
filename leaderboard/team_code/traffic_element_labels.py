"""Build auditable traffic-element labels from CARLA actors and waypoints."""

import fnmatch
import math

from leaderboard_stop_targets import build_stop_targets


SCHEMA_VERSION = 2
DEFAULT_MAX_DISTANCE = 80.0


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


def _traffic_light_label(light, ego_transform, ego_lane, active_light_id, errors):
    transform = light.get_transform()
    location = light.get_location()
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
    is_active = active_light_id == int(light.id)
    controls_ego_lane = is_active or (
        ego_lane is not None and ego_lane in affected_lanes
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
    }
    return label


def collect_traffic_element_labels(
    hero,
    world,
    frame_id,
    route_waypoints,
    max_distance=DEFAULT_MAX_DISTANCE,
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
    light_actors = []
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
            light_actors.append(light)
        except Exception as exc:
            errors.append(
                {
                    "actor_id": int(light.id),
                    "field": "traffic_light",
                    "error": str(exc),
                }
            )

    traffic_lights.sort(key=lambda item: item["actor_id"])
    map_name = str(world_map.name).split("/")[-1]
    stop_targets = build_stop_targets(
        light_actors,
        world_map,
        route_waypoints,
        ego_transform,
        hero.bounding_box,
        map_name,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "frame_id": str(frame_id),
        "map_name": map_name,
        "ego": {
            "actor_id": int(hero.id),
            "location": _location_dict(ego_location),
            "rotation": _rotation_dict(ego_transform.rotation),
            "lane": _waypoint_metadata(ego_waypoint),
        },
        "active_traffic_light_id": active_light_id,
        "traffic_lights": traffic_lights,
        "stop_targets": stop_targets,
        "errors": errors,
    }


def legacy_affordances_from_labels(labels):
    active_lights = [
        item for item in labels["traffic_lights"] if item["is_active_for_ego"]
    ]
    return {
        "traffic_light": active_lights[0]["state"] if active_lights else None,
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
    if not isinstance(record.get("frame_id"), str):
        errors.append("frame_id must be a string")
    if not isinstance(record.get("map_name"), str):
        errors.append("map_name must be a string")
    if not isinstance(record.get("traffic_lights"), list):
        errors.append("traffic_lights must be a list")
    if not isinstance(record.get("stop_targets"), list):
        errors.append("stop_targets must be a list")
    if "stop_signs" in record:
        errors.append("stop_signs is forbidden")
    if not isinstance(record.get("errors"), list):
        errors.append("errors must be a list")
    if not isinstance(record.get("ego"), dict):
        errors.append("ego must be an object")
    return errors
