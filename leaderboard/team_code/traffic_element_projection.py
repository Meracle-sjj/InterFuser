"""Project CARLA traffic-element geometry into camera images."""

import math

import numpy as np


IMAGE_SCHEMA_VERSION = 2
ASSOCIATION = {
    "roi_expand_pixels": 6,
    "minimum_semantic_pixels": 3,
    "traffic_light": {
        "semantic_tag": 7,
        "depth_tolerance_m": 4.0,
    },
    "stop_sign": {
        "semantic_tag": 8,
        "depth_tolerance_m": 6.0,
    },
}


def camera_intrinsics(width, height, fov_degrees):
    """Return a pinhole intrinsic matrix for a CARLA camera."""
    focal = float(width) / (
        2.0 * math.tan(math.radians(float(fov_degrees)) / 2.0)
    )
    return np.array(
        [
            [focal, 0.0, float(width) / 2.0],
            [0.0, focal, float(height) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _rotation_matrix(pitch, yaw, roll):
    cy = math.cos(math.radians(float(yaw)))
    sy = math.sin(math.radians(float(yaw)))
    cr = math.cos(math.radians(float(roll)))
    sr = math.sin(math.radians(float(roll)))
    cp = math.cos(math.radians(float(pitch)))
    sp = math.sin(math.radians(float(pitch)))
    return np.array(
        [
            [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
            [sp, -cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def transform_matrix(transform):
    """Return the CARLA local-to-world homogeneous transform matrix."""
    rotation = transform.rotation
    location = transform.location
    matrix = np.identity(4, dtype=np.float64)
    matrix[:3, 3] = [
        float(location.x),
        float(location.y),
        float(location.z),
    ]
    matrix[:3, :3] = _rotation_matrix(
        rotation.pitch,
        rotation.yaw,
        rotation.roll,
    )
    return matrix


def world_to_camera(world_points, camera_transform):
    """Transform Nx3 CARLA world points into camera forward/right/up axes."""
    points = np.asarray(world_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("world_points must have shape Nx3")
    homogeneous = np.column_stack([points, np.ones(len(points))])
    world_to_sensor = np.linalg.inv(transform_matrix(camera_transform))
    return (world_to_sensor @ homogeneous.T).T[:, :3]


def project_camera_points(camera_points, intrinsic):
    """Project camera forward/right/up points into image pixel coordinates."""
    points = np.asarray(camera_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("camera_points must have shape Nx3")

    in_front = points[:, 0] > 1e-6
    image_axes = np.column_stack([points[:, 1], -points[:, 2], points[:, 0]])
    pixels_h = (np.asarray(intrinsic, dtype=np.float64) @ image_axes.T).T
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    pixels[in_front] = pixels_h[in_front, :2] / pixels_h[in_front, 2:3]
    return pixels, in_front


def decode_carla_depth(raw_bgr):
    """Decode CARLA's 24-bit depth representation into meters."""
    data = np.asarray(raw_bgr, dtype=np.float64)
    if data.ndim != 3 or data.shape[2] < 3:
        raise ValueError("raw_bgr must have shape HxWx3 or HxWx4")
    normalized = np.dot(
        data[..., :3],
        np.array([65536.0, 256.0, 1.0], dtype=np.float64),
    )
    return 1000.0 * normalized / (256.0**3 - 1.0)


def projected_roi(
    world_vertices,
    camera_transform,
    intrinsic,
    width,
    height,
    expand=6,
):
    """Project world vertices into a clipped, right/bottom-exclusive ROI."""
    camera_points = world_to_camera(world_vertices, camera_transform)
    pixels, in_front = project_camera_points(camera_points, intrinsic)
    visible = pixels[in_front]
    visible = visible[np.isfinite(visible).all(axis=1)]
    if not len(visible):
        return None

    x1 = max(0, int(math.floor(np.min(visible[:, 0]))) - int(expand))
    y1 = max(0, int(math.floor(np.min(visible[:, 1]))) - int(expand))
    x2 = min(
        int(width),
        int(math.ceil(np.max(visible[:, 0]))) + int(expand) + 1,
    )
    y2 = min(
        int(height),
        int(math.ceil(np.max(visible[:, 1]))) + int(expand) + 1,
    )
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def associate_semantic_box(
    roi,
    semantic,
    depth_m,
    actor_distance_m,
    semantic_tag,
    depth_tolerance_m,
    min_pixels,
):
    """Derive a visible box from semantic pixels near an actor's depth."""
    empty = {
        "visibility": "not_visible",
        "bbox_xyxy": None,
        "semantic_pixel_count": 0,
        "median_depth_residual_m": None,
    }
    if roi is None:
        return empty

    semantic = np.asarray(semantic)
    depth_m = np.asarray(depth_m, dtype=np.float64)
    if semantic.shape != depth_m.shape:
        raise ValueError("semantic and depth_m must have the same shape")

    x1, y1, x2, y2 = (int(value) for value in roi)
    height, width = semantic.shape
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("roi must be clipped inside the image")

    semantic_roi = semantic[y1:y2, x1:x2]
    expected_depths = np.asarray(actor_distance_m, dtype=np.float64)
    if expected_depths.ndim == 0:
        expected_depths = expected_depths.reshape(1)
    if (
        expected_depths.ndim != 1
        or not len(expected_depths)
        or not np.isfinite(expected_depths).all()
    ):
        raise ValueError("actor_distance_m must contain finite distances")
    depth_roi = depth_m[y1:y2, x1:x2]
    residual = np.min(
        np.abs(depth_roi[..., np.newaxis] - expected_depths),
        axis=-1,
    )
    mask = (
        (semantic_roi == int(semantic_tag))
        & np.isfinite(residual)
        & (residual <= float(depth_tolerance_m))
    )
    ys, xs = np.where(mask)
    pixel_count = int(len(xs))
    if pixel_count < int(min_pixels):
        result = dict(empty)
        result["semantic_pixel_count"] = pixel_count
        return result

    return {
        "visibility": "visible",
        "bbox_xyxy": [
            int(x1 + xs.min()),
            int(y1 + ys.min()),
            int(x1 + xs.max() + 1),
            int(y1 + ys.max() + 1),
        ],
        "semantic_pixel_count": pixel_count,
        "median_depth_residual_m": float(np.median(residual[mask])),
    }


def clip_image_segment(start, end, width, height):
    """Clip a 2D segment to image bounds using Liang-Barsky."""
    x1, y1 = (float(value) for value in start)
    x2, y2 = (float(value) for value in end)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise ValueError("segment coordinates must be finite")
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")

    dx = x2 - x1
    dy = y2 - y1
    p = (-dx, dx, -dy, dy)
    q = (x1, float(width - 1) - x1, y1, float(height - 1) - y1)
    lower = 0.0
    upper = 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0.0:
                return None
            continue
        ratio = qi / pi
        if pi < 0.0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower > upper:
            return None

    clipped = [
        [x1 + lower * dx, y1 + lower * dy],
        [x1 + upper * dx, y1 + upper * dy],
    ]
    for point in clipped:
        point[0] = min(max(point[0], 0.0), float(width - 1))
        point[1] = min(max(point[1], 0.0), float(height - 1))
    return clipped


def _dict_location(location):
    return np.array(
        [float(location["x"]), float(location["y"]), float(location["z"])],
        dtype=np.float64,
    )


def _world_bounding_box_vertices(bounding_box):
    extent = bounding_box.extent
    local = np.array(
        [
            [x, y, z]
            for x in (-float(extent.x), float(extent.x))
            for y in (-float(extent.y), float(extent.y))
            for z in (-float(extent.z), float(extent.z))
        ],
        dtype=np.float64,
    )
    rotation = bounding_box.rotation
    rotated = (
        _rotation_matrix(rotation.pitch, rotation.yaw, rotation.roll)
        @ local.T
    ).T
    center = np.array(
        [
            float(bounding_box.location.x),
            float(bounding_box.location.y),
            float(bounding_box.location.z),
        ],
        dtype=np.float64,
    )
    return rotated + center, center


def _actor_world_vertices(actor, element, element_type):
    if element_type == "traffic_light" and hasattr(actor, "get_light_boxes"):
        light_boxes = list(actor.get_light_boxes())
        if not light_boxes:
            raise ValueError("traffic light has no light boxes")
        geometry = [_world_bounding_box_vertices(box) for box in light_boxes]
        return (
            np.concatenate([vertices for vertices, _center in geometry]),
            "traffic_light_boxes",
            np.array([center for _vertices, center in geometry]),
        )

    bounding_box = getattr(actor, "bounding_box", None)
    if bounding_box is not None and hasattr(bounding_box, "get_world_vertices"):
        vertices = bounding_box.get_world_vertices(actor.get_transform())
        if vertices:
            world_vertices = np.array(
                [[item.x, item.y, item.z] for item in vertices],
                dtype=np.float64,
            )
            return (
                world_vertices,
                "actor_bounding_box",
                np.mean(world_vertices, axis=0, keepdims=True),
            )

    trigger = element.get("trigger_volume")
    if not isinstance(trigger, dict):
        raise ValueError("actor has no bounding box or trigger volume")
    center = _dict_location(trigger["center"])
    extent = trigger["extent"]
    rotation = trigger["rotation"]
    local = np.array(
        [
            [x, y, z]
            for x in (-float(extent["x"]), float(extent["x"]))
            for y in (-float(extent["y"]), float(extent["y"]))
            for z in (-float(extent["z"]), float(extent["z"]))
        ],
        dtype=np.float64,
    )
    rotated = (
        _rotation_matrix(
            rotation.get("pitch", 0.0),
            rotation.get("yaw", 0.0),
            rotation.get("roll", 0.0),
        )
        @ local.T
    ).T
    return rotated + center, "trigger_volume", center.reshape(1, 3)


def _unknown_element(element, element_type, association_source):
    result = {
        "actor_id": int(element["actor_id"]),
        "element_type": element_type,
        "visibility": "unknown",
        "bbox_xyxy": None,
        "geometry_roi_xyxy": None,
        "semantic_pixel_count": 0,
        "median_depth_residual_m": None,
        "association_source": association_source,
        "geometry_source": None,
    }
    if element_type == "traffic_light":
        result.update(
            {
                "state": element["state"],
                "is_active_for_ego": bool(element["is_active_for_ego"]),
                "controls_ego_lane": bool(element["controls_ego_lane"]),
                "relevant_to_ego": bool(element["relevant_to_ego"]),
            }
        )
    else:
        result["affects_ego_route"] = bool(element["affects_ego_route"])
    return result


def _build_element_view(
    element,
    element_type,
    actor,
    camera,
    intrinsic,
    depth_m,
):
    result = _unknown_element(element, element_type, "not_evaluated")
    if actor is None:
        result["association_source"] = "actor_missing"
        return result, f"actor {int(element['actor_id'])} unavailable"

    try:
        vertices, geometry_source, geometry_centers = _actor_world_vertices(
            actor,
            element,
            element_type,
        )
        roi = projected_roi(
            vertices,
            camera["transform"],
            intrinsic,
            camera["width"],
            camera["height"],
            ASSOCIATION["roi_expand_pixels"],
        )
        camera_location = np.array(
            [
                camera["transform"].location.x,
                camera["transform"].location.y,
                camera["transform"].location.z,
            ],
            dtype=np.float64,
        )
        actor_distances = np.linalg.norm(
            geometry_centers - camera_location,
            axis=1,
        )
        settings = ASSOCIATION[element_type]
        associated = associate_semantic_box(
            roi,
            camera["semantic"],
            depth_m,
            actor_distances,
            settings["semantic_tag"],
            settings["depth_tolerance_m"],
            ASSOCIATION["minimum_semantic_pixels"],
        )
    except Exception as exc:
        result["association_source"] = "projection_error"
        return result, f"actor {int(element['actor_id'])} projection failed: {exc}"

    result.update(associated)
    result["geometry_roi_xyxy"] = roi
    result["geometry_source"] = geometry_source
    result["association_source"] = (
        "semantic_depth_confirmed"
        if associated["visibility"] == "visible"
        else "semantic_depth_no_support"
    )
    return result, None


def _project_stop_line(
    line,
    owner_actor_id,
    owner_type,
    camera,
    intrinsic,
):
    result = {
        "owner_actor_id": int(owner_actor_id),
        "owner_type": owner_type,
        "geometry_source": line["geometry_source"],
        "longitudinal_distance": float(line["longitudinal_distance"]),
        "ego_before_line": bool(line["ego_before_line"]),
        "projected_endpoints": None,
        "image_segment": None,
        "projection_status": "not_projected",
    }
    points = np.array(
        [
            _dict_location(line["left_endpoint"]),
            _dict_location(line["right_endpoint"]),
        ],
        dtype=np.float64,
    )
    camera_points = world_to_camera(points, camera["transform"])
    pixels, in_front = project_camera_points(camera_points, intrinsic)
    if not bool(np.all(in_front)):
        result["projection_status"] = "behind_camera"
        return result

    endpoints = pixels.tolist()
    result["projected_endpoints"] = endpoints
    segment = clip_image_segment(
        endpoints[0],
        endpoints[1],
        camera["width"],
        camera["height"],
    )
    result["image_segment"] = segment
    result["projection_status"] = "projected" if segment is not None else "outside_image"
    return result


def _build_camera_record(camera, traffic_elements, actors_by_id):
    record = {
        "width": camera.get("width"),
        "height": camera.get("height"),
        "fov_degrees": camera.get("fov_degrees"),
        "traffic_lights": [],
        "stop_signs": [],
        "stop_lines": [],
        "errors": [],
    }
    camera_error = camera.get("error")
    if camera_error:
        record["errors"].append(str(camera_error))
        for element in traffic_elements["traffic_lights"]:
            record["traffic_lights"].append(
                _unknown_element(element, "traffic_light", "sensor_unavailable")
            )
        for element in traffic_elements["stop_signs"]:
            record["stop_signs"].append(
                _unknown_element(element, "stop_sign", "sensor_unavailable")
            )
        return record

    intrinsic = camera_intrinsics(
        camera["width"],
        camera["height"],
        camera["fov_degrees"],
    )
    depth_m = (
        np.asarray(camera["depth_m"], dtype=np.float64)
        if "depth_m" in camera
        else decode_carla_depth(camera["depth_raw"])
    )
    for element_type, source_key, target_key in (
        ("traffic_light", "traffic_lights", "traffic_lights"),
        ("stop_sign", "stop_signs", "stop_signs"),
    ):
        for element in traffic_elements[source_key]:
            actor_id = int(element["actor_id"])
            view, error = _build_element_view(
                element,
                element_type,
                actors_by_id.get(actor_id),
                camera,
                intrinsic,
                depth_m,
            )
            record[target_key].append(view)
            if error:
                record["errors"].append(error)
            for line in element["stop_lines"]:
                try:
                    record["stop_lines"].append(
                        _project_stop_line(
                            line,
                            actor_id,
                            element_type,
                            camera,
                            intrinsic,
                        )
                    )
                except Exception as exc:
                    record["errors"].append(
                        f"actor {actor_id} stop line projection failed: {exc}"
                    )
    return record


def build_traffic_element_view_record(
    frame_id,
    traffic_elements,
    actors_by_id,
    camera_frames,
):
    """Build a versioned image-label record for one saved sensor frame."""
    return {
        "schema_version": IMAGE_SCHEMA_VERSION,
        "source_traffic_element_schema_version": traffic_elements["schema_version"],
        "frame_id": str(frame_id),
        "association": {
            "roi_expand_pixels": ASSOCIATION["roi_expand_pixels"],
            "minimum_semantic_pixels": ASSOCIATION["minimum_semantic_pixels"],
            "traffic_light": dict(ASSOCIATION["traffic_light"]),
            "stop_sign": dict(ASSOCIATION["stop_sign"]),
        },
        "cameras": {
            camera_name: _build_camera_record(
                camera,
                traffic_elements,
                actors_by_id,
            )
            for camera_name, camera in camera_frames.items()
        },
        "errors": list(traffic_elements.get("errors", [])),
    }
