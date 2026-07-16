"""Project CARLA traffic-element geometry into camera images."""

import math

import cv2
import numpy as np


IMAGE_SCHEMA_VERSION = 3
EVIDENCE = {
    "roi_expand_pixels": 6,
    "minimum_semantic_pixels": 3,
    "traffic_light": {
        "semantic_tag": 7,
        "depth_tolerance_m": 4.0,
    },
    "road_lines_semantic_tag": 24,
    "corridor_depth_tolerance_m": 2.0,
    "lidar_min_height_m": -0.5,
    "lidar_max_height_m": 3.0,
    "lidar_road_surface_tolerance_m": 0.25,
}

# Kept as a temporary import compatibility alias for audit tooling migrated later.
ASSOCIATION = EVIDENCE


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


def world_to_sensor_xyz(world_points, sensor_transform):
    """Transform Nx3 CARLA world points into a sensor's local axes."""
    points = np.asarray(world_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("world_points must have shape Nx3")
    homogeneous = np.column_stack([points, np.ones(len(points))])
    return (
        np.linalg.inv(transform_matrix(sensor_transform)) @ homogeneous.T
    ).T[:, :3]


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


def _undirected_angle_degrees(segment):
    (x1, y1), (x2, y2) = segment
    return math.degrees(math.atan2(float(y2) - y1, float(x2) - x1)) % 180.0


def _undirected_angle_error(first, second):
    difference = abs(float(first) - float(second)) % 180.0
    return min(difference, 180.0 - difference)


def _sample_segment_pixels(segment, width, height):
    (x1, y1), (x2, y2) = segment
    length = math.hypot(float(x2) - x1, float(y2) - y1)
    sample_count = max(2, int(math.ceil(length)) + 1)
    xs = np.rint(np.linspace(x1, x2, sample_count)).astype(np.int64)
    ys = np.rint(np.linspace(y1, y2, sample_count)).astype(np.int64)
    valid = (xs >= 0) & (xs < int(width)) & (ys >= 0) & (ys < int(height))
    return xs[valid], ys[valid]


def find_painted_line_candidate(
    rgb,
    depth_m,
    corridor_polygon,
    expected_boundary_segment,
    expected_depth_m,
    semantic=None,
):
    """Return a review-only transverse painted-line candidate."""
    unknown = {"status": "unknown", "image_segment": None, "score": None}
    try:
        image = np.asarray(rgb)
        depth = np.asarray(depth_m, dtype=np.float64)
        polygon = np.asarray(corridor_polygon, dtype=np.float64)
        boundary = np.asarray(expected_boundary_segment, dtype=np.float64)
        if image.ndim != 3 or image.shape[2] < 3:
            return unknown
        height, width = image.shape[:2]
        if depth.shape != (height, width):
            return unknown
        if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
            return unknown
        if boundary.shape != (2, 2) or not np.isfinite(boundary).all():
            return unknown
        if not math.isfinite(float(expected_depth_m)):
            return unknown

        boundary_length = float(np.linalg.norm(boundary[1] - boundary[0]))
        if boundary_length <= 1e-6:
            return unknown
        expected_angle = _undirected_angle_degrees(boundary)

        gray = cv2.cvtColor(image[..., :3].astype(np.uint8), cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 180)
        corridor_mask = np.zeros((height, width), dtype=np.uint8)
        polygon_pixels = np.rint(polygon).astype(np.int32)
        cv2.fillPoly(corridor_mask, [polygon_pixels], 255)
        edges = cv2.bitwise_and(edges, corridor_mask)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=20,
            minLineLength=12,
            maxLineGap=6,
        )
        if lines is None:
            return unknown

        semantic_array = None if semantic is None else np.asarray(semantic)
        if semantic_array is not None and semantic_array.shape != (height, width):
            semantic_array = None
        candidates = []
        for raw_line in lines[:, 0, :]:
            segment = np.asarray(
                [[raw_line[0], raw_line[1]], [raw_line[2], raw_line[3]]],
                dtype=np.float64,
            )
            length = float(np.linalg.norm(segment[1] - segment[0]))
            if length < 0.25 * boundary_length:
                continue
            angle_error = _undirected_angle_error(
                _undirected_angle_degrees(segment),
                expected_angle,
            )
            if angle_error > 15.0:
                continue

            xs, ys = _sample_segment_pixels(segment, width, height)
            if not len(xs):
                continue
            depths = depth[ys, xs]
            finite = np.isfinite(depths)
            if not bool(np.any(finite)):
                continue
            residual = float(
                np.median(np.abs(depths[finite] - float(expected_depth_m)))
            )
            if residual > 2.0:
                continue

            line_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.line(
                line_mask,
                tuple(np.rint(segment[0]).astype(int)),
                tuple(np.rint(segment[1]).astype(int)),
                255,
                3,
            )
            line_pixels = line_mask > 0
            road_line_count = 0
            road_line_fraction = None
            if semantic_array is not None:
                road_line_count = int(
                    np.count_nonzero(
                        line_pixels
                        & (semantic_array == int(EVIDENCE["road_lines_semantic_tag"]))
                    )
                )
                total = int(np.count_nonzero(line_pixels))
                road_line_fraction = (
                    float(road_line_count) / total if total else 0.0
                )

            score = length / boundary_length - residual / 2.0
            candidates.append(
                {
                    "status": "candidate",
                    "image_segment": segment.tolist(),
                    "score": float(score),
                    "angle_error_degrees": float(angle_error),
                    "median_depth_residual_m": residual,
                    "road_lines_semantic_pixel_count": road_line_count,
                    "road_lines_semantic_fraction": road_line_fraction,
                }
            )
    except (TypeError, ValueError, cv2.error):
        return unknown

    return max(candidates, key=lambda item: item["score"]) if candidates else unknown


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
            EVIDENCE["roi_expand_pixels"],
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
        settings = EVIDENCE[element_type]
        associated = associate_semantic_box(
            roi,
            camera["semantic"],
            depth_m,
            actor_distances,
            settings["semantic_tag"],
            settings["depth_tolerance_m"],
            EVIDENCE["minimum_semantic_pixels"],
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


def _unknown_target_view(target, reason):
    return {
        "target_id": str(target["target_id"]),
        "geometry_source": target.get("geometry_source"),
        "owner_traffic_light_actor_ids": list(
            target.get("owner_traffic_light_actor_ids", [])
        ),
        "status": "unknown",
        "unknown_reason": reason,
        "geometry_unknown_reason": target.get("unknown_reason"),
        "boundary": {
            "projection_status": "unknown",
            "projected_endpoints": None,
            "image_segment": None,
        },
        "recommended_stop_pose": {
            "projection_status": "unknown",
            "image_point": None,
            "camera_forward_depth_m": None,
        },
        "corridor": {
            "projection_status": "unknown",
            "image_polyline": [],
            "image_envelope": [],
            "finite_depth_sample_count": 0,
            "depth_supported_sample_count": 0,
            "median_depth_residual_m": None,
            "occlusion_status": "unknown",
        },
        "painted_line": {
            "status": "unknown",
            "image_segment": None,
            "score": None,
        },
    }


def _project_segment(world_points, camera, intrinsic):
    result = {
        "projection_status": "unknown",
        "projected_endpoints": None,
        "image_segment": None,
        "camera_forward_depth_m": None,
    }
    camera_points = world_to_camera(world_points, camera["transform"])
    pixels, in_front = project_camera_points(camera_points, intrinsic)
    if not bool(np.all(in_front)):
        result["projection_status"] = "behind_camera"
        return result

    endpoints = pixels.tolist()
    result["projected_endpoints"] = endpoints
    result["camera_forward_depth_m"] = float(np.mean(camera_points[:, 0]))
    segment = clip_image_segment(
        endpoints[0],
        endpoints[1],
        camera["width"],
        camera["height"],
    )
    result["image_segment"] = segment
    result["projection_status"] = (
        "projected" if segment is not None else "outside_image"
    )
    return result


def _project_point(world_point, camera, intrinsic):
    camera_points = world_to_camera(
        np.asarray(world_point, dtype=np.float64).reshape(1, 3),
        camera["transform"],
    )
    pixels, in_front = project_camera_points(camera_points, intrinsic)
    result = {
        "projection_status": "behind_camera",
        "image_point": None,
        "camera_forward_depth_m": float(camera_points[0, 0]),
    }
    if not bool(in_front[0]):
        return result

    pixel = pixels[0]
    result["image_point"] = pixel.tolist()
    inside = (
        0.0 <= pixel[0] < float(camera["width"])
        and 0.0 <= pixel[1] < float(camera["height"])
    )
    result["projection_status"] = "projected" if inside else "outside_image"
    return result


def _corridor_edges(target, centerline, locations):
    fallback_yaw = float(
        target.get("recommended_ego_stop_pose", {})
        .get("rotation", {})
        .get("yaw", 0.0)
    )
    fallback = np.array(
        [math.cos(math.radians(fallback_yaw)), math.sin(math.radians(fallback_yaw))]
    )
    left = []
    right = []
    for index, (sample, location) in enumerate(zip(centerline, locations)):
        if len(locations) == 1:
            tangent = fallback
        elif index == 0:
            tangent = locations[1, :2] - location[:2]
        elif index == len(locations) - 1:
            tangent = location[:2] - locations[index - 1, :2]
        else:
            tangent = locations[index + 1, :2] - locations[index - 1, :2]
        norm = float(np.linalg.norm(tangent))
        tangent = fallback if norm <= 1e-9 else tangent / norm
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        half_width = float(sample["lane_width"]) / 2.0
        left.append(
            [
                location[0] + normal[0] * half_width,
                location[1] + normal[1] * half_width,
                location[2],
            ]
        )
        right.append(
            [
                location[0] - normal[0] * half_width,
                location[1] - normal[1] * half_width,
                location[2],
            ]
        )
    return np.asarray(left), np.asarray(right)


def _visible_pixel_indices(pixels, in_front, width, height):
    finite = np.isfinite(pixels).all(axis=1)
    inside = (
        in_front
        & finite
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < float(width))
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < float(height))
    )
    return np.flatnonzero(inside)


def _project_corridor(target, camera, intrinsic, depth_m):
    centerline = list(target["stop_evidence_corridor"]["centerline"])
    if not centerline:
        raise ValueError("corridor centerline is empty")
    locations = np.asarray(
        [_dict_location(sample["location"]) for sample in centerline],
        dtype=np.float64,
    )
    camera_points = world_to_camera(locations, camera["transform"])
    pixels, in_front = project_camera_points(camera_points, intrinsic)
    visible = _visible_pixel_indices(
        pixels,
        in_front,
        camera["width"],
        camera["height"],
    )

    left, right = _corridor_edges(target, centerline, locations)
    left_pixels, left_front = project_camera_points(
        world_to_camera(left, camera["transform"]),
        intrinsic,
    )
    right_pixels, right_front = project_camera_points(
        world_to_camera(right, camera["transform"]),
        intrinsic,
    )
    left_visible = _visible_pixel_indices(
        left_pixels,
        left_front,
        camera["width"],
        camera["height"],
    )
    right_visible = _visible_pixel_indices(
        right_pixels,
        right_front,
        camera["width"],
        camera["height"],
    )

    residuals = []
    supported = 0
    for index in visible:
        x = int(np.clip(round(float(pixels[index, 0])), 0, camera["width"] - 1))
        y = int(np.clip(round(float(pixels[index, 1])), 0, camera["height"] - 1))
        measured = float(depth_m[y, x])
        if not math.isfinite(measured):
            continue
        residual = abs(measured - float(camera_points[index, 0]))
        residuals.append(residual)
        if residual <= float(EVIDENCE["corridor_depth_tolerance_m"]):
            supported += 1

    if supported:
        occlusion = "supported"
    elif residuals:
        occlusion = "occluded"
    elif len(visible):
        occlusion = "no_finite_depth"
    elif bool(np.any(in_front)):
        occlusion = "outside_image"
    else:
        occlusion = "behind_camera"

    if len(visible):
        projection_status = "projected"
    elif bool(np.any(in_front)):
        projection_status = "outside_image"
    else:
        projection_status = "behind_camera"
    return {
        "projection_status": projection_status,
        "image_polyline": pixels[visible].tolist(),
        "image_envelope": (
            left_pixels[left_visible].tolist()
            + right_pixels[right_visible][::-1].tolist()
        ),
        "projected_sample_count": int(len(visible)),
        "finite_depth_sample_count": int(len(residuals)),
        "depth_supported_sample_count": int(supported),
        "median_depth_residual_m": (
            float(np.median(residuals)) if residuals else None
        ),
        "occlusion_status": occlusion,
    }


def _project_target_camera(target, camera, intrinsic, depth_m):
    if target.get("status") != "valid":
        return _unknown_target_view(target, "geometry_unknown")
    result = {
        "target_id": str(target["target_id"]),
        "geometry_source": target.get("geometry_source"),
        "owner_traffic_light_actor_ids": list(
            target.get("owner_traffic_light_actor_ids", [])
        ),
        "status": "available",
        "unknown_reason": None,
        "geometry_unknown_reason": None,
    }
    boundary = target["leaderboard_infraction_boundary"]
    result["boundary"] = _project_segment(
        np.asarray(
            [
                _dict_location(boundary["left_endpoint"]),
                _dict_location(boundary["right_endpoint"]),
            ]
        ),
        camera,
        intrinsic,
    )
    result["recommended_stop_pose"] = _project_point(
        _dict_location(target["recommended_ego_stop_pose"]["location"]),
        camera,
        intrinsic,
    )
    result["corridor"] = _project_corridor(
        target,
        camera,
        intrinsic,
        depth_m,
    )
    result["painted_line"] = {
        "status": "unknown",
        "image_segment": None,
        "score": None,
    }
    if (
        result["boundary"]["projection_status"] == "projected"
        and result["corridor"]["projection_status"] == "projected"
        and len(result["corridor"]["image_envelope"]) >= 3
        and camera.get("rgb") is not None
    ):
        result["painted_line"] = find_painted_line_candidate(
            camera["rgb"],
            depth_m,
            result["corridor"]["image_envelope"],
            result["boundary"]["image_segment"],
            result["boundary"]["camera_forward_depth_m"],
            semantic=camera.get("semantic"),
        )
    return result


def build_lidar_target_evidence(
    target,
    points,
    lidar_transform,
    ego_transform,
):
    """Count lidar returns inside a route-associated stop-evidence corridor."""
    target_id = str(target["target_id"])
    if target.get("status") != "valid":
        return {
            "target_id": target_id,
            "status": "unknown",
            "unknown_reason": "geometry_unknown",
        }
    if points is None or lidar_transform is None or ego_transform is None:
        return {
            "target_id": target_id,
            "status": "unknown",
            "unknown_reason": "sensor_unavailable",
        }

    try:
        raw = np.asarray(points, dtype=np.float64)
        if (
            raw.ndim != 2
            or raw.shape[1] < 3
            or not np.isfinite(raw[:, :3]).all()
        ):
            raise ValueError("lidar points must be finite Nx3 or Nx4")
        samples = list(target["stop_evidence_corridor"]["centerline"])
        if not samples:
            raise ValueError("corridor centerline is empty")
        world_centerline = np.asarray(
            [
                [sample["location"][axis] for axis in ("x", "y", "z")]
                for sample in samples
            ],
            dtype=np.float64,
        )
        centerline = world_to_sensor_xyz(world_centerline, lidar_transform)
        delta = raw[:, np.newaxis, :3] - centerline[np.newaxis, :, :]
        nearest = np.argmin(np.sum(delta[:, :, :2] ** 2, axis=2), axis=1)
        nearest_delta = delta[np.arange(len(raw)), nearest]
        widths = np.asarray(
            [float(sample["lane_width"]) / 2.0 for sample in samples]
        )
        lateral = np.linalg.norm(nearest_delta[:, :2], axis=1)
        relative_z = nearest_delta[:, 2]
        inside = (
            (lateral <= widths[nearest])
            & (relative_z >= float(EVIDENCE["lidar_min_height_m"]))
            & (relative_z <= float(EVIDENCE["lidar_max_height_m"]))
        )
        surface = inside & (
            np.abs(relative_z)
            <= float(EVIDENCE["lidar_road_surface_tolerance_m"])
        )
        sensor_to_world = transform_matrix(lidar_transform)
        ego_to_world = transform_matrix(ego_transform)
    except Exception:
        return {
            "target_id": target_id,
            "status": "unknown",
            "unknown_reason": "projection_error",
        }

    return {
        "target_id": target_id,
        "status": "available",
        "unknown_reason": None,
        "sensor_to_ego": (
            np.linalg.inv(ego_to_world) @ sensor_to_world
        ).tolist(),
        "ego_to_world": ego_to_world.tolist(),
        "corridor_centerline_xyz": centerline.tolist(),
        "corridor_half_width_m": widths.tolist(),
        "corridor_road_height_m": centerline[:, 2].tolist(),
        "in_corridor_point_count": int(np.count_nonzero(inside)),
        "road_surface_point_count": int(np.count_nonzero(surface)),
    }


def _build_camera_record(camera, traffic_elements, actors_by_id):
    record = {
        "width": camera.get("width"),
        "height": camera.get("height"),
        "fov_degrees": camera.get("fov_degrees"),
        "traffic_lights": [],
        "stop_targets": [],
        "errors": [],
    }
    camera_error = camera.get("error")
    if camera_error:
        record["errors"].append(str(camera_error))
        for element in traffic_elements["traffic_lights"]:
            record["traffic_lights"].append(
                _unknown_element(element, "traffic_light", "sensor_unavailable")
            )
        for target in traffic_elements["stop_targets"]:
            record["stop_targets"].append(
                _unknown_target_view(target, "sensor_unavailable")
            )
        return record

    try:
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
        if depth_m.shape != (
            int(camera["height"]),
            int(camera["width"]),
        ):
            raise ValueError("camera depth shape does not match image dimensions")
    except Exception as exc:
        record["errors"].append(f"camera projection setup failed: {exc}")
        for element in traffic_elements["traffic_lights"]:
            record["traffic_lights"].append(
                _unknown_element(element, "traffic_light", "projection_error")
            )
        for target in traffic_elements["stop_targets"]:
            record["stop_targets"].append(
                _unknown_target_view(target, "projection_error")
            )
        return record

    for element in traffic_elements["traffic_lights"]:
        actor_id = int(element["actor_id"])
        view, error = _build_element_view(
            element,
            "traffic_light",
            actors_by_id.get(actor_id),
            camera,
            intrinsic,
            depth_m,
        )
        record["traffic_lights"].append(view)
        if error:
            record["errors"].append(error)

    for target in traffic_elements["stop_targets"]:
        try:
            record["stop_targets"].append(
                _project_target_camera(target, camera, intrinsic, depth_m)
            )
        except Exception as exc:
            record["stop_targets"].append(
                _unknown_target_view(target, "projection_error")
            )
            record["errors"].append(
                f"target {target.get('target_id')} projection failed: {exc}"
            )
    return record


def _build_lidar_record(targets, lidar_frame):
    record = {"targets": [], "errors": []}
    frame = lidar_frame or {}
    frame_error = frame.get("error") or (
        "required sensor unavailable: lidar" if not lidar_frame else None
    )
    if frame_error:
        record["errors"].append(str(frame_error))
    for target in targets:
        evidence = build_lidar_target_evidence(
            target,
            None if frame_error else frame.get("points"),
            None if frame_error else frame.get("transform"),
            None if frame_error else frame.get("ego_transform"),
        )
        record["targets"].append(evidence)
        if evidence["unknown_reason"] == "projection_error":
            record["errors"].append(
                f"target {target.get('target_id')} lidar projection failed"
            )
    return record


def _evidence_config_record():
    return {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in EVIDENCE.items()
    }


def build_traffic_element_view_record(
    frame_id,
    traffic_elements,
    actors_by_id,
    camera_frames,
    lidar_frame=None,
):
    """Build a versioned camera/lidar evidence record for one sensor frame."""
    return {
        "schema_version": IMAGE_SCHEMA_VERSION,
        "source_traffic_element_schema_version": traffic_elements["schema_version"],
        "frame_id": str(frame_id),
        "association": _evidence_config_record(),
        "cameras": {
            camera_name: _build_camera_record(
                camera,
                traffic_elements,
                actors_by_id,
            )
            for camera_name, camera in camera_frames.items()
        },
        "lidar": _build_lidar_record(
            traffic_elements["stop_targets"],
            lidar_frame,
        ),
        "errors": list(traffic_elements.get("errors", [])),
    }
