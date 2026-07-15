"""Project CARLA traffic-element geometry into camera images."""

import math

import numpy as np


IMAGE_SCHEMA_VERSION = 1


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


def transform_matrix(transform):
    """Return the CARLA local-to-world homogeneous transform matrix."""
    rotation = transform.rotation
    location = transform.location
    cy = math.cos(math.radians(float(rotation.yaw)))
    sy = math.sin(math.radians(float(rotation.yaw)))
    cr = math.cos(math.radians(float(rotation.roll)))
    sr = math.sin(math.radians(float(rotation.roll)))
    cp = math.cos(math.radians(float(rotation.pitch)))
    sp = math.sin(math.radians(float(rotation.pitch)))

    matrix = np.identity(4, dtype=np.float64)
    matrix[:3, 3] = [
        float(location.x),
        float(location.y),
        float(location.z),
    ]
    matrix[:3, :3] = [
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
        [sp, -cp * sr, cp * cr],
    ]
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
    residual = np.abs(depth_m[y1:y2, x1:x2] - float(actor_distance_m))
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

    return [
        [x1 + lower * dx, y1 + lower * dy],
        [x1 + upper * dx, y1 + upper * dy],
    ]
