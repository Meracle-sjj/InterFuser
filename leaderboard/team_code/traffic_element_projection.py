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
