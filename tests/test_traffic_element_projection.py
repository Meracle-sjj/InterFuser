import math
import unittest

import numpy as np

from team_code.traffic_element_projection import (
    camera_intrinsics,
    decode_carla_depth,
    project_camera_points,
    transform_matrix,
    world_to_camera,
)


class FakeLocation:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class FakeRotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll


class FakeTransform:
    def __init__(self, location=None, rotation=None):
        self.location = location or FakeLocation()
        self.rotation = rotation or FakeRotation()


class ProjectionMathTests(unittest.TestCase):
    def test_camera_intrinsics_for_400x300_fov100(self):
        matrix = camera_intrinsics(400, 300, 100.0)
        expected_focal = 400.0 / (2.0 * math.tan(math.radians(50.0)))

        self.assertAlmostEqual(matrix[0, 0], expected_focal)
        self.assertAlmostEqual(matrix[1, 1], expected_focal)
        self.assertEqual(matrix[0, 2], 200.0)
        self.assertEqual(matrix[1, 2], 150.0)

    def test_transform_matrix_contains_translation(self):
        matrix = transform_matrix(
            FakeTransform(FakeLocation(10.0, 20.0, 2.0), FakeRotation())
        )

        np.testing.assert_allclose(matrix[:3, 3], [10.0, 20.0, 2.0])

    def test_world_to_camera_respects_carla_yaw(self):
        camera = FakeTransform(
            FakeLocation(10.0, 20.0, 2.0),
            FakeRotation(yaw=90.0),
        )
        camera_points = world_to_camera(
            np.array([[10.0, 25.0, 2.0]]),
            camera,
        )

        np.testing.assert_allclose(camera_points[0], [5.0, 0.0, 0.0], atol=1e-6)

    def test_project_camera_point_uses_forward_right_up_axes(self):
        intrinsic = camera_intrinsics(400, 300, 100.0)
        projected, in_front = project_camera_points(
            np.array([[10.0, 2.0, 1.0]]),
            intrinsic,
        )

        self.assertTrue(in_front[0])
        self.assertGreater(projected[0, 0], 200.0)
        self.assertLess(projected[0, 1], 150.0)

    def test_project_camera_point_behind_camera_is_nan(self):
        intrinsic = camera_intrinsics(400, 300, 100.0)
        projected, in_front = project_camera_points(
            np.array([[-1.0, 0.0, 0.0]]),
            intrinsic,
        )

        self.assertFalse(in_front[0])
        self.assertTrue(np.isnan(projected[0]).all())

    def test_depth_decoder_matches_carla_24_bit_encoding(self):
        raw = np.array([[[128, 0, 0]]], dtype=np.uint8)
        expected = 1000.0 * (128.0 * 65536.0) / (256.0**3 - 1.0)

        self.assertAlmostEqual(float(decode_carla_depth(raw)[0, 0]), expected)


if __name__ == "__main__":
    unittest.main()
