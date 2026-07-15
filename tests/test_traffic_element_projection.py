import math
import unittest

import numpy as np

from team_code.traffic_element_projection import (
    associate_semantic_box,
    camera_intrinsics,
    clip_image_segment,
    decode_carla_depth,
    project_camera_points,
    projected_roi,
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


class AssociationTests(unittest.TestCase):
    def test_projected_roi_clips_and_expands_visible_vertices(self):
        camera = FakeTransform()
        intrinsic = camera_intrinsics(400, 300, 100.0)
        vertices = np.array(
            [
                [10.0, -1.0, -1.0],
                [10.0, 1.0, -1.0],
                [10.0, -1.0, 1.0],
                [10.0, 1.0, 1.0],
            ]
        )

        roi = projected_roi(vertices, camera, intrinsic, 400, 300, expand=6)

        self.assertIsNotNone(roi)
        self.assertGreater(roi[2], roi[0])
        self.assertGreater(roi[3], roi[1])
        self.assertTrue(all(0 <= value <= limit for value, limit in zip(
            roi,
            (400, 300, 400, 300),
        )))

    def test_semantic_depth_pixels_create_tight_xyxy_box(self):
        semantic = np.zeros((10, 12), dtype=np.uint8)
        semantic[3:6, 4:8] = 18
        depth = np.full((10, 12), 100.0, dtype=np.float64)
        depth[3:6, 4:8] = 20.0

        result = associate_semantic_box(
            roi=[2, 1, 10, 9],
            semantic=semantic,
            depth_m=depth,
            actor_distance_m=20.0,
            semantic_tag=18,
            depth_tolerance_m=4.0,
            min_pixels=3,
        )

        self.assertEqual(result["bbox_xyxy"], [4, 3, 8, 6])
        self.assertEqual(result["semantic_pixel_count"], 12)
        self.assertEqual(result["visibility"], "visible")
        self.assertEqual(result["median_depth_residual_m"], 0.0)

    def test_absent_semantic_support_is_not_visible(self):
        result = associate_semantic_box(
            [0, 0, 8, 8],
            np.zeros((8, 8), dtype=np.uint8),
            np.full((8, 8), 20.0, dtype=np.float64),
            20.0,
            18,
            4.0,
            3,
        )

        self.assertIsNone(result["bbox_xyxy"])
        self.assertEqual(result["semantic_pixel_count"], 0)
        self.assertEqual(result["visibility"], "not_visible")

    def test_semantic_pixels_outside_depth_tolerance_are_not_visible(self):
        semantic = np.full((5, 5), 18, dtype=np.uint8)
        depth = np.full((5, 5), 40.0, dtype=np.float64)

        result = associate_semantic_box(
            [0, 0, 5, 5],
            semantic,
            depth,
            20.0,
            18,
            4.0,
            3,
        )

        self.assertEqual(result["visibility"], "not_visible")
        self.assertEqual(result["semantic_pixel_count"], 0)

    def test_stop_line_segment_is_clipped_to_image(self):
        self.assertEqual(
            clip_image_segment((-10.0, 5.0), (20.0, 5.0), 12, 10),
            [[0.0, 5.0], [11.0, 5.0]],
        )

    def test_segment_outside_image_returns_none(self):
        self.assertIsNone(
            clip_image_segment((-10.0, -5.0), (-2.0, -1.0), 12, 10)
        )


if __name__ == "__main__":
    unittest.main()
