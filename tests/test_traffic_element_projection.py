import copy
import json
import math
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from team_code.traffic_element_projection import (
    EVIDENCE,
    IMAGE_SCHEMA_VERSION,
    associate_semantic_box,
    build_lidar_target_evidence,
    build_traffic_element_view_record,
    camera_intrinsics,
    clip_image_segment,
    decode_carla_depth,
    find_painted_line_candidate,
    project_camera_points,
    projected_roi,
    transform_matrix,
    world_to_camera,
    world_to_sensor_xyz,
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


class FakeBoundingBox:
    def __init__(self, vertices):
        self._vertices = vertices

    def get_world_vertices(self, transform):
        return list(self._vertices)


class FakeWorldBoundingBox:
    def __init__(self, location, extent, rotation=None):
        self.location = location
        self.extent = extent
        self.rotation = rotation or FakeRotation()


class FakeActor:
    def __init__(self, actor_id, vertices):
        self.id = actor_id
        self.bounding_box = FakeBoundingBox(vertices)
        self._transform = FakeTransform()

    def get_transform(self):
        return self._transform


class FakeTrafficLightActor(FakeActor):
    def __init__(self, actor_id, vertices, light_boxes):
        super().__init__(actor_id, vertices)
        self._light_boxes = light_boxes

    def get_light_boxes(self):
        return list(self._light_boxes)


def _valid_stop_target():
    return {
        "target_id": "Town01_Opt:3:0:-1:10.0",
        "status": "valid",
        "unknown_reason": None,
        "geometry_source": "scenario_runner_running_red_light_test_v1",
        "owner_traffic_light_actor_ids": [11],
        "signed_route_distance_m": 10.0,
        "trigger_stop_waypoint": {
            "location": {"x": 8.0, "y": 0.0, "z": 0.0},
        },
        "leaderboard_infraction_boundary": {
            "left_endpoint": {"x": 10.0, "y": -1.0, "z": 0.0},
            "right_endpoint": {"x": 10.0, "y": 1.0, "z": 0.0},
        },
        "recommended_ego_stop_pose": {
            "location": {"x": 7.0, "y": 0.0, "z": 0.0},
            "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
        },
        "stop_evidence_corridor": {
            "centerline": [
                {
                    "location": {"x": x, "y": 0.0, "z": 0.0},
                    "lane_width": 4.0,
                }
                for x in (8.0, 10.0, 12.0)
            ],
        },
    }


def _phase2_record(target=None):
    return {
        "schema_version": 2,
        "frame_id": "0052",
        "map_name": "Town01_Opt",
        "ego": {
            "actor_id": 1,
            "location": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            "lane": {"road_id": 3, "section_id": 0, "lane_id": -1},
        },
        "active_traffic_light_id": 11,
        "traffic_lights": [
            {
                "actor_id": 11,
                "state": "Red",
                "is_active_for_ego": True,
                "controls_ego_lane": True,
                "relevant_to_ego": True,
                "location": {"x": 20.0, "y": 0.0, "z": 0.0},
            }
        ],
        "stop_targets": [target or _valid_stop_target()],
        "errors": [],
    }


def _camera_frame():
    semantic = np.zeros((10, 12), dtype=np.uint8)
    semantic[3:6, 4:8] = 7
    depth = np.full((10, 12), 100.0, dtype=np.float64)
    depth[3:6, 4:8] = 20.0
    return {
        "transform": FakeTransform(),
        "rgb": np.zeros((10, 12, 3), dtype=np.uint8),
        "semantic": semantic,
        "depth_m": depth,
        "width": 12,
        "height": 10,
        "fov_degrees": 90.0,
    }


def _lidar_frame(points=None):
    if points is None:
        points = np.array(
            [
                [10.0, 0.0, 0.0, 1.0],
                [10.0, 0.0, 1.0, 1.0],
                [10.0, 8.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    return {
        "transform": FakeTransform(),
        "ego_transform": FakeTransform(),
        "points": points,
    }


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
    def test_association_uses_carla_0916_semantic_tags(self):
        self.assertEqual(IMAGE_SCHEMA_VERSION, 3)
        self.assertEqual(EVIDENCE["traffic_light"]["semantic_tag"], 7)
        self.assertEqual(EVIDENCE["road_lines_semantic_tag"], 24)
        self.assertNotIn("stop_sign", EVIDENCE)

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

    def test_semantic_pixels_can_match_multiple_geometry_depths(self):
        result = associate_semantic_box(
            [0, 0, 3, 1],
            np.full((1, 3), 7, dtype=np.uint8),
            np.array([[10.0, 20.0, 30.0]]),
            [10.0, 20.0],
            7,
            0.1,
            2,
        )

        self.assertEqual(result["visibility"], "visible")
        self.assertEqual(result["bbox_xyxy"], [0, 0, 2, 1])
        self.assertEqual(result["semantic_pixel_count"], 2)

    def test_transverse_segment_is_clipped_to_image(self):
        self.assertEqual(
            clip_image_segment((-10.0, 5.0), (20.0, 5.0), 12, 10),
            [[0.0, 5.0], [11.0, 5.0]],
        )

    def test_segment_outside_image_returns_none(self):
        self.assertIsNone(
            clip_image_segment((-10.0, -5.0), (-2.0, -1.0), 12, 10)
        )

    def test_clipped_segment_is_clamped_inside_numeric_image_bounds(self):
        segment = clip_image_segment(
            (30.067727895143836, 171.74269465800492),
            (-11.93178824196188, 177.0732625136505),
            400,
            300,
        )

        self.assertEqual(segment[1][0], 0.0)
        for point in segment:
            self.assertTrue(0.0 <= point[0] <= 399.0)
            self.assertTrue(0.0 <= point[1] <= 299.0)


class PaintedLineCandidateTests(unittest.TestCase):
    corridor = [[20, 50], [180, 50], [180, 100], [20, 100]]
    boundary = [[30, 80], [170, 80]]

    def test_transverse_line_is_candidate_not_verified(self):
        rgb = np.zeros((120, 200, 3), dtype=np.uint8)
        cv2.line(rgb, (40, 80), (160, 80), (255, 255, 255), 4)

        result = find_painted_line_candidate(
            rgb,
            np.full((120, 200), 12.0),
            corridor_polygon=self.corridor,
            expected_boundary_segment=self.boundary,
            expected_depth_m=12.0,
        )

        self.assertEqual(result["status"], "candidate")
        self.assertNotEqual(result["status"], "verified")
        self.assertLessEqual(result["angle_error_degrees"], 15.0)
        self.assertLessEqual(result["median_depth_residual_m"], 2.0)

    def test_blank_corridor_remains_unknown(self):
        result = find_painted_line_candidate(
            np.zeros((120, 200, 3), dtype=np.uint8),
            np.full((120, 200), 12.0),
            self.corridor,
            self.boundary,
            12.0,
        )

        self.assertEqual(
            result,
            {"status": "unknown", "image_segment": None, "score": None},
        )

    def test_parallel_lane_marking_is_rejected(self):
        rgb = np.zeros((120, 200, 3), dtype=np.uint8)
        cv2.line(rgb, (100, 55), (100, 98), (255, 255, 255), 4)

        result = find_painted_line_candidate(
            rgb,
            np.full((120, 200), 12.0),
            self.corridor,
            self.boundary,
            12.0,
        )

        self.assertEqual(result["status"], "unknown")

    def test_depth_inconsistent_line_is_rejected(self):
        rgb = np.zeros((120, 200, 3), dtype=np.uint8)
        cv2.line(rgb, (40, 80), (160, 80), (255, 255, 255), 4)

        result = find_painted_line_candidate(
            rgb,
            np.full((120, 200), 30.0),
            self.corridor,
            self.boundary,
            12.0,
        )

        self.assertEqual(result["status"], "unknown")


class EvidenceSchemaV3Tests(unittest.TestCase):
    def setUp(self):
        self.vertices = [
            FakeLocation(20.0, -4.0, -4.0),
            FakeLocation(20.0, 4.0, -4.0),
            FakeLocation(20.0, -4.0, 4.0),
            FakeLocation(20.0, 4.0, 4.0),
        ]

    def _build_view(self, target=None, camera=None, lidar=None):
        return build_traffic_element_view_record(
            frame_id="0052",
            traffic_elements=_phase2_record(target),
            actors_by_id={11: FakeActor(11, self.vertices)},
            camera_frames={"front": camera or _camera_frame()},
            lidar_frame=lidar or _lidar_frame(),
        )

    def test_v3_has_traffic_lights_targets_and_no_stop_sign_keys(self):
        record = self._build_view()

        self.assertEqual(IMAGE_SCHEMA_VERSION, 3)
        self.assertEqual(record["schema_version"], 3)
        self.assertEqual(record["source_traffic_element_schema_version"], 2)
        self.assertEqual(len(record["cameras"]["front"]["stop_targets"]), 1)
        self.assertNotIn("stop_sign", json.dumps(record).lower())

    def test_visible_light_preserves_phase2_semantics(self):
        record = build_traffic_element_view_record(
            frame_id="0052",
            traffic_elements=_phase2_record(),
            actors_by_id={11: FakeActor(11, self.vertices)},
            camera_frames={"front": _camera_frame()},
            lidar_frame=_lidar_frame(),
        )

        self.assertEqual(record["schema_version"], 3)
        self.assertEqual(record["frame_id"], "0052")
        light = record["cameras"]["front"]["traffic_lights"][0]
        self.assertEqual(light["actor_id"], 11)
        self.assertEqual(light["state"], "Red")
        self.assertTrue(light["is_active_for_ego"])
        self.assertTrue(light["controls_ego_lane"])
        self.assertEqual(light["association_source"], "semantic_depth_confirmed")
        self.assertEqual(light["bbox_xyxy"], [4, 3, 8, 6])

    def test_traffic_light_uses_light_boxes_instead_of_trigger_box(self):
        actor = FakeTrafficLightActor(
            11,
            [FakeLocation(-5.0, 0.0, 0.0)],
            [
                FakeWorldBoundingBox(
                    FakeLocation(20.0, 0.0, 0.0),
                    FakeLocation(4.0, 4.0, 4.0),
                )
            ],
        )

        record = build_traffic_element_view_record(
            "0052",
            _phase2_record(),
            {11: actor},
            {"front": _camera_frame()},
            lidar_frame=_lidar_frame(),
        )

        light = record["cameras"]["front"]["traffic_lights"][0]
        self.assertEqual(light["visibility"], "visible")
        self.assertEqual(light["geometry_source"], "traffic_light_boxes")

    def test_camera_projects_boundary_pose_and_corridor(self):
        target = self._build_view()["cameras"]["front"]["stop_targets"][0]

        self.assertEqual(target["signed_route_distance_m"], 10.0)
        self.assertEqual(
            target["trigger_waypoint"]["projection_status"],
            "projected",
        )
        self.assertEqual(target["boundary"]["projection_status"], "projected")
        self.assertIsNotNone(target["boundary"]["image_segment"])
        self.assertEqual(
            target["recommended_stop_pose"]["projection_status"],
            "projected",
        )
        self.assertGreaterEqual(len(target["corridor"]["image_polyline"]), 2)
        self.assertGreaterEqual(len(target["corridor"]["image_envelope"]), 4)

    def test_projected_target_runs_review_only_candidate_extraction(self):
        candidate = {
            "status": "candidate",
            "image_segment": [[4, 5], [8, 5]],
            "score": 0.8,
        }
        with patch(
            "team_code.traffic_element_projection.find_painted_line_candidate",
            return_value=candidate,
        ) as find_candidate:
            target = self._build_view()["cameras"]["front"]["stop_targets"][0]

        self.assertIs(target["painted_line"], candidate)
        find_candidate.assert_called_once()

    def test_camera_depth_support_is_counted_per_corridor_sample(self):
        camera = _camera_frame()
        camera["depth_m"][:] = 10.0

        corridor = self._build_view(camera=camera)["cameras"]["front"][
            "stop_targets"
        ][0]["corridor"]

        self.assertGreater(corridor["finite_depth_sample_count"], 0)
        self.assertGreater(corridor["depth_supported_sample_count"], 0)
        self.assertIsNotNone(corridor["median_depth_residual_m"])
        self.assertEqual(corridor["occlusion_status"], "supported")

    def test_finite_inconsistent_depth_marks_corridor_occluded(self):
        corridor = self._build_view()["cameras"]["front"]["stop_targets"][0][
            "corridor"
        ]

        self.assertGreater(corridor["finite_depth_sample_count"], 0)
        self.assertEqual(corridor["depth_supported_sample_count"], 0)
        self.assertEqual(corridor["occlusion_status"], "occluded")

    def test_outside_image_target_is_known_not_unknown(self):
        target = copy.deepcopy(_valid_stop_target())
        for endpoint in target["leaderboard_infraction_boundary"].values():
            endpoint["y"] += 100.0
        target["recommended_ego_stop_pose"]["location"]["y"] += 100.0
        for sample in target["stop_evidence_corridor"]["centerline"]:
            sample["location"]["y"] += 100.0

        view = self._build_view(target=target)["cameras"]["front"]["stop_targets"][0]

        self.assertEqual(view["status"], "available")
        self.assertEqual(view["boundary"]["projection_status"], "outside_image")
        self.assertEqual(view["corridor"]["projection_status"], "outside_image")

    def test_behind_camera_target_is_reported(self):
        target = copy.deepcopy(_valid_stop_target())
        for endpoint in target["leaderboard_infraction_boundary"].values():
            endpoint["x"] *= -1.0
        target["recommended_ego_stop_pose"]["location"]["x"] *= -1.0
        for sample in target["stop_evidence_corridor"]["centerline"]:
            sample["location"]["x"] *= -1.0

        view = self._build_view(target=target)["cameras"]["front"]["stop_targets"][0]

        self.assertEqual(view["boundary"]["projection_status"], "behind_camera")
        self.assertEqual(view["corridor"]["projection_status"], "behind_camera")

    def test_missing_actor_is_unknown_not_negative(self):
        record = build_traffic_element_view_record(
            "0052",
            _phase2_record(),
            {},
            {"front": _camera_frame()},
            lidar_frame=_lidar_frame(),
        )

        light = record["cameras"]["front"]["traffic_lights"][0]
        self.assertEqual(light["visibility"], "unknown")
        self.assertIsNone(light["bbox_xyxy"])
        self.assertEqual(light["association_source"], "actor_missing")
        self.assertIn("actor 11 unavailable", record["cameras"]["front"]["errors"][0])

    def test_missing_camera_evidence_is_unknown(self):
        record = build_traffic_element_view_record(
            "0052",
            _phase2_record(),
            {11: FakeActor(11, self.vertices)},
            {"front": {"error": "required sensor unavailable"}},
            lidar_frame=_lidar_frame(),
        )

        light = record["cameras"]["front"]["traffic_lights"][0]
        self.assertEqual(light["visibility"], "unknown")
        self.assertIsNone(light["bbox_xyxy"])
        self.assertEqual(
            record["cameras"]["front"]["errors"],
            ["required sensor unavailable"],
        )
        target = record["cameras"]["front"]["stop_targets"][0]
        self.assertEqual(target["status"], "unknown")
        self.assertEqual(target["unknown_reason"], "sensor_unavailable")

    def test_unknown_geometry_stays_unknown_for_camera_and_lidar(self):
        target = copy.deepcopy(_valid_stop_target())
        target["status"] = "unknown"
        target["unknown_reason"] = "waypoint_branch"

        record = self._build_view(target=target)

        camera_target = record["cameras"]["front"]["stop_targets"][0]
        lidar_target = record["lidar"]["targets"][0]
        self.assertEqual(camera_target["status"], "unknown")
        self.assertEqual(camera_target["unknown_reason"], "geometry_unknown")
        self.assertEqual(lidar_target["status"], "unknown")
        self.assertEqual(lidar_target["unknown_reason"], "geometry_unknown")

    def test_lidar_counts_corridor_and_surface_band_points(self):
        evidence = build_lidar_target_evidence(
            _valid_stop_target(),
            points=_lidar_frame()["points"],
            lidar_transform=FakeTransform(),
            ego_transform=FakeTransform(),
        )

        self.assertEqual(evidence["status"], "available")
        self.assertEqual(evidence["in_corridor_point_count"], 2)
        self.assertEqual(evidence["road_surface_point_count"], 1)

    def test_world_to_sensor_transform_round_trip(self):
        sensor = FakeTransform(
            FakeLocation(2.0, 0.0, 0.0),
            FakeRotation(yaw=90.0),
        )
        world = np.array([[2.0, 1.0, 0.0]])
        local = world_to_sensor_xyz(world, sensor)
        homogeneous = np.column_stack([local, np.ones(len(local))])
        restored = (transform_matrix(sensor) @ homogeneous.T).T[:, :3]

        np.testing.assert_allclose(local, [[1.0, 0.0, 0.0]], atol=1e-6)
        np.testing.assert_allclose(restored, world, atol=1e-6)

    def test_missing_lidar_is_unknown(self):
        evidence = build_lidar_target_evidence(
            _valid_stop_target(),
            points=None,
            lidar_transform=None,
            ego_transform=None,
        )

        self.assertEqual(evidence["status"], "unknown")
        self.assertEqual(evidence["unknown_reason"], "sensor_unavailable")

    def test_non_finite_lidar_points_are_unknown(self):
        evidence = build_lidar_target_evidence(
            _valid_stop_target(),
            points=np.array([[np.nan, 0.0, 0.0, 1.0]]),
            lidar_transform=FakeTransform(),
            ego_transform=FakeTransform(),
        )

        self.assertEqual(evidence["status"], "unknown")
        self.assertEqual(evidence["unknown_reason"], "projection_error")


if __name__ == "__main__":
    unittest.main()
