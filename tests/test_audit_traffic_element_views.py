"""
[INPUT]: 依赖 tools.data.audit_traffic_element_views 的公开审计 API，并使用临时目录构造 schema v2/v3 多传感器证据。
[OUTPUT]: 提供 evidence v3 结构、帧对齐、可见性、投影、LiDAR 与人工复核状态的回归测试。
[POS]: tests 的 schema v3 审计契约测试，阻止采集数据校验规则在重构中静默放宽或改变。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from tools.data.audit_traffic_element_views import (
    AuditError,
    audit_traffic_element_views,
)


EVIDENCE = {
    "roi_expand_pixels": 6,
    "minimum_semantic_pixels": 3,
    "traffic_light": {"semantic_tag": 7, "depth_tolerance_m": 4.0},
    "road_lines_semantic_tag": 24,
    "corridor_depth_tolerance_m": 2.0,
    "lidar_min_height_m": -0.5,
    "lidar_max_height_m": 3.0,
    "lidar_road_surface_tolerance_m": 0.25,
}
GEOMETRY_SOURCE = "scenario_runner_running_red_light_test_v1"
VALID_ID = "Town01_Opt:7:0:-1:20.0"
UNKNOWN_ID = "Town01_Opt:7:0:-1:40.0"


def _phase_target(target_id, status):
    return {
        "target_id": target_id,
        "status": status,
        "unknown_reason": None if status == "valid" else "waypoint_branch",
        "geometry_source": GEOMETRY_SOURCE,
        "owner_traffic_light_actor_ids": [11],
    }


def _phase_record():
    return {
        "schema_version": 2,
        "frame_id": "0000",
        "map_name": "Town01_Opt",
        "ego": {},
        "active_traffic_light_id": 11,
        "traffic_lights": [
            {
                "actor_id": 11,
                "state": "Red",
                "is_active_for_ego": True,
                "controls_ego_lane": True,
                "relevant_to_ego": True,
            }
        ],
        "stop_targets": [
            _phase_target(VALID_ID, "valid"),
            _phase_target(UNKNOWN_ID, "unknown"),
        ],
        "errors": [],
    }


def _light_view(visibility="visible"):
    visible = visibility == "visible"
    return {
        "actor_id": 11,
        "element_type": "traffic_light",
        "state": "Red",
        "is_active_for_ego": True,
        "controls_ego_lane": True,
        "relevant_to_ego": True,
        "visibility": visibility,
        "bbox_xyxy": [80, 20, 100, 40] if visible else None,
        "geometry_roi_xyxy": [70, 10, 110, 50],
        "semantic_pixel_count": 20 if visible else 0,
        "median_depth_residual_m": 0.25 if visible else None,
        "association_source": (
            "semantic_depth_confirmed"
            if visible
            else "semantic_depth_no_support"
        ),
        "geometry_source": "traffic_light_boxes",
    }


def _unknown_camera_target():
    return {
        "target_id": UNKNOWN_ID,
        "geometry_source": GEOMETRY_SOURCE,
        "owner_traffic_light_actor_ids": [11],
        "status": "unknown",
        "unknown_reason": "geometry_unknown",
        "geometry_unknown_reason": "waypoint_branch",
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


def _available_camera_target(candidate=False):
    painted_line = {
        "status": "unknown",
        "image_segment": None,
        "score": None,
    }
    if candidate:
        painted_line = {
            "status": "candidate",
            "image_segment": [[40.0, 80.0], [160.0, 80.0]],
            "score": 0.8,
            "angle_error_degrees": 1.0,
            "median_depth_residual_m": 0.2,
            "road_lines_semantic_pixel_count": 30,
            "road_lines_semantic_fraction": 0.5,
        }
    return {
        "target_id": VALID_ID,
        "geometry_source": GEOMETRY_SOURCE,
        "owner_traffic_light_actor_ids": [11],
        "status": "available",
        "unknown_reason": None,
        "geometry_unknown_reason": None,
        "boundary": {
            "projection_status": "projected",
            "projected_endpoints": [[30.0, 80.0], [170.0, 80.0]],
            "image_segment": [[30.0, 80.0], [170.0, 80.0]],
            "camera_forward_depth_m": 12.0,
        },
        "recommended_stop_pose": {
            "projection_status": "projected",
            "image_point": [100.0, 95.0],
            "camera_forward_depth_m": 10.0,
        },
        "corridor": {
            "projection_status": "projected",
            "image_polyline": [[100.0, 95.0], [100.0, 80.0]],
            "image_envelope": [
                [20.0, 100.0],
                [20.0, 50.0],
                [180.0, 50.0],
                [180.0, 100.0],
            ],
            "projected_sample_count": 2,
            "finite_depth_sample_count": 2,
            "depth_supported_sample_count": 2,
            "median_depth_residual_m": 0.2,
            "occlusion_status": "supported",
        },
        "painted_line": painted_line,
    }


def _camera_record(visible=False, candidate=False):
    return {
        "width": 200,
        "height": 120,
        "fov_degrees": 100.0,
        "traffic_lights": [_light_view("visible" if visible else "not_visible")],
        "stop_targets": [
            _available_camera_target(candidate=candidate),
            _unknown_camera_target(),
        ],
        "errors": [],
    }


def _view_record():
    identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return {
        "schema_version": 3,
        "source_traffic_element_schema_version": 2,
        "frame_id": "0000",
        "association": copy.deepcopy(EVIDENCE),
        "cameras": {
            "front": _camera_record(visible=True, candidate=True),
            "left": _camera_record(),
            "right": _camera_record(),
        },
        "lidar": {
            "targets": [
                {
                    "target_id": VALID_ID,
                    "status": "available",
                    "unknown_reason": None,
                    "sensor_to_ego": identity,
                    "ego_to_world": identity,
                    "corridor_centerline_xyz": [[8.0, 0.0, 0.0], [12.0, 0.0, 0.0]],
                    "corridor_half_width_m": [2.0, 2.0],
                    "corridor_road_height_m": [0.0, 0.0],
                    "in_corridor_point_count": 20,
                    "road_surface_point_count": 12,
                },
                {
                    "target_id": UNKNOWN_ID,
                    "status": "unknown",
                    "unknown_reason": "geometry_unknown",
                },
            ],
            "errors": [],
        },
        "errors": [],
    }


class TrafficElementViewAuditTests(unittest.TestCase):
    def _write_fixture(self, root, view=None):
        route = Path(root) / "route_00"
        for name in (
            "rgb_front",
            "rgb_left",
            "rgb_right",
            "traffic_elements",
            "traffic_element_views",
            "lidar",
        ):
            (route / name).mkdir(parents=True, exist_ok=True)
        for camera in ("front", "left", "right"):
            (route / f"rgb_{camera}" / "0000.jpg").write_bytes(b"image")
        (route / "lidar" / "0000.npy").write_bytes(b"lidar")
        (route / "traffic_elements" / "0000.json").write_text(
            json.dumps(_phase_record()),
            encoding="utf-8",
        )
        (route / "traffic_element_views" / "0000.json").write_text(
            json.dumps(view if view is not None else _view_record()),
            encoding="utf-8",
        )
        return route

    def test_summarizes_v3_camera_and_lidar_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_fixture(tmp)
            summary = audit_traffic_element_views(tmp)

        self.assertEqual(summary["frames"], 1)
        self.assertEqual(summary["visible_traffic_light_frames"], 1)
        self.assertEqual(summary["semantic_confirmed_traffic_lights"], 1)
        self.assertEqual(summary["projected_stop_boundaries"], 3)
        self.assertEqual(summary["projected_stop_corridors"], 3)
        self.assertEqual(summary["painted_line_candidates"], 1)
        self.assertEqual(summary["verified_painted_lines"], 0)
        self.assertEqual(summary["unknown_stop_targets"], 1)
        self.assertEqual(summary["lidar_available_targets"], 1)
        self.assertEqual(summary["lidar_unknown_targets"], 1)
        self.assertEqual(summary["in_corridor_point_counts"], [20])

    def test_separates_geometry_unknown_from_sensor_unknown(self):
        record = _view_record()
        target = record["cameras"]["front"]["stop_targets"][0]
        target.clear()
        target.update(_unknown_camera_target())
        target["target_id"] = VALID_ID
        target["unknown_reason"] = "sensor_unavailable"
        target["geometry_unknown_reason"] = None
        with tempfile.TemporaryDirectory() as tmp:
            self._write_fixture(tmp, record)
            summary = audit_traffic_element_views(tmp)

        self.assertEqual(summary["geometry_unknown_camera_evidence"], 3)
        self.assertEqual(summary["sensor_unknown_camera_evidence"], 1)

    def test_rejects_legacy_evidence_schema(self):
        record = _view_record()
        record["schema_version"] = 2
        with tempfile.TemporaryDirectory() as tmp:
            self._write_fixture(tmp, record)
            with self.assertRaisesRegex(
                AuditError,
                "unsupported evidence schema_version",
            ):
                audit_traffic_element_views(tmp)

    def test_rejects_camera_segment_outside_image(self):
        record = _view_record()
        record["cameras"]["front"]["stop_targets"][0]["boundary"][
            "image_segment"
        ][1][0] = 200.0
        with tempfile.TemporaryDirectory() as tmp:
            self._write_fixture(tmp, record)
            with self.assertRaisesRegex(AuditError, "inside image bounds"):
                audit_traffic_element_views(tmp)

    def test_rejects_negative_lidar_count(self):
        record = _view_record()
        record["lidar"]["targets"][0]["in_corridor_point_count"] = -1
        with tempfile.TemporaryDirectory() as tmp:
            self._write_fixture(tmp, record)
            with self.assertRaisesRegex(AuditError, "non-negative integer"):
                audit_traffic_element_views(tmp)

    def test_rejects_invalid_lidar_transform_shape(self):
        record = _view_record()
        record["lidar"]["targets"][0]["sensor_to_ego"] = [[1.0]]
        with tempfile.TemporaryDirectory() as tmp:
            self._write_fixture(tmp, record)
            with self.assertRaisesRegex(AuditError, "4x4 finite matrix"):
                audit_traffic_element_views(tmp)

    def test_rejects_unreviewed_verified_candidate(self):
        record = _view_record()
        painted = record["cameras"]["front"]["stop_targets"][0]["painted_line"]
        painted["status"] = "verified"
        with tempfile.TemporaryDirectory() as tmp:
            self._write_fixture(tmp, record)
            with self.assertRaisesRegex(AuditError, "manual_manifest"):
                audit_traffic_element_views(tmp)

    def test_accepts_manually_verified_candidate(self):
        record = _view_record()
        painted = record["cameras"]["front"]["stop_targets"][0]["painted_line"]
        painted["status"] = "verified"
        painted["review_source"] = "manual_manifest"
        with tempfile.TemporaryDirectory() as tmp:
            self._write_fixture(tmp, record)
            summary = audit_traffic_element_views(tmp)

        self.assertEqual(summary["painted_line_candidates"], 0)
        self.assertEqual(summary["verified_painted_lines"], 1)

    def test_accepts_manually_rejected_candidate(self):
        record = _view_record()
        painted = record["cameras"]["front"]["stop_targets"][0]["painted_line"]
        painted["status"] = "unknown"
        painted["review_source"] = "manual_manifest"
        painted["review_decision"] = "rejected"
        with tempfile.TemporaryDirectory() as tmp:
            self._write_fixture(tmp, record)
            summary = audit_traffic_element_views(tmp)

        self.assertEqual(summary["painted_line_candidates"], 0)
        self.assertEqual(summary["verified_painted_lines"], 0)

    def test_rejects_missing_lidar_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            route = self._write_fixture(tmp)
            (route / "lidar" / "0000.npy").unlink()
            with self.assertRaisesRegex(AuditError, "missing lidar frames: 0000"):
                audit_traffic_element_views(tmp)

    def test_rejects_unknown_traffic_light_actor(self):
        record = _view_record()
        record["cameras"]["front"]["traffic_lights"][0]["actor_id"] = 99
        with tempfile.TemporaryDirectory() as tmp:
            self._write_fixture(tmp, record)
            with self.assertRaisesRegex(AuditError, "actor 99 missing"):
                audit_traffic_element_views(tmp)

    def test_rejects_association_metadata_mismatch(self):
        record = _view_record()
        record["association"]["corridor_depth_tolerance_m"] = 3.0
        with tempfile.TemporaryDirectory() as tmp:
            self._write_fixture(tmp, record)
            with self.assertRaisesRegex(AuditError, "association metadata mismatch"):
                audit_traffic_element_views(tmp)


if __name__ == "__main__":
    unittest.main()
