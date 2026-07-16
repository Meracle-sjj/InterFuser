import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from tools.data.audit_traffic_element_views import (
    AuditError,
    audit_traffic_element_views,
)


EXPECTED_ASSOCIATION = {
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


def _phase1_record():
    return {
        "schema_version": 1,
        "active_traffic_light_id": 11,
        "traffic_lights": [
            {
                "actor_id": 11,
                "state": "Red",
                "is_active_for_ego": True,
                "controls_ego_lane": True,
                "stop_lines": [],
            }
        ],
        "stop_signs": [
            {
                "actor_id": 21,
                "affects_ego_route": True,
                "stop_lines": [
                    {
                        "geometry_source": (
                            "trigger_volume_route_entry_approximation"
                        ),
                    }
                ],
            }
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
        "bbox_xyxy": [4, 3, 8, 6] if visible else None,
        "geometry_roi_xyxy": [2, 1, 10, 9],
        "semantic_pixel_count": 12 if visible else 0,
        "median_depth_residual_m": 0.25 if visible else None,
        "association_source": (
            "semantic_depth_confirmed"
            if visible
            else "semantic_depth_no_support"
        ),
        "geometry_source": "actor_bounding_box",
    }


def _stop_view(visibility="visible"):
    visible = visibility == "visible"
    return {
        "actor_id": 21,
        "element_type": "stop_sign",
        "affects_ego_route": True,
        "visibility": visibility,
        "bbox_xyxy": [7, 2, 10, 7] if visible else None,
        "geometry_roi_xyxy": [5, 0, 12, 9],
        "semantic_pixel_count": 8 if visible else 0,
        "median_depth_residual_m": 0.5 if visible else None,
        "association_source": (
            "semantic_depth_confirmed"
            if visible
            else "semantic_depth_no_support"
        ),
        "geometry_source": "actor_bounding_box",
    }


def _camera_record(visible=False):
    return {
        "width": 12,
        "height": 10,
        "fov_degrees": 90.0,
        "traffic_lights": [_light_view("visible" if visible else "not_visible")],
        "stop_signs": [_stop_view("visible" if visible else "not_visible")],
        "stop_lines": (
            [
                {
                    "owner_actor_id": 21,
                    "owner_type": "stop_sign",
                    "geometry_source": (
                        "trigger_volume_route_entry_approximation"
                    ),
                    "longitudinal_distance": 10.0,
                    "ego_before_line": True,
                    "projected_endpoints": [[1.0, 5.0], [11.0, 5.0]],
                    "image_segment": [[1.0, 5.0], [11.0, 5.0]],
                    "projection_status": "projected",
                }
            ]
            if visible
            else []
        ),
        "errors": [],
    }


def _view_record():
    return {
        "schema_version": 2,
        "source_traffic_element_schema_version": 1,
        "frame_id": "0000",
        "association": copy.deepcopy(EXPECTED_ASSOCIATION),
        "cameras": {
            "front": _camera_record(visible=True),
            "left": _camera_record(visible=False),
            "right": _camera_record(visible=False),
        },
        "errors": [],
    }


class TrafficElementViewAuditTests(unittest.TestCase):
    def _write_fixture(self, root):
        route = Path(root) / "route_00"
        for name in ("rgb_front", "traffic_elements", "traffic_element_views"):
            (route / name).mkdir(parents=True, exist_ok=True)
        (route / "rgb_front" / "0000.jpg").write_bytes(b"image")
        (route / "traffic_elements" / "0000.json").write_text(
            json.dumps(_phase1_record()),
            encoding="utf-8",
        )
        (route / "traffic_element_views" / "0000.json").write_text(
            json.dumps(_view_record()),
            encoding="utf-8",
        )
        return route

    def test_summarizes_valid_image_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_fixture(tmp)

            summary = audit_traffic_element_views(tmp)

        self.assertEqual(summary["frames"], 1)
        self.assertEqual(summary["invalid_frames"], 0)
        self.assertEqual(summary["visible_traffic_light_frames"], 1)
        self.assertEqual(summary["semantic_confirmed_traffic_lights"], 1)
        self.assertEqual(summary["semantic_confirmed_stop_signs"], 1)
        self.assertEqual(summary["projected_stop_lines"], 1)

    def test_counts_projected_relevant_stop_line_without_visible_sign_box(self):
        record = _view_record()
        record["cameras"]["front"]["stop_signs"][0] = _stop_view(
            "not_visible"
        )
        with tempfile.TemporaryDirectory() as tmp:
            route = self._write_fixture(tmp)
            (route / "traffic_element_views" / "0000.json").write_text(
                json.dumps(record),
                encoding="utf-8",
            )

            summary = audit_traffic_element_views(tmp)

        self.assertEqual(summary["route_relevant_stop_sign_frames"], 1)
        self.assertEqual(summary["visible_stop_sign_frames"], 0)

    def test_rejects_missing_view_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            route = self._write_fixture(tmp)
            (route / "traffic_element_views" / "0000.json").unlink()

            with self.assertRaisesRegex(AuditError, "missing view frames: 0000"):
                audit_traffic_element_views(tmp)

    def test_rejects_unknown_actor_id(self):
        record = _view_record()
        record["cameras"]["front"]["traffic_lights"][0]["actor_id"] = 99
        with tempfile.TemporaryDirectory() as tmp:
            route = self._write_fixture(tmp)
            (route / "traffic_element_views" / "0000.json").write_text(
                json.dumps(record),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(AuditError, "actor 99 missing"):
                audit_traffic_element_views(tmp)

    def test_rejects_zero_area_visible_box(self):
        record = _view_record()
        record["cameras"]["front"]["traffic_lights"][0]["bbox_xyxy"] = [4, 3, 4, 6]
        with tempfile.TemporaryDirectory() as tmp:
            route = self._write_fixture(tmp)
            (route / "traffic_element_views" / "0000.json").write_text(
                json.dumps(record),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(AuditError, "positive area"):
                audit_traffic_element_views(tmp)

    def test_rejects_visible_box_without_minimum_semantic_pixels(self):
        record = _view_record()
        record["cameras"]["front"]["traffic_lights"][0][
            "semantic_pixel_count"
        ] = 2
        with tempfile.TemporaryDirectory() as tmp:
            route = self._write_fixture(tmp)
            (route / "traffic_element_views" / "0000.json").write_text(
                json.dumps(record),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(AuditError, "at least 3 semantic pixels"):
                audit_traffic_element_views(tmp)

    def test_rejects_non_finite_stop_line_point(self):
        record = _view_record()
        record["cameras"]["front"]["stop_lines"][0]["image_segment"][0][0] = math.nan
        with tempfile.TemporaryDirectory() as tmp:
            route = self._write_fixture(tmp)
            (route / "traffic_element_views" / "0000.json").write_text(
                json.dumps(record),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(AuditError, "finite image point"):
                audit_traffic_element_views(tmp)

    def test_rejects_association_threshold_mismatch(self):
        record = _view_record()
        record["association"]["traffic_light"]["depth_tolerance_m"] = 5.0
        with tempfile.TemporaryDirectory() as tmp:
            route = self._write_fixture(tmp)
            (route / "traffic_element_views" / "0000.json").write_text(
                json.dumps(record),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(AuditError, "association metadata mismatch"):
                audit_traffic_element_views(tmp)


if __name__ == "__main__":
    unittest.main()
