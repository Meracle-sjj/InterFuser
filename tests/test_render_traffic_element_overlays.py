import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools.data.render_traffic_element_overlays import (
    BOUNDARY_COLOR,
    CANDIDATE_COLOR,
    CORRIDOR_COLOR,
    STOP_POSE_COLOR,
    TRIGGER_COLOR,
    build_review_manifest_entries,
    render_overlay,
    select_records,
)


def _light(relevant=True):
    return {
        "actor_id": 11,
        "state": "Red",
        "visibility": "visible",
        "bbox_xyxy": [20, 20, 50, 60],
        "is_active_for_ego": relevant,
        "controls_ego_lane": relevant,
        "relevant_to_ego": relevant,
    }


def _target(status="available", candidate=False, distance=10.0, projected=True):
    unknown = status == "unknown"
    painted = {"status": "unknown", "image_segment": None, "score": None}
    if candidate:
        painted = {
            "status": "candidate",
            "image_segment": [[120.0, 190.0], [280.0, 190.0]],
            "score": 0.8,
        }
    return {
        "target_id": "Town01_Opt:7:0:-1:20.0",
        "status": status,
        "unknown_reason": "geometry_unknown" if unknown else None,
        "geometry_unknown_reason": "waypoint_branch" if unknown else None,
        "signed_route_distance_m": distance,
        "trigger_waypoint": {
            "projection_status": "unknown" if unknown else "projected",
            "image_point": None if unknown else [50.0, 220.0],
        },
        "boundary": {
            "projection_status": "unknown" if unknown else ("projected" if projected else "behind_camera"),
            "image_segment": None if unknown or not projected else [[100.0, 200.0], [300.0, 200.0]],
        },
        "recommended_stop_pose": {
            "projection_status": "unknown" if unknown else "projected",
            "image_point": None if unknown else [200.0, 240.0],
        },
        "corridor": {
            "projection_status": "unknown" if unknown else ("projected" if projected else "outside_image"),
            "image_polyline": [] if unknown or not projected else [[200.0, 240.0], [200.0, 150.0]],
            "image_envelope": (
                []
                if unknown or not projected
                else [[80.0, 250.0], [100.0, 150.0], [300.0, 150.0], [320.0, 250.0]]
            ),
        },
        "painted_line": painted,
    }


def _record(*, target=None, light=None):
    return {
        "frame_id": "0000",
        "cameras": {
            "front": {
                "traffic_lights": [] if light is None else [light],
                "stop_targets": [] if target is None else [target],
            }
        },
    }


class TrafficElementOverlayTests(unittest.TestCase):
    def test_render_overlay_draws_all_geometry_layers(self):
        record = _record(target=_target(candidate=True), light=_light())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rgb_path = root / "0000.png"
            output_path = root / "overlay.png"
            cv2.imwrite(str(rgb_path), np.full((300, 400, 3), 255, dtype=np.uint8))

            render_overlay(rgb_path, record, "front", output_path)
            image = cv2.imread(str(output_path))

        self.assertTrue(np.array_equal(image[220, 50], TRIGGER_COLOR))
        self.assertTrue(np.array_equal(image[200, 200], BOUNDARY_COLOR))
        self.assertTrue(np.array_equal(image[240, 200], STOP_POSE_COLOR))
        self.assertTrue(np.array_equal(image[250, 80], CORRIDOR_COLOR))
        self.assertTrue(np.array_equal(image[190, 200], CANDIDATE_COLOR))

    def test_select_records_prioritizes_target_and_negative_coverage(self):
        records = [
            ("route/traffic_element_views/0005", _record(light=_light())),
            ("route/traffic_element_views/0003", _record(light=_light(False))),
            ("route/traffic_element_views/0001", _record(target=_target())),
            (
                "route/traffic_element_views/0002",
                _record(target=_target(status="unknown")),
            ),
            ("route/traffic_element_views/0004", _record()),
            ("route/traffic_element_views/0000", _record(light=_light())),
        ]

        selected = select_records(records, camera_name="front", limit=5)

        self.assertEqual(
            [key for key, _record in selected],
            [
                "route/traffic_element_views/0001",
                "route/traffic_element_views/0002",
                "route/traffic_element_views/0003",
                "route/traffic_element_views/0004",
                "route/traffic_element_views/0000",
            ],
        )

    def test_review_manifest_contains_only_candidates(self):
        records = [
            (
                "route/traffic_element_views/0001",
                _record(target=_target(candidate=True)),
            ),
            ("route/traffic_element_views/0002", _record(target=_target())),
        ]

        entries = build_review_manifest_entries(records, camera_name="front")

        self.assertEqual(
            entries,
            [
                {
                    "view_path": "route/traffic_element_views/0001.json",
                    "camera": "front",
                    "target_id": "Town01_Opt:7:0:-1:20.0",
                    "decision": "unreviewed",
                }
            ],
        )

    def test_fill_round_robins_across_route_groups(self):
        records = [
            (f"route_{route}/traffic_element_views/000{frame}", _record())
            for route in ("a", "b", "c")
            for frame in (0, 1)
        ]

        selected = select_records(records, camera_name="front", limit=6)

        self.assertEqual(
            [key for key, _record in selected],
            [
                "route_a/traffic_element_views/0000",
                "route_b/traffic_element_views/0000",
                "route_c/traffic_element_views/0000",
                "route_a/traffic_element_views/0001",
                "route_b/traffic_element_views/0001",
                "route_c/traffic_element_views/0001",
            ],
        )

    def test_valid_target_prefers_nearest_signed_route_distance(self):
        records = [
            (
                "route/traffic_element_views/0001",
                _record(target=_target(distance=50.0)),
            ),
            (
                "route/traffic_element_views/0002",
                _record(target=_target(distance=5.0)),
            ),
        ]

        selected = select_records(records, camera_name="front", limit=1)

        self.assertEqual(selected[0][0], "route/traffic_element_views/0002")

    def test_valid_target_prefers_visible_geometry_before_distance(self):
        records = [
            (
                "route/traffic_element_views/0001",
                _record(target=_target(distance=0.0, projected=False)),
            ),
            (
                "route/traffic_element_views/0002",
                _record(target=_target(distance=7.0, projected=True)),
            ),
        ]

        selected = select_records(records, camera_name="front", limit=1)

        self.assertEqual(selected[0][0], "route/traffic_element_views/0002")


if __name__ == "__main__":
    unittest.main()
