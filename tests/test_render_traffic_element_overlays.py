import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools.data.render_traffic_element_overlays import (
    render_overlay,
    select_records,
)


def _camera_record(
    *,
    active_light=False,
    irrelevant_light=False,
    relevant_stop=False,
):
    lights = []
    if active_light:
        lights.append(
            {
                "actor_id": 11,
                "visibility": "visible",
                "is_active_for_ego": True,
                "controls_ego_lane": True,
                "relevant_to_ego": True,
            }
        )
    if irrelevant_light:
        lights.append(
            {
                "actor_id": 12,
                "visibility": "visible",
                "is_active_for_ego": False,
                "controls_ego_lane": False,
                "relevant_to_ego": False,
            }
        )
    stops = []
    if relevant_stop:
        stops.append(
            {
                "actor_id": 21,
                "visibility": "visible",
                "affects_ego_route": True,
            }
        )
    return {
        "traffic_lights": lights,
        "stop_signs": stops,
        "stop_lines": [],
    }


def _record(**camera_options):
    return {"cameras": {"front": _camera_record(**camera_options)}}


class TrafficElementOverlayTests(unittest.TestCase):
    def test_render_overlay_draws_boxes_lines_and_labels(self):
        record = {
            "frame_id": "0000",
            "cameras": {
                "front": {
                    "traffic_lights": [
                        {
                            "actor_id": 11,
                            "state": "Red",
                            "visibility": "visible",
                            "bbox_xyxy": [40, 30, 80, 70],
                            "is_active_for_ego": True,
                            "controls_ego_lane": True,
                        }
                    ],
                    "stop_signs": [
                        {
                            "actor_id": 21,
                            "visibility": "visible",
                            "bbox_xyxy": [100, 50, 130, 100],
                            "affects_ego_route": True,
                        }
                    ],
                    "stop_lines": [
                        {
                            "owner_actor_id": 11,
                            "geometry_source": "carla_stop_waypoint",
                            "longitudinal_distance": 8.0,
                            "projection_status": "projected",
                            "image_segment": [[20.0, 200.0], [300.0, 200.0]],
                        }
                    ],
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rgb_path = root / "0000.jpg"
            output_path = root / "overlay.jpg"
            cv2.imwrite(
                str(rgb_path),
                np.full((300, 400, 3), 255, dtype=np.uint8),
            )

            result = render_overlay(
                rgb_path,
                record,
                camera_name="front",
                output_path=output_path,
            )

            self.assertEqual(result, output_path)
            self.assertTrue(output_path.exists())
            image = cv2.imread(str(output_path))
            self.assertEqual(image.shape[:2], (300, 400))
            self.assertTrue(np.any(image != 255))

    def test_select_records_prioritizes_coverage_then_fills_by_frame_id(self):
        records = [
            ("route_b/0005", _record()),
            ("route_a/0003", _record(irrelevant_light=True)),
            ("route_a/0001", _record(active_light=True)),
            ("route_a/0002", _record(irrelevant_light=True)),
            ("route_a/0004", _record(relevant_stop=True)),
            ("route_a/0000", _record()),
        ]

        selected = select_records(records, camera_name="front", limit=5)

        self.assertEqual(
            [key for key, _record in selected],
            [
                "route_a/0001",
                "route_a/0002",
                "route_a/0004",
                "route_a/0003",
                "route_a/0000",
            ],
        )


if __name__ == "__main__":
    unittest.main()
