import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools.data.export_stop_boundary_labels import (
    ExportError,
    export_stop_boundary_labels,
)


TARGET_ID = "Town01_Opt:7:0:-1:20.0"


def _phase_record(primary=True, status="valid"):
    return {
        "schema_version": 2,
        "frame_id": "0000",
        "ego": {
            "location": {"x": 10.0, "y": 20.0, "z": 0.0},
            "rotation": {"yaw": 90.0},
        },
        "stop_targets": [
            {
                "target_id": TARGET_ID,
                "status": status,
                "primary_for_ego": primary,
                "geometry_source": "scenario_runner_running_red_light_test_v1",
                "leaderboard_infraction_boundary": {
                    "left_endpoint": {"x": 10.0, "y": 25.0, "z": 0.0},
                    "right_endpoint": {"x": 8.0, "y": 25.0, "z": 0.0},
                },
                "recommended_ego_stop_pose": {
                    "location": {"x": 10.0, "y": 24.0, "z": 0.0}
                },
                "signed_route_distance_m": 5.0,
                "relative_heading_degrees": 0.0,
                "ego_before_boundary": True,
                "owner_traffic_light_actor_ids": [11],
                "state_by_actor_id": {"11": "Red"},
            }
        ],
    }


def _view_record(projected=True, status="available"):
    return {
        "schema_version": 3,
        "frame_id": "0000",
        "cameras": {
            "front": {
                "width": 20,
                "height": 12,
                "stop_targets": [
                    {
                        "target_id": TARGET_ID,
                        "status": status,
                        "boundary": {
                            "projection_status": (
                                "projected" if projected else "outside_image"
                            ),
                            "image_segment": (
                                [[3.0, 6.0], [16.0, 6.0]] if projected else None
                            ),
                        },
                        "painted_line": {
                            "status": "unknown",
                            "image_segment": None,
                            "score": None,
                        },
                    }
                ],
            }
        },
    }


def _write_fixture(root, phase=None, view=None):
    route = Path(root) / "route_group" / "route_run"
    for name in ("traffic_elements", "traffic_element_views", "rgb_front"):
        (route / name).mkdir(parents=True, exist_ok=True)
    (route / "traffic_elements" / "0000.json").write_text(
        json.dumps(phase if phase is not None else _phase_record()),
        encoding="utf-8",
    )
    (route / "traffic_element_views" / "0000.json").write_text(
        json.dumps(view if view is not None else _view_record()),
        encoding="utf-8",
    )
    cv2.imwrite(
        str(route / "rgb_front" / "0000.jpg"),
        np.full((12, 20, 3), 127, dtype=np.uint8),
    )


class StopBoundaryExportTests(unittest.TestCase):
    def test_exports_mask_and_geometry_manifest_without_copying_rgb(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            output = Path(tmp) / "export"
            _write_fixture(root)
            source_rgb = (
                root / "route_group" / "route_run" / "rgb_front" / "0000.jpg"
            )
            source_bytes = source_rgb.read_bytes()

            result = export_stop_boundary_labels(root, output)
            entries = [
                json.loads(line)
                for line in (output / "manifest.jsonl").read_text().splitlines()
            ]
            entry = entries[0]
            mask = cv2.imread(
                str(output / entry["mask_path"]), cv2.IMREAD_UNCHANGED
            )

            self.assertEqual(result["labels_exported"], 1)
            self.assertEqual(
                entry["label_type"], "leaderboard_virtual_stop_boundary"
            )
            self.assertEqual(entry["painted_line_status"], "unknown")
            self.assertEqual(entry["ego_bev_segment_m"][0]["forward"], 5.0)
            self.assertAlmostEqual(entry["ego_bev_segment_m"][0]["right"], 0.0)
            self.assertEqual(entry["ego_bev_segment_m"][1]["forward"], 5.0)
            self.assertAlmostEqual(entry["ego_bev_segment_m"][1]["right"], 2.0)
            self.assertEqual(mask.shape, (12, 20))
            self.assertEqual(mask[6, 3], 255)
            self.assertEqual(mask[6, 16], 255)
            self.assertEqual(mask[0, 0], 0)
            self.assertFalse((output / entry["source_rgb"]).exists())
            self.assertEqual(source_rgb.read_bytes(), source_bytes)

    def test_primary_only_rejects_a_batch_without_primary_projected_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            _write_fixture(root, phase=_phase_record(primary=False))

            with self.assertRaisesRegex(ExportError, "no projected"):
                export_stop_boundary_labels(
                    root, Path(tmp) / "export", primary_only=True
                )

    def test_unprojected_boundary_is_not_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            _write_fixture(root, view=_view_record(projected=False))

            with self.assertRaisesRegex(ExportError, "no projected"):
                export_stop_boundary_labels(root, Path(tmp) / "export")

    def test_rejects_nonempty_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            output = Path(tmp) / "export"
            _write_fixture(root)
            output.mkdir()
            (output / "keep.txt").write_text("user data", encoding="utf-8")

            with self.assertRaisesRegex(ExportError, "must be empty"):
                export_stop_boundary_labels(root, output)

            self.assertEqual((output / "keep.txt").read_text(), "user data")

    def test_missing_dataset_raises_export_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ExportError, "not a directory"):
                export_stop_boundary_labels(
                    Path(tmp) / "missing", Path(tmp) / "export"
                )


if __name__ == "__main__":
    unittest.main()
