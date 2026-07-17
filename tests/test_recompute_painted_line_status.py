import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools.data.recompute_painted_line_status import (
    RecomputeError,
    recompute_painted_line_status,
)


HEIGHT = 300
WIDTH = 400


def _write_images(route_run, camera, frame_id, rgb=None, depth_m=10.0, write_seg=True):
    """Write synthetic RGB/depth/semantic frames matching the collector layout."""
    rgb_dir = route_run / f"rgb_{camera}"
    depth_dir = route_run / f"depth_{camera}"
    seg_dir = route_run / f"seg_{camera}"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    if rgb is None:
        rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    # Collector stored RGB; cv2.imwrite expects BGR, and the recompute loader
    # swaps channels back to BGR, so write the array as-is via imwrite.
    cv2.imwrite(str(rgb_dir / f"{frame_id}.jpg"), rgb)

    # Depth: encode meters into CARLA's 24-bit layout. decode_carla_depth does
    # dot(bgr, [65536, 256, 1]) * 1000 / (256^3 - 1), so the B channel carries
    # the high byte and the R channel the low byte of `normalized`.
    normalized = depth_m * (256.0**3 - 1.0) / 1000.0
    hi = int(round(normalized // 65536.0)) % 256   # → B channel (x65536)
    mid = int(round(normalized // 256.0)) % 256     # → G channel (x256)
    lo = int(round(normalized)) % 256               # → R channel (x1)
    depth_bgr = np.full((HEIGHT, WIDTH, 3), 0, dtype=np.uint8)
    depth_bgr[..., 0] = hi   # B
    depth_bgr[..., 1] = mid  # G
    depth_bgr[..., 2] = lo   # R
    cv2.imwrite(str(depth_dir / f"{frame_id}.png"), depth_bgr)

    if write_seg:
        seg = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        cv2.imwrite(str(seg_dir / f"{frame_id}.png"), seg)


def _view_with_projected_target():
    """A view record whose front stop target has boundary+corridor projected."""
    return {
        "schema_version": 3,
        "frame_id": "0000",
        "cameras": {
            "front": {
                "stop_targets": [
                    {
                        "target_id": "Town01_Opt:7:0:-1:20.0",
                        "status": "available",
                        "boundary": {
                            "projection_status": "projected",
                            "image_segment": [[100.0, 150.0], [300.0, 150.0]],
                            "camera_forward_depth_m": 10.0,
                        },
                        "corridor": {
                            "projection_status": "projected",
                            # A rectangle enclosing the horizontal boundary line.
                            "image_envelope": [
                                [100.0, 140.0],
                                [300.0, 140.0],
                                [300.0, 160.0],
                                [100.0, 160.0],
                            ],
                        },
                        "painted_line": {
                            "status": "unknown",
                            "image_segment": None,
                            "score": None,
                        },
                    }
                ]
            }
        },
    }


def _write_view(root, view, rel="route/traffic_element_views/0000.json"):
    view_path = Path(root) / rel
    view_path.parent.mkdir(parents=True, exist_ok=True)
    view_path.write_text(json.dumps(view), encoding="utf-8")
    return view_path


class RecomputePaintedLineStatusTests(unittest.TestCase):
    def test_eligible_target_with_crossing_line_becomes_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_view(tmp, _view_with_projected_target())
            route_run = Path(tmp) / "route"
            # Draw a bright horizontal line along y=150 that crosses the corridor.
            rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            cv2.line(rgb, (100, 150), (300, 150), (255, 255, 255), 3)
            _write_images(route_run, "front", "0000", rgb=rgb, depth_m=10.0)

            result = recompute_painted_line_status(tmp)

        self.assertEqual(result["eligible_for_recompute"], 1)
        self.assertEqual(result["candidate_hits"], 1)
        self.assertEqual(result["unknown_after_recompute"], 0)
        self.assertEqual(result["load_or_compute_errors"], 0)
        self.assertEqual(len(result["candidate_samples"]), 1)
        self.assertEqual(result["candidate_samples"][0]["camera"], "front")

    def test_ineligible_target_is_not_recomputed(self):
        view = _view_with_projected_target()
        # Boundary off-screen: gate fails, target is skipped entirely.
        view["cameras"]["front"]["stop_targets"][0]["boundary"][
            "projection_status"
        ] = "outside_image"
        view["cameras"]["front"]["stop_targets"][0]["boundary"]["image_segment"] = None
        view["cameras"]["front"]["stop_targets"][0]["corridor"][
            "projection_status"
        ] = "outside_image"
        view["cameras"]["front"]["stop_targets"][0]["corridor"]["image_envelope"] = []
        with tempfile.TemporaryDirectory() as tmp:
            _write_view(tmp, view)
            # No images on disk — if the gate failed to skip, this would crash.
            result = recompute_painted_line_status(tmp)

        self.assertEqual(result["total_stop_targets"], 1)
        self.assertEqual(result["eligible_for_recompute"], 0)
        self.assertEqual(result["candidate_hits"], 0)
        self.assertEqual(result["load_or_compute_errors"], 0)

    def test_missing_rgb_file_counted_as_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_view(tmp, _view_with_projected_target())
            # Write only depth+seg, deliberately omit RGB.
            route_run = Path(tmp) / "route"
            (route_run / "depth_front").mkdir(parents=True)
            (route_run / "seg_front").mkdir(parents=True)
            # depth/seg presence is irrelevant; the loader hits RGB first and fails.
            result = recompute_painted_line_status(tmp)

        self.assertEqual(result["eligible_for_recompute"], 1)
        self.assertEqual(result["candidate_hits"], 0)
        self.assertEqual(result["load_or_compute_errors"], 1)

    def test_summary_has_required_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_view(tmp, _view_with_projected_target())
            route_run = Path(tmp) / "route"
            _write_images(route_run, "front", "0000")
            result = recompute_painted_line_status(tmp)

        required = {
            "dataset_root",
            "cameras",
            "views_scanned",
            "total_stop_targets",
            "eligible_for_recompute",
            "candidate_hits",
            "unknown_after_recompute",
            "load_or_compute_errors",
            "candidate_rate_of_eligible",
            "candidate_rate_of_all_targets",
            "candidate_samples",
        }
        self.assertTrue(required.issubset(result.keys()))

    def test_limit_caps_scanned_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            for frame in ("0000", "0001", "0002"):
                _write_view(
                    tmp,
                    _view_with_projected_target(),
                    rel=f"route/traffic_element_views/{frame}.json",
                )
            result = recompute_painted_line_status(tmp, limit=2)

        self.assertEqual(result["views_scanned"], 2)

    def test_missing_root_raises_recompute_error(self):
        with self.assertRaises(RecomputeError):
            recompute_painted_line_status("/nonexistent/path/should/not/exist")


if __name__ == "__main__":
    unittest.main()
