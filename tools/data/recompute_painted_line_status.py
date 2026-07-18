#!/usr/bin/env python3
"""Audit optional painted-line candidates from stored sensor evidence.

This dry-run tool reruns the visible road-marking detector near each projected
Leaderboard boundary. A virtual infraction boundary is valid even when no
painted line exists, so a zero candidate rate is not a label or collection
failure. This tool never creates virtual stop-boundary labels and never modifies
stored JSON.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
TEAM_CODE = REPO_ROOT / "leaderboard" / "team_code"
if str(TEAM_CODE) not in sys.path:
    sys.path.insert(0, str(TEAM_CODE))

from traffic_element_projection import (  # noqa: E402  (after sys.path setup)
    decode_carla_depth,
    find_painted_line_candidate,
)


CAMERAS = ("front", "left", "right")
MAX_CANDIDATE_SAMPLES = 5


class RecomputeError(Exception):
    """Raised when the dataset root is unusable."""


def _iter_view_records(root):
    """Yield ``(view_path, route_run, frame_id)`` for every stored view record."""
    root = Path(root)
    for view_path in sorted(root.rglob("traffic_element_views/*.json")):
        route_run = view_path.parent.parent
        frame_id = view_path.stem
        yield view_path, route_run, frame_id


def _load_rgb_bgr(route_run, camera, frame_id):
    """Load a stored RGB jpg and restore the collector's BGR layout.

    The collector saved jpgs after ``cvtColor(BGR2RGB)``, so the stored file is
    RGB. ``cv2.imread`` returns it as a BGR-labelled array of that RGB data; we
    swap it back so the array matches what the in-memory collector would have
    passed to ``find_painted_line_candidate``.
    """
    rgb_path = route_run / f"rgb_{camera}" / f"{frame_id}.jpg"
    raw = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if raw is None:
        raise FileNotFoundError(f"unable to read RGB image: {rgb_path}")
    return cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)


def _load_depth_m(route_run, camera, frame_id):
    """Load a stored CARLA 24-bit depth png and decode it to meters."""
    depth_path = route_run / f"depth_{camera}" / f"{frame_id}.png"
    raw = cv2.imread(str(depth_path), cv2.IMREAD_COLOR)
    if raw is None:
        raise FileNotFoundError(f"unable to read depth image: {depth_path}")
    return decode_carla_depth(raw)


def _load_semantic(route_run, camera, frame_id):
    """Load a stored single-channel semantic png (uint8 CityObjectLabel tags)."""
    seg_path = route_run / f"seg_{camera}" / f"{frame_id}.png"
    semantic = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)
    return None if semantic is None else semantic


def _eligible_for_recompute(target):
    """Return whether stored evidence can be checked for a painted line."""
    boundary = target.get("boundary", {})
    corridor = target.get("corridor", {})
    return (
        boundary.get("projection_status") == "projected"
        and corridor.get("projection_status") == "projected"
        and len(corridor.get("image_envelope", [])) >= 3
    )


def _recompute_target(target, route_run, camera, frame_id):
    """Reload evidence and rerun the candidate detector for one stop target."""
    rgb_bgr = _load_rgb_bgr(route_run, camera, frame_id)
    depth_m = _load_depth_m(route_run, camera, frame_id)
    semantic = _load_semantic(route_run, camera, frame_id)
    corridor = target["corridor"]
    boundary = target["boundary"]
    return find_painted_line_candidate(
        rgb_bgr,
        depth_m,
        np.asarray(corridor["image_envelope"], dtype=np.float64),
        np.asarray(boundary["image_segment"], dtype=np.float64),
        boundary["camera_forward_depth_m"],
        semantic=semantic,
    )


def recompute_painted_line_status(root, cameras=None, limit=None):
    """Scan ``root`` and report the would-be painted-line candidate hit rate.

    Read-only: no files are modified. ``cameras`` defaults to all three;
    ``limit`` caps the number of view records scanned (``None`` = all).
    """
    root = Path(root)
    if not root.is_dir():
        raise RecomputeError(f"dataset root is not a directory: {root}")
    cameras = tuple(CAMERAS) if cameras is None else tuple(cameras)

    views_scanned = 0
    total_stop_targets = 0
    eligible = 0
    candidate_hits = 0
    unknown_after = 0
    errors = 0
    candidate_samples = []

    records = _iter_view_records(root)
    for view_path, route_run, frame_id in records:
        if limit is not None and views_scanned >= limit:
            break
        views_scanned += 1
        try:
            with view_path.open("r", encoding="utf-8") as handle:
                view = json.load(handle)
        except (OSError, ValueError):
            errors += 1
            continue
        cameras_record = view.get("cameras", {}) if isinstance(view, dict) else {}
        for camera in cameras:
            camera_record = cameras_record.get(camera)
            if not isinstance(camera_record, dict):
                continue
            stop_targets = camera_record.get("stop_targets")
            if not isinstance(stop_targets, list):
                continue
            for target in stop_targets:
                if not isinstance(target, dict):
                    continue
                total_stop_targets += 1
                if not _eligible_for_recompute(target):
                    continue
                eligible += 1
                try:
                    result = _recompute_target(target, route_run, camera, frame_id)
                except (OSError, ValueError, cv2.error):
                    errors += 1
                    continue
                if result.get("status") == "candidate":
                    candidate_hits += 1
                    if len(candidate_samples) < MAX_CANDIDATE_SAMPLES:
                        candidate_samples.append(
                            {
                                "view_path": str(
                                    view_path.relative_to(root)
                                    if root in view_path.parents
                                    else view_path
                                ),
                                "camera": camera,
                                "target_id": target.get("target_id"),
                                "score": result.get("score"),
                                "median_depth_residual_m": result.get(
                                    "median_depth_residual_m"
                                ),
                            }
                        )
                else:
                    unknown_after += 1

    return {
        "dataset_root": str(root),
        "cameras": list(cameras),
        "views_scanned": views_scanned,
        "total_stop_targets": total_stop_targets,
        "eligible_for_recompute": eligible,
        "candidate_hits": candidate_hits,
        "unknown_after_recompute": unknown_after,
        "load_or_compute_errors": errors,
        "candidate_rate_of_eligible": (
            f"{candidate_hits * 100.0 / eligible:.1f}%" if eligible else "0.0%"
        ),
        "candidate_rate_of_all_targets": (
            f"{candidate_hits * 100.0 / total_stop_targets:.2f}%"
            if total_stop_targets
            else "0.00%"
        ),
        "candidate_samples": candidate_samples,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--camera",
        action="append",
        dest="cameras",
        choices=("front", "left", "right"),
        help="restrict to one or more cameras (default: all three)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap the number of view records scanned (default: all)",
    )
    args = parser.parse_args(argv)
    try:
        result = recompute_painted_line_status(
            args.root, cameras=args.cameras, limit=args.limit
        )
    except RecomputeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
