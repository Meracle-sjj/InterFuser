#!/usr/bin/env python3
"""Render deterministic traffic-light and stop-target evidence overlays."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


LIGHT_COLORS = {
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "green": (0, 255, 0),
}
UNKNOWN_LIGHT_COLOR = (192, 192, 192)
TRIGGER_COLOR = (255, 255, 0)
BOUNDARY_COLOR = (255, 0, 255)
STOP_POSE_COLOR = (255, 0, 0)
CORRIDOR_COLOR = (0, 180, 0)
CANDIDATE_COLOR = (0, 255, 255)


def _visible(item):
    return item.get("visibility") == "visible"


def _camera(record, camera_name):
    cameras = record.get("cameras", {})
    if camera_name not in cameras:
        raise ValueError(f"camera {camera_name!r} is absent from the view record")
    return cameras[camera_name]


def _frame_flags(record, camera_name):
    camera = _camera(record, camera_name)
    targets = camera.get("stop_targets", [])
    lights = camera.get("traffic_lights", [])
    visible_lights = [item for item in lights if _visible(item)]
    valid_target = any(item.get("status") == "available" for item in targets)
    unknown_target = any(
        item.get("unknown_reason") == "geometry_unknown"
        or item.get("geometry_unknown_reason") is not None
        for item in targets
    )
    irrelevant_light = any(
        item.get("relevant_to_ego") is False for item in visible_lights
    )
    return {
        "valid_target": valid_target,
        "unknown_target": unknown_target,
        "irrelevant_light": irrelevant_light,
        "hard_negative": not targets and not visible_lights,
    }


def select_records(records, camera_name="front", limit=12):
    """Select deterministic geometry and negative coverage samples."""
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    ordered = sorted(records, key=lambda entry: str(entry[0]))
    selected = []
    selected_keys = set()
    selected_group_counts = {}
    flags = {key: _frame_flags(record, camera_name) for key, record in ordered}

    def group_key(entry):
        return str(entry[0]).split("/traffic_element_views/", 1)[0]

    def add(entry):
        selected.append(entry)
        selected_keys.add(entry[0])
        group = group_key(entry)
        selected_group_counts[group] = selected_group_counts.get(group, 0) + 1

    def category_rank(entry, category):
        if category != "valid_target":
            return (0, 0.0)
        target_ranks = [
            (
                0
                if target.get("boundary", {}).get("projection_status") == "projected"
                and target.get("corridor", {}).get("projection_status") == "projected"
                else 1,
                abs(float(target["signed_route_distance_m"])),
            )
            for target in _camera(entry[1], camera_name).get("stop_targets", [])
            if target.get("status") == "available"
            and isinstance(target.get("signed_route_distance_m"), (int, float))
        ]
        return min(target_ranks) if target_ranks else (1, float("inf"))

    for category in (
        "valid_target",
        "unknown_target",
        "irrelevant_light",
        "hard_negative",
    ):
        candidates = [
            entry
            for entry in ordered
            if entry[0] not in selected_keys and flags[entry[0]][category]
        ]
        if candidates and len(selected) < limit:
            add(
                min(
                    candidates,
                    key=lambda entry: (
                        selected_group_counts.get(group_key(entry), 0),
                        category_rank(entry, category),
                        str(entry[0]),
                    ),
                )
            )
    while len(selected) < limit:
        remaining = [entry for entry in ordered if entry[0] not in selected_keys]
        if not remaining:
            break
        add(
            min(
                remaining,
                key=lambda entry: (
                    selected_group_counts.get(group_key(entry), 0),
                    str(entry[0]),
                ),
            )
        )
    return selected


def _pixel(point):
    return tuple(int(round(value)) for value in point)


def _draw_box_and_label(image, item):
    box = item.get("bbox_xyxy")
    if not _visible(item) or not isinstance(box, list) or len(box) != 4:
        return
    color = LIGHT_COLORS.get(str(item.get("state", "")).lower(), UNKNOWN_LIGHT_COLOR)
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), color, 2)
    label = "TL#{} {} {}".format(
        item.get("actor_id", "?"),
        item.get("state", "unknown"),
        "route" if item.get("relevant_to_ego") else "irrelevant",
    )
    cv2.putText(
        image,
        label,
        (max(0, x1), max(12, y1 - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_projected_point(image, projection, color, marker="circle"):
    if projection.get("projection_status") != "projected":
        return
    point = projection.get("image_point")
    if not isinstance(point, list) or len(point) != 2:
        return
    if marker == "cross":
        cv2.drawMarker(image, _pixel(point), color, cv2.MARKER_CROSS, 12, 2)
    else:
        cv2.circle(image, _pixel(point), 5, color, -1)


def _draw_segment(image, projection, color, thickness=2):
    if projection.get("projection_status") != "projected":
        return
    segment = projection.get("image_segment")
    if not isinstance(segment, list) or len(segment) != 2:
        return
    cv2.line(image, _pixel(segment[0]), _pixel(segment[1]), color, thickness)


def _draw_target(image, target):
    corridor = target.get("corridor", {})
    if corridor.get("projection_status") == "projected":
        envelope = corridor.get("image_envelope", [])
        if len(envelope) >= 3:
            points = np.rint(np.asarray(envelope)).astype(np.int32)
            cv2.polylines(image, [points], True, CORRIDOR_COLOR, 2)
        polyline = corridor.get("image_polyline", [])
        if len(polyline) >= 2:
            points = np.rint(np.asarray(polyline)).astype(np.int32)
            cv2.polylines(image, [points], False, CORRIDOR_COLOR, 1)

    _draw_projected_point(image, target.get("trigger_waypoint", {}), TRIGGER_COLOR)
    _draw_segment(image, target.get("boundary", {}), BOUNDARY_COLOR, 2)
    _draw_projected_point(
        image,
        target.get("recommended_stop_pose", {}),
        STOP_POSE_COLOR,
        marker="cross",
    )
    painted = target.get("painted_line", {})
    if painted.get("status") in {"candidate", "verified"}:
        segment = painted.get("image_segment")
        if isinstance(segment, list) and len(segment) == 2:
            cv2.line(
                image,
                _pixel(segment[0]),
                _pixel(segment[1]),
                CANDIDATE_COLOR,
                2,
            )

    anchor = target.get("recommended_stop_pose", {}).get("image_point")
    if not isinstance(anchor, list):
        segment = target.get("boundary", {}).get("image_segment")
        anchor = segment[0] if isinstance(segment, list) and segment else [8, 18]
    distance = target.get("signed_route_distance_m")
    distance_text = f" {float(distance):+.1f}m" if isinstance(distance, (int, float)) else ""
    reason = target.get("geometry_unknown_reason") or target.get("unknown_reason")
    text = "{}{} {}{}".format(
        target.get("target_id", "target"),
        distance_text,
        target.get("status", "unknown"),
        f" {reason}" if reason else "",
    )
    cv2.putText(
        image,
        text,
        (max(0, int(anchor[0])), max(12, int(anchor[1]) - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        BOUNDARY_COLOR,
        1,
        cv2.LINE_AA,
    )


def render_overlay(rgb_path, view_record, camera_name, output_path):
    """Draw v3 traffic-light and stop-target evidence over one RGB frame."""
    rgb_path = Path(rgb_path)
    output_path = Path(output_path)
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unable to read RGB image: {rgb_path}")
    camera = _camera(view_record, camera_name)
    for light in camera.get("traffic_lights", []):
        _draw_box_and_label(image, light)
    for target in camera.get("stop_targets", []):
        _draw_target(image, target)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"unable to write overlay image: {output_path}")
    return output_path


def build_review_manifest_entries(records, camera_name="front"):
    entries = []
    for key, record in records:
        for target in _camera(record, camera_name).get("stop_targets", []):
            if target.get("painted_line", {}).get("status") != "candidate":
                continue
            entries.append(
                {
                    "view_path": f"{key}.json",
                    "camera": camera_name,
                    "target_id": target["target_id"],
                    "decision": "unreviewed",
                }
            )
    return entries


def _discover_records(root, camera_name):
    root = Path(root)
    records = []
    assets = {}
    for view_path in sorted(root.rglob("traffic_element_views/*.json")):
        route = view_path.parent.parent
        rgb_path = route / f"rgb_{camera_name}" / f"{view_path.stem}.jpg"
        if not rgb_path.is_file():
            raise FileNotFoundError(f"RGB frame for {view_path} is missing: {rgb_path}")
        key = view_path.relative_to(root).with_suffix("").as_posix()
        record = json.loads(view_path.read_text(encoding="utf-8"))
        records.append((key, record))
        assets[key] = rgb_path
    if not records:
        raise ValueError(f"no traffic_element_views records under {root}")
    return records, assets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--camera", choices=("front", "left", "right"), default="front")
    parser.add_argument("--review-manifest-output")
    args = parser.parse_args(argv)

    records, assets = _discover_records(args.root, args.camera)
    selected = select_records(records, args.camera, args.limit)
    outputs = []
    output_dir = Path(args.output_dir)
    for index, (key, record) in enumerate(selected):
        output = render_overlay(
            assets[key],
            record,
            camera_name=args.camera,
            output_path=output_dir / f"{index:02d}_{key.replace('/', '__')}.jpg",
        )
        outputs.append(str(output))
    manifest = build_review_manifest_entries(selected, args.camera)
    if args.review_manifest_output:
        manifest_path = Path(args.review_manifest_output)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"rendered": len(outputs), "outputs": outputs, "candidates": len(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
