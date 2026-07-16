#!/usr/bin/env python3
"""Render deterministic RGB samples for traffic-element label review."""

import argparse
import json
from pathlib import Path

import cv2


LIGHT_COLORS = {
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "green": (0, 255, 0),
}
STOP_SIGN_COLOR = (255, 255, 0)
EXACT_STOP_LINE_COLOR = (255, 0, 255)
APPROXIMATE_STOP_LINE_COLOR = (0, 165, 255)
UNKNOWN_LIGHT_COLOR = (192, 192, 192)


def _visible(item):
    return item.get("visibility") == "visible"


def _camera(record, camera_name):
    cameras = record.get("cameras", {})
    if camera_name not in cameras:
        raise ValueError(f"camera {camera_name!r} is absent from the view record")
    return cameras[camera_name]


def _frame_flags(record, camera_name):
    camera = _camera(record, camera_name)
    lights = camera.get("traffic_lights", [])
    stops = camera.get("stop_signs", [])
    visible_lights = [item for item in lights if _visible(item)]
    visible_stops = [item for item in stops if _visible(item)]
    active_light = any(
        item.get("is_active_for_ego") is True for item in visible_lights
    )
    irrelevant_light = any(
        item.get("is_active_for_ego") is False
        and item.get("controls_ego_lane") is False
        and item.get("relevant_to_ego") is False
        for item in visible_lights
    )
    relevant_stop = any(
        item.get("affects_ego_route") is True for item in stops
    )
    visible_infrastructure = bool(visible_lights or visible_stops)
    return {
        "active_light": active_light,
        "irrelevant_light": irrelevant_light,
        "relevant_stop": relevant_stop,
        "hard_negative": (
            visible_infrastructure and not active_light and not relevant_stop
        ),
    }


def select_records(records, camera_name="front", limit=12):
    """Select coverage samples deterministically, then fill by record key."""
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    ordered = sorted(records, key=lambda entry: str(entry[0]))
    selected = []
    selected_keys = set()
    flags = {
        key: _frame_flags(record, camera_name) for key, record in ordered
    }

    for category in (
        "active_light",
        "irrelevant_light",
        "relevant_stop",
        "hard_negative",
    ):
        if len(selected) >= limit:
            break
        for entry in ordered:
            key = entry[0]
            if key not in selected_keys and flags[key][category]:
                selected.append(entry)
                selected_keys.add(key)
                break

    for entry in ordered:
        if len(selected) >= limit:
            break
        key = entry[0]
        if key not in selected_keys:
            selected.append(entry)
            selected_keys.add(key)
    return selected


def _actor_distances(camera):
    distances = {}
    for line in camera.get("stop_lines", []):
        distance = line.get("longitudinal_distance")
        actor_id = line.get("owner_actor_id")
        if not isinstance(distance, (int, float)) or actor_id is None:
            continue
        current = distances.get(actor_id)
        if current is None or abs(distance) < abs(current):
            distances[actor_id] = float(distance)
    return distances


def _label_for_actor(prefix, item, distance):
    parts = [f"{prefix}#{item.get('actor_id', '?')}"]
    state = item.get("state")
    if state is not None:
        parts.append(str(state))
    parts.append(str(item.get("visibility", "unknown")))
    if distance is not None:
        parts.append(f"{distance:+.1f}m")
    return " ".join(parts)


def _draw_box_and_label(image, item, color, label):
    box = item.get("bbox_xyxy")
    if not _visible(item) or not isinstance(box, list) or len(box) != 4:
        return
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), color, 2)
    text_y = max(12, y1 - 4)
    cv2.putText(
        image,
        label,
        (max(0, x1), text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_stop_line(image, line):
    if line.get("projection_status") != "projected":
        return
    segment = line.get("image_segment")
    if not isinstance(segment, list) or len(segment) != 2:
        return
    start = tuple(int(round(value)) for value in segment[0])
    end = tuple(int(round(value)) for value in segment[1])
    exact = line.get("geometry_source") == "carla_stop_waypoint"
    color = EXACT_STOP_LINE_COLOR if exact else APPROXIMATE_STOP_LINE_COLOR
    cv2.line(image, start, end, color, 2, cv2.LINE_AA)
    midpoint = (
        int(round((start[0] + end[0]) / 2.0)),
        max(12, int(round((start[1] + end[1]) / 2.0)) - 4),
    )
    distance = line.get("longitudinal_distance")
    distance_text = (
        f" {float(distance):+.1f}m"
        if isinstance(distance, (int, float))
        else ""
    )
    cv2.putText(
        image,
        f"line#{line.get('owner_actor_id', '?')}{distance_text}",
        midpoint,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        color,
        1,
        cv2.LINE_AA,
    )


def render_overlay(rgb_path, view_record, camera_name, output_path):
    """Draw traffic-element labels over one RGB frame and return its path."""
    rgb_path = Path(rgb_path)
    output_path = Path(output_path)
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unable to read RGB image: {rgb_path}")

    camera = _camera(view_record, camera_name)
    distances = _actor_distances(camera)
    for item in camera.get("traffic_lights", []):
        state = str(item.get("state", "unknown"))
        color = LIGHT_COLORS.get(state.lower(), UNKNOWN_LIGHT_COLOR)
        label = _label_for_actor(
            "TL", item, distances.get(item.get("actor_id"))
        )
        _draw_box_and_label(image, item, color, label)
    for item in camera.get("stop_signs", []):
        label = _label_for_actor(
            "STOP", item, distances.get(item.get("actor_id"))
        )
        _draw_box_and_label(image, item, STOP_SIGN_COLOR, label)
    for line in camera.get("stop_lines", []):
        _draw_stop_line(image, line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"unable to write overlay image: {output_path}")
    return output_path


def _discover_records(root, camera_name):
    root = Path(root)
    records = []
    assets = {}
    for view_path in sorted(root.rglob("traffic_element_views/*.json")):
        route = view_path.parent.parent
        rgb_path = route / f"rgb_{camera_name}" / f"{view_path.stem}.jpg"
        if not rgb_path.is_file():
            raise FileNotFoundError(
                f"RGB frame for {view_path} is missing: {rgb_path}"
            )
        key = view_path.relative_to(root).with_suffix("").as_posix()
        record = json.loads(view_path.read_text(encoding="utf-8"))
        records.append((key, record))
        assets[key] = rgb_path
    if not records:
        raise ValueError(f"no traffic_element_views records under {root}")
    return records, assets


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Render deterministic traffic-element label overlays"
    )
    parser.add_argument("root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument(
        "--camera", choices=("front", "left", "right"), default="front"
    )
    args = parser.parse_args(argv)

    records, assets = _discover_records(args.root, args.camera)
    selected = select_records(records, args.camera, args.limit)
    outputs = []
    output_dir = Path(args.output_dir)
    for index, (key, record) in enumerate(selected):
        output_name = f"{index:02d}_{key.replace('/', '__')}.jpg"
        output = render_overlay(
            assets[key],
            record,
            camera_name=args.camera,
            output_path=output_dir / output_name,
        )
        outputs.append(str(output))
    print(json.dumps({"rendered": len(outputs), "outputs": outputs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
