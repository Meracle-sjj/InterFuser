"""
[INPUT]: 依赖行人 sequence inventory、三相机 CARLA 原始语义 mask/RGB，以及 measurements 中由 AutoPilot 碰撞预测生成的 is_pedestrian_present。
[OUTPUT]: 提供行人威胁 sequence 的触发覆盖、RGB 可见阶段、相机贡献、天气重复签名与代表帧 JSON 审计报告。
[POS]: tools/data 的 M1 特权监督审计器，把“行人均为横穿威胁”的采集先验转化为可复现证据，但绝不把 birdview 或测量真值送入模型推理。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


REPORT_SCHEMA_VERSION = 1
DEFAULT_CAMERAS = ("front", "left", "right")
TOWN_PATTERN = re.compile(r"town(?P<town>\d+)", re.IGNORECASE)
ROUTE_PATTERN = re.compile(r"route(?P<route>\d+)", re.IGNORECASE)
WEATHER_PATTERN = re.compile(r"_w(?P<weather>\d+)(?:_|$)", re.IGNORECASE)


class PedestrianVisibilityAuditError(ValueError):
    """Raised when the requested audit contract is malformed."""


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PedestrianVisibilityAuditError(f"unable to read JSON {path}: {exc}") from exc


def _frame_paths(directory, suffixes):
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    accepted = {suffix.lower() for suffix in suffixes}
    result = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in accepted:
            continue
        if path.stem in result:
            raise PedestrianVisibilityAuditError(
                f"duplicate frame stem {path.stem} in {directory}"
            )
        result[path.stem] = path
    return result


def _load_inventory(dataset_root, inventory_path):
    dataset_root = Path(dataset_root).resolve()
    inventory_path = Path(inventory_path).resolve()
    payload = _read_json(inventory_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("sequences"), list):
        raise PedestrianVisibilityAuditError(
            "inventory must be an object containing a sequences list"
        )

    sequences = []
    seen = set()
    for index, item in enumerate(payload["sequences"]):
        if not isinstance(item, dict) or not isinstance(item.get("sequence"), str):
            raise PedestrianVisibilityAuditError(
                f"inventory sequences[{index}] must contain a string sequence"
            )
        relative = Path(item["sequence"])
        if relative.is_absolute() or ".." in relative.parts:
            raise PedestrianVisibilityAuditError(
                f"inventory sequence escapes dataset root: {relative}"
            )
        key = relative.as_posix()
        if key in seen:
            raise PedestrianVisibilityAuditError(f"duplicate inventory sequence: {key}")
        seen.add(key)
        route = (dataset_root / relative).resolve()
        try:
            route.relative_to(dataset_root)
        except ValueError as exc:
            raise PedestrianVisibilityAuditError(
                f"inventory sequence escapes dataset root: {relative}"
            ) from exc
        if not route.is_dir():
            raise PedestrianVisibilityAuditError(f"sequence directory is missing: {route}")
        sequences.append((key, route))
    if not sequences:
        raise PedestrianVisibilityAuditError("inventory contains no sequences")
    return {
        "path": inventory_path,
        "sha256": _sha256(inventory_path),
        "declared_total_scanned": payload.get("total_scanned"),
        "declared_pedestrian_sequences": payload.get("pedestrian_sequences"),
        "sequences": sequences,
    }


def _sequence_identity(relative_path):
    town_match = TOWN_PATTERN.search(relative_path)
    route_match = ROUTE_PATTERN.search(relative_path)
    weather_match = WEATHER_PATTERN.search(relative_path)
    town = f"Town{int(town_match.group('town')):02d}" if town_match else None
    route_id = int(route_match.group("route")) if route_match else None
    weather = int(weather_match.group("weather")) if weather_match else None
    group = f"{town}:route{route_id:02d}" if town is not None and route_id is not None else None
    return town, route_id, weather, group


def _contiguous_intervals(frame_ids):
    numeric = sorted(int(frame) for frame in frame_ids)
    if not numeric:
        return []
    intervals = []
    start = previous = numeric[0]
    for value in numeric[1:]:
        if value != previous + 1:
            intervals.append([f"{start:04d}", f"{previous:04d}"])
            start = value
        previous = value
    intervals.append([f"{start:04d}", f"{previous:04d}"])
    return intervals


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _representative_frame(frame_ids, frame_pixels, candidates):
    candidates = set(candidates)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda frame: (
            sum(frame_pixels.get(frame, {}).values()),
            -frame_ids.index(frame),
        ),
    )


def _frame_evidence(route, frame, cameras, seg_paths, rgb_paths, birdview_paths):
    if frame is None:
        return None
    return {
        "frame": frame,
        "rgb": {
            camera: str(rgb_paths[camera].get(frame))
            for camera in cameras
            if rgb_paths[camera].get(frame) is not None
        },
        "semantic": {
            camera: str(seg_paths[camera].get(frame))
            for camera in cameras
            if seg_paths[camera].get(frame) is not None
        },
        "birdview": str(birdview_paths.get(frame))
        if birdview_paths.get(frame) is not None
        else None,
        "measurement": str(route / "measurements" / f"{frame}.json"),
    }


def _audit_sequence(
    relative_path,
    route,
    cameras,
    pedestrian_tag,
    minimum_pixels,
):
    errors = []
    town, route_id, weather, town_route_group = _sequence_identity(relative_path)
    if town_route_group is None:
        errors.append(f"{relative_path}: unable to infer Town and route ID")

    measurement_paths = _frame_paths(route / "measurements", {".json"})
    if not measurement_paths:
        errors.append(f"{relative_path}: missing measurements JSON files")

    active_frames = set()
    measurement_frames = set(measurement_paths)
    missing_hazard_field = []
    for frame, path in sorted(measurement_paths.items()):
        try:
            measurement = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{relative_path}/measurements/{path.name}: unreadable: {exc}")
            continue
        if "is_pedestrian_present" not in measurement:
            missing_hazard_field.append(frame)
            continue
        actor_ids = measurement["is_pedestrian_present"]
        if not isinstance(actor_ids, list):
            errors.append(
                f"{relative_path}/measurements/{path.name}: "
                "is_pedestrian_present must be a list"
            )
        elif actor_ids:
            active_frames.add(frame)
    if missing_hazard_field:
        errors.append(
            f"{relative_path}: measurements missing is_pedestrian_present for "
            + ", ".join(missing_hazard_field)
        )

    seg_paths = {}
    rgb_paths = {}
    frame_pixels = defaultdict(dict)
    camera_stats = {}
    semantic_frames = set()
    for camera in cameras:
        seg_paths[camera] = _frame_paths(route / f"seg_{camera}", {".png"})
        rgb_paths[camera] = _frame_paths(
            route / f"rgb_{camera}", {".jpg", ".jpeg", ".png"}
        )
        if not seg_paths[camera]:
            errors.append(f"{relative_path}: missing seg_{camera} PNG files")
        if not rgb_paths[camera]:
            errors.append(f"{relative_path}: missing rgb_{camera} images")
        semantic_frames.update(seg_paths[camera])
        pixels_total = 0
        masks_with_any = 0
        qualified_masks = 0
        maximum_pixels = 0
        for frame, path in sorted(seg_paths[camera].items()):
            try:
                labels = np.asarray(Image.open(path))
            except OSError as exc:
                errors.append(f"{relative_path}/{path.name}: unreadable mask: {exc}")
                continue
            if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
                errors.append(
                    f"{relative_path}/{path.name}: semantic mask must be 2D integer"
                )
                continue
            pixels = int(np.count_nonzero(labels == pedestrian_tag))
            frame_pixels[frame][camera] = pixels
            pixels_total += pixels
            maximum_pixels = max(maximum_pixels, pixels)
            if pixels:
                masks_with_any += 1
            if pixels >= minimum_pixels:
                qualified_masks += 1
                if frame not in rgb_paths[camera]:
                    errors.append(
                        f"{relative_path}: qualified seg_{camera}/{frame} has no RGB"
                    )
        camera_stats[camera] = {
            "pixels": pixels_total,
            "masks_with_any": masks_with_any,
            "qualified_masks": qualified_masks,
            "maximum_pixels_per_mask": maximum_pixels,
        }

    if measurement_frames != semantic_frames:
        missing_measurement = sorted(semantic_frames - measurement_frames)
        missing_semantic = sorted(measurement_frames - semantic_frames)
        if missing_measurement:
            errors.append(
                f"{relative_path}: semantic frames without measurements: "
                + ", ".join(missing_measurement)
            )
        if missing_semantic:
            errors.append(
                f"{relative_path}: measurement frames without semantic masks: "
                + ", ".join(missing_semantic)
            )

    frame_ids = sorted(semantic_frames | measurement_frames, key=lambda value: int(value))
    visible_frames = {
        frame
        for frame, pixels in frame_pixels.items()
        if any(value >= minimum_pixels for value in pixels.values())
    }
    active_visible = active_frames & visible_frames
    active_not_visible = active_frames - visible_frames
    visible_not_active = visible_frames - active_frames
    first_active = min(active_frames, key=int) if active_frames else None
    last_active = max(active_frames, key=int) if active_frames else None
    if active_frames:
        first_value = int(first_active)
        last_value = int(last_active)
        visible_before = {frame for frame in visible_frames if int(frame) < first_value}
        visible_after = {frame for frame in visible_frames if int(frame) > last_value}
        visible_inside_inactive = {
            frame
            for frame in visible_not_active
            if first_value <= int(frame) <= last_value
        }
    else:
        visible_before = set()
        visible_after = set()
        visible_inside_inactive = set()

    signature_payload = [
        {
            "frame": frame,
            "hazard": frame in active_frames,
            "pixels": [frame_pixels.get(frame, {}).get(camera, 0) for camera in cameras],
        }
        for frame in frame_ids
    ]
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    birdview_paths = _frame_paths(route / "birdview", {".jpg", ".jpeg", ".png"})
    peak_hazard = _representative_frame(frame_ids, frame_pixels, active_visible)
    peak_before = _representative_frame(frame_ids, frame_pixels, visible_before)
    peak_overall = _representative_frame(frame_ids, frame_pixels, visible_frames)
    representative_frames = {
        "peak_before_hazard": _frame_evidence(
            route, peak_before, cameras, seg_paths, rgb_paths, birdview_paths
        ),
        "first_hazard": _frame_evidence(
            route, first_active, cameras, seg_paths, rgb_paths, birdview_paths
        ),
        "peak_visible_hazard": _frame_evidence(
            route, peak_hazard, cameras, seg_paths, rgb_paths, birdview_paths
        ),
        "last_hazard": _frame_evidence(
            route, last_active, cameras, seg_paths, rgb_paths, birdview_paths
        ),
        "peak_visible_overall": _frame_evidence(
            route, peak_overall, cameras, seg_paths, rgb_paths, birdview_paths
        ),
    }

    return {
        "sequence": relative_path,
        "town": town,
        "route_id": route_id,
        "weather": weather,
        "town_route_group": town_route_group,
        "trajectory_visibility_signature": signature,
        "measurement_frames": len(measurement_frames),
        "semantic_logical_frames": len(semantic_frames),
        "qualified_visible_frames": len(visible_frames),
        "hazard_frames": len(active_frames),
        "hazard_intervals": _contiguous_intervals(active_frames),
        "hazard_visible_frames": len(active_visible),
        "hazard_not_visible_frames": len(active_not_visible),
        "visible_not_hazard_frames": len(visible_not_active),
        "visible_before_hazard_frames": len(visible_before),
        "visible_inside_inactive_frames": len(visible_inside_inactive),
        "visible_after_hazard_frames": len(visible_after),
        "hazard_frame_visibility_ratio": _ratio(len(active_visible), len(active_frames)),
        "has_privileged_collision_hazard": bool(active_frames),
        "has_visible_hazard_frame": bool(active_visible),
        "camera_stats": camera_stats,
        "representative_frames": representative_frames,
        "valid": not errors,
        "errors": errors,
    }


def audit_pedestrian_hazard_visibility(
    dataset_root,
    inventory_path,
    cameras=DEFAULT_CAMERAS,
    pedestrian_tag=12,
    minimum_pixels=10,
    hazard_source_path=None,
):
    """Return deterministic privileged-hazard and RGB-visibility facts."""
    dataset_root = Path(dataset_root)
    if not dataset_root.is_dir():
        raise PedestrianVisibilityAuditError(
            f"dataset root is not a directory: {dataset_root}"
        )
    cameras = tuple(cameras)
    if not cameras or len(set(cameras)) != len(cameras):
        raise PedestrianVisibilityAuditError(
            "cameras must be a non-empty list without duplicates"
        )
    if pedestrian_tag < 0 or pedestrian_tag > 255:
        raise PedestrianVisibilityAuditError("pedestrian_tag must be in [0, 255]")
    if minimum_pixels <= 0:
        raise PedestrianVisibilityAuditError("minimum_pixels must be positive")

    inventory = _load_inventory(dataset_root, inventory_path)
    sequence_reports = [
        _audit_sequence(
            relative,
            route,
            cameras,
            pedestrian_tag,
            minimum_pixels,
        )
        for relative, route in inventory["sequences"]
    ]
    errors = [error for item in sequence_reports for error in item["errors"]]

    camera_totals = {
        camera: {
            key: sum(item["camera_stats"][camera][key] for item in sequence_reports)
            for key in (
                "pixels",
                "masks_with_any",
                "qualified_masks",
            )
        }
        for camera in cameras
    }
    for camera in cameras:
        camera_totals[camera]["maximum_pixels_per_mask"] = max(
            item["camera_stats"][camera]["maximum_pixels_per_mask"]
            for item in sequence_reports
        )

    by_town = defaultdict(Counter)
    signatures = defaultdict(list)
    town_route_groups = set()
    for item in sequence_reports:
        by_town[item["town"]]["sequences"] += 1
        by_town[item["town"]]["hazard_sequences"] += int(
            item["has_privileged_collision_hazard"]
        )
        by_town[item["town"]]["visible_hazard_sequences"] += int(
            item["has_visible_hazard_frame"]
        )
        by_town[item["town"]]["hazard_frames"] += item["hazard_frames"]
        by_town[item["town"]]["hazard_visible_frames"] += item[
            "hazard_visible_frames"
        ]
        signatures[item["trajectory_visibility_signature"]].append(item)
        if item["town_route_group"] is not None:
            town_route_groups.add(item["town_route_group"])

    signature_groups = []
    for signature, members in sorted(signatures.items()):
        representative = min(
            members,
            key=lambda item: (
                item["weather"] != 0,
                item["weather"] if item["weather"] is not None else 999,
                item["sequence"],
            ),
        )
        signature_groups.append(
            {
                "signature": signature,
                "member_count": len(members),
                "town_route_groups": sorted(
                    {
                        item["town_route_group"]
                        for item in members
                        if item["town_route_group"] is not None
                    }
                ),
                "representative_sequence": representative["sequence"],
                "representative_frames": representative["representative_frames"],
                "members": [item["sequence"] for item in members],
            }
        )

    sequence_count = len(sequence_reports)
    hazard_sequences = sum(
        item["has_privileged_collision_hazard"] for item in sequence_reports
    )
    visible_hazard_sequences = sum(
        item["has_visible_hazard_frame"] for item in sequence_reports
    )
    hazard_frames = sum(item["hazard_frames"] for item in sequence_reports)
    hazard_visible_frames = sum(
        item["hazard_visible_frames"] for item in sequence_reports
    )
    source = None
    if hazard_source_path is not None:
        source_path = Path(hazard_source_path).resolve()
        if not source_path.is_file():
            raise PedestrianVisibilityAuditError(
                f"hazard source is not a file: {source_path}"
            )
        source = {"path": str(source_path), "sha256": _sha256(source_path)}

    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "dataset_root": str(dataset_root.resolve()),
        "inventory": {
            "path": str(inventory["path"]),
            "sha256": inventory["sha256"],
            "declared_total_scanned": inventory["declared_total_scanned"],
            "declared_pedestrian_sequences": inventory[
                "declared_pedestrian_sequences"
            ],
        },
        "hazard_contract": {
            "field": "is_pedestrian_present",
            "meaning": (
                "walker actor IDs whose projected motion intersects the ego "
                "vehicle trajectory according to AutoPilot"
            ),
            "source": source,
            "privileged_only": True,
            "model_input": False,
        },
        "cameras": list(cameras),
        "pedestrian_source_tag": pedestrian_tag,
        "minimum_pixels_per_mask": minimum_pixels,
        "summary": {
            "sequence_count": sequence_count,
            "town_route_group_count": len(town_route_groups),
            "trajectory_visibility_signature_count": len(signatures),
            "sequences_with_privileged_collision_hazard": hazard_sequences,
            "sequences_with_visible_hazard_frame": visible_hazard_sequences,
            "visible_hazard_sequence_ratio": _ratio(
                visible_hazard_sequences, sequence_count
            ),
            "all_sequences_have_privileged_collision_hazard": (
                hazard_sequences == sequence_count
            ),
            "all_sequences_have_visible_hazard_frame": (
                visible_hazard_sequences == sequence_count
            ),
            "measurement_frames": sum(
                item["measurement_frames"] for item in sequence_reports
            ),
            "qualified_visible_frames": sum(
                item["qualified_visible_frames"] for item in sequence_reports
            ),
            "hazard_frames": hazard_frames,
            "hazard_visible_frames": hazard_visible_frames,
            "hazard_frame_visibility_ratio": _ratio(
                hazard_visible_frames, hazard_frames
            ),
        },
        "camera_stats": camera_totals,
        "by_town": {
            town: {
                **dict(sorted(counts.items())),
                "hazard_frame_visibility_ratio": _ratio(
                    counts["hazard_visible_frames"], counts["hazard_frames"]
                ),
            }
            for town, counts in sorted(by_town.items())
        },
        "signature_groups": signature_groups,
        "sequences": sequence_reports,
        "valid": not errors,
        "errors": errors,
    }


def _parse_cameras(value):
    cameras = tuple(item.strip() for item in value.split(",") if item.strip())
    if not cameras:
        raise argparse.ArgumentTypeError("at least one camera is required")
    return cameras


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether pedestrian-visible sequences contain privileged "
            "collision hazards and whether those hazards are visible in RGB masks."
        )
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument(
        "--cameras", type=_parse_cameras, default=DEFAULT_CAMERAS
    )
    parser.add_argument("--pedestrian-tag", type=int, default=12)
    parser.add_argument("--minimum-pixels", type=int, default=10)
    parser.add_argument("--hazard-source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-all-hazard", action="store_true")
    parser.add_argument("--require-visible-hazard", action="store_true")
    args = parser.parse_args(argv)

    report = audit_pedestrian_hazard_visibility(
        args.dataset_root,
        args.inventory,
        cameras=args.cameras,
        pedestrian_tag=args.pedestrian_tag,
        minimum_pixels=args.minimum_pixels,
        hazard_source_path=args.hazard_source,
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)

    summary = report["summary"]
    rejected = not report["valid"]
    rejected = rejected or (
        args.require_all_hazard
        and not summary["all_sequences_have_privileged_collision_hazard"]
    )
    rejected = rejected or (
        args.require_visible_hazard
        and not summary["all_sequences_have_visible_hazard_frame"]
    )
    return 2 if rejected else 0


if __name__ == "__main__":
    raise SystemExit(main())
