"""
[INPUT]: 依赖 B0/V D7 run manifest、每个 attempt 的 Leaderboard result 与可选 control.csv，要求路线、seed 和 pipeline-valid 契约完全配对。
[OUTPUT]: 提供状态/指标差值、首碰撞时空位置及起步后转向、车道偏移、路口分数的连续控制帧 JSON 诊断。
[POS]: tools/evaluation 的 H1 失败归因器，在汇总分数之后定位 V 的闭环退化发生在路线何处，不修改 runner、模型或原始结果。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


REPORT_SCHEMA_VERSION = 1
CONTROL_SUMMARY_FIELDS = (
    "mean_abs_steer",
    "max_abs_steer",
    "mean_lane_offset",
    "mean_abs_lane_offset",
    "max_abs_lane_offset",
    "mean_net_is_junction",
    "junction_positive_fraction",
    "mean_aux_junction",
    "aux_junction_positive_fraction",
    "mean_junction_consensus",
    "raw_aux_junction_disagreement_fraction",
    "mean_abs_ctrl0_x",
    "mean_abs_ctrl1_x",
    "mean_abs_lane_center_correction",
    "lane_center_correction_saturation_fraction",
)
TIME_PATTERN = re.compile(r"at time=(?P<time>-?\d+(?:\.\d+)?)")
LOCATION_PATTERN = re.compile(
    r"at \(x=(?P<x>-?\d+(?:\.\d+)?), "
    r"y=(?P<y>-?\d+(?:\.\d+)?), z=(?P<z>-?\d+(?:\.\d+)?)\)"
)
COLLISION_CATEGORIES = (
    "collisions_pedestrian",
    "collisions_vehicle",
    "collisions_layout",
)


class VisualD7FailureAnalysisError(ValueError):
    """Raised when paired D7 evidence is incomplete or incomparable."""


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path, label):
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VisualD7FailureAnalysisError(
            f"unable to read {label} JSON {path}: {exc}"
        ) from exc


def _load_manifest(path, variant):
    path = Path(path).resolve()
    payload = _read_json(path, f"{variant} manifest")
    attempts = payload.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise VisualD7FailureAnalysisError(f"{variant} manifest has no attempts")
    by_id = {}
    for attempt in attempts:
        attempt_id = attempt.get("attempt_id") if isinstance(attempt, dict) else None
        if not isinstance(attempt_id, str) or not attempt_id:
            raise VisualD7FailureAnalysisError(
                f"{variant} manifest contains an invalid attempt_id"
            )
        if attempt_id in by_id:
            raise VisualD7FailureAnalysisError(
                f"{variant} manifest duplicates attempt {attempt_id}"
            )
        if not attempt.get("pipeline_valid"):
            raise VisualD7FailureAnalysisError(
                f"{variant} attempt is pipeline-invalid: {attempt_id}"
            )
        by_id[attempt_id] = attempt
    return {
        "path": path,
        "sha256": _sha256(path),
        "attempts": by_id,
    }


def _result_record(manifest_path, attempt_id):
    result_path = manifest_path.parent / "attempts" / attempt_id / "leaderboard_result.json"
    payload = _read_json(result_path, f"attempt {attempt_id} result")
    records = payload.get("_checkpoint", {}).get("records")
    if not isinstance(records, list) or len(records) != 1:
        raise VisualD7FailureAnalysisError(
            f"attempt {attempt_id} result must contain exactly one route record"
        )
    record = records[0]
    infractions = record.get("infractions")
    if not isinstance(infractions, dict):
        raise VisualD7FailureAnalysisError(
            f"attempt {attempt_id} result has no infraction dictionary"
        )
    return result_path, record


def _event(category, message):
    time_match = TIME_PATTERN.search(message)
    location_match = LOCATION_PATTERN.search(message)
    return {
        "category": category,
        "message": message,
        "game_time": float(time_match.group("time")) if time_match else None,
        "location": (
            {
                axis: float(location_match.group(axis))
                for axis in ("x", "y", "z")
            }
            if location_match
            else None
        ),
    }


def _extract_events(record):
    result = []
    for category, messages in sorted(record["infractions"].items()):
        if not isinstance(messages, list):
            raise VisualD7FailureAnalysisError(
                f"infraction category {category} must contain a list"
            )
        result.extend(_event(category, message) for message in messages)
    return result


def _first_timed_collision(events, category=None):
    candidates = [
        item
        for item in events
        if item["category"] in COLLISION_CATEGORIES
        and item["game_time"] is not None
        and (category is None or item["category"] == category)
    ]
    return min(candidates, key=lambda item: item["game_time"]) if candidates else None


def _metrics(attempt):
    result = attempt.get("leaderboard_result")
    if not isinstance(result, dict) or not result.get("valid"):
        raise VisualD7FailureAnalysisError(
            f"attempt {attempt.get('attempt_id')} lacks a valid leaderboard result"
        )
    scores = result.get("scores")
    if not isinstance(scores, dict):
        raise VisualD7FailureAnalysisError(
            f"attempt {attempt.get('attempt_id')} lacks scores"
        )
    return {
        "driving_score": float(scores["score_composed"]),
        "route_completion": float(scores["score_route"]),
        "infraction_score": float(scores["score_penalty"]),
    }


def _mean(values):
    return sum(values) / len(values) if values else None


def _control_csv_path(manifest_path, attempt_id, required):
    pattern_root = manifest_path.parent / "attempts" / attempt_id / "sensor_data"
    matches = sorted(pattern_root.glob("*/control.csv"))
    if len(matches) > 1:
        raise VisualD7FailureAnalysisError(
            f"attempt {attempt_id} has multiple control.csv files"
        )
    if not matches:
        if required:
            raise VisualD7FailureAnalysisError(
                f"attempt {attempt_id} has no control.csv"
            )
        return None
    return matches[0]


def _float_field(row, field, path, line_number):
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise VisualD7FailureAnalysisError(
            f"invalid {field} at {path}:{line_number}"
        ) from exc


def _control_summary(path, moving_speed_threshold, early_control_steps):
    if path is None:
        return None
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required_fields = {
            "step",
            "speed",
            "steer",
            "lane_offset",
            "net_is_junction",
            "aux_junction",
            "ctrl0_x",
            "ctrl1_x",
            "lane_center_correction",
        }
        missing = sorted(required_fields - set(reader.fieldnames or ()))
        if missing:
            raise VisualD7FailureAnalysisError(
                f"control CSV {path} misses fields: {missing}"
            )
        rows = []
        for line_number, raw in enumerate(reader, start=2):
            rows.append(
                {
                    field: _float_field(raw, field, path, line_number)
                    for field in required_fields
                }
            )
    if not rows:
        raise VisualD7FailureAnalysisError(f"control CSV {path} has no rows")
    steps = [int(row["step"]) for row in rows]
    if steps != sorted(steps) or len(steps) != len(set(steps)):
        raise VisualD7FailureAnalysisError(
            f"control CSV {path} must have unique monotonic steps"
        )
    moving_index = next(
        (
            index
            for index, row in enumerate(rows)
            if row["speed"] >= moving_speed_threshold
        ),
        None,
    )
    if moving_index is None:
        raise VisualD7FailureAnalysisError(
            f"control CSV {path} never reaches moving speed threshold"
        )
    window = rows[moving_index : moving_index + early_control_steps]
    abs_values = {
        field: [abs(row[field]) for row in window]
        for field in (
            "steer",
            "lane_offset",
            "ctrl0_x",
            "ctrl1_x",
            "lane_center_correction",
        )
    }
    summary = {
        "source": str(path.resolve()),
        "source_sha256": _sha256(path),
        "total_control_rows": len(rows),
        "moving_control_rows": sum(
            row["speed"] >= moving_speed_threshold for row in rows
        ),
        "movement_start_step": steps[moving_index],
        "early_window_first_step": int(window[0]["step"]),
        "early_window_last_step": int(window[-1]["step"]),
        "early_window_rows": len(window),
        "mean_abs_steer": _mean(abs_values["steer"]),
        "max_abs_steer": max(abs_values["steer"]),
        "mean_lane_offset": _mean([row["lane_offset"] for row in window]),
        "mean_abs_lane_offset": _mean(abs_values["lane_offset"]),
        "max_abs_lane_offset": max(abs_values["lane_offset"]),
        "mean_net_is_junction": _mean(
            [row["net_is_junction"] for row in window]
        ),
        "junction_positive_fraction": _mean(
            [row["net_is_junction"] >= 0.5 for row in window]
        ),
        "mean_aux_junction": _mean([row["aux_junction"] for row in window]),
        "aux_junction_positive_fraction": _mean(
            [row["aux_junction"] >= 0.5 for row in window]
        ),
        "mean_junction_consensus": _mean(
            [min(row["net_is_junction"], row["aux_junction"]) for row in window]
        ),
        "raw_aux_junction_disagreement_fraction": _mean(
            [
                abs(row["net_is_junction"] - row["aux_junction"]) >= 0.5
                for row in window
            ]
        ),
        "mean_abs_ctrl0_x": _mean(abs_values["ctrl0_x"]),
        "mean_abs_ctrl1_x": _mean(abs_values["ctrl1_x"]),
        "mean_abs_lane_center_correction": _mean(
            abs_values["lane_center_correction"]
        ),
        "lane_center_correction_saturation_fraction": _mean(
            [
                value >= 0.249
                for value in abs_values["lane_center_correction"]
            ]
        ),
    }
    for threshold in (0.5, 1.0):
        row = next(
            (item for item in rows[moving_index:] if abs(item["lane_offset"]) >= threshold),
            None,
        )
        summary[f"first_abs_lane_offset_ge_{threshold:g}_step"] = (
            int(row["step"]) if row else None
        )
    return summary


def analyze_visual_d7_failures(
    b0_manifest_path,
    v_manifest_path,
    early_collision_seconds=10.0,
    early_control_steps=40,
    moving_speed_threshold=0.5,
    require_control=False,
):
    """Return deterministic paired failure facts for two valid D7 matrices."""
    if early_collision_seconds <= 0:
        raise VisualD7FailureAnalysisError(
            "early_collision_seconds must be positive"
        )
    if early_control_steps <= 0:
        raise VisualD7FailureAnalysisError("early_control_steps must be positive")
    if moving_speed_threshold < 0:
        raise VisualD7FailureAnalysisError(
            "moving_speed_threshold must be non-negative"
        )
    variants = {
        "b0": _load_manifest(b0_manifest_path, "b0"),
        "v": _load_manifest(v_manifest_path, "v"),
    }
    b0_ids = set(variants["b0"]["attempts"])
    v_ids = set(variants["v"]["attempts"])
    if b0_ids != v_ids:
        raise VisualD7FailureAnalysisError(
            f"attempt matrices differ: b0-only={sorted(b0_ids-v_ids)}, "
            f"v-only={sorted(v_ids-b0_ids)}"
        )

    pairs = []
    per_route = defaultdict(list)
    status_transitions = Counter()
    for attempt_id in sorted(
        b0_ids,
        key=lambda item: (
            variants["b0"]["attempts"][item]["route_id"],
            variants["b0"]["attempts"][item]["traffic_manager_seed"],
        ),
    ):
        b0_attempt = variants["b0"]["attempts"][attempt_id]
        v_attempt = variants["v"]["attempts"][attempt_id]
        for field in ("route_id", "traffic_manager_seed"):
            if b0_attempt.get(field) != v_attempt.get(field):
                raise VisualD7FailureAnalysisError(
                    f"attempt {attempt_id} differs in {field}"
                )

        variant_facts = {}
        for variant, attempt in (("b0", b0_attempt), ("v", v_attempt)):
            result_path, record = _result_record(
                variants[variant]["path"], attempt_id
            )
            events = _extract_events(record)
            control_path = _control_csv_path(
                variants[variant]["path"], attempt_id, require_control
            )
            variant_facts[variant] = {
                "status": attempt["leaderboard_result"]["status"],
                "metrics": _metrics(attempt),
                "duration_game": float(record.get("meta", {}).get("duration_game", 0)),
                "events": events,
                "first_collision": _first_timed_collision(events),
                "first_layout_collision": _first_timed_collision(
                    events, "collisions_layout"
                ),
                "result": str(result_path),
                "result_sha256": _sha256(result_path),
                "early_control": _control_summary(
                    control_path,
                    moving_speed_threshold=moving_speed_threshold,
                    early_control_steps=early_control_steps,
                ),
            }
        delta = {
            key: variant_facts["v"]["metrics"][key]
            - variant_facts["b0"]["metrics"][key]
            for key in ("driving_score", "route_completion", "infraction_score")
        }
        control_delta = None
        if all(variant_facts[key]["early_control"] for key in ("b0", "v")):
            control_delta = {
                field: (
                    variant_facts["v"]["early_control"][field]
                    - variant_facts["b0"]["early_control"][field]
                )
                for field in CONTROL_SUMMARY_FIELDS
            }
        b0_layout = variant_facts["b0"]["first_layout_collision"]
        v_layout = variant_facts["v"]["first_layout_collision"]
        early_layout_regression = bool(
            v_layout
            and v_layout["game_time"] <= early_collision_seconds
            and (not b0_layout or b0_layout["game_time"] > early_collision_seconds)
        )
        pair = {
            "attempt_id": attempt_id,
            "route_id": b0_attempt["route_id"],
            "seed": b0_attempt["traffic_manager_seed"],
            "b0": variant_facts["b0"],
            "v": variant_facts["v"],
            "v_minus_b0": delta,
            "early_control_v_minus_b0": control_delta,
            "status_transition": (
                f"{variant_facts['b0']['status']} -> {variant_facts['v']['status']}"
            ),
            "early_layout_collision_regression": early_layout_regression,
        }
        pairs.append(pair)
        per_route[pair["route_id"]].append(pair)
        status_transitions[pair["status_transition"]] += 1

    route_reports = []
    repeatable_early_routes = []
    for route_id, route_pairs in sorted(per_route.items()):
        early_count = sum(
            item["early_layout_collision_regression"] for item in route_pairs
        )
        route_report = {
            "route_id": route_id,
            "pair_count": len(route_pairs),
            "mean_v_minus_b0": {
                metric: _mean(
                    [item["v_minus_b0"][metric] for item in route_pairs]
                )
                for metric in (
                    "driving_score",
                    "route_completion",
                    "infraction_score",
                )
            },
            "early_layout_collision_regression_pairs": early_count,
            "mean_early_control_v_minus_b0": {
                field: _mean(
                    [
                        item["early_control_v_minus_b0"][field]
                        for item in route_pairs
                        if item["early_control_v_minus_b0"] is not None
                    ]
                )
                for field in CONTROL_SUMMARY_FIELDS
            },
            "status_transitions": dict(
                sorted(Counter(item["status_transition"] for item in route_pairs).items())
            ),
        }
        route_reports.append(route_report)
        if early_count == len(route_pairs):
            repeatable_early_routes.append(route_id)

    driving_deltas = [item["v_minus_b0"]["driving_score"] for item in pairs]
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "contract": {
            "delta_direction": "V minus B0",
            "early_collision_seconds": early_collision_seconds,
            "attempt_count": len(pairs),
            "early_control_steps_after_movement": early_control_steps,
            "moving_speed_threshold": moving_speed_threshold,
            "control_csv_required": require_control,
        },
        "sources": {
            variant: {
                "manifest": str(data["path"]),
                "manifest_sha256": data["sha256"],
            }
            for variant, data in variants.items()
        },
        "summary": {
            "driving_score_improved_pairs": sum(value > 0 for value in driving_deltas),
            "driving_score_tied_pairs": sum(value == 0 for value in driving_deltas),
            "driving_score_worse_pairs": sum(value < 0 for value in driving_deltas),
            "early_layout_collision_regression_pairs": sum(
                item["early_layout_collision_regression"] for item in pairs
            ),
            "routes_with_repeatable_early_layout_regression": repeatable_early_routes,
            "largest_route_completion_regressions": [
                {
                    "attempt_id": item["attempt_id"],
                    "route_id": item["route_id"],
                    "seed": item["seed"],
                    "v_minus_b0": item["v_minus_b0"]["route_completion"],
                }
                for item in sorted(
                    pairs, key=lambda item: item["v_minus_b0"]["route_completion"]
                )[:5]
            ],
            "status_transitions": dict(sorted(status_transitions.items())),
        },
        "per_route": route_reports,
        "pairs": pairs,
        "valid": True,
        "errors": [],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Diagnose paired B0/V D7 route failures from immutable results."
    )
    parser.add_argument("--b0-manifest", type=Path, required=True)
    parser.add_argument("--v-manifest", type=Path, required=True)
    parser.add_argument("--early-collision-seconds", type=float, default=10.0)
    parser.add_argument("--early-control-steps", type=int, default=40)
    parser.add_argument("--moving-speed-threshold", type=float, default=0.5)
    parser.add_argument("--require-control", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = analyze_visual_d7_failures(
        args.b0_manifest,
        args.v_manifest,
        early_collision_seconds=args.early_collision_seconds,
        early_control_steps=args.early_control_steps,
        moving_speed_threshold=args.moving_speed_threshold,
        require_control=args.require_control,
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is None:
        print(serialized)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
