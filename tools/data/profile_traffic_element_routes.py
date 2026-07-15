#!/usr/bin/env python3
"""Profile dense CARLA routes for traffic-control coverage."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
for dependency in (
    REPO_ROOT / "carla" / "PythonAPI",
    REPO_ROOT / "carla" / "PythonAPI" / "carla",
    REPO_ROOT / "leaderboard",
    REPO_ROOT / "scenario_runner",
):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))


def _point_array(points, name):
    values = np.asarray(list(points), dtype=np.float64)
    if values.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"{name} must contain 2D points")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite coordinates")
    return values


def _remove_consecutive_duplicates(points):
    if len(points) < 2:
        return points
    changed = np.any(np.diff(points, axis=0) != 0.0, axis=1)
    keep = np.concatenate(([True], changed))
    return points[keep]


def _distance_summary(route, actors, relevant_radius_m, nearby_radius_m):
    if not len(actors):
        return {
            "nearby": 0,
            "relevant": 0,
            "minimum_distance_m": None,
            "route_minimum_distances": np.full(len(route), np.inf),
        }
    distances = np.linalg.norm(
        route[:, np.newaxis, :] - actors[np.newaxis, :, :], axis=2
    )
    actor_minimum = np.min(distances, axis=0)
    return {
        "nearby": int(np.count_nonzero(actor_minimum <= nearby_radius_m)),
        "relevant": int(np.count_nonzero(actor_minimum <= relevant_radius_m)),
        "minimum_distance_m": float(np.min(actor_minimum)),
        "route_minimum_distances": np.min(distances, axis=1),
    }


def score_dense_route(
    route_points,
    traffic_lights,
    stop_signs,
    relevant_radius_m=30.0,
    nearby_radius_m=80.0,
):
    """Score a dense 2D route against traffic-element trigger centers."""
    relevant_radius_m = float(relevant_radius_m)
    nearby_radius_m = float(nearby_radius_m)
    if not math.isfinite(relevant_radius_m) or relevant_radius_m < 0.0:
        raise ValueError("relevant_radius_m must be finite and nonnegative")
    if not math.isfinite(nearby_radius_m) or nearby_radius_m < relevant_radius_m:
        raise ValueError(
            "nearby_radius_m must be finite and at least relevant_radius_m"
        )

    route = _remove_consecutive_duplicates(
        _point_array(route_points, "route_points")
    )
    if not len(route):
        raise ValueError("route_points must not be empty")
    lights = _point_array(traffic_lights, "traffic_lights")
    stops = _point_array(stop_signs, "stop_signs")
    light_summary = _distance_summary(
        route, lights, relevant_radius_m, nearby_radius_m
    )
    stop_summary = _distance_summary(
        route, stops, relevant_radius_m, nearby_radius_m
    )
    nearest_infrastructure = np.minimum(
        light_summary["route_minimum_distances"],
        stop_summary["route_minimum_distances"],
    )

    return {
        "dense_route_points": int(len(route)),
        "traffic_light_actors": int(len(lights)),
        "stop_sign_actors": int(len(stops)),
        "nearby_traffic_lights": light_summary["nearby"],
        "nearby_stop_signs": stop_summary["nearby"],
        "relevant_traffic_lights": light_summary["relevant"],
        "relevant_stop_signs": stop_summary["relevant"],
        "minimum_traffic_light_distance_m": light_summary[
            "minimum_distance_m"
        ],
        "minimum_stop_sign_distance_m": stop_summary["minimum_distance_m"],
        "hard_negative_points": int(
            np.count_nonzero(nearest_infrastructure > relevant_radius_m)
        ),
        "relevant_radius_m": relevant_radius_m,
        "nearby_radius_m": nearby_radius_m,
    }


def _optimized_town_name(town):
    town = str(town).split("/")[-1]
    return town if town.endswith("_Opt") else f"{town}_Opt"


def _route_id(config):
    prefix = "RouteScenario_"
    return config.name[len(prefix) :] if config.name.startswith(prefix) else config.name


def _route_sort_key(config):
    route_id = _route_id(config)
    return (0, int(route_id)) if route_id.isdigit() else (1, route_id)


def _trigger_centers(actors):
    centers = []
    for actor in actors:
        center = actor.get_transform().transform(actor.trigger_volume.location)
        centers.append((float(center.x), float(center.y)))
    return centers


def profile_routes(
    routes_file,
    scenario_file,
    host="127.0.0.1",
    port=2400,
    timeout_s=60.0,
    route_ids=None,
    relevant_radius_m=30.0,
    nearby_radius_m=80.0,
):
    """Load optimized maps and profile selected routes through CARLA."""
    import carla
    from leaderboard.utils.route_manipulation import interpolate_trajectory
    from leaderboard.utils.route_parser import RouteParser

    configs = RouteParser.parse_routes_file(
        str(routes_file), str(scenario_file)
    )
    requested = {str(value) for value in route_ids or []}
    if requested:
        configs = [config for config in configs if _route_id(config) in requested]
        found = {_route_id(config) for config in configs}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"route IDs absent from {routes_file}: {missing}")

    grouped = defaultdict(list)
    for config in configs:
        grouped[config.town].append(config)
    client = carla.Client(str(host), int(port))
    client.set_timeout(float(timeout_s))
    summaries = []
    for town in sorted(grouped):
        loaded_map = _optimized_town_name(town)
        world = client.load_world(loaded_map)
        actors = world.get_actors()
        traffic_lights = _trigger_centers(
            actors.filter("traffic.traffic_light*")
        )
        stop_signs = _trigger_centers(actors.filter("traffic.stop*"))
        for config in sorted(grouped[town], key=_route_sort_key):
            _, dense_route = interpolate_trajectory(
                world, config.trajectory, hop_resolution=1.0
            )
            route_points = [
                (transform.location.x, transform.location.y)
                for transform, _road_option in dense_route
            ]
            score = score_dense_route(
                route_points,
                traffic_lights,
                stop_signs,
                relevant_radius_m,
                nearby_radius_m,
            )
            summaries.append(
                {
                    "route_id": _route_id(config),
                    "route_name": config.name,
                    "town": config.town,
                    "loaded_map": loaded_map,
                    **score,
                }
            )
    return {
        "route_file": str(Path(routes_file)),
        "scenario_file": str(Path(scenario_file)),
        "host": str(host),
        "port": int(port),
        "routes": summaries,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Profile CARLA routes for traffic-element coverage"
    )
    parser.add_argument("routes_file")
    parser.add_argument(
        "--scenarios",
        default=str(REPO_ROOT / "leaderboard/data/42routes/42scenarios.json"),
    )
    parser.add_argument("--route-id", action="append", dest="route_ids")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2400)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--relevant-radius", type=float, default=30.0)
    parser.add_argument("--nearby-radius", type=float, default=80.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    report = profile_routes(
        args.routes_file,
        args.scenarios,
        host=args.host,
        port=args.port,
        timeout_s=args.timeout,
        route_ids=args.route_ids,
        relevant_radius_m=args.relevant_radius,
        nearby_radius_m=args.nearby_radius,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
