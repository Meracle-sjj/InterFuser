# Leaderboard-Aligned Stop Target Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace STOP-sign supervision with route-associated, Leaderboard-aligned traffic-light stop targets and aligned RGB, depth, semantic, and lidar evidence, then validate the result with a bounded four-route collection.

**Architecture:** A new pure `leaderboard_stop_targets.py` module mirrors `RunningRedLightTest` geometry and owns route association, signed distance, stop-pose, and corridor construction. `traffic_element_labels.py` emits traffic-light schema v2, while `traffic_element_projection.py` emits evidence schema v3 and remains responsible for camera/lidar transforms. Collector wiring supplies the cached BehaviorAgent route and aligned sensors; separate audits, overlays, and a live-CARLA parity tool enforce provenance and prohibit every STOP-sign label.

**Tech Stack:** Python 3.10, NumPy, OpenCV, PIL, CARLA 0.9.16 Python API, Scenario Runner/Leaderboard 1.0, `unittest`, JSON, Bash.

**Execution root:** `/data/shijj/interfuser_origin` on `ghbserver02-frpMe`, branch `codex/fix-background-traffic`.

---

## File Map

- Create `leaderboard/team_code/leaderboard_stop_targets.py`: evaluator-parity geometry, route association, stable target IDs, recommended stop pose, and route corridor.
- Create `tests/test_leaderboard_stop_targets.py`: pure fake-waypoint tests for evaluator geometry and route targets.
- Modify `leaderboard/team_code/traffic_element_labels.py`: schema v2, traffic-light-only labels, and stop-target integration.
- Modify `leaderboard/team_code/interfuser_data_collector.py`: cache the dense planner route, disable STOP-sign behavior, pass frame/route/lidar context, and write both schemas atomically.
- Modify `leaderboard/team_code/interfuser_collector_complete.py`: remove STOP actor discovery and STOP measurement fields.
- Modify `tests/test_traffic_element_labels.py`: replace v1 stop-sign expectations with v2 stop-target and no-STOP assertions.
- Modify `tests/test_traffic_element_collector.py`: verify route/lidar wiring and atomic v2/v3 records.
- Create `tests/test_collector_stop_exclusion.py`: verify collector options and side-channel JSON fields exclude STOP labels.
- Modify `leaderboard/team_code/traffic_element_projection.py`: evidence schema v3, stop-target camera projection, lidar transforms, corridor point statistics, and optional painted-line candidates.
- Modify `tests/test_traffic_element_projection.py`: v3 camera/lidar/candidate tests.
- Rewrite `tools/data/audit_traffic_element_labels.py`: validate v2 targets and scan all generated JSON for forbidden STOP labels.
- Rewrite `tools/data/audit_traffic_element_views.py`: validate v3 camera/lidar evidence and per-town summaries.
- Modify `tests/test_audit_traffic_element_labels.py`: v2 and forbidden-side-channel tests.
- Modify `tests/test_audit_traffic_element_views.py`: v3 evidence validation tests.
- Modify `tools/data/render_traffic_element_overlays.py`: draw trigger, evaluator boundary, stop pose, corridor, candidate line, and lamp boxes only.
- Modify `tests/test_render_traffic_element_overlays.py`: deterministic v3 selection and drawing tests.
- Create `tools/data/apply_painted_line_reviews.py`: apply explicit manual candidate decisions atomically without inferring verification.
- Create `tests/test_apply_painted_line_reviews.py`: manifest validation and candidate-promotion tests.
- Modify `tools/data/profile_traffic_element_routes.py`: profile traffic lights only.
- Modify `tests/test_profile_traffic_element_routes.py`: remove every STOP-sign input and metric.
- Create `tools/data/check_leaderboard_stop_target_geometry.py`: compare collector geometry with an independent `RunningRedLightTest` reproduction on live CARLA towns.
- Create `tests/test_check_leaderboard_stop_target_geometry.py`: comparison and mismatch tests.
- Modify `tools/data/run_traffic_element_small_batch.sh`: run geometry checks and v2/v3 acceptance without STOP metrics.
- Rewrite `docs/traffic_element_label_schema.md`: traffic-element schema v2.
- Rewrite `docs/traffic_element_image_label_schema.md`: evidence schema v3.

Do not modify `leaderboard/team_code/interfuser_agent.py`,
`leaderboard/team_code/interfuser_controller.py`, Scenario Runner criteria, or
model-training code. Do not stage existing data, results, caches, or unrelated
working-tree changes.

### Task 1: Mirror Leaderboard infraction-boundary geometry

**Files:**
- Create: `leaderboard/team_code/leaderboard_stop_targets.py`
- Create: `tests/test_leaderboard_stop_targets.py`

- [ ] **Step 1: Write failing evaluator-geometry tests**

Create fake `Location`, `Rotation`, `Transform`, `Trigger`, `Waypoint`, `Map`,
and `TrafficLight` classes in the test file. The waypoint fake must expose both
`is_intersection` and `is_junction`, retain the distance passed to `next()`, and
allow zero, one, or multiple successors. Add these tests:

```python
from team_code.leaderboard_stop_targets import (
    BOUNDARY_HALF_LANE_RATIO,
    BOUNDARY_STEP_M,
    GEOMETRY_SOURCE,
    advance_to_infraction_boundary,
    boundary_from_waypoint,
    sample_trigger_lane_waypoints,
)


class LeaderboardBoundaryTests(unittest.TestCase):
    def test_samples_ninety_percent_trigger_extent_at_one_meter(self):
        light = FakeTrafficLight(extent_x=2.0, yaw=0.0)
        world_map = FakeMap(default_lane=FakeWaypoint(7, -1, 0.0))

        samples = sample_trigger_lane_waypoints(light, world_map)

        self.assertEqual([round(x, 1) for x in world_map.queries_x], [-1.8, -0.8, 0.2, 1.2])
        self.assertEqual(len(samples), 1)  # consecutive road/lane duplicates collapse

    def test_advances_by_half_meter_to_last_non_intersection_waypoint(self):
        start, middle, boundary, junction = linked_waypoints(
            [0.0, 0.5, 1.0, 1.5], intersection_index=3
        )

        result = advance_to_infraction_boundary(start)

        self.assertEqual(result["status"], "valid")
        self.assertIs(result["waypoint"], boundary)
        self.assertEqual(result["steps"], 2)
        self.assertEqual(start.next_distances, [BOUNDARY_STEP_M])

    def test_branch_is_unknown_even_though_first_branch_matches_evaluator(self):
        start = FakeWaypoint(7, -1, 0.0)
        first = FakeWaypoint(7, -1, 0.5, is_intersection=True)
        second = FakeWaypoint(8, -1, 0.5, is_intersection=True)
        start.successors = [first, second]

        result = advance_to_infraction_boundary(start)

        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["unknown_reason"], "waypoint_branch")
        self.assertIs(result["waypoint"], start)

    def test_boundary_uses_evaluator_lane_width_ratio(self):
        waypoint = FakeWaypoint(7, -1, 10.0, y=2.0, yaw=0.0, lane_width=4.0)

        boundary = boundary_from_waypoint(waypoint)

        self.assertEqual(BOUNDARY_HALF_LANE_RATIO, 0.4)
        self.assertEqual(GEOMETRY_SOURCE, "scenario_runner_running_red_light_test_v1")
        self.assertAlmostEqual(boundary["left_endpoint"]["y"], 3.6)
        self.assertAlmostEqual(boundary["right_endpoint"]["y"], 0.4)
```

Also test `waypoint_loop`, `intersection_not_found`, an initially intersecting
waypoint, and preservation of road/section/lane/`s` metadata.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
export PYTHONPATH=$PWD/interfuser:$PWD/carla/PythonAPI:$PWD/carla/PythonAPI/carla:$PWD/leaderboard:$PWD/leaderboard/team_code:$PWD/scenario_runner:$PWD
/data1/shijj/conda_envs/interfuser_origin/bin/python -m unittest discover \
  -s tests -p 'test_leaderboard_stop_targets.py' -v
```

Expected: `ModuleNotFoundError` for `team_code.leaderboard_stop_targets`.

- [ ] **Step 3: Implement the evaluator-parity API**

Create the module with these versioned constants and public functions:

```python
import math


GEOMETRY_SOURCE = "scenario_runner_running_red_light_test_v1"
TRIGGER_SOURCE = "carla_traffic_light_trigger_waypoint"
TRIGGER_EXTENT_RATIO = 0.9
TRIGGER_SAMPLE_STEP_M = 1.0
BOUNDARY_STEP_M = 0.5
BOUNDARY_HALF_LANE_RATIO = 0.4
MAX_BOUNDARY_STEPS = 400


def _lane_id(waypoint):
    return (
        int(waypoint.road_id),
        int(getattr(waypoint, "section_id", 0)),
        int(waypoint.lane_id),
    )


def _is_intersection(waypoint):
    return bool(
        getattr(waypoint, "is_intersection", getattr(waypoint, "is_junction", False))
    )


def _location_like(reference, x, y, z):
    try:
        return type(reference)(x=x, y=y, z=z)
    except TypeError:
        value = type("LocationValue", (), {})()
        value.x, value.y, value.z = x, y, z
        return value


def sample_trigger_lane_waypoints(light, world_map):
    transform = light.get_transform()
    trigger = light.trigger_volume
    center = transform.transform(trigger.location)
    yaw = math.radians(float(transform.rotation.yaw))
    current = -TRIGGER_EXTENT_RATIO * float(trigger.extent.x)
    maximum = TRIGGER_EXTENT_RATIO * float(trigger.extent.x)
    result = []
    while current < maximum:
        query = _location_like(
            center,
            center.x + current * math.cos(yaw),
            center.y + current * math.sin(yaw),
            center.z,
        )
        waypoint = world_map.get_waypoint(query)
        if waypoint is not None and (not result or _lane_id(result[-1]) != _lane_id(waypoint)):
            result.append(waypoint)
        current += TRIGGER_SAMPLE_STEP_M
    return result


def advance_to_infraction_boundary(start):
    waypoint = start
    visited = set()
    branch_seen = False
    steps = 0
    while not _is_intersection(waypoint):
        identity = _lane_id(waypoint) + (round(float(getattr(waypoint, "s", 0.0)), 2),)
        if identity in visited:
            return {"status": "unknown", "unknown_reason": "waypoint_loop", "waypoint": waypoint, "steps": steps}
        visited.add(identity)
        if steps >= MAX_BOUNDARY_STEPS:
            return {"status": "unknown", "unknown_reason": "intersection_not_found", "waypoint": waypoint, "steps": steps}
        successors = list(waypoint.next(BOUNDARY_STEP_M) or [])
        if not successors:
            return {"status": "unknown", "unknown_reason": "intersection_not_found", "waypoint": waypoint, "steps": steps}
        branch_seen = branch_seen or len(successors) > 1
        following = successors[0]
        if _is_intersection(following):
            break
        waypoint = following
        steps += 1
    return {
        "status": "unknown" if branch_seen else "valid",
        "unknown_reason": "waypoint_branch" if branch_seen else None,
        "waypoint": waypoint,
        "steps": steps,
    }


def boundary_from_waypoint(waypoint):
    center = waypoint.transform.location
    yaw = math.radians(float(waypoint.transform.rotation.yaw))
    offset = BOUNDARY_HALF_LANE_RATIO * float(waypoint.lane_width)
    right_x, right_y = -math.sin(yaw), math.cos(yaw)
    point = lambda sign: {
        "x": float(center.x + sign * offset * right_x),
        "y": float(center.y + sign * offset * right_y),
        "z": float(center.z),
    }
    return {
        "geometry_source": GEOMETRY_SOURCE,
        "road_id": int(waypoint.road_id),
        "section_id": int(getattr(waypoint, "section_id", 0)),
        "lane_id": int(waypoint.lane_id),
        "s": float(getattr(waypoint, "s", 0.0)),
        "lane_width": float(waypoint.lane_width),
        "center": {"x": float(center.x), "y": float(center.y), "z": float(center.z)},
        "left_endpoint": point(1.0),
        "right_endpoint": point(-1.0),
    }
```

Keep internal helpers CARLA-object-compatible but independent of
`BaseAgent`, `InterfuserAgent`, and Scenario Runner imports.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Expected: all `LeaderboardBoundaryTests` pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add leaderboard/team_code/leaderboard_stop_targets.py tests/test_leaderboard_stop_targets.py
git commit -m "Add Leaderboard stop boundary geometry"
```

### Task 2: Build route-associated targets, stop poses, and corridors

**Files:**
- Modify: `leaderboard/team_code/leaderboard_stop_targets.py`
- Modify: `tests/test_leaderboard_stop_targets.py`

- [ ] **Step 1: Add failing route-target tests**

Add tests that construct a straight dense route with 0.5 m waypoint spacing
and two traffic lights resolving to the same lane/boundary:

```python
from team_code.leaderboard_stop_targets import (
    CORRIDOR_EXTENSION_M,
    ROUTE_MAX_DISTANCE_M,
    ROUTE_MIN_DISTANCE_M,
    SAFETY_MARGIN_M,
    build_stop_targets,
)


class RouteTargetTests(unittest.TestCase):
    def test_groups_owner_lights_and_uses_map_stable_target_id(self):
        route = dense_route(0.0, 100.0, step=0.5, road_id=7, lane_id=-1)
        lights = [
            FakeTrafficLight(actor_id=41, state="Red", trigger_x=20.0),
            FakeTrafficLight(actor_id=99, state="Red", trigger_x=20.0),
        ]

        targets = build_stop_targets(
            lights, FakeMap(route), route, ego_transform_at(10.0),
            ego_bounding_box=FakeBoundingBox(location_x=0.2, extent_x=2.0),
            map_name="Town01_Opt",
        )

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0]["owner_traffic_light_actor_ids"], [41, 99])
        self.assertNotIn("41", targets[0]["target_id"])
        self.assertTrue(targets[0]["target_id"].startswith("Town01_Opt:7:0:-1:"))

    def test_signed_route_distance_and_primary_target(self):
        targets = build_two_target_route(ego_x=10.0, boundary_xs=(20.0, 50.0))
        self.assertAlmostEqual(targets[0]["signed_route_distance_m"], 10.0, places=3)
        self.assertTrue(targets[0]["primary_for_ego"])
        self.assertFalse(targets[1]["primary_for_ego"])

    def test_recently_crossed_target_is_retained(self):
        target = build_one_target(ego_x=25.0, boundary_x=20.0)
        self.assertGreaterEqual(target["signed_route_distance_m"], ROUTE_MIN_DISTANCE_M)
        self.assertFalse(target["ego_before_boundary"])

    def test_stop_pose_uses_front_offset_and_one_meter_margin(self):
        target = build_one_target(ego_x=0.0, boundary_x=20.0, bbox_location_x=0.2, bbox_extent_x=2.0)
        self.assertEqual(SAFETY_MARGIN_M, 1.0)
        self.assertAlmostEqual(target["vehicle_front_offset_m"], 2.2)
        self.assertAlmostEqual(target["recommended_ego_stop_pose"]["location"]["x"], 16.8)

    def test_corridor_extends_three_meters_beyond_trigger_and_boundary(self):
        target = build_one_target(ego_x=0.0, trigger_x=10.0, boundary_x=20.0)
        xs = [point["location"]["x"] for point in target["stop_evidence_corridor"]["centerline"]]
        self.assertEqual(CORRIDOR_EXTENSION_M, 3.0)
        self.assertAlmostEqual(xs[0], 7.0, places=3)
        self.assertAlmostEqual(xs[-1], 23.0, places=3)
```

Also test the `[-10 m, 80 m]` inclusion window, heading mismatch, route/lane
ambiguity, branch propagation to unknown, and empty route handling.

- [ ] **Step 2: Run the route-target tests and verify RED**

Run the `RouteTargetTests` class. Expected: import errors for the new constants
and `build_stop_targets`.

- [ ] **Step 3: Implement route indexing and target construction**

Add these constants and functions. Use cumulative route-arc length, never
ego-forward distance, for inclusion and primary selection:

```python
ROUTE_MIN_DISTANCE_M = -10.0
ROUTE_MAX_DISTANCE_M = 80.0
ROUTE_MATCH_DISTANCE_M = 4.0
CORRIDOR_EXTENSION_M = 3.0
CORRIDOR_STEP_M = 0.5
SAFETY_MARGIN_M = 1.0


def _distance(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _route_index(route_waypoints):
    route = list(route_waypoints or [])
    cumulative = [0.0]
    for previous, current in zip(route, route[1:]):
        cumulative.append(cumulative[-1] + _distance(previous.transform.location, current.transform.location))
    return route, cumulative


def _nearest_route_index(route, location, lane=None):
    candidates = [
        (index, _distance(waypoint.transform.location, location))
        for index, waypoint in enumerate(route)
        if lane is None or _lane_id(waypoint) == lane
    ]
    if not candidates:
        return None
    index, distance = min(candidates, key=lambda item: item[1])
    return index if distance <= ROUTE_MATCH_DISTANCE_M else None


def _interpolate_route_point(route, cumulative, distance_m):
    distance_m = min(max(float(distance_m), cumulative[0]), cumulative[-1])
    for index in range(1, len(route)):
        if cumulative[index] < distance_m:
            continue
        span = cumulative[index] - cumulative[index - 1]
        ratio = 0.0 if span == 0.0 else (distance_m - cumulative[index - 1]) / span
        a, b = route[index - 1], route[index]
        location = {
            axis: float(getattr(a.transform.location, axis) + ratio * (getattr(b.transform.location, axis) - getattr(a.transform.location, axis)))
            for axis in ("x", "y", "z")
        }
        return {"location": location, "lane_width": float(a.lane_width)}
    last = route[-1]
    return {"location": {axis: float(getattr(last.transform.location, axis)) for axis in ("x", "y", "z")}, "lane_width": float(last.lane_width)}
```

`build_stop_targets()` must then:

1. Build the cumulative route and current ego route index.
2. Sample each light trigger and advance every controlled lane to an evaluator
   boundary using Task 1.
3. Match trigger and boundary waypoints to route indices with lane and heading
   checks.
4. Compute signed route distance and discard values outside `[-10, 80]`.
5. Group equal `(map, road, section, lane, round(boundary_s, 1))` targets.
6. Use target IDs formatted as
   `map:road:section:lane:quantized_s`; store sorted owner actor IDs and a
   string-keyed state map.
7. Interpolate the stop pose at
   `boundary_route_s - (bbox.location.x + bbox.extent.x + 1.0)`.
8. Sample the route corridor from 3 m before the lower of trigger/boundary arc
   coordinates through 3 m after the higher coordinate at 0.5 m resolution.
9. Mark the nearest non-negative valid target `primary_for_ego=True`.

Return unknown targets with their geometry diagnostics when route ownership is
known but boundary traversal is ambiguous. Do not synthesize a negative when
the route is missing.

- [ ] **Step 4: Run all stop-target tests and verify GREEN**

Expected: evaluator geometry and route-target tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add leaderboard/team_code/leaderboard_stop_targets.py tests/test_leaderboard_stop_targets.py
git commit -m "Associate Leaderboard stop targets with routes"
```

### Task 3: Emit traffic-element schema v2 and remove STOP side channels

**Files:**
- Modify: `leaderboard/team_code/traffic_element_labels.py`
- Modify: `leaderboard/team_code/interfuser_data_collector.py`
- Modify: `leaderboard/team_code/interfuser_collector_complete.py`
- Modify: `tests/test_traffic_element_labels.py`
- Modify: `tests/test_traffic_element_collector.py`
- Create: `tests/test_collector_stop_exclusion.py`

- [ ] **Step 1: Replace v1 tests with failing v2/no-STOP tests**

Change label fixtures to pass a dense `route_waypoints` list and add:

```python
class TrafficElementSchemaV2Tests(unittest.TestCase):
    def test_record_contains_stop_targets_and_no_stop_sign_schema(self):
        record = collect_traffic_element_labels(
            FakeHero(), FakeWorld([FakeTrafficLight(11, "Red")]),
            frame_id="0042", route_waypoints=dense_route(0.0, 60.0),
        )
        self.assertEqual(record["schema_version"], 2)
        self.assertEqual(record["frame_id"], "0042")
        self.assertIn("stop_targets", record)
        self.assertNotIn("stop_signs", record)
        self.assertNotIn("stop_lines", record["traffic_lights"][0])

    def test_traffic_stop_actor_is_never_queried_or_serialized(self):
        actors = RecordingActorList([FakeStopSign(21), FakeTrafficLight(11, "Green")])
        record = collect_traffic_element_labels(
            FakeHero(), FakeWorld(actors), frame_id="0000", route_waypoints=dense_route(0.0, 60.0)
        )
        self.assertNotIn("traffic.stop*", actors.filter_patterns)
        self.assertNotIn("stop_signs", record)
        self.assertNotIn("traffic.stop", json.dumps(record).lower())
        self.assertNotIn("trigger_volume_route_entry_approximation", json.dumps(record))

    def test_legacy_affordance_merge_does_not_emit_stop_sign(self):
        merged = merge_legacy_affordances(
            {"hazard_vehicle": False},
            {"traffic_lights": [{"is_active_for_ego": True, "state": "Red"}]},
        )
        self.assertEqual(merged, {"hazard_vehicle": False, "traffic_light": "Red"})
```

In `tests/test_collector_stop_exclusion.py`, patch the base measurement method,
use empty actor filters, and assert:

```python
def test_complete_collector_measurements_have_no_stop_fields(self):
    with patch.object(InterfuserDataCollector, "_get_measurements", return_value={"speed": 0.0}):
        record = InterfuserCollectorComplete._get_measurements(
            object.__new__(InterfuserCollectorComplete), FakeHeroWithWorld(), {}, None, 0.0
        )
    self.assertNotIn("is_stop_sign_present", record)
    self.assertNotIn("stop_sign", json.dumps(record).lower())


def test_agent_options_ignore_stop_signs(self):
    self.assertIs(AGENT_OPTIONS["ignore_stop_signs"], True)
```

- [ ] **Step 2: Run label/collector tests and verify RED**

Run:

```bash
/data1/shijj/conda_envs/interfuser_origin/bin/python -m unittest discover \
  -s tests -p 'test_traffic_element_labels.py' -v
/data1/shijj/conda_envs/interfuser_origin/bin/python -m unittest discover \
  -s tests -p 'test_traffic_element_collector.py' -v
/data1/shijj/conda_envs/interfuser_origin/bin/python -m unittest discover \
  -s tests -p 'test_collector_stop_exclusion.py' -v
```

Expected: schema remains v1, STOP keys are present, and `AGENT_OPTIONS` is
undefined.

- [ ] **Step 3: Implement v2 labels and collector route caching**

In `traffic_element_labels.py`:

```python
from leaderboard_stop_targets import build_stop_targets

SCHEMA_VERSION = 2


def collect_traffic_element_labels(hero, world, frame_id, route_waypoints, max_distance=80.0):
    errors = []
    ego_transform = hero.get_transform()
    ego_location = hero.get_location()
    world_map = world.get_map()
    ego_waypoint = world_map.get_waypoint(ego_location)
    active_light = hero.get_traffic_light()
    active_light_id = int(active_light.id) if active_light is not None else None
    light_actors = []
    traffic_lights = []
    for light in _actor_filter(world.get_actors(), "traffic.traffic_light*"):
        is_active = active_light_id == int(light.id)
        if ego_location.distance(light.get_location()) > max_distance and not is_active:
            continue
        try:
            traffic_lights.append(
                _traffic_light_label(light, ego_transform, _lane_identity(ego_waypoint), active_light_id, errors)
            )
            light_actors.append(light)
        except Exception as exc:
            errors.append({"actor_id": int(light.id), "field": "traffic_light", "error": str(exc)})
    stop_targets = build_stop_targets(
        light_actors,
        world_map,
        route_waypoints,
        ego_transform,
        hero.bounding_box,
        str(world_map.name).split("/")[-1],
    )
    ego_record = {
        "actor_id": int(hero.id),
        "location": _location_dict(ego_location),
        "rotation": _rotation_dict(ego_transform.rotation),
        "lane": _waypoint_metadata(ego_waypoint),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "frame_id": str(frame_id),
        "map_name": str(world_map.name).split("/")[-1],
        "ego": ego_record,
        "active_traffic_light_id": active_light_id,
        "traffic_lights": traffic_lights,
        "stop_targets": stop_targets,
        "errors": errors,
    }


def legacy_affordances_from_labels(labels):
    active = [item for item in labels["traffic_lights"] if item["is_active_for_ego"]]
    return {"traffic_light": active[0]["state"] if active else None}
```

Delete `_stop_sign_label`, `_route_entry_stop_line`, STOP actor filtering,
`stop_signs`, and stop-sign validation. Traffic-light records retain lamp/state,
trigger, affected-lane, active, and relevance fields but no longer own
`stop_lines`; stop geometry lives only in `stop_targets`.

In `interfuser_data_collector.py`, define and use:

```python
AGENT_OPTIONS = {
    "base_tlight_threshold": 3.0,
    "base_vehicle_threshold": 8.0,
    "base_min_distance": 3.0,
    "max_brake": 0.8,
    "ignore_stop_signs": True,
}
```

Pass `opt_dict=dict(AGENT_OPTIONS)` to both `BehaviorAgent` and the
`BasicAgent` fallback. Initialize `self._traffic_route_waypoints = []` in
`setup()` and `set_global_plan()`. Immediately after `set_destination()`, copy
the complete local plan before it is consumed:

```python
self._traffic_route_waypoints = [
    waypoint
    for waypoint, _road_option in self._agent.get_local_planner().get_plan()
]
```

Pass `frame_id=f"{frame:04d}"` and this cached route to
`collect_traffic_element_labels()`. Build `actors_by_id` only from
`traffic_lights`. Write `traffic_elements` with `_write_json_atomic()` just as
views are written.

In `interfuser_collector_complete.py`, remove the `traffic.stop*` filter,
`is_stop_sign_present`, and every stop-sign actor loop. This collector may
still encounter STOP graphics in RGB; it must not emit a structured STOP
signal or use one for Basic/BehaviorAgent control.

- [ ] **Step 4: Run v2 label/collector tests and verify GREEN**

Expected: the three modules pass and recursive JSON assertions find no STOP
side-channel keys.

- [ ] **Step 5: Commit Task 3**

```bash
git add leaderboard/team_code/traffic_element_labels.py \
  leaderboard/team_code/interfuser_data_collector.py \
  leaderboard/team_code/interfuser_collector_complete.py \
  tests/test_traffic_element_labels.py tests/test_traffic_element_collector.py \
  tests/test_collector_stop_exclusion.py
git commit -m "Emit traffic-light stop target schema v2"
```

### Task 4: Project camera corridors and compute lidar evidence

**Files:**
- Modify: `leaderboard/team_code/traffic_element_projection.py`
- Modify: `leaderboard/team_code/interfuser_data_collector.py`
- Modify: `tests/test_traffic_element_projection.py`
- Modify: `tests/test_traffic_element_collector.py`

- [ ] **Step 1: Add failing schema-v3 camera/lidar tests**

Replace the phase fixture with schema v2 and one valid stop target. Add tests:

```python
from team_code.traffic_element_projection import (
    EVIDENCE,
    IMAGE_SCHEMA_VERSION,
    build_lidar_target_evidence,
    build_traffic_element_view_record,
)


class EvidenceSchemaV3Tests(unittest.TestCase):
    def test_v3_has_traffic_lights_targets_and_no_stop_sign_keys(self):
        record = build_traffic_element_view_record(
            "0007", phase2_record(), {11: light_actor()},
            {"front": camera_frame()}, lidar_frame=lidar_frame(),
        )
        self.assertEqual(IMAGE_SCHEMA_VERSION, 3)
        self.assertEqual(record["source_traffic_element_schema_version"], 2)
        self.assertEqual(len(record["cameras"]["front"]["stop_targets"]), 1)
        self.assertNotIn("stop_sign", json.dumps(record).lower())

    def test_camera_projects_boundary_pose_and_corridor(self):
        target = build_view()["cameras"]["front"]["stop_targets"][0]
        self.assertEqual(target["boundary"]["projection_status"], "projected")
        self.assertIsNotNone(target["boundary"]["image_segment"])
        self.assertEqual(target["recommended_stop_pose"]["projection_status"], "projected")
        self.assertGreaterEqual(len(target["corridor"]["image_polyline"]), 2)

    def test_lidar_counts_corridor_and_surface_band_points(self):
        evidence = build_lidar_target_evidence(
            valid_stop_target(),
            points=np.array([[10.0, 0.0, 0.0, 1.0], [10.0, 0.0, 1.0, 1.0], [10.0, 8.0, 0.0, 1.0]]),
            lidar_transform=identity_transform(),
            ego_transform=identity_transform(),
        )
        self.assertEqual(evidence["status"], "available")
        self.assertEqual(evidence["in_corridor_point_count"], 2)
        self.assertEqual(evidence["road_surface_point_count"], 1)
```

Also test outside-image, behind-camera, missing camera, missing lidar, transform
round-trip, non-finite points, and geometry-valid/sensor-unknown separation.

- [ ] **Step 2: Run projection/collector tests and verify RED**

Expected: schema is still 2 and lidar evidence functions are absent.

- [ ] **Step 3: Implement evidence schema v3**

Change constants to:

```python
IMAGE_SCHEMA_VERSION = 3
EVIDENCE = {
    "roi_expand_pixels": 6,
    "minimum_semantic_pixels": 3,
    "traffic_light": {"semantic_tag": 7, "depth_tolerance_m": 4.0},
    "road_lines_semantic_tag": 24,
    "corridor_depth_tolerance_m": 2.0,
    "lidar_min_height_m": -0.5,
    "lidar_max_height_m": 3.0,
    "lidar_road_surface_tolerance_m": 0.25,
}
```

Remove `ASSOCIATION["stop_sign"]`, stop-sign element views, and generic
`stop_lines`. Add `_project_target_camera()` that projects the evaluator
boundary, recommended stop pose, and every corridor centerline sample. It must
derive left/right corridor edges from each sample's local route tangent and
half lane width, then store the projected envelope as left edge followed by the
reversed right edge. It must return a target-scoped object even when projection
is outside or unknown. For valid projected samples, compare decoded camera
depth with the sample's camera-forward depth, count residuals at or below
`corridor_depth_tolerance_m`, store the median finite residual, and mark the
evidence occluded when finite depth exists but no sample meets the tolerance.
An unknown geometry target gets sensor evidence status `unknown` with reason
`geometry_unknown`; camera or lidar processing must not reinterpret it.

Add pure lidar transforms and evidence counting:

```python
def world_to_sensor_xyz(world_points, sensor_transform):
    points = np.asarray(world_points, dtype=np.float64)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    return (np.linalg.inv(transform_matrix(sensor_transform)) @ homogeneous.T).T[:, :3]


def build_lidar_target_evidence(target, points, lidar_transform, ego_transform):
    if points is None or lidar_transform is None or ego_transform is None:
        return {"target_id": target["target_id"], "status": "unknown", "unknown_reason": "sensor_unavailable"}
    raw = np.asarray(points, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] < 3 or not np.isfinite(raw[:, :3]).all():
        return {"target_id": target["target_id"], "status": "unknown", "unknown_reason": "projection_error"}
    world_centerline = np.array([
        [sample["location"][axis] for axis in ("x", "y", "z")]
        for sample in target["stop_evidence_corridor"]["centerline"]
    ])
    centerline = world_to_sensor_xyz(world_centerline, lidar_transform)
    delta = raw[:, None, :3] - centerline[None, :, :]
    nearest = np.argmin(np.sum(delta[:, :, :2] ** 2, axis=2), axis=1)
    nearest_delta = delta[np.arange(len(raw)), nearest]
    widths = np.array([
        sample["lane_width"] / 2.0
        for sample in target["stop_evidence_corridor"]["centerline"]
    ])
    lateral = np.linalg.norm(nearest_delta[:, :2], axis=1)
    relative_z = nearest_delta[:, 2]
    inside = (
        (lateral <= widths[nearest])
        & (relative_z >= EVIDENCE["lidar_min_height_m"])
        & (relative_z <= EVIDENCE["lidar_max_height_m"])
    )
    surface = inside & (np.abs(relative_z) <= EVIDENCE["lidar_road_surface_tolerance_m"])
    sensor_to_world = transform_matrix(lidar_transform)
    ego_to_world = transform_matrix(ego_transform)
    return {
        "target_id": target["target_id"],
        "status": "available",
        "unknown_reason": None,
        "sensor_to_ego": (np.linalg.inv(ego_to_world) @ sensor_to_world).tolist(),
        "ego_to_world": ego_to_world.tolist(),
        "corridor_centerline_xyz": centerline.tolist(),
        "in_corridor_point_count": int(np.count_nonzero(inside)),
        "road_surface_point_count": int(np.count_nonzero(surface)),
    }
```

If nearest-sample width counting proves too permissive at a curve in the new
unit test, use point-to-segment XY distance while preserving the same public
record fields.

Update `build_traffic_element_view_record()` to accept `lidar_frame`, emit
only traffic-light views plus per-camera `stop_targets`, and add a top-level
`lidar` object containing target evidence and errors.

In `_get_traffic_element_camera_frames()`, retain the existing transforms and
depth/semantic arrays. Add `_get_traffic_element_lidar_frame()`:

```python
def _get_traffic_element_lidar_frame(self, input_data, hero):
    sensor = getattr(self.sensor_interface, "_sensors_objects", {}).get("lidar")
    if sensor is None or "lidar" not in input_data:
        return {"error": "required sensor unavailable: lidar"}
    return {
        "transform": sensor.get_transform(),
        "ego_transform": hero.get_transform(),
        "points": input_data["lidar"][1],
    }
```

Pass that object to the v3 builder before saving the unchanged complete lidar
array.

- [ ] **Step 4: Run projection/collector tests and verify GREEN**

Expected: all v3 camera/lidar tests pass; existing traffic-light box tests
remain green with semantic tag 7.

- [ ] **Step 5: Commit Task 4**

```bash
git add leaderboard/team_code/traffic_element_projection.py \
  leaderboard/team_code/interfuser_data_collector.py \
  tests/test_traffic_element_projection.py tests/test_traffic_element_collector.py
git commit -m "Add camera and lidar stop target evidence"
```

### Task 5: Add optional painted-line candidates without automatic positives

**Files:**
- Modify: `leaderboard/team_code/traffic_element_projection.py`
- Modify: `leaderboard/team_code/interfuser_data_collector.py`
- Modify: `tests/test_traffic_element_projection.py`

- [ ] **Step 1: Add failing synthetic candidate tests**

Add a 200 x 120 synthetic RGB/depth image with a horizontal white line inside
the projected corridor and a blank control image:

```python
from team_code.traffic_element_projection import find_painted_line_candidate


def test_transverse_line_is_candidate_not_verified(self):
    rgb = np.zeros((120, 200, 3), dtype=np.uint8)
    cv2.line(rgb, (40, 80), (160, 80), (255, 255, 255), 4)
    result = find_painted_line_candidate(
        rgb, np.full((120, 200), 12.0),
        corridor_polygon=[[20, 50], [180, 50], [180, 100], [20, 100]],
        expected_boundary_segment=[[30, 80], [170, 80]],
        expected_depth_m=12.0,
    )
    self.assertEqual(result["status"], "candidate")
    self.assertNotEqual(result["status"], "verified")


def test_blank_corridor_remains_unknown(self):
    result = find_painted_line_candidate(
        np.zeros((120, 200, 3), dtype=np.uint8),
        np.full((120, 200), 12.0),
        [[20, 50], [180, 50], [180, 100], [20, 100]],
        [[30, 80], [170, 80]],
        12.0,
    )
    self.assertEqual(result, {"status": "unknown", "image_segment": None, "score": None})
```

- [ ] **Step 2: Verify RED for the missing candidate function**

Run the two candidate tests. Expected: import failure.

- [ ] **Step 3: Implement constrained candidate extraction**

Use grayscale, `cv2.Canny(60, 180)`, a filled corridor mask, and
`cv2.HoughLinesP(..., threshold=20, minLineLength=12, maxLineGap=6)`. Reject
segments whose undirected angle differs by more than 15 degrees from the
projected evaluator boundary or whose median finite depth residual exceeds
2.0 m. Also require segment length to be at least 25 percent of the projected
evaluator-boundary length. Rank remaining segments by normalized length minus
normalized depth residual. Return only `candidate` or `unknown`; no automatic
code path may return `verified`.

Add raw BGR RGB arrays to camera frames and call the function only when the
corridor and evaluator boundary both project. Persist segment, score, angle
error, depth residual, and semantic `RoadLines` support as review diagnostics.

- [ ] **Step 4: Run all projection tests and verify GREEN**

Expected: synthetic line is `candidate`, blank image is `unknown`, and no test
or implementation assigns `verified`.

- [ ] **Step 5: Commit Task 5**

```bash
git add leaderboard/team_code/traffic_element_projection.py \
  leaderboard/team_code/interfuser_data_collector.py \
  tests/test_traffic_element_projection.py
git commit -m "Add review-only painted stop line candidates"
```

### Task 6: Enforce v2/v3 schemas with audits

**Files:**
- Modify: `tools/data/audit_traffic_element_labels.py`
- Modify: `tools/data/audit_traffic_element_views.py`
- Modify: `tests/test_audit_traffic_element_labels.py`
- Modify: `tests/test_audit_traffic_element_views.py`

- [ ] **Step 1: Replace audit fixtures and add failing forbidden-label tests**

Build one valid phase-v2 fixture with a grouped target and one valid evidence-v3
fixture with three cameras and lidar evidence. Add:

```python
FORBIDDEN_STOP_KEYS = {"stop_sign", "stop_signs", "is_stop_sign_present"}


def test_phase_audit_summarizes_valid_and_unknown_targets_by_town(self):
    summary = audit_dataset(write_v2_fixture())
    self.assertEqual(summary["valid_stop_targets"], 1)
    self.assertEqual(summary["unknown_stop_targets"], 1)
    self.assertEqual(summary["unknown_reasons"], {"waypoint_branch": 1})
    self.assertEqual(summary["forbidden_stop_occurrences"], 0)


def test_phase_audit_rejects_stop_sign_key_anywhere_in_generated_json(self):
    root = write_v2_fixture()
    write_json(root / "route_00/measurements/0000.json", {"is_stop_sign_present": []})
    with self.assertRaisesRegex(AuditError, "forbidden STOP label"):
        audit_dataset(root)


def test_view_audit_separates_geometry_unknown_from_sensor_unknown(self):
    summary = audit_traffic_element_views(write_v3_fixture(sensor_unknown=True))
    self.assertEqual(summary["unknown_stop_targets"], 0)
    self.assertEqual(summary["unknown_camera_evidence"], 1)


def test_view_audit_rejects_legacy_schema(self):
    root = write_v3_fixture()
    mutate_view(root, schema_version=2)
    with self.assertRaisesRegex(AuditError, "unsupported evidence schema_version"):
        audit_traffic_element_views(root)
```

Also test stable target-ID uniqueness, required geometry source, signed-distance
finiteness, primary uniqueness, camera segment bounds, lidar counts, transform
shape, candidate-never-verified-by-collector metadata, and exact frame
alignment across RGB/phase/evidence/lidar.

- [ ] **Step 2: Run both audit modules and verify RED**

Expected: old stop-sign metrics appear and v2/v3 fixtures fail.

- [ ] **Step 3: Rewrite validators and summaries**

For phase v2, validate required top-level fields and recursively reject these
case-insensitive keys/values in every generated JSON directory under a route:

```python
FORBIDDEN_KEY_FRAGMENT = "stop_sign"
FORBIDDEN_VALUE_PREFIX = "traffic.stop"
FORBIDDEN_PROVENANCE = "trigger_volume_route_entry_approximation"


def _forbidden_occurrences(value, path="record"):
    errors = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if FORBIDDEN_KEY_FRAGMENT in str(key).lower():
                errors.append(child_path)
            errors.extend(_forbidden_occurrences(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_forbidden_occurrences(child, f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        if lowered.startswith(FORBIDDEN_VALUE_PREFIX) or lowered == FORBIDDEN_PROVENANCE:
            errors.append(path)
    return errors
```

Scan `traffic_elements`, `traffic_element_views`, `measurements`,
`affordances`, `3d_bbs`, `2d_bbs_*`, and `other_actors`. Do not scan raw image,
depth, lidar, log, result, or documentation files.

Phase summary fields must include frames, map/town, traffic-light states,
active-light frames, valid/unknown targets, unknown reasons, primary frames,
before/after targets, trigger-to-boundary route-distance distribution, and
forbidden occurrence count.

Evidence summary fields must include per-camera visible lights, projected
boundaries/corridors, candidate/verified lines, camera unknown reasons, lidar
available/unknown targets, in-corridor/road-surface point distributions, hard
negatives, invalid frames, and error frames. Geometry unknown and sensor
unknown counts remain separate.

- [ ] **Step 4: Run audit tests and verify GREEN**

Expected: all phase/evidence fixtures validate or fail for their asserted
reason; no old STOP metrics remain.

- [ ] **Step 5: Commit Task 6**

```bash
git add tools/data/audit_traffic_element_labels.py \
  tools/data/audit_traffic_element_views.py \
  tests/test_audit_traffic_element_labels.py \
  tests/test_audit_traffic_element_views.py
git commit -m "Audit Leaderboard stop target evidence"
```

### Task 7: Update overlays, route profiling, live geometry parity, and docs

**Files:**
- Modify: `tools/data/render_traffic_element_overlays.py`
- Modify: `tests/test_render_traffic_element_overlays.py`
- Create: `tools/data/apply_painted_line_reviews.py`
- Create: `tests/test_apply_painted_line_reviews.py`
- Modify: `tools/data/profile_traffic_element_routes.py`
- Modify: `tests/test_profile_traffic_element_routes.py`
- Create: `tools/data/check_leaderboard_stop_target_geometry.py`
- Create: `tests/test_check_leaderboard_stop_target_geometry.py`
- Modify: `docs/traffic_element_label_schema.md`
- Modify: `docs/traffic_element_image_label_schema.md`

- [ ] **Step 1: Write failing overlay/profile/parity tests**

Overlay tests must assert pixels change for all four geometry layers and that
selection order is valid target, unknown target, irrelevant visible light,
hard negative, then frame-key fill. Remove all stop-sign fixture objects.

Change profile tests to this traffic-light-only API:

```python
summary = score_dense_route(
    route_points=[(0.0, 0.0), (10.0, 0.0), (200.0, 0.0)],
    traffic_lights=[(8.0, 1.0)],
    relevant_radius_m=30.0,
    nearby_radius_m=80.0,
)
self.assertEqual(summary["nearby_traffic_lights"], 1)
self.assertEqual(summary["hard_negative_points"], 1)
self.assertNotIn("stop_sign_actors", summary)
```

For the parity tool, expose `compare_boundaries(actual, expected,
tolerance_m=1e-3)` and test exact match, lane mismatch, count mismatch, and
coordinate mismatch.

For manual review, test that an explicit manifest decision can change only an
existing `candidate` to `verified` or `unknown`, that `unreviewed` never changes
the dataset, and that missing target IDs or direct verification of an
`unknown` candidate fail without writing a partial JSON file.

- [ ] **Step 2: Run overlay/profile/parity tests and verify RED**

Expected: overlay expects v2 fields, profiler still requires `stop_signs`, and
the parity module is missing.

- [ ] **Step 3: Implement v3 review tooling and traffic-light-only profiling**

Use these BGR overlay colors:

```python
TRIGGER_COLOR = (255, 255, 0)
BOUNDARY_COLOR = (255, 0, 255)
STOP_POSE_COLOR = (255, 0, 0)
CORRIDOR_COLOR = (0, 180, 0)
CANDIDATE_COLOR = (0, 255, 255)
```

Draw lamp boxes as before, but delete all STOP-sign loops and labels. Draw
target ID, signed route distance, status, and unknown reason with geometry.
When `--review-manifest-output PATH` is supplied, also write candidate entries
with route-relative view path, camera, target ID, and
`"decision":"unreviewed"`.

Implement `apply_painted_line_reviews.py` with an atomic read/validate/write
loop. A manifest entry has exactly:

```json
{
  "view_path": "route/traffic_element_views/0000.json",
  "camera": "front",
  "target_id": "Town01_Opt:7:0:-1:42.0",
  "decision": "verified"
}
```

Accepted decisions are `verified`, `rejected`, and `unreviewed`. `verified`
changes an existing `candidate` status to `verified` and adds
`review_source="manual_manifest"`; `rejected` changes it to `unknown` and adds
the same review source; `unreviewed` makes no change. Validate the entire
manifest and all target lookups before replacing any file. Use the collector's
write-temp, flush, `fsync`, `os.replace` pattern.

Remove the `stop_signs` parameter and every STOP metric from
`score_dense_route()` and `profile_routes()`. Hard-negative distance is the
route distance to the nearest traffic-light trigger only.

The live parity tool must:

1. Connect to a caller-supplied CARLA host/port without starting or stopping a
   server.
2. Load each requested town.
3. Enumerate traffic lights and compute actual boundaries with
   `leaderboard_stop_targets.py`.
4. Independently reproduce the exact `RunningRedLightTest` trigger sampling,
   first-successor advancement, and 0.4-lane-width endpoints inside the tool.
5. Compare road/lane identity and endpoint/center distance at `1e-3 m`.
6. Emit JSON with per-town actors, boundaries, valid/unknown counts,
   mismatches, branches, and trigger-to-boundary distances.
7. Exit 2 if any valid collector boundary differs from the independent
   reference.

Rewrite both schema docs from the accepted design, including every versioned
constant, coordinate convention, unknown policy, forbidden STOP field, CLI
audit command, and overlay legend. State explicitly that learned junction
outputs never generate labels.

- [ ] **Step 4: Run tooling tests and verify GREEN**

Expected: overlay/review/profile/parity unit tests pass and a repository search
of the two schema docs contains STOP only in explicit
forbidden/compatibility text.

- [ ] **Step 5: Commit Task 7**

```bash
git add tools/data/render_traffic_element_overlays.py \
  tests/test_render_traffic_element_overlays.py \
  tools/data/apply_painted_line_reviews.py \
  tests/test_apply_painted_line_reviews.py \
  tools/data/profile_traffic_element_routes.py \
  tests/test_profile_traffic_element_routes.py \
  tools/data/check_leaderboard_stop_target_geometry.py \
  tests/test_check_leaderboard_stop_target_geometry.py \
  docs/traffic_element_label_schema.md docs/traffic_element_image_label_schema.md
git commit -m "Document and review stop target evidence"
```

### Task 8: Update bounded runner and verify the implementation

**Files:**
- Modify: `tools/data/run_traffic_element_small_batch.sh`

- [ ] **Step 1: Add failing shell assertions for v2/v3 acceptance fields**

Before launching CARLA, make the runner reject source constants other than
phase schema 2 and evidence schema 3:

```bash
PHASE_SCHEMA=$(${PYTHON_BIN} -c 'from traffic_element_labels import SCHEMA_VERSION; print(SCHEMA_VERSION)')
VIEW_SCHEMA=$(${PYTHON_BIN} -c 'from traffic_element_projection import IMAGE_SCHEMA_VERSION; print(IMAGE_SCHEMA_VERSION)')
[ "${PHASE_SCHEMA}" = 2 ] || { echo "expected phase schema 2" >&2; exit 65; }
[ "${VIEW_SCHEMA}" = 3 ] || { echo "expected evidence schema 3" >&2; exit 65; }
```

Add a shell-test mode, `TRAFFIC_BATCH_VALIDATE_ONLY=1`, that runs these checks
and exits before CARLA startup. Run it against the pre-Task-8 script and verify
the mode is absent/nonzero.

- [ ] **Step 2: Update runner geometry checks and acceptance summaries**

After every `preload_route_map`, run the live parity tool for that town and
write `results/.../geometry/<town>.json`. Keep one file per town and do not
rerun it for route36 after route13 already checked Town03.

Replace all stop-sign acceptance fields with:

```text
valid_stop_targets
unknown_stop_targets
unknown_reasons
primary_stop_target_frames
projected_stop_boundaries
projected_stop_corridors
painted_line_candidates
verified_painted_lines
lidar_available_targets
lidar_unknown_targets
forbidden_stop_occurrences
```

The runner fails when forbidden occurrences are nonzero, geometry parity has a
mismatch, either audit is nonzero, or any valid target lacks camera/lidar
statistics. Unknown targets are reported and do not fail unless their reason
is missing.

- [ ] **Step 3: Run validate-only mode and focused/full tests**

Run:

```bash
TRAFFIC_BATCH_VALIDATE_ONLY=1 tools/data/run_traffic_element_small_batch.sh \
  results/full42_eval/routes/route_00_Town01_Opt.xml \
  results/full42_eval/routes/route_39_Town04_Opt.xml

/data1/shijj/conda_envs/interfuser_origin/bin/python -m unittest discover \
  -s tests -p 'test_*.py' -v
```

Expected: validate-only exits 0; all prior 59 tracked tests plus new
tests pass. Record the exact new total rather than predicting it in docs.

- [ ] **Step 4: Check scoped diffs and commit the runner**

```bash
git diff --check -- \
  leaderboard/team_code/leaderboard_stop_targets.py \
  leaderboard/team_code/traffic_element_labels.py \
  leaderboard/team_code/traffic_element_projection.py \
  leaderboard/team_code/interfuser_data_collector.py \
  leaderboard/team_code/interfuser_collector_complete.py \
  tools/data tests docs/traffic_element_label_schema.md \
  docs/traffic_element_image_label_schema.md
git add tools/data/run_traffic_element_small_batch.sh
git commit -m "Validate bounded stop target collection"
```

- [ ] **Step 5: Run the bounded four-route collection**

First record the port-2000 CARLA PIDs. Then run:

```bash
BATCH_RUN_ID=20260716_leaderboard_stop_targets \
  tools/data/run_traffic_element_small_batch.sh \
  results/full42_eval/routes/route_00_Town01_Opt.xml \
  results/full42_eval/routes/route_39_Town04_Opt.xml
```

Expected: route13/bg0, route36/bg0, route00/bg20, and route39/bg20 complete;
both audits and live geometry parity exit 0; output stays below 2,000 frames and
2 GiB; no evaluator or CARLA process remains on 2400/8400; port-2000 PIDs are
unchanged.

- [ ] **Step 6: Run aggregate audits and render deterministic review samples**

```bash
PY=/data1/shijj/conda_envs/interfuser_origin/bin/python
ROOT=data/traffic_element_small_batch/20260716_leaderboard_stop_targets
OUT=results/traffic_element_small_batch/20260716_leaderboard_stop_targets

${PY} tools/data/audit_traffic_element_labels.py "${ROOT}" > "${OUT}/aggregate_phase2_audit.json"
${PY} tools/data/audit_traffic_element_views.py "${ROOT}" > "${OUT}/aggregate_evidence_audit.json"
${PY} tools/data/render_traffic_element_overlays.py "${ROOT}" \
  --output-dir data/traffic_element_small_batch_review/20260716_leaderboard_stop_targets \
  --limit 16 --camera front \
  --review-manifest-output "${OUT}/painted_line_review_manifest.json"
```

Expected: zero invalid records, unexplained errors, and forbidden STOP
occurrences. Review 16 overlays spanning Town01/03/04, valid and unknown
targets, before/after frames, candidates, irrelevant lights, and hard
negatives. Change only visually certain manifest decisions from `unreviewed`.
If at least one decision changes, run
`${PY} tools/data/apply_painted_line_reviews.py "${ROOT}" "${OUT}/painted_line_review_manifest.json"`
and rerun the evidence audit. Record observed counts and rejected candidate
frames in
`results/traffic_element_small_batch/20260716_leaderboard_stop_targets/acceptance.md`.

- [ ] **Step 7: Final repository and process verification**

```bash
git status --short
git log --oneline -8
ps -eo pid=,args= | awk '/CarlaUE4/ && (/--world-port=2000/ || /--world-port=2400/) && !/awk/ {print}'
```

Expected: only the user's pre-existing controller/log/data changes remain
unstaged; implementation files are committed; port 2400 has no CARLA process;
port 2000 still shows the original PIDs.

Do not commit collected data, generated overlays, result logs, caches, or the
acceptance report unless the user separately requests artifact tracking.
