# Traffic Element Label Schema

Schema version: `1`

`InterfuserDataCollector` writes one JSON document per saved sensor frame to:

```text
<route>/traffic_elements/0000.json
<route>/traffic_elements/0001.json
...
```

The record is the source of truth for traffic-light, stop-sign, and stop-line
supervision. The legacy `affordances` file is derived from this record.

## Coordinate Conventions

- World positions use CARLA world coordinates in meters.
- Rotations use CARLA pitch/yaw/roll in degrees.
- Ego-relative positions use meters with these axes:
  - `forward`: positive in the ego vehicle's forward direction;
  - `right`: positive to the ego vehicle's right;
  - `up`: positive upward.
- `longitudinal_distance` is the stop-line center's `forward` coordinate.
- `lateral_offset` is the stop-line center's `right` coordinate.
- `ego_before_line` is `true` when `longitudinal_distance >= 0`.

The sign of `ego_before_line` is a geometric observation, not proof that the
line controls the ego route. Consumers must also check the parent element's
route relevance.

## Top-Level Record

```json
{
  "schema_version": 1,
  "ego": {},
  "active_traffic_light_id": null,
  "traffic_lights": [],
  "stop_signs": [],
  "errors": []
}
```

### `schema_version`

Integer schema version. Readers must reject unsupported versions instead of
guessing field semantics.

### `ego`

```json
{
  "actor_id": 1,
  "location": {"x": 0.0, "y": 0.0, "z": 0.0},
  "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
  "lane": {
    "road_id": 7,
    "section_id": 0,
    "lane_id": -1,
    "s": 12.5,
    "lane_width": 3.5
  }
}
```

`lane` is nullable when `Map.get_waypoint` fails. Such a failure is also added
to `errors`.

### `active_traffic_light_id`

Actor ID returned by `Vehicle.get_traffic_light()`, or `null` when CARLA does
not report an active light for the ego vehicle.

### `errors`

Extraction errors are structured objects containing `field`, `error`, and,
when applicable, `actor_id`. Missing optional geometry must create an error;
it must not be silently converted into a negative label.

If the ego waypoint is unavailable, stop-sign route relevance is not emitted.
If a stop-sign waypoint traversal fails, that stop-sign item is omitted and the
failure is recorded. Consumers must exclude frames with extraction errors from
negative-label training unless the affected field is explicitly irrelevant to
the task.

## Traffic-Light Object

Version 1 includes traffic lights whose actor origin is within 80 m of the ego
vehicle, plus the CARLA-active light even if its actor origin is farther away.
This is a collection radius, not a visibility claim; an item can be occluded,
outside a camera view, or irrelevant to the ego lane.

Each item in `traffic_lights` contains:

| Field | Meaning |
| --- | --- |
| `actor_id`, `type_id` | CARLA actor identity |
| `state` | CARLA state name such as `Red`, `Yellow`, `Green`, `Off`, or `Unknown` |
| `location`, `rotation` | Actor pose in world coordinates |
| `relative_position` | Actor pose origin in ego forward/right/up coordinates |
| `relative_heading` | Actor yaw minus ego yaw, normalized to `[-180, 180)` degrees |
| `distance` | Euclidean 3D distance from ego to actor origin in meters |
| `trigger_volume` | Trigger center, extent, and rotation in world and ego coordinates |
| `is_active_for_ego` | Actor ID equals `active_traffic_light_id` |
| `controls_ego_lane` | Active for ego, or a stop/affected waypoint matches ego road/section/lane |
| `relevant_to_ego` | Version 1 alias of `controls_ego_lane` |
| `affected_lanes` | CARLA `get_affected_lane_waypoints()` metadata |
| `stop_lines` | Cross-lane line segments derived from CARLA stop waypoints |

`controls_ego_lane=false` does not mean the light is visually absent. It means
the current version could not associate that nearby light with the ego lane.

## Stop-Sign Object

Version 1 applies the same 80 m actor-origin collection radius to stop signs.

Each item in `stop_signs` contains:

| Field | Meaning |
| --- | --- |
| `actor_id`, `type_id` | CARLA actor identity |
| `location`, `rotation` | Actor pose in world coordinates |
| `relative_position`, `relative_heading`, `distance` | Ego-relative geometry |
| `trigger_volume` | Oriented CARLA trigger volume in world and ego coordinates |
| `affects_ego_route` | A forward waypoint branch intersects the trigger volume within 30 m |
| `stop_lines` | Zero or one derived route-entry line in schema version 1 |

`affects_ego_route` is a route-horizon approximation. At a junction,
`Waypoint.next()` can return several branches; version 1 checks all branches
and may therefore mark a stop sign relevant before the final route branch is
known. Training code should retain this provenance and may filter or refine it
with the planned route.

## Stop-Line Object

```json
{
  "geometry_source": "carla_stop_waypoint",
  "is_exact_carla_stop_position": true,
  "road_id": 3,
  "section_id": 0,
  "lane_id": -1,
  "s": 42.0,
  "lane_width": 3.5,
  "center": {"x": 12.0, "y": 4.0, "z": 0.0},
  "left_endpoint": {"x": 12.0, "y": 2.25, "z": 0.0},
  "right_endpoint": {"x": 12.0, "y": 5.75, "z": 0.0},
  "relative_center": {"forward": 10.0, "right": 0.2, "up": 0.0},
  "longitudinal_distance": 10.0,
  "lateral_offset": 0.2,
  "ego_before_line": true
}
```

The example assumes a waypoint yaw of 0 degrees. In CARLA's left-handed
coordinates the waypoint right vector is world `+Y`, so the left endpoint has
the smaller `y` value.

### Provenance: `carla_stop_waypoint`

Used only for waypoints returned by
`TrafficLight.get_stop_waypoints()`. CARLA 0.9.16 documents these as stop
positions computed from the traffic-light trigger boxes. This is the exact
simulator-provided stop position available through the Python API.

### Provenance: `trigger_volume_route_entry_approximation`

Used for stop signs because `carla.TrafficSign` has no
`get_stop_waypoints()` API. The collector walks forward waypoints, finds the
first point inside the oriented trigger volume, and uses the preceding
waypoint as the stop-line center. This is a derived training label and sets
`is_exact_carla_stop_position=false`.

The approximation must be audited visually before model training. It can be
replaced in a later schema version by route-aware map geometry without changing
the meaning of existing records.

## Legacy Affordances

The collector derives:

```json
{
  "traffic_light": "Red",
  "stop_sign": true,
  "hazard_vehicle": false,
  "hazard_pedestrian": false
}
```

- `traffic_light` is the state of the CARLA-active traffic light, otherwise
  `null`.
- `stop_sign` is true when any collected stop sign has
  `affects_ego_route=true`.
- Vehicle and pedestrian hazards retain their existing collector logic.

New training code should read `traffic_elements` directly. The legacy file is
kept only for compatibility with the current dataset loader.

## Audit Command

```bash
/data1/shijj/conda_envs/interfuser_origin/bin/python \
  tools/data/audit_traffic_element_labels.py <dataset-or-route-root>
```

The command rejects missing files, invalid JSON, unsupported schema versions,
malformed nested geometry, non-finite stop-line values, provenance mismatches,
inconsistent active-light or stop-sign relevance fields, and missing RGB/label
frame pairs. It reports state counts, route relevance, exact traffic-light stop
lines, approximate stop-sign stop lines, stop-sign frame and unique-actor
counts, and frames containing extraction errors.
