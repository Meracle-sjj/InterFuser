# Traffic-Element Label Schema v2

This schema records route-associated traffic-light stop targets. It replaces
the legacy roadside-sign schema and every approximate sign-derived line.

The target is a safe driving reference near a signalized junction, not a claim
that CARLA contains a visible painted white line at that location. RGB, depth,
semantic, and lidar evidence are recorded separately under evidence schema v3.

## Source of truth

The boundary geometry mirrors Leaderboard 1.0 Scenario Runner
`RunningRedLightTest`:

1. Transform the traffic-light trigger volume into world coordinates.
2. Sample 90 percent of its lateral extent at 1.0 m intervals.
3. Map samples to lane waypoints and collapse consecutive lane duplicates.
4. Advance with `waypoint.next(0.5)[0]` until the next waypoint is inside an
   intersection.
5. Place boundary endpoints at 0.4 times the lane width on either side of the
   final non-intersection waypoint.

The provenance string is
`scenario_runner_running_red_light_test_v1`. The trigger waypoint provenance
is `carla_traffic_light_trigger_waypoint`.

`net_is_junction`, `aux_junction`, and `new_junction_prediction` are learned
outputs. They never create, suppress, or modify labels. OpenDRIVE
`Waypoint.is_junction` / `is_intersection` is used only because the Leaderboard
criterion uses it to locate its infraction boundary.

## Versioned constants

| Constant | Value |
| --- | ---: |
| Trigger extent ratio | 0.9 |
| Trigger sample step | 1.0 m |
| Boundary traversal step | 0.5 m |
| Boundary half-width ratio | 0.4 lane width |
| Maximum traversal | 400 steps |
| Route inclusion window | -10.0 m to +80.0 m |
| Route match radius | 4.0 m |
| Stop-pose safety margin | 1.0 m |
| Corridor extension | 3.0 m |
| Corridor sample step | 0.5 m |

Changing a geometry constant requires a new provenance string and schema
version.

## Top-level record

Each `traffic_elements/<frame>.json` file contains:

```json
{
  "schema_version": 2,
  "frame_id": "0000",
  "map_name": "Town03_Opt",
  "ego": {},
  "active_traffic_light_id": 123,
  "traffic_lights": [],
  "stop_targets": [],
  "errors": []
}
```

`frame_id` must equal the file stem. `errors` contains collection failures and
must be empty for an accepted batch.

## Traffic lights

Every nearby light retains:

- actor ID, CARLA type, state, world location, and rotation;
- ego-relative forward/right/up position and relative heading;
- trigger-volume geometry;
- affected OpenDRIVE lanes;
- `is_active_for_ego`, `controls_ego_lane`, and `relevant_to_ego`.

An active light is retained even when its actor origin is outside the normal
80 m actor radius. A nearby light that does not own the planned route remains
useful visual hard-negative evidence, but it cannot create a stop target.

Traffic-light records do not contain a stop-line field. CARLA
`get_stop_waypoints()` is not used as a physical-line source.

## Stop targets

Only a traffic-light boundary associated with the dense planner route is
included. Route association uses road, section, lane, heading, and cumulative
route arc length.

The stable ID is:

```text
<map>:<road_id>:<section_id>:<lane_id>:<boundary_s rounded to 0.1 m>
```

It never contains a run-specific actor ID. Multiple traffic-light actors that
own the same lane boundary are grouped under one ID.

Each target records:

- `owner_traffic_light_actor_ids` and `state_by_actor_id`;
- `status`, `unknown_reason`, and `primary_for_ego`;
- `route_lane` and geometry provenance;
- `trigger_stop_waypoint`;
- `leaderboard_infraction_boundary`, including center and endpoints;
- `recommended_ego_stop_pose`;
- vehicle front offset and 1.0 m safety margin;
- `stop_evidence_corridor` centerline and lane width samples;
- signed route distance, Euclidean distance, relative heading, and crossing
  state;
- trigger-to-boundary route distance and traversal diagnostics.

The recommended ego-center pose is upstream of the boundary by:

```text
bounding_box.location.x + bounding_box.extent.x + 1.0 m
```

The signed route distance is authoritative. A target between -10 m and 0 m is
retained as recently crossed. The nearest valid non-negative target is the
only target allowed to set `primary_for_ego=true`.

## Unknown policy

Ambiguous or incomplete geometry is stored with `status="unknown"` and an
explicit reason such as:

- `waypoint_branch`;
- `waypoint_loop`;
- `intersection_not_found`;
- `direction_mismatch`;
- `route_lane_not_found`;
- `route_light_ambiguous`;
- `trigger_waypoint_unavailable`.

Unknown geometry is never converted into either a positive or a hard negative.
Missing route context produces no synthetic negative target.

## Forbidden legacy content

Schema v2 rejects any generated JSON containing a key with the fragment
`stop_sign`, an actor value beginning with `traffic.stop`, or the legacy
provenance `trigger_volume_route_entry_approximation`. This explicitly covers
roadside red signs, road-surface `STOP` text, old sign actors, and old
sign-derived approximate lines.

The audit scans JSON under `traffic_elements`, `traffic_element_views`,
`measurements`, `affordances`, `3d_bbs`, `2d_bbs_*`, and `other_actors`. It does
not scan raw images, lidar arrays, logs, results, or documentation.

## Validation

From the repository root:

```bash
python tools/data/audit_traffic_element_labels.py <dataset-root>
```

The audit enforces schema version, frame alignment, stable IDs, owner-light
references, provenance, finite geometry, finite signed distances, unique
primary selection, explained unknowns, and zero forbidden occurrences.
