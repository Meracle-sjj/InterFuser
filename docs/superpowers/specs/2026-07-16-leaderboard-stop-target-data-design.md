# Leaderboard-Aligned Traffic-Light Stop Target Data Design

Date: 2026-07-16

Branch: `codex/fix-background-traffic`

## Context

The current traffic-element collector mixes two different supervision goals:

- traffic-light lamp visibility and state;
- stop geometry from CARLA traffic-light and `traffic.stop` actors.

The `traffic.stop` concept is outside the intended driving policy. Roadside red
STOP signs, road-surface `STOP` text, and trigger-volume-derived stop-sign lines
must not be collected as training targets.

The current traffic-light line is also not a physical stop-line ground truth.
CARLA 0.9.16 `TrafficLight.get_stop_waypoints()` samples lane waypoints across
the traffic-light trigger volume. It does not verify a painted transverse line
or the junction boundary. Measurements on Town01, Town03, and Town04 showed:

- trigger waypoint to traffic-light actor origin: 7.482 to 43.903 m;
- trigger waypoint to trigger-volume center: median 0.343 m, p95 1.886 m;
- trigger waypoint to the last non-intersection waypoint: 6.5 to 13.0 m in
  Town01, 0 to 6.0 m in Town03, and 0 to 7.5 m in Town04;
- 11 of 60 Town03 trigger waypoints and 12 of 52 Town04 trigger waypoints were
  already marked as junction waypoints.

The existing image projections do not reliably identify painted stop lines.
Of 1,115 front-camera trigger-line projections between 0 and 30 m before the
ego vehicle, 821 contained no `RoadLines` semantic pixels, and none had at
least 50 percent `RoadLines` support. Map-level `RoadLines` environment objects
are also merged inconsistently between towns and cannot be used as a portable
stop-line source.

The useful target is therefore a Leaderboard-aligned stopping region with
aligned RGB, depth, semantic, and lidar evidence. A painted stop line is
optional evidence, not a requirement for a valid target.

## Goals

1. Reproduce the stop boundary used by Leaderboard's
   `RunningRedLightTest` and expose its provenance explicitly.
2. Derive a conservative ego-center stop pose upstream of that boundary.
3. Associate the boundary and its surrounding route corridor with aligned
   camera, depth, semantic, and lidar evidence.
4. Retain nearby traffic lights for lamp/state learning and hard negatives.
5. Exclude every STOP-sign concept and approximate stop-sign line from new
   training schemas, audits, overlays, and acceptance reports.
6. Treat ambiguous geometry and missing evidence as unknown, never as a
   negative label.

## Non-Goals

- Training or changing the driving model in this phase.
- Treating a learned junction prediction as label ground truth.
- Guaranteeing that the Leaderboard boundary coincides with painted road
  markings.
- Reconstructing v3 targets from the existing v2 batch.
- Changing Scenario Runner or Leaderboard evaluation behavior.

## Source-of-Truth Boundaries

The repository contains three similarly named junction signals. They must not
be conflated:

- `net_is_junction` is the Interfuser model output.
- `aux_junction` / `new_junction_prediction` is an external learned linear
  classifier output.
- `carla.Waypoint.is_junction` / `is_intersection` is OpenDRIVE map topology.

Neither learned output is permitted in label generation. The OpenDRIVE flag is
used only because the Leaderboard evaluator itself uses it to construct the
red-light infraction boundary. The schema names that boundary
`leaderboard_infraction_boundary`; it does not call it a physical stop line.

### Trigger Stop Waypoint

`trigger_stop_waypoint` records the traffic-light trigger lane waypoint. It is
an association seed and corridor endpoint, not a training target by itself and
not necessarily upstream of the evaluator boundary. Its provenance is
`carla_traffic_light_trigger_waypoint`.

### Leaderboard Infraction Boundary

The collector mirrors `RunningRedLightTest.get_traffic_light_waypoints()`:

1. Transform the traffic-light trigger volume to world coordinates.
2. Sample across 90 percent of its lateral extent at 1 m intervals.
3. Map samples to lane waypoints and deduplicate consecutive road/lane pairs.
4. Advance each waypoint in 0.5 m increments while the next waypoint remains
   outside an intersection.
5. Build the transverse boundary from 0.4 times the lane width on each side,
   matching `RunningRedLightTest`.

The collector records all branch counts. It follows the evaluator's first
branch for parity, but marks the target unknown when any step has more than one
successor because route ownership is ambiguous. It also detects repeated
waypoints and limits the search to 400 steps (200 m). A loop, exhausted search,
missing waypoint, or direction mismatch produces an unknown target with an
explicit reason.

The provenance string is
`scenario_runner_running_red_light_test_v1`. Any change to sampling, step size,
line width, or ambiguity handling requires a new provenance value and schema
version.

### Route Association

Only a boundary associated with the ego vehicle's planned route becomes a stop
target. Association requires:

- matching road, section, and lane identity along the forward route;
- a positive heading dot product between ego-route and boundary waypoints;
- a unique route/lane boundary after equivalent owner lights are grouped.

All unique route-associated boundaries with signed route distance in
`[-10 m, 80 m]` are retained. This preserves approach frames and a short
post-crossing audit window. The nearest non-negative valid target is marked
`primary_for_ego`; recently crossed targets remain non-primary records. Lights
that resolve to the same map, road, section, lane, and quantized boundary `s`
are grouped into one target with all owner actor IDs recorded.

Nearby lights that fail route association remain in `traffic_lights` as visual
hard negatives. They do not produce stop targets.

### Recommended Ego Stop Pose

`recommended_ego_stop_pose` is placed upstream of the Leaderboard boundary by:

```text
vehicle_front_offset_m + safety_margin_m
```

The front offset is `bounding_box.location.x + bounding_box.extent.x` in
vehicle-local forward coordinates. The initial version uses a recorded,
versioned safety margin of 1.0 m. The pose is an auxiliary control target, not
an evaluator rule.

## Stop Evidence Corridor

The `stop_evidence_corridor` covers the route interval between the trigger stop
waypoint and Leaderboard boundary, regardless of which endpoint is upstream.
It extends 3.0 m beyond each endpoint. Its centerline is sampled from route
waypoints at 0.5 m spacing and retains lane width and road elevation per
sample. This polyline representation supports curved approaches without
reducing them to an axis-aligned box.

The corridor is the primary supervision region. It may contain a painted stop
line, road texture, curb geometry, traffic-light structures, depth cues, or
lidar road-surface returns. A valid target does not require any one cue.

## Schema Changes

### Traffic-Element Schema Version 2

`traffic_elements/*.json` moves from version 1 to version 2. Its top-level
training fields are:

```json
{
  "schema_version": 2,
  "frame_id": "0000",
  "ego": {},
  "traffic_lights": [],
  "stop_targets": [],
  "errors": []
}
```

There is no `stop_signs` field. Records containing `stop_signs`,
`owner_type="stop_sign"`, `traffic.stop`, or
`trigger_volume_route_entry_approximation` are invalid under version 2.

Each stop target contains:

- stable target ID derived from map, road, section, lane, and quantized
  boundary `s`, never from a run-specific actor ID;
- owner traffic-light actor IDs and their current states;
- `primary_for_ego` selection status;
- route/lane identity and association evidence;
- `status`: `valid` or `unknown`;
- `unknown_reason`, null for valid targets;
- trigger stop waypoint;
- Leaderboard infraction boundary and endpoints;
- recommended ego stop pose and offset constants;
- stop-evidence corridor centerline and width samples;
- signed route-arc distance, Euclidean distance, relative heading, and
  before/after state;
- geometry provenance and branch/search diagnostics.

Signed route distance is authoritative for approach/crossing state. Euclidean
and ego-forward distances remain diagnostics because they are misleading on
curved approaches.

### Image/Evidence Schema Version 3

`traffic_element_views/*.json` moves from image schema version 2 to evidence
schema version 3 and references traffic-element schema version 2.

It retains traffic-light views and their semantic/depth-confirmed lamp boxes.
It removes all stop-sign association constants and all stop-sign views.

For every stop target and camera, the record stores:

- projected Leaderboard boundary endpoints and clipped segment;
- projected recommended stop pose;
- projected corridor road-surface polygon or polyline envelope;
- projection status: `projected`, `outside_image`, `behind_camera`, or
  `unknown`;
- finite-depth support count, residual statistics, and occlusion status;
- optional visible-stop-line status: `unknown`, `candidate`, or `verified`.

Only manually reviewed `verified` stop lines are positive painted-line labels.
`candidate` is review input, and `unknown` is ignored by painted-line losses.

For lidar, the record stores:

- the exact sensor-to-ego and ego-to-world transforms used for the frame;
- the corridor centerline transformed into lidar coordinates;
- per-sample half-width and road-height information;
- total in-corridor point count and road-surface-band point count;
- evidence status and extraction errors.

The complete aligned lidar array remains in `lidar/<frame>.npy`. Point indices
or duplicate cropped point clouds are not stored; the training loader can
reproduce the crop from the versioned corridor geometry.

## Painted-Line Candidate Generation

Painted-line discovery is optional and cannot invalidate a stop target. A
candidate must:

- lie inside the projected stop-evidence corridor;
- be transverse to the local route heading;
- have depth consistent with the corridor road surface;
- span a meaningful fraction of the controlled lane;
- be associated with a valid traffic-light stop target.

The transverse-line geometry and traffic-light route association reject most
road-surface `STOP` text, but they are not treated as a proof of exclusion.
Semantic `RoadLines` support is recorded but is not sufficient by itself. An
automatic result remains `candidate` until manual review confirms that it is a
painted transverse stop line; only then may it become `verified`. This review
creates a small gold set. A later model may use that set for pseudo-labeling,
but pseudo-label generation is outside this collection phase.

## Error Handling

Frame-level sensor or actor failures remain in the top-level `errors` array.
Target-specific topology and association failures use `status="unknown"` with
one of these explicit reason categories:

- `trigger_waypoint_unavailable`;
- `route_lane_not_found`;
- `route_light_ambiguous`;
- `direction_mismatch`;
- `waypoint_branch`;
- `waypoint_loop`;
- `intersection_not_found`.

Projection and sensor failures do not invalidate otherwise valid geometry.
Each camera and lidar evidence object has its own `status="unknown"` and an
explicit `projection_error` or `sensor_unavailable` reason.

Unknown targets remain auditable but are excluded from positive and negative
training losses. A camera outside the field of view is a known visibility
state; a missing sensor or failed projection is unknown evidence.

## Audit and Review

The phase-1 and evidence audits report per town and route:

- frames, traffic lights, valid stop targets, and unknown targets by reason;
- trigger-to-boundary and ego-to-boundary route-distance distributions;
- branch, loop, and missing-intersection counts;
- before/after-crossing counts;
- per-camera projected-corridor and valid-depth counts;
- lidar in-corridor and road-surface point distributions;
- visible-line candidate and verified counts;
- hard-negative frames with visible irrelevant traffic lights;
- forbidden STOP keys, values, actors, and provenance occurrences.

Forbidden STOP occurrences make the audit fail. Invalid geometry, non-finite
coordinates, mismatched frame IDs, unsupported schema constants, unexplained
errors, and silently converted unknowns also fail.

Review overlays use distinct colors for:

- trigger waypoint;
- Leaderboard infraction boundary;
- recommended ego stop pose;
- stop-evidence corridor;
- optional painted-line candidate;
- route-relevant and irrelevant traffic-light lamp boxes.

Review samples cover multiple towns, near/far approaches, curved lanes,
occlusion, before/after crossing, branches, and hard negatives.

## Testing

Unit tests cover:

- exact 1 m trigger sampling and road/lane deduplication;
- exact 0.5 m evaluator-boundary advancement;
- 0.4-lane-width boundary endpoints;
- route association and signed route distance;
- vehicle-front offset and 1.0 m safety margin;
- branch, loop, missing intersection, and direction mismatch handling;
- camera projection and clipping of curved corridor samples;
- lidar coordinate transforms and reproducible corridor crops;
- schema rejection of every stop-sign field and provenance;
- audit summary and overlay behavior.

CARLA integration checks run on Town01, Town03, and Town04. They compare the
collector boundary against an independent reproduction of
`RunningRedLightTest`, validate finite sensor projections, and report all
per-town distance distributions. The current 63-test traffic-element baseline
must remain green in addition to the new tests.

## Small-Batch Rollout

After implementation, rerun route13, route36, route00, and route39 with the
existing bounded runner:

- CARLA RPC 2400;
- Traffic Manager 8400;
- GPU 2;
- approximately 1,000 frames;
- maximum 2,000 frames and 2 GiB;
- port-2000 process identity checked before and after collection.

The batch is accepted only when:

- both audits exit zero;
- there are zero invalid records and zero unexplained collection errors;
- there are zero STOP-sign targets or approximate stop-sign lines;
- every valid target has traffic-light ownership and the required Leaderboard
  provenance;
- unknown targets are grouped by reason and town rather than converted to
  negatives;
- camera and lidar evidence statistics are present for every valid target;
- deterministic overlays from each covered town pass manual review.

The existing v2 batch remains diagnostic evidence only. Any future training
reader for this dataset must reject traffic-element schema version 1 and image
schema version 2 so legacy STOP labels cannot enter the new training set
accidentally. Implementing that future model-training reader is outside this
collection phase.

## Implementation Boundaries

The implementation is limited to traffic-element geometry, evidence
projection, audits, overlays, documentation, tests, and the bounded batch
runner. It does not modify `interfuser_agent.py`,
`interfuser_controller.py`, Scenario Runner criteria, or model training code.
