# Traffic-Element Evidence Schema v3

Evidence schema v3 associates traffic-element schema v2 with aligned RGB,
semantic, depth, and lidar observations. A valid route target remains valid
when a camera is occluded or a sensor is unavailable; geometry status and
sensor-evidence status are intentionally separate.

## Top-level record

Each `traffic_element_views/<frame>.json` file contains:

```json
{
  "schema_version": 3,
  "source_traffic_element_schema_version": 2,
  "frame_id": "0000",
  "association": {},
  "cameras": {
    "front": {},
    "left": {},
    "right": {}
  },
  "lidar": {"targets": [], "errors": []},
  "errors": []
}
```

The evidence file, phase-v2 file, front/left/right RGB images, and complete
lidar array must share the same frame stem.

## Evidence constants

The serialized `association` object must exactly match:

```json
{
  "roi_expand_pixels": 6,
  "minimum_semantic_pixels": 3,
  "traffic_light": {
    "semantic_tag": 7,
    "depth_tolerance_m": 4.0
  },
  "road_lines_semantic_tag": 24,
  "corridor_depth_tolerance_m": 2.0,
  "lidar_min_height_m": -0.5,
  "lidar_max_height_m": 3.0,
  "lidar_road_surface_tolerance_m": 0.25
}
```

## Coordinates

CARLA world and sensor transforms use homogeneous local-to-world 4x4 matrices.
Camera-local axes are forward, right, and up. Projection maps these to image
right, image down, and depth. Pixel boxes are `xyxy` with an exclusive right
and bottom edge; projected points and segments must lie within numeric image
bounds.

Lidar points remain in the complete `lidar/<frame>.npy` array. The evidence
record stores enough transforms and corridor geometry to reproduce the crop;
it does not duplicate points or save point indices.

## Traffic-light evidence

Each camera retains one view per phase-v2 traffic-light actor. CARLA 0.9.16
light-box geometry defines the projection ROI where available. A lamp box is
`visible` only when at least three semantic-tag-7 pixels also agree with one of
the light-box depths within 4.0 m.

Missing actors or sensors produce `visibility="unknown"`. A valid projection
without semantic/depth support produces `not_visible`, which is a known visual
state rather than missing evidence.

## Stop-target camera evidence

Each phase-v2 target appears exactly once under every camera and records:

- `target_id`, owner lights, geometry provenance, and signed route distance;
- projected trigger waypoint;
- projected Leaderboard infraction boundary endpoints and clipped segment;
- projected recommended ego stop pose;
- corridor center polyline and lane-width envelope;
- finite-depth sample count, supported-depth count, median residual, and
  occlusion status;
- optional `painted_line` review evidence.

Projection states are:

- `projected`: geometry intersects the image;
- `outside_image`: in front of the camera but outside its bounds;
- `behind_camera`: not in the forward half-space;
- `unknown`: geometry or sensor processing could not be evaluated.

For valid projected corridor samples, decoded depth is compared with the
camera-forward sample depth. Residuals at or below 2.0 m count as support. If
finite depth exists but no sample meets the threshold, the corridor is marked
`occluded`.

If the phase target has unknown topology, camera evidence must use
`unknown_reason="geometry_unknown"` and preserve the phase reason separately.
Camera processing cannot reinterpret it as a negative. For valid geometry,
missing or failed sensors use `sensor_unavailable` or `projection_error`.

## Painted-line review evidence

A painted line is optional. It is not required for a valid stop target.

Automatic extraction is constrained to the projected corridor. It uses Canny
edges and probabilistic Hough segments, then requires:

- undirected angle within 15 degrees of the projected evaluator boundary;
- length at least 25 percent of the projected boundary length;
- median finite depth residual at most 2.0 m.

Semantic `RoadLines` support is recorded as a diagnostic and is not proof by
itself. Automatic code returns only `candidate` or `unknown`; it never returns
`verified`.

Generate review overlays and a manifest with:

```bash
python tools/data/render_traffic_element_overlays.py <dataset-root> \
  --output-dir <overlay-dir> --camera front --limit 16 \
  --review-manifest-output <manifest.json>
```

Manifest decisions are `verified`, `rejected`, or `unreviewed`. Apply explicit
decisions with:

```bash
python tools/data/apply_painted_line_reviews.py \
  <dataset-root> <manifest.json>
```

Only an existing `candidate` can become `verified` or be rejected. A verified
entry must carry `review_source="manual_manifest"`. The entire manifest is
validated before any evidence file is replaced.

## Lidar target evidence

For each target, the record stores:

- `sensor_to_ego` and `ego_to_world` 4x4 matrices;
- corridor centerline in lidar coordinates;
- per-sample half widths and road heights;
- total in-corridor point count;
- road-surface-band point count.

The crop accepts points within the nearest corridor sample half width and the
height interval [-0.5 m, 3.0 m]. Road-surface support uses an absolute relative
height tolerance of 0.25 m. Counts are diagnostics and may be zero. Missing or
invalid data is `unknown`, never a negative.

## Forbidden legacy content

Evidence schema v3 has no roadside-sign views, sign association constants, or
generic sign-derived lines. Generated JSON containing a `stop_sign` key,
`traffic.stop` actor value, or
`trigger_volume_route_entry_approximation` provenance is invalid. The word is
mentioned here only to define the compatibility rejection rule.

## Validation and overlay legend

Run:

```bash
python tools/data/audit_traffic_element_views.py <dataset-root>
```

The audit checks schema linkage, exact evidence constants, frame alignment,
camera bounds, phase target/actor ownership, geometry-versus-sensor unknown
states, lidar transforms/counts, and manual verification provenance.

Overlay colors are BGR:

| Layer | Color |
| --- | --- |
| Trigger waypoint | `(255, 255, 0)` |
| Leaderboard boundary | `(255, 0, 255)` |
| Recommended stop pose | `(255, 0, 0)` |
| Route corridor | `(0, 180, 0)` |
| Painted-line candidate | `(0, 255, 255)` |

Recommended stop pose markers (`(255, 0, 0)`, BGR red cross) are only drawn
when the target projects inside the image (`projection_status: projected`). When
the target is close and low, the recommended stop pose may project beyond the
bottom edge of the frame (`projection_status: outside_image`); in that case no
marker is drawn and the absence is not an overlay bug. Inspect the source JSON
`projection_status` field to confirm. (Observed on frame 0046 of the
`20260716_leaderboard_stop_targets_camera_fix` batch.)

Learned junction outputs never generate or validate evidence.
