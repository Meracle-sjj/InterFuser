# Traffic Element Image Projection and Small-Batch Collection Design

Date: 2026-07-15

## Goal

Extend the Phase 1 CARLA traffic-element labels with auditable image-space
supervision for traffic lights, stop signs, and stop lines. Prove label quality
with a bounded multi-route collection before any large-scale visual pretraining
dataset is generated.

The work must keep the current collection entry point:

```text
leaderboard/team_code/interfuser_collector_complete.py
```

It must not switch the run to `BaseAgent`, `MapAgent`, or
`CarlaAutopilotAgent`.

## Verified Repository Context

- `InterfuserAgent` directly inherits `AutonomousAgent`.
- `InterfuserDataCollector` directly inherits `AutonomousAgent`.
- Current evaluation scripts invoke `interfuser_agent.py`; the completed
  Phase 1 smoke runs invoked `interfuser_collector_complete.py`.
- `base_agent.py` contains projection, semantic-segmentation, and depth
  association code, but it is not in either active inheritance chain.
- The collector currently writes placeholder 2D boxes `[0, 0, 0, 0]`.
- The available front, left, and right camera groups each contain aligned RGB,
  semantic segmentation, and depth sensors at 400 x 300 with 100 degree FOV.
- The rear camera has RGB only.

The implementation may use `base_agent.py` as a checked mathematical
reference, but it must not import from it or depend on its runtime state.

## Alternatives

### A. Independent semantic/depth-gated projection

Create a standalone projection module. Associate CARLA semantic pixels with
nearby traffic-element actors using decoded depth and projected geometry. Save
object boxes, visibility evidence, and projected stop-line segments.

This is the selected approach. It reuses sensors already present in the
collector, accounts for occlusion better than geometry alone, and does not
change the active agent hierarchy.

### B. Geometry-only 3D projection

Project actor or trigger-volume corners without checking semantic or depth
images. This is simpler, but it produces positive boxes for occluded objects
and for structures outside the visible part of the image.

This is rejected as the primary training label. Geometry-only results may be
retained as diagnostics when semantic visibility is absent.

### C. CARLA instance-segmentation cameras

Add instance-segmentation sensors and derive masks from simulator instance
IDs. This can provide better masks but increases the sensor suite, disk usage,
and evaluator risk. It also needs a separate validation of CARLA 0.9.16 pixel
encoding.

This is deferred until the small batch shows that semantic/depth association is
insufficient.

## Architecture

### Pure projection module

Add `leaderboard/team_code/traffic_element_projection.py`. It owns:

- camera intrinsic matrix construction from width, height, and FOV;
- CARLA world-to-camera coordinate conversion;
- projection and clipping of world points;
- CARLA depth-image decoding;
- semantic/depth actor association;
- bounding-box extraction from matched pixels;
- stop-line endpoint projection;
- schema validation helpers.

The module accepts duck-typed transforms, points, image arrays, and Phase 1
records. It has no agent inheritance and no global CARLA state.

### Collector integration

`InterfuserDataCollector` continues to collect Phase 1 world labels once per
saved frame. It then reads the actual front/left/right sensor transforms from
the collector's existing sensor interface and calls the pure projection
module.

If a required sensor transform or image is unavailable, the frame receives a
structured error and no positive image label is synthesized. Missing evidence
must never become a negative label.

Add one directory:

```text
<route>/traffic_element_views/0000.json
```

Existing `traffic_elements`, RGB, segmentation, depth, affordance, and
measurement files remain unchanged.

### Audit and visualization

Extend the data audit or add a focused image-label audit that checks:

- one-to-one frame IDs across RGB, Phase 1 labels, and image labels;
- source actor IDs exist in the matching Phase 1 record;
- visible boxes have finite integer coordinates, positive area, and lie inside
  the 400 x 300 image;
- visible boxes contain semantic support pixels;
- stop-line image points are finite and preserve Phase 1 provenance;
- errors and unknown visibility are counted separately from negatives.

Add an offline overlay command. It draws sampled boxes and stop-line segments
on existing RGB images and writes review images outside the training data
directories.

## Image Label Schema

Schema version 1 is separate from the Phase 1 world-label schema:

```json
{
  "schema_version": 1,
  "source_traffic_element_schema_version": 1,
  "frame_id": "0052",
  "cameras": {
    "front": {
      "width": 400,
      "height": 300,
      "fov_degrees": 100.0,
      "traffic_lights": [],
      "stop_signs": [],
      "stop_lines": [],
      "errors": []
    }
  }
}
```

Each traffic-light item records:

- actor ID and Phase 1 state;
- active-light and ego-lane relevance flags;
- `bbox_xyxy` or `null`;
- visibility state: `visible`, `not_visible`, or `unknown`;
- semantic pixel count;
- median depth residual in meters;
- geometry and association provenance.

Each stop-sign item records the same visibility evidence plus
`affects_ego_route`.

Each stop-line item records:

- owner actor ID and type;
- Phase 1 geometry source;
- projected endpoints when both are in front of the camera;
- clipped image segment when it intersects the image;
- world-label longitudinal distance and `ego_before_line`;
- projection status and structured errors.

All nearby Phase 1 actors remain represented even when not visible. A null box
with `not_visible` is different from a missing or failed label.

## Association Rules

For each camera and actor:

1. Project the actor bounding box into an image ROI. If the actor has no usable
   bounding box, project its trigger volume and record that fallback
   provenance. Expand the clipped ROI by 6 pixels on each side.
2. Decode CARLA's three-channel depth image with the repository's existing
   24-bit formula and compute the camera-to-actor distance.
3. Select semantic tag 18 for traffic lights or tag 12 for stop signs.
4. Inside the ROI, retain pixels whose decoded depth differs from actor
   distance by at most 4 m for traffic lights or 6 m for stop signs.
5. Derive a box only when at least 3 pixels remain. The visible box is the
   tight bounds of those retained pixels, not the projected actor ROI.

The ROI expansion, semantic tags, depth tolerances, and minimum pixel count are
written into every image-label record as association metadata. Changing any of
them requires a new image-label schema version.

Projection-only boxes are diagnostics and must carry a different provenance
from semantic/depth-confirmed boxes.

## Error Handling

- Sensor image or transform missing: record a camera error and mark affected
  labels `unknown`.
- Point behind camera: `not_visible`, not an extraction error.
- Projected geometry outside the image: `not_visible`.
- Semantic pixels absent for an in-frame projection: `not_visible` with
  diagnostic projection retained.
- Invalid or non-finite geometry: frame fails audit.
- Phase 1 actor ID mismatch: frame fails audit.
- Image-label write failure: frame-set alignment fails audit.

The collector must calculate labels before writing a completed image-label
file. It writes a sibling temporary file and publishes it with an atomic rename
only after JSON serialization succeeds. Temporary or partial files must not be
treated as valid frames.

## Testing

Implementation follows red-green TDD:

- intrinsic matrix and CARLA axis-convention tests;
- world-to-camera tests at yaw 0, +90, and -90 degrees;
- point and stop-line projection tests;
- depth decoding tests with known encoded values;
- semantic/depth association positive, occluded, off-screen, and ambiguous
  cases;
- schema and frame-alignment audit failures;
- collector integration test proving one projection call per saved frame.

Live verification must include:

- an active red-light frame;
- an irrelevant visible light;
- a route-relevant stop-sign frame;
- a no-active-light hard negative;
- a partially or fully occluded actor when background traffic is enabled.

## Small-Batch Collection

The first batch is a label-quality batch, not a training release.

Budget:

- at most four routes;
- at most 2,000 saved frames;
- at most 2 GB total output;
- front, left, and right image labels only;
- no change to the 2 Hz save rate.

Route mix:

1. Town03 route13, known traffic-light coverage, background traffic 0.
2. Town03 route36, known route-relevant stop-sign coverage, background traffic
   0.
3. One Town01 or Town04 route selected by a topology preflight for traffic-light
   diversity, with 20 background vehicles.
4. One Town01 or Town04 route selected for hard negatives and occlusion, with
   20 background vehicles.

The preflight records why routes 3 and 4 were selected. It must inspect CARLA
map and actor geometry rather than infer coverage from route names.

The Phase 1 smoke data used about 66 MB for 96 frames and 311 MB for 463
frames, approximately 0.67 MB per saved frame. The 2,000-frame budget therefore
projects to roughly 1.35 GB before overlays and remains below the 2 GB cap.

Acceptance criteria:

- zero invalid frames;
- zero unmatched RGB/world-label/image-label frame IDs;
- zero non-finite or zero-area visible boxes;
- at least one semantic/depth-confirmed active traffic light;
- at least one semantic/depth-confirmed route-relevant stop sign;
- at least one projected relevant stop line before and after crossing;
- at least 20 no-active-light hard-negative frames;
- twelve manually reviewed overlays covering positive, irrelevant, negative,
  and occlusion cases;
- the batch remains inside the frame and storage budgets.

Failure of any criterion stops expansion. Thresholds or projection logic are
fixed and the same bounded batch is rerun before collecting more data.

## Non-Goals

- No visual model training in this phase.
- No controller changes.
- No replacement of InterFuser classification heads.
- No runtime dependency on `BaseAgent`.
- No pixel-perfect stop-line mask; Phase 2 emits projected line geometry.
- No large-scale collection until the bounded batch passes.
