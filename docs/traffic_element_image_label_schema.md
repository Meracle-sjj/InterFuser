# Traffic Element Image Label Schema

Image schema version: `2`

Source traffic-element schema: `traffic_elements/*.json`, version `1`

`InterfuserDataCollector` writes one image-label record for every saved sensor
frame:

```text
<route>/traffic_element_views/0000.json
<route>/traffic_element_views/0001.json
...
```

The record projects the Phase 1 traffic-light, stop-sign, and stop-line labels
into the existing front, left, and right cameras. It is an auditable view of
the Phase 1 actors, not an independent source of traffic-control relevance.

## Association Constants

The version 2 constants are copied into every record:

```json
{
  "roi_expand_pixels": 6,
  "minimum_semantic_pixels": 3,
  "traffic_light": {
    "semantic_tag": 7,
    "depth_tolerance_m": 4.0
  },
  "stop_sign": {
    "semantic_tag": 8,
    "depth_tolerance_m": 6.0
  }
}
```

Readers must reject records whose constants differ from the values above.
Changing a tag, tolerance, minimum support, or ROI expansion changes label
semantics and therefore requires a new `schema_version`.

## Coordinate and Image Conventions

World geometry uses CARLA's left-handed coordinates in meters. Camera-space
coordinates use `forward`, `right`, and `up`. Projection uses:

```text
u = cx + focal * right / forward
v = cy - focal * up / forward
```

Only points with positive camera `forward` are projectable. Camera intrinsics
are calculated from the image width and horizontal field of view. The current
collector uses 400 x 300 images and a 100-degree field of view for front,
left, and right cameras.

`bbox_xyxy` and `geometry_roi_xyxy` use pixel coordinates in
`[x1, y1, x2, y2]` order. Left and top are inclusive; right and bottom are
exclusive. A valid box satisfies:

```text
0 <= x1 < x2 <= width
0 <= y1 < y2 <= height
```

For traffic lights, the geometry ROI is the union of the world-space boxes
returned by CARLA 0.9.16 `TrafficLight.get_light_boxes()`. The traffic-light
actor `bounding_box` is its road trigger volume and does not enclose the
visible lamps. Other actors use their bounding box or trigger volume. The ROI
is expanded by six pixels and clipped to the image. The final `bbox_xyxy` is
the tight box around semantic pixels that also satisfy the depth test.

CARLA depth images are received in OpenCV BGR byte order. Version 2 decodes
meters as:

```text
depth_m = 1000 * (B * 65536 + G * 256 + R) / (256^3 - 1)
```

The depth residual is the minimum difference to the camera distance of any
projected geometry center. A traffic-light actor can have several lamp boxes,
so each box center is retained as a valid expected depth.

## Top-Level Record

```json
{
  "schema_version": 2,
  "source_traffic_element_schema_version": 1,
  "frame_id": "0000",
  "association": {},
  "cameras": {
    "front": {},
    "left": {},
    "right": {}
  },
  "errors": []
}
```

| Field | Meaning |
| --- | --- |
| `schema_version` | Image-label schema version. Version 2 readers must reject other values. |
| `source_traffic_element_schema_version` | Version of the matching Phase 1 record. |
| `frame_id` | String frame ID; it must match the JSON and RGB filename stem. |
| `association` | Exact versioned constants shown above. |
| `cameras` | Exactly `front`, `left`, and `right` for audited collector output. |
| `errors` | Phase 1 extraction errors copied without reinterpretation. |

Each camera object contains `width`, `height`, `fov_degrees`,
`traffic_lights`, `stop_signs`, `stop_lines`, and `errors`. Camera errors
include missing sensors, missing actors, actor projection failures, and
stop-line projection failures.

## Traffic-Light and Stop-Sign Views

All Phase 1 actors are retained in each camera, including off-screen,
occluded, irrelevant, and unavailable actors. Actor identity and relevance
are never inferred again from image pixels.

Common fields are:

| Field | Meaning |
| --- | --- |
| `actor_id` | CARLA actor ID copied from Phase 1. |
| `element_type` | `traffic_light` or `stop_sign`. |
| `visibility` | `visible`, `not_visible`, or `unknown`. |
| `bbox_xyxy` | Tight semantic/depth-confirmed box, otherwise `null`. |
| `geometry_roi_xyxy` | Expanded projected geometry ROI; nullable when geometry is off-screen or unavailable. |
| `semantic_pixel_count` | Matching semantic/depth pixels inside the ROI. |
| `median_depth_residual_m` | Median absolute residual for matching pixels; non-null only for a visible item. |
| `association_source` | Evidence or failure state listed below. |
| `geometry_source` | `traffic_light_boxes`, `actor_bounding_box`, `trigger_volume`, or `null` when geometry was unavailable. |

Traffic-light views also copy `state`, `is_active_for_ego`,
`controls_ego_lane`, and `relevant_to_ego`. Stop-sign views copy
`affects_ego_route`. Their definitions remain those in
`docs/traffic_element_label_schema.md`.

### Visibility

`visible` means at least three pixels within the projected ROI have the
expected semantic tag and a depth residual within the configured tolerance.
Its `association_source` is `semantic_depth_confirmed`.

`not_visible` means projection and association were evaluated, but the actor
was off-screen or did not have enough semantic/depth support. Its
`association_source` is `semantic_depth_no_support`. This is a per-camera
visibility result, not a negative traffic-control relevance label.

`unknown` means the evidence required to evaluate visibility was unavailable.
Its `association_source` is one of `actor_missing`, `sensor_unavailable`, or
`projection_error`. Unknown items must not be used as negative examples.

`bbox_xyxy` is null for both `not_visible` and `unknown`. Extraction or camera
errors must be filtered explicitly before negative-label training.

CARLA 0.9.16 identifies traffic-light pixels with `CityObjectLabel` value 7
and traffic-sign pixels with value 8. Values 18 and 12 are Motorcycle and
Pedestrians in this build and must not be used for these labels.

A Phase 1 `traffic.stop` actor can exist without a visible vertical stop-sign
asset. Town03 route36 is such a case: RGB contains a road-surface `STOP`
marking, while the semantic image contains no tag-8 sign pixels. The actor is
retained as `not_visible`; its route-relevant projected stop line remains valid
supervision. Consumers must not synthesize a stop-sign visual positive from
the trigger volume.

## Stop-Line Views

Each Phase 1 stop line is projected independently into every camera:

```json
{
  "owner_actor_id": 11,
  "owner_type": "traffic_light",
  "geometry_source": "carla_stop_waypoint",
  "longitudinal_distance": 8.0,
  "ego_before_line": true,
  "projected_endpoints": [[12.5, 220.0], [387.0, 220.0]],
  "image_segment": [[12.5, 220.0], [387.0, 220.0]],
  "projection_status": "projected"
}
```

`owner_actor_id`, `owner_type`, `geometry_source`,
`longitudinal_distance`, and `ego_before_line` are copied from the matching
Phase 1 actor and stop line. Distances are ego-forward meters and may be
negative after crossing.

Traffic-light lines use `carla_stop_waypoint`. Stop-sign lines use
`trigger_volume_route_entry_approximation`. The latter remains approximate
and must not be presented as an exact simulator stop position.

`projected_endpoints` contains the raw image projections when both world
endpoints are in front of the camera. These points may lie outside the image.
`image_segment` contains the line clipped to inclusive image bounds and is
non-null only when `projection_status` is `projected`.

Valid projection statuses are:

| Status | Meaning |
| --- | --- |
| `projected` | Both endpoints are in front and the clipped segment intersects the image. |
| `outside_image` | Both endpoints are in front, but the segment does not intersect the image. |
| `behind_camera` | At least one endpoint is not in front of the camera. |

If projection raises an exception, the line is omitted and the camera error
is recorded. A missing line with an error is unknown evidence, not proof that
the stop line is absent.

## Audit and Overlay Commands

```bash
/data1/shijj/conda_envs/interfuser_origin/bin/python \
  tools/data/audit_traffic_element_views.py <dataset-or-route-root>

/data1/shijj/conda_envs/interfuser_origin/bin/python \
  tools/data/render_traffic_element_overlays.py <dataset-or-route-root> \
  --output-dir <review-directory> --limit 12 --camera front
```

The audit requires matching RGB, Phase 1, and image-view frame IDs; exact
front/left/right camera keys; Phase 1 actor identity preservation; finite,
in-bounds positive-area boxes and line segments; minimum semantic support;
and consistent association constants and stop-line provenance.
