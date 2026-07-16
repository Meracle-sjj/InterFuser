# Traffic Element Image Projection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add auditable front/left/right image labels for CARLA traffic lights, stop signs, and stop lines, then validate them with a bounded four-route collection.

**Architecture:** A new pure `traffic_element_projection.py` module owns camera math, depth decoding, semantic/depth association, and schema construction without importing `BaseAgent`. The existing `interfuser_collector_complete.py` remains the run entry point; its base collector writes one atomic `traffic_element_views/*.json` record per saved frame. Separate audit, overlay, and route-profile tools keep validation and batch selection outside the driving agent.

**Tech Stack:** Python 3.10, NumPy, OpenCV, PIL, CARLA 0.9.16 Python API, `unittest`, JSON, existing Leaderboard sensor interface.

**Execution root:** `/data/shijj/interfuser_origin` on `ghbserver02-frpMe`, branch `codex/fix-background-traffic`.

---

## File Map

- Create `leaderboard/team_code/traffic_element_projection.py`: pure camera transforms, projection, depth decoding, semantic association, stop-line clipping, and image-label schema construction.
- Modify `leaderboard/team_code/interfuser_data_collector.py`: create `traffic_element_views/`, obtain existing sensor transforms, build one image-label record, and publish JSON atomically.
- Create `tools/data/audit_traffic_element_views.py`: validate image-label schema, frame alignment, actor identity, boxes, semantic evidence, and stop-line projections.
- Create `tools/data/render_traffic_element_overlays.py`: draw sampled labels on RGB images for manual QA.
- Create `tools/data/profile_traffic_element_routes.py`: score dense CARLA routes for traffic-control coverage and hard-negative segments.
- Create `tools/data/run_traffic_element_small_batch.sh`: run the bounded collection on an isolated CARLA port.
- Create `tests/test_traffic_element_projection.py`: pure projection and association tests.
- Create `tests/test_audit_traffic_element_views.py`: audit and alignment tests.
- Create `tests/test_render_traffic_element_overlays.py`: overlay rendering test.
- Create `tests/test_profile_traffic_element_routes.py`: pure route-scoring tests.
- Modify `tests/test_traffic_element_collector.py`: collector integration and one-call-per-frame tests.
- Create `docs/traffic_element_image_label_schema.md`: field semantics, thresholds, units, and provenance.

The implementation must not import `team_code.base_agent` and must not change
`InterfuserAgent`, `InterfuserController`, or the collector entry point.

### Task 1: Camera geometry and CARLA depth decoding

**Files:**
- Create: `leaderboard/team_code/traffic_element_projection.py`
- Create: `tests/test_traffic_element_projection.py`

- [ ] **Step 1: Write failing intrinsic, transform, projection, and depth tests**

```python
import math
import unittest

import numpy as np

from team_code.traffic_element_projection import (
    camera_intrinsics,
    decode_carla_depth,
    project_camera_points,
    transform_matrix,
    world_to_camera,
)


class Location:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class Rotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch, self.yaw, self.roll = pitch, yaw, roll


class Transform:
    def __init__(self, location=None, rotation=None):
        self.location = location or Location()
        self.rotation = rotation or Rotation()


class ProjectionMathTests(unittest.TestCase):
    def test_camera_intrinsics_for_400x300_fov100(self):
        matrix = camera_intrinsics(400, 300, 100.0)
        expected_focal = 400.0 / (2.0 * math.tan(math.radians(50.0)))
        self.assertAlmostEqual(matrix[0, 0], expected_focal)
        self.assertAlmostEqual(matrix[1, 1], expected_focal)
        self.assertEqual(matrix[0, 2], 200.0)
        self.assertEqual(matrix[1, 2], 150.0)

    def test_world_to_camera_respects_carla_yaw(self):
        camera = Transform(Location(10.0, 20.0, 2.0), Rotation(yaw=90.0))
        points = np.array([[10.0, 25.0, 2.0]])
        camera_points = world_to_camera(points, camera)
        np.testing.assert_allclose(camera_points[0], [5.0, 0.0, 0.0], atol=1e-6)

    def test_project_camera_point_uses_forward_right_up_axes(self):
        intrinsic = camera_intrinsics(400, 300, 100.0)
        projected, in_front = project_camera_points(
            np.array([[10.0, 2.0, 1.0]]), intrinsic
        )
        self.assertTrue(in_front[0])
        self.assertGreater(projected[0, 0], 200.0)
        self.assertLess(projected[0, 1], 150.0)

    def test_depth_decoder_matches_carla_24_bit_encoding(self):
        raw = np.array([[[128, 0, 0]]], dtype=np.uint8)
        expected = 1000.0 * (128.0 * 65536.0) / (256.0**3 - 1.0)
        self.assertAlmostEqual(float(decode_carla_depth(raw)[0, 0]), expected)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
export PYTHONPATH=$PWD/carla/PythonAPI:$PWD/carla/PythonAPI/carla:$PWD/leaderboard:$PWD/leaderboard/team_code:$PWD/scenario_runner:$PWD
/data1/shijj/conda_envs/interfuser_origin/bin/python -m unittest tests.test_traffic_element_projection.ProjectionMathTests -v
```

Expected: import failure for `team_code.traffic_element_projection`.

- [ ] **Step 3: Implement the minimal pure math API**

```python
import math

import numpy as np


IMAGE_SCHEMA_VERSION = 1


def camera_intrinsics(width, height, fov_degrees):
    focal = float(width) / (2.0 * math.tan(math.radians(float(fov_degrees)) / 2.0))
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def transform_matrix(transform):
    rotation = transform.rotation
    location = transform.location
    cy, sy = math.cos(math.radians(rotation.yaw)), math.sin(math.radians(rotation.yaw))
    cr, sr = math.cos(math.radians(rotation.roll)), math.sin(math.radians(rotation.roll))
    cp, sp = math.cos(math.radians(rotation.pitch)), math.sin(math.radians(rotation.pitch))
    matrix = np.identity(4, dtype=np.float64)
    matrix[:3, 3] = [location.x, location.y, location.z]
    matrix[:3, :3] = [
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
        [sp, -cp * sr, cp * cr],
    ]
    return matrix


def world_to_camera(world_points, camera_transform):
    points = np.asarray(world_points, dtype=np.float64)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    return (np.linalg.inv(transform_matrix(camera_transform)) @ homogeneous.T).T[:, :3]


def project_camera_points(camera_points, intrinsic):
    points = np.asarray(camera_points, dtype=np.float64)
    in_front = points[:, 0] > 1e-6
    image_axes = np.column_stack([points[:, 1], -points[:, 2], points[:, 0]])
    pixels_h = (intrinsic @ image_axes.T).T
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    pixels[in_front] = pixels_h[in_front, :2] / pixels_h[in_front, 2:3]
    return pixels, in_front


def decode_carla_depth(raw_bgr):
    data = np.asarray(raw_bgr, dtype=np.float32)[..., :3]
    normalized = np.dot(data, np.array([65536.0, 256.0, 1.0], dtype=np.float32))
    return 1000.0 * normalized / (256.0**3 - 1.0)
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Expected: four projection math tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add leaderboard/team_code/traffic_element_projection.py tests/test_traffic_element_projection.py
git commit -m "Add CARLA camera projection primitives"
```

### Task 2: ROI projection, semantic/depth association, and stop-line clipping

**Files:**
- Modify: `leaderboard/team_code/traffic_element_projection.py`
- Modify: `tests/test_traffic_element_projection.py`

- [ ] **Step 1: Add failing association and line-clipping tests**

```python
from team_code.traffic_element_projection import (
    associate_semantic_box,
    clip_image_segment,
    projected_roi,
)


class AssociationTests(unittest.TestCase):
    def test_semantic_depth_pixels_create_tight_xyxy_box(self):
        semantic = np.zeros((10, 12), dtype=np.uint8)
        semantic[3:6, 4:8] = 7
        depth = np.full((10, 12), 100.0, dtype=np.float32)
        depth[3:6, 4:8] = 20.0
        result = associate_semantic_box(
            roi=[2, 1, 10, 9],
            semantic=semantic,
            depth_m=depth,
            actor_distance_m=20.0,
            semantic_tag=7,
            depth_tolerance_m=4.0,
            min_pixels=3,
        )
        self.assertEqual(result["bbox_xyxy"], [4, 3, 8, 6])
        self.assertEqual(result["semantic_pixel_count"], 12)
        self.assertEqual(result["visibility"], "visible")

    def test_absent_semantic_support_is_not_visible(self):
        result = associate_semantic_box(
            [0, 0, 8, 8],
            np.zeros((8, 8), dtype=np.uint8),
            np.full((8, 8), 20.0, dtype=np.float32),
            20.0,
            7,
            4.0,
            3,
        )
        self.assertIsNone(result["bbox_xyxy"])
        self.assertEqual(result["visibility"], "not_visible")

    def test_stop_line_segment_is_clipped_to_image(self):
        self.assertEqual(
            clip_image_segment((-10.0, 5.0), (20.0, 5.0), 12, 10),
            [[0.0, 5.0], [11.0, 5.0]],
        )
```

- [ ] **Step 2: Verify RED for missing association functions**

Run the full `tests.test_traffic_element_projection` module. Expected:
imports fail for the new functions.

- [ ] **Step 3: Implement ROI, association, and Liang-Barsky clipping**

```python
def projected_roi(world_vertices, camera_transform, intrinsic, width, height, expand=6):
    camera_points = world_to_camera(world_vertices, camera_transform)
    pixels, in_front = project_camera_points(camera_points, intrinsic)
    visible = pixels[in_front]
    if not len(visible):
        return None
    x1 = max(0, int(math.floor(np.min(visible[:, 0]))) - expand)
    y1 = max(0, int(math.floor(np.min(visible[:, 1]))) - expand)
    x2 = min(width, int(math.ceil(np.max(visible[:, 0]))) + expand + 1)
    y2 = min(height, int(math.ceil(np.max(visible[:, 1]))) + expand + 1)
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


def associate_semantic_box(
    roi,
    semantic,
    depth_m,
    actor_distance_m,
    semantic_tag,
    depth_tolerance_m,
    min_pixels,
):
    if roi is None:
        return {"visibility": "not_visible", "bbox_xyxy": None,
                "semantic_pixel_count": 0, "median_depth_residual_m": None}
    x1, y1, x2, y2 = roi
    semantic_roi = semantic[y1:y2, x1:x2]
    residual = np.abs(depth_m[y1:y2, x1:x2] - actor_distance_m)
    mask = (semantic_roi == semantic_tag) & (residual <= depth_tolerance_m)
    ys, xs = np.where(mask)
    if len(xs) < min_pixels:
        return {"visibility": "not_visible", "bbox_xyxy": None,
                "semantic_pixel_count": int(len(xs)),
                "median_depth_residual_m": None}
    return {
        "visibility": "visible",
        "bbox_xyxy": [
            int(x1 + xs.min()), int(y1 + ys.min()),
            int(x1 + xs.max() + 1), int(y1 + ys.max() + 1),
        ],
        "semantic_pixel_count": int(len(xs)),
        "median_depth_residual_m": float(np.median(residual[mask])),
    }


def clip_image_segment(start, end, width, height):
    x1, y1 = map(float, start)
    x2, y2 = map(float, end)
    dx, dy = x2 - x1, y2 - y1
    p = (-dx, dx, -dy, dy)
    q = (x1, width - 1.0 - x1, y1, height - 1.0 - y1)
    lower, upper = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0:
                return None
            continue
        ratio = qi / pi
        if pi < 0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower > upper:
            return None
    return [[x1 + lower * dx, y1 + lower * dy],
            [x1 + upper * dx, y1 + upper * dy]]
```

- [ ] **Step 4: Verify GREEN**

Expected: all projection tests pass and boxes use right/bottom-exclusive
`xyxy` coordinates.

- [ ] **Step 5: Commit Task 2**

```bash
git add leaderboard/team_code/traffic_element_projection.py tests/test_traffic_element_projection.py
git commit -m "Add semantic depth traffic element association"
```

### Task 3: Build the versioned image-label record

**Files:**
- Modify: `leaderboard/team_code/traffic_element_projection.py`
- Modify: `tests/test_traffic_element_projection.py`

- [ ] **Step 1: Write a failing record-construction test**

Create fake actors whose `bounding_box.get_world_vertices(transform)` returns
known vertices. Assert that `build_traffic_element_view_record`:

```python
record = build_traffic_element_view_record(
    frame_id="0052",
    traffic_elements=phase1_record,
    actors_by_id={11: fake_light},
    camera_frames={"front": camera_frame},
)
self.assertEqual(record["schema_version"], 2)
self.assertEqual(record["frame_id"], "0052")
light = record["cameras"]["front"]["traffic_lights"][0]
self.assertEqual(light["actor_id"], 11)
self.assertEqual(light["state"], "Red")
self.assertEqual(light["association_source"], "semantic_depth_confirmed")
self.assertEqual(light["bbox_xyxy"], [4, 3, 8, 6])
```

Also add cases for:

- actor ID missing from `actors_by_id` -> `visibility="unknown"` plus error;
- off-screen actor -> `visibility="not_visible"`;
- stop-line endpoints -> clipped segment with Phase 1 provenance;
- source Phase 1 extraction errors -> propagated top-level errors.

- [ ] **Step 2: Run the focused tests and verify RED**

Expected: missing `build_traffic_element_view_record`.

- [ ] **Step 3: Implement the schema constructor**

Add constants:

```python
ASSOCIATION = {
    "roi_expand_pixels": 6,
    "minimum_semantic_pixels": 3,
    "traffic_light": {"semantic_tag": 7, "depth_tolerance_m": 4.0},
    "stop_sign": {"semantic_tag": 8, "depth_tolerance_m": 6.0},
}
```

Implement:

```python
def build_traffic_element_view_record(
    frame_id,
    traffic_elements,
    actors_by_id,
    camera_frames,
):
    result = {
        "schema_version": IMAGE_SCHEMA_VERSION,
        "source_traffic_element_schema_version": traffic_elements["schema_version"],
        "frame_id": str(frame_id),
        "association": ASSOCIATION,
        "cameras": {},
        "errors": list(traffic_elements.get("errors", [])),
    }
    for camera_name, camera in camera_frames.items():
        result["cameras"][camera_name] = _build_camera_record(
            camera_name, camera, traffic_elements, actors_by_id
        )
    return result
```

`_build_camera_record` must preserve every Phase 1 actor, use
`actor.bounding_box.get_world_vertices(actor.get_transform())` when
available, use trigger-volume vertices otherwise, and copy state/relevance
fields without reinterpretation. Stop-line segments use the Phase 1
`left_endpoint` and `right_endpoint` world coordinates.

- [ ] **Step 4: Run all projection tests and verify GREEN**

Expected: record, missing-actor, off-screen, and stop-line tests pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add leaderboard/team_code/traffic_element_projection.py tests/test_traffic_element_projection.py
git commit -m "Build versioned traffic element image labels"
```

### Task 4: Integrate image labels into the existing collector

**Files:**
- Modify: `leaderboard/team_code/interfuser_data_collector.py`
- Modify: `tests/test_traffic_element_collector.py`

- [ ] **Step 1: Add a failing one-call and atomic-write test**

Patch `collect_traffic_element_labels` and
`build_traffic_element_view_record`. Supply fake front/left/right sensor
objects through `collector.sensor_interface._sensors_objects`. Assert:

```python
collector._save_all_data(input_data, control=None, timestamp=0.0)
build_views.assert_called_once()
self.assertTrue((save_path / "traffic_element_views" / "0000.json").exists())
self.assertFalse((save_path / "traffic_element_views" / "0000.json.tmp").exists())
```

Add a missing-sensor test asserting the image record contains an error and no
positive box rather than writing a negative label.

- [ ] **Step 2: Run collector tests and verify RED**

Expected: missing directory/integration or missing projection import.

- [ ] **Step 3: Integrate without changing the collector entry point**

Add `traffic_element_views` to `data_dirs`. Import the new constructor.
Build:

```python
camera_frames = {}
for position in ("front", "left", "right"):
    seg_id = f"seg_{position}"
    depth_id = f"depth_{position}"
    sensor = self.sensor_interface._sensors_objects.get(seg_id)
    if sensor is None or seg_id not in input_data or depth_id not in input_data:
        camera_frames[position] = {"error": "required sensor unavailable"}
        continue
    camera_frames[position] = {
        "transform": sensor.get_transform(),
        "semantic": input_data[seg_id][1][:, :, 2],
        "depth_raw": input_data[depth_id][1][:, :, :3],
        "width": 400,
        "height": 300,
        "fov_degrees": 100.0,
    }
```

Build `actors_by_id` only from traffic lights and stop signs already present
in the Phase 1 record. Write JSON to `0000.json.tmp`, flush and close it, then
publish with `os.replace(tmp_path, final_path)`.

- [ ] **Step 4: Run collector, projection, and full unit tests**

Expected: integration tests pass; the existing 29-test baseline remains green.

- [ ] **Step 5: Commit Task 4**

```bash
git add leaderboard/team_code/interfuser_data_collector.py tests/test_traffic_element_collector.py
git commit -m "Collect traffic element image labels"
```

### Task 5: Add a strict image-label audit

**Files:**
- Create: `tools/data/audit_traffic_element_views.py`
- Create: `tests/test_audit_traffic_element_views.py`

- [ ] **Step 1: Write failing audit tests**

Create a valid temporary route with matching `rgb_front`,
`traffic_elements`, and `traffic_element_views` frame 0000. Assert summary
fields:

```python
self.assertEqual(summary["frames"], 1)
self.assertEqual(summary["invalid_frames"], 0)
self.assertEqual(summary["visible_traffic_light_frames"], 1)
self.assertEqual(summary["semantic_confirmed_traffic_lights"], 1)
self.assertEqual(summary["projected_stop_lines"], 1)
```

Add invalid cases for:

- missing view frame;
- unknown actor ID;
- zero-area or out-of-bounds visible box;
- visible box with fewer than 3 semantic pixels;
- non-finite stop-line point;
- schema or association-threshold mismatch.

- [ ] **Step 2: Verify RED**

Run:

```bash
/data1/shijj/conda_envs/interfuser_origin/bin/python -m unittest tests.test_audit_traffic_element_views -v
```

Expected: import failure for the new audit module.

- [ ] **Step 3: Implement audit and CLI**

Expose:

```python
def audit_traffic_element_views(root):
    """Return a JSON-serializable summary or raise AuditError with file paths."""
```

The summary must contain frame counts, invalid/error/unknown frames, visible
and semantic-confirmed element counts, active-light frames, route-relevant
stop-sign frames, projected stop lines before/after crossing, hard-negative
frames, unique actor IDs, and per-camera counts. CLI exits 2 on any invalid
record or frame mismatch.

- [ ] **Step 4: Verify GREEN and audit the fixture through the CLI**

Expected: all audit tests pass and CLI prints sorted JSON.

- [ ] **Step 5: Commit Task 5**

```bash
git add tools/data/audit_traffic_element_views.py tests/test_audit_traffic_element_views.py
git commit -m "Audit traffic element image labels"
```

### Task 6: Add deterministic RGB overlays

**Files:**
- Create: `tools/data/render_traffic_element_overlays.py`
- Create: `tests/test_render_traffic_element_overlays.py`

- [ ] **Step 1: Write a failing overlay test**

Create a 400 x 300 white RGB fixture and one view record. Call:

```python
output = render_overlay(
    rgb_path,
    view_record,
    camera_name="front",
    output_path=target,
)
self.assertTrue(target.exists())
image = cv2.imread(str(target))
self.assertTrue(np.any(image != 255))
```

- [ ] **Step 2: Verify RED**

Expected: missing overlay module.

- [ ] **Step 3: Implement overlay rendering**

Use OpenCV to draw:

- red/yellow/green traffic-light boxes by state;
- cyan stop-sign boxes;
- magenta exact CARLA stop lines;
- orange approximate stop-sign lines;
- actor ID, state, visibility, and longitudinal distance labels.

CLI arguments:

```text
root --output-dir PATH --limit 12 --camera front
```

Selection order is deterministic and attempts one active light, one irrelevant
light, one route-relevant stop sign, one hard negative, and then fills
remaining slots by frame ID. Output is outside the route dataset.

- [ ] **Step 4: Verify GREEN**

Expected: overlay test passes; generated image dimensions remain 400 x 300.

- [ ] **Step 5: Commit Task 6**

```bash
git add tools/data/render_traffic_element_overlays.py tests/test_render_traffic_element_overlays.py
git commit -m "Render traffic element label overlays"
```

### Task 7: Add topology-based route profiling

**Files:**
- Create: `tools/data/profile_traffic_element_routes.py`
- Create: `tests/test_profile_traffic_element_routes.py`

- [ ] **Step 1: Write failing pure scoring tests**

Given dense route points and traffic-element trigger centers, assert:

```python
summary = score_dense_route(
    route_points=[(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (200.0, 0.0)],
    traffic_lights=[(8.0, 1.0)],
    stop_signs=[(19.0, 0.5)],
    relevant_radius_m=30.0,
    nearby_radius_m=80.0,
)
self.assertEqual(summary["nearby_traffic_lights"], 1)
self.assertEqual(summary["nearby_stop_signs"], 1)
self.assertGreater(summary["hard_negative_points"], 0)
```

Also test empty actor lists and repeated route points.

- [ ] **Step 2: Verify RED**

Expected: missing profile module.

- [ ] **Step 3: Implement scoring and CARLA CLI**

Use `RouteParser.parse_routes_file` and
`leaderboard.utils.route_manipulation.interpolate_trajectory(world,
trajectory, hop_resolution=1.0)`. For each Town group, explicitly preload the
`*_Opt` map on the isolated CARLA port. Transform actor trigger centers to
world coordinates, score the dense route, and write a JSON report with route
file, Town, dense point count, nearby actor counts, minimum distances, and hard
negative point count.

- [ ] **Step 4: Verify GREEN**

Expected: pure scoring tests pass. The CARLA CLI is exercised during Task 9.

- [ ] **Step 5: Commit Task 7**

```bash
git add tools/data/profile_traffic_element_routes.py tests/test_profile_traffic_element_routes.py
git commit -m "Profile routes for traffic element coverage"
```

### Task 8: Document schema and run two live smoke routes

**Files:**
- Create: `docs/traffic_element_image_label_schema.md`
- Output: `data/traffic_element_image_smoke/`

- [ ] **Step 1: Document all fields and thresholds**

Copy the exact schema constants from code. Document right/bottom-exclusive
`bbox_xyxy`, CARLA axes, 24-bit depth units, semantic tags 7/8, 6-pixel ROI
expansion, 4 m/6 m depth tolerances, 3-pixel minimum, visibility states,
projection provenance, nullable fields, and schema version rules.

- [ ] **Step 2: Scan schema/code for contradictions and placeholders**

```bash
grep -R -n -E 'TODO|TBD|PLACEHOLDER|FIXME' leaderboard/team_code/traffic_element_projection.py docs/traffic_element_image_label_schema.md
```

Expected: no matches.

- [ ] **Step 3: Start an isolated CARLA server and preload Town03_Opt**

Use GPU 2, RPC port 2400, and TM port 8400. Confirm the existing port-2000
server PIDs remain unchanged. Set `INTERFUSER_REUSE_CURRENT_WORLD=1`.

- [ ] **Step 4: Run route13/bg0 and route36/bg0**

Use `interfuser_collector_complete.py`, separate output/checkpoint/log paths,
and a 600-second external timeout per route. Do not overwrite Phase 1 smoke
data.

- [ ] **Step 5: Audit and render overlays**

```bash
python tools/data/audit_traffic_element_labels.py data/traffic_element_image_smoke
python tools/data/audit_traffic_element_views.py data/traffic_element_image_smoke
python tools/data/render_traffic_element_overlays.py data/traffic_element_image_smoke --output-dir data/traffic_element_image_smoke_review --limit 12 --camera front
```

Require zero invalid frames, a semantic-confirmed active traffic light, a
Phase 1 route-relevant stop sign with a projected line, and stop-line
projections before/after crossing. A tag-8 stop-sign box is optional when the
map has no visible vertical sign asset. Inspect all twelve overlay images.

- [ ] **Step 6: Commit code and documentation after smoke passes**

```bash
git add docs/traffic_element_image_label_schema.md
git commit -m "Document traffic element image labels"
```

Smoke data and overlays stay untracked.

### Task 9: Profile and run the bounded small batch

**Files:**
- Create: `tools/data/run_traffic_element_small_batch.sh`
- Output: `data/traffic_element_small_batch/`
- Output: `results/traffic_element_small_batch/`

- [ ] **Step 1: Profile Town01 and Town04 route candidates**

Run the profile CLI on routes 00-05 and 18-23/39-41 using the isolated server.
Save `results/traffic_element_small_batch/route_profile.json`. Select one
route maximizing traffic-light coverage and one maximizing hard-negative
points while still containing projected traffic infrastructure.

- [ ] **Step 2: Write the bounded runner**

The shell script must:

- use only port 2400/TM 8400 and GPU 2;
- start and stop only CARLA processes containing `--world-port=2400`;
- preload each `*_Opt` map;
- export `INTERFUSER_REUSE_CURRENT_WORLD=1`;
- run route13/bg0, route36/bg0, and the two profiled routes/bg20;
- stop before launching a fifth route;
- stop if cumulative image-label frames reach 2,000;
- stop if output reaches 2 GB;
- write per-route exit codes and logs;
- run both audits after each route;
- preserve the user's port-2000 server.

- [ ] **Step 3: Validate and commit the runner**

```bash
bash -n tools/data/run_traffic_element_small_batch.sh
git add tools/data/run_traffic_element_small_batch.sh
git commit -m "Add bounded traffic element batch runner"
```

Expected: shell syntax check exits 0. The script requires exactly two profiled
route paths as arguments in addition to the fixed route13 and route36 inputs.

- [ ] **Step 4: Run the small batch**

Monitor every route. Do not leave evaluator or CARLA sessions running after
completion or failure.

- [ ] **Step 5: Verify acceptance criteria**

Record exact batch counts and ensure:

- zero invalid or unmatched frames;
- zero invalid visible boxes;
- semantic active-light positives and route-relevant stop-line projections;
- stop-line projections before and after crossing;
- at least 20 hard-negative frames;
- twelve reviewed overlays;
- no more than 2,000 frames and 2 GB.

- [ ] **Step 6: Run final verification**

```bash
export PYTHONPATH=$PWD/carla/PythonAPI:$PWD/carla/PythonAPI/carla:$PWD/leaderboard:$PWD/leaderboard/team_code:$PWD/scenario_runner:$PWD
/data1/shijj/conda_envs/interfuser_origin/bin/python -m unittest discover -s tests -p 'test_*.py' -v
/data1/shijj/conda_envs/interfuser_origin/bin/python -m py_compile leaderboard/team_code/traffic_element_projection.py tools/data/audit_traffic_element_views.py tools/data/render_traffic_element_overlays.py tools/data/profile_traffic_element_routes.py
git diff --check
```

Because the branch contains unrelated user changes, also run scoped
`git diff --check -- <Phase 2 files>` and report any unrelated global failure
without modifying it.

## Plan Self-Review

- Tasks 1-3 cover pure projection, explicit thresholds, visibility, and schema.
- Task 4 preserves the current collector and entry point without importing
  `BaseAgent`.
- Tasks 5-6 provide strict machine audit and manual visual QA.
- Task 7 selects routes from dense CARLA topology rather than filenames.
- Tasks 8-9 enforce smoke-before-batch, frame/storage caps, isolated ports, and
  cleanup.
- No task changes the controller, Interfuser model, or active evaluation agent.
- No large-scale collection or model training is included.
