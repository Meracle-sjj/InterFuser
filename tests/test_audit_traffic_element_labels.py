import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from tools.data.audit_traffic_element_labels import AuditError, audit_dataset


GEOMETRY_SOURCE = "scenario_runner_running_red_light_test_v1"


def _point(x, y=0.0, z=0.0):
    return {"x": float(x), "y": float(y), "z": float(z)}


def _target(boundary_s, status="valid", primary=False, signed_distance=10.0):
    target_id = f"Town01_Opt:7:0:-1:{boundary_s:.1f}"
    return {
        "target_id": target_id,
        "map_name": "Town01_Opt",
        "owner_traffic_light_actor_ids": [11],
        "state_by_actor_id": {"11": "Red"},
        "primary_for_ego": primary,
        "status": status,
        "unknown_reason": None if status == "valid" else "waypoint_branch",
        "route_lane": {"road_id": 7, "section_id": 0, "lane_id": -1},
        "geometry_source": GEOMETRY_SOURCE,
        "trigger_stop_waypoint": {
            "geometry_source": "carla_traffic_light_trigger_waypoint",
            "road_id": 7,
            "section_id": 0,
            "lane_id": -1,
            "s": boundary_s - 5.0,
            "location": _point(boundary_s - 5.0),
        },
        "leaderboard_infraction_boundary": {
            "geometry_source": GEOMETRY_SOURCE,
            "road_id": 7,
            "section_id": 0,
            "lane_id": -1,
            "s": boundary_s,
            "lane_width": 4.0,
            "center": _point(boundary_s),
            "left_endpoint": _point(boundary_s, -1.6),
            "right_endpoint": _point(boundary_s, 1.6),
        },
        "recommended_ego_stop_pose": {
            "location": _point(boundary_s - 3.0),
            "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            "lane_width": 4.0,
        },
        "vehicle_front_offset_m": 2.0,
        "safety_margin_m": 1.0,
        "stop_evidence_corridor": {
            "sample_step_m": 0.5,
            "extension_m": 3.0,
            "centerline": [
                {"location": _point(x), "lane_width": 4.0}
                for x in (boundary_s - 8.0, boundary_s, boundary_s + 3.0)
            ],
        },
        "signed_route_distance_m": signed_distance,
        "euclidean_distance_m": abs(signed_distance),
        "relative_heading_degrees": 0.0,
        "ego_before_boundary": signed_distance >= 0.0,
        "trigger_to_boundary_route_distance_m": 5.0,
        "search_steps": 10,
    }


def valid_record():
    return {
        "schema_version": 2,
        "frame_id": "0000",
        "map_name": "Town01_Opt",
        "ego": {},
        "active_traffic_light_id": 11,
        "traffic_lights": [
            {
                "actor_id": 11,
                "state": "Red",
                "is_active_for_ego": True,
                "controls_ego_lane": True,
                "relevant_to_ego": True,
            }
        ],
        "stop_targets": [
            _target(20.0, status="valid", primary=True, signed_distance=10.0),
            _target(40.0, status="unknown", signed_distance=30.0),
        ],
        "errors": [],
    }


class TrafficElementAuditTests(unittest.TestCase):
    def _write_fixture(self, root, record=None):
        route = Path(root) / "route_00"
        labels = route / "traffic_elements"
        labels.mkdir(parents=True)
        (route / "rgb_front").mkdir()
        (route / "rgb_front" / "0000.jpg").write_bytes(b"image")
        (labels / "0000.json").write_text(
            json.dumps(record if record is not None else valid_record()),
            encoding="utf-8",
        )
        return route

    def test_summarizes_valid_and_unknown_targets_by_town(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_fixture(tmp)
            summary = audit_dataset(tmp)

        self.assertEqual(summary["frames"], 1)
        self.assertEqual(summary["traffic_lights"], 1)
        self.assertEqual(summary["traffic_light_states"], {"Red": 1})
        self.assertEqual(summary["active_traffic_light_frames"], 1)
        self.assertEqual(summary["valid_stop_targets"], 1)
        self.assertEqual(summary["unknown_stop_targets"], 1)
        self.assertEqual(summary["unknown_reasons"], {"waypoint_branch": 1})
        self.assertEqual(summary["primary_stop_target_frames"], 1)
        self.assertEqual(summary["towns"]["Town01_Opt"]["frames"], 1)
        self.assertEqual(summary["forbidden_stop_occurrences"], 0)

    def test_rejects_forbidden_stop_key_in_generated_side_channel(self):
        with tempfile.TemporaryDirectory() as tmp:
            route = self._write_fixture(tmp)
            (route / "measurements").mkdir()
            (route / "measurements" / "0000.json").write_text(
                json.dumps({"is_stop_sign_present": []}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(AuditError, "forbidden STOP label"):
                audit_dataset(tmp)

    def test_rejects_forbidden_actor_value_in_generated_side_channel(self):
        with tempfile.TemporaryDirectory() as tmp:
            route = self._write_fixture(tmp)
            (route / "other_actors").mkdir()
            (route / "other_actors" / "0000.json").write_text(
                json.dumps([{"type_id": "traffic.stop"}]),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(AuditError, "forbidden STOP label"):
                audit_dataset(tmp)

    def test_rejects_legacy_schema(self):
        record = valid_record()
        record["schema_version"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            self._write_fixture(tmp, record)
            with self.assertRaisesRegex(AuditError, "unsupported schema_version"):
                audit_dataset(tmp)

    def test_rejects_duplicate_target_ids_within_frame(self):
        record = valid_record()
        record["stop_targets"][1]["target_id"] = record["stop_targets"][0][
            "target_id"
        ]
        with tempfile.TemporaryDirectory() as tmp:
            self._write_fixture(tmp, record)
            with self.assertRaisesRegex(AuditError, "target_id must be unique"):
                audit_dataset(tmp)

    def test_rejects_wrong_geometry_source(self):
        record = valid_record()
        record["stop_targets"][0]["geometry_source"] = "unknown_source"
        with tempfile.TemporaryDirectory() as tmp:
            self._write_fixture(tmp, record)
            with self.assertRaisesRegex(AuditError, "geometry_source"):
                audit_dataset(tmp)

    def test_rejects_non_finite_signed_distance(self):
        record = valid_record()
        record["stop_targets"][0]["signed_route_distance_m"] = math.nan
        with tempfile.TemporaryDirectory() as tmp:
            self._write_fixture(tmp, record)
            with self.assertRaisesRegex(AuditError, "signed_route_distance_m"):
                audit_dataset(tmp)

    def test_rejects_multiple_primary_targets(self):
        record = valid_record()
        second = copy.deepcopy(record["stop_targets"][0])
        second["target_id"] = "Town01_Opt:7:0:-1:25.0"
        second["leaderboard_infraction_boundary"]["s"] = 25.0
        record["stop_targets"].append(second)
        with tempfile.TemporaryDirectory() as tmp:
            self._write_fixture(tmp, record)
            with self.assertRaisesRegex(AuditError, "at most one primary"):
                audit_dataset(tmp)

    def test_rejects_label_without_matching_rgb_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            route = self._write_fixture(tmp)
            (route / "rgb_front" / "0000.jpg").unlink()
            with self.assertRaisesRegex(AuditError, "missing rgb_front frames: 0000"):
                audit_dataset(tmp)

    def test_rejects_directory_without_label_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(AuditError, "no traffic element label files"):
                audit_dataset(tmp)

    def test_non_object_record_is_an_audit_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_fixture(tmp, [])
            with self.assertRaisesRegex(AuditError, "record must be an object"):
                audit_dataset(tmp)


if __name__ == "__main__":
    unittest.main()
