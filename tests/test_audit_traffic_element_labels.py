import json
import tempfile
import unittest
from pathlib import Path

from tools.data.audit_traffic_element_labels import AuditError, audit_dataset


def _valid_stop_line(source):
    return {
        "road_id": 3,
        "section_id": 0,
        "lane_id": -1,
        "s": 42.0,
        "lane_width": 3.5,
        "geometry_source": source,
        "is_exact_carla_stop_position": source == "carla_stop_waypoint",
        "center": {"x": 12.0, "y": 0.0, "z": 0.0},
        "left_endpoint": {"x": 12.0, "y": -1.75, "z": 0.0},
        "right_endpoint": {"x": 12.0, "y": 1.75, "z": 0.0},
        "relative_center": {"forward": 10.0, "right": 0.0, "up": 0.0},
        "longitudinal_distance": 10.0,
        "lateral_offset": 0.0,
        "ego_before_line": True,
    }


def _valid_record():
    return {
        "schema_version": 1,
        "ego": {},
        "active_traffic_light_id": 11,
        "traffic_lights": [
            {
                "actor_id": 11,
                "state": "Red",
                "is_active_for_ego": True,
                "stop_lines": [_valid_stop_line("carla_stop_waypoint")],
            }
        ],
        "stop_signs": [
            {
                "actor_id": 21,
                "affects_ego_route": True,
                "stop_lines": [
                    _valid_stop_line("trigger_volume_route_entry_approximation")
                ],
            }
        ],
        "errors": [],
    }


class TrafficElementAuditTests(unittest.TestCase):
    def _write_record(self, root, record):
        target = Path(root) / "route_00" / "traffic_elements" / "0000.json"
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps(record), encoding="utf-8")
        rgb = target.parents[1] / "rgb_front" / "0000.jpg"
        rgb.parent.mkdir()
        rgb.write_bytes(b"test image placeholder")

    def test_summarizes_valid_traffic_element_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_record(tmp, _valid_record())

            summary = audit_dataset(tmp)

        self.assertEqual(summary["frames"], 1)
        self.assertEqual(summary["invalid_frames"], 0)
        self.assertEqual(summary["traffic_lights"], 1)
        self.assertEqual(summary["traffic_light_states"], {"Red": 1})
        self.assertEqual(summary["active_traffic_light_frames"], 1)
        self.assertEqual(summary["exact_traffic_light_stop_lines"], 1)
        self.assertEqual(summary["stop_signs"], 1)
        self.assertEqual(summary.get("stop_sign_frames"), 1)
        self.assertEqual(summary.get("unique_stop_sign_actors"), 1)
        self.assertEqual(summary["route_relevant_stop_sign_frames"], 1)
        self.assertEqual(summary["approximate_stop_sign_stop_lines"], 1)

    def test_rejects_directory_without_label_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(AuditError, "no traffic element label files"):
                audit_dataset(tmp)

    def test_rejects_unsupported_schema_version(self):
        record = _valid_record()
        record["schema_version"] = 99
        with tempfile.TemporaryDirectory() as tmp:
            self._write_record(tmp, record)

            with self.assertRaisesRegex(AuditError, "unsupported schema_version"):
                audit_dataset(tmp)

    def test_rejects_missing_top_level_lists(self):
        record = _valid_record()
        del record["traffic_lights"]
        del record["stop_signs"]
        with tempfile.TemporaryDirectory() as tmp:
            self._write_record(tmp, record)

            with self.assertRaisesRegex(AuditError, "traffic_lights must be a list"):
                audit_dataset(tmp)

    def test_rejects_non_object_record_with_audit_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_record(tmp, [])

            try:
                audit_dataset(tmp)
            except AuditError as exc:
                self.assertRegex(str(exc), "record must be an object")
            except Exception as exc:
                self.fail(f"expected AuditError, got {type(exc).__name__}: {exc}")
            else:
                self.fail("expected AuditError")

    def test_rejects_non_object_stop_line_with_audit_error(self):
        record = _valid_record()
        record["traffic_lights"][0]["stop_lines"] = [None]
        with tempfile.TemporaryDirectory() as tmp:
            self._write_record(tmp, record)

            try:
                audit_dataset(tmp)
            except AuditError as exc:
                self.assertRegex(str(exc), "stop_lines\\[0\\] must be an object")
            except Exception as exc:
                self.fail(f"expected AuditError, got {type(exc).__name__}: {exc}")
            else:
                self.fail("expected AuditError")

    def test_rejects_stop_line_without_required_geometry(self):
        record = _valid_record()
        del record["traffic_lights"][0]["stop_lines"][0]["center"]
        with tempfile.TemporaryDirectory() as tmp:
            self._write_record(tmp, record)

            with self.assertRaisesRegex(AuditError, "center must be an object"):
                audit_dataset(tmp)

    def test_rejects_label_without_matching_rgb_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_record(tmp, _valid_record())
            (Path(tmp) / "route_00" / "rgb_front" / "0000.jpg").unlink()

            with self.assertRaisesRegex(AuditError, "missing rgb_front frames: 0000"):
                audit_dataset(tmp)

    def test_rejects_non_string_traffic_light_state_with_audit_error(self):
        record = _valid_record()
        record["traffic_lights"][0]["state"] = []
        with tempfile.TemporaryDirectory() as tmp:
            self._write_record(tmp, record)

            try:
                audit_dataset(tmp)
            except AuditError as exc:
                self.assertRegex(str(exc), "state must be a non-empty string")
            except Exception as exc:
                self.fail(f"expected AuditError, got {type(exc).__name__}: {exc}")
            else:
                self.fail("expected AuditError")

    def test_rejects_stop_sign_relevance_without_matching_stop_line(self):
        record = _valid_record()
        record["stop_signs"][0]["affects_ego_route"] = False
        with tempfile.TemporaryDirectory() as tmp:
            self._write_record(tmp, record)

            with self.assertRaisesRegex(
                AuditError,
                "stop_lines must be empty when affects_ego_route is false",
            ):
                audit_dataset(tmp)

    def test_rejects_active_light_id_without_matching_active_item(self):
        record = _valid_record()
        record["traffic_lights"][0]["is_active_for_ego"] = False
        with tempfile.TemporaryDirectory() as tmp:
            self._write_record(tmp, record)

            with self.assertRaisesRegex(
                AuditError,
                "active_traffic_light_id must match exactly one active item",
            ):
                audit_dataset(tmp)


if __name__ == "__main__":
    unittest.main()
