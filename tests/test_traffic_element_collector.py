import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, patch

import numpy as np

from team_code import interfuser_data_collector as collector_module


class TrafficElementCollectorTests(unittest.TestCase):
    @staticmethod
    def _make_save_dirs(root):
        names = [
            "rgb_front", "rgb_left", "rgb_right", "rgb_rear",
            "seg_front", "seg_left", "seg_right",
            "depth_front", "depth_left", "depth_right",
            "lidar", "traffic_elements", "traffic_element_views", "birdview",
            "3d_bbs", "2d_bbs_front", "2d_bbs_left", "2d_bbs_right",
            "2d_bbs_rear", "affordances", "measurements", "other_actors",
        ]
        for name in names:
            (root / name).mkdir()

    @staticmethod
    def _input_data():
        camera = np.zeros((4, 4, 4), dtype=np.uint8)
        return {
            "rgb_front": (0, camera.copy()),
            "rgb_left": (0, camera.copy()),
            "rgb_right": (0, camera.copy()),
            "rgb_rear": (0, camera.copy()),
            "seg_front": (0, camera.copy()),
            "seg_left": (0, camera.copy()),
            "seg_right": (0, camera.copy()),
            "depth_front": (0, camera.copy()),
            "depth_left": (0, camera.copy()),
            "depth_right": (0, camera.copy()),
            "lidar": (0, np.zeros((2, 4), dtype=np.float32)),
        }

    def test_missing_label_context_does_not_write_sensor_frame(self):
        collector = object.__new__(collector_module.InterfuserDataCollector)
        collector.step = 0
        collector.save_freq = 10

        with tempfile.TemporaryDirectory() as tmp:
            collector.save_path = Path(tmp)
            (collector.save_path / "rgb_front").mkdir()
            input_data = {
                "rgb_front": (0, np.zeros((4, 4, 4), dtype=np.uint8)),
            }

            with patch.object(
                collector_module.CarlaDataProvider,
                "get_hero_actor",
                return_value=None,
            ), patch.object(
                collector_module.CarlaDataProvider,
                "get_world",
                return_value=None,
            ):
                collector._save_all_data(input_data, control=None, timestamp=0.0)

            self.assertFalse((collector.save_path / "rgb_front" / "0000.jpg").exists())

    def test_writes_one_atomic_image_label_record_per_saved_frame(self):
        collector = object.__new__(collector_module.InterfuserDataCollector)
        collector.step = 0
        collector.save_freq = 10
        route_marker = SimpleNamespace(name="route-waypoint")
        collector._traffic_route_waypoints = [route_marker]
        camera_transform = SimpleNamespace(name="camera-transform")
        lidar_transform = SimpleNamespace(name="lidar-transform")
        ego_transform = SimpleNamespace(name="ego-transform")
        camera_sensor = SimpleNamespace(get_transform=lambda: camera_transform)
        lidar_sensor = SimpleNamespace(get_transform=lambda: lidar_transform)
        collector.sensor_interface = SimpleNamespace(
            _sensors_objects={
                "seg_front": camera_sensor,
                "seg_left": camera_sensor,
                "seg_right": camera_sensor,
                "lidar": lidar_sensor,
            }
        )
        world = SimpleNamespace(get_actors=lambda: [])
        hero = SimpleNamespace(get_transform=lambda: ego_transform)
        phase1 = {
            "schema_version": 2,
            "frame_id": "0000",
            "map_name": "Town01_Opt",
            "ego": {},
            "traffic_lights": [],
            "stop_targets": [],
            "errors": [],
        }
        view_record = {
            "schema_version": 3,
            "source_traffic_element_schema_version": 2,
            "frame_id": "0000",
            "association": {},
            "cameras": {},
            "errors": [],
        }
        input_data = self._input_data()

        with tempfile.TemporaryDirectory() as tmp:
            collector.save_path = Path(tmp)
            self._make_save_dirs(collector.save_path)
            with patch.object(
                collector_module.CarlaDataProvider,
                "get_hero_actor",
                return_value=hero,
            ), patch.object(
                collector_module.CarlaDataProvider,
                "get_world",
                return_value=world,
            ), patch.object(
                collector_module,
                "collect_traffic_element_labels",
                return_value=phase1,
            ) as collect_labels, patch.object(
                collector_module,
                "build_traffic_element_view_record",
                return_value=view_record,
                create=True,
            ) as build_views, patch.object(
                collector,
                "_generate_birdview",
                return_value=np.zeros((4, 4, 3), dtype=np.uint8),
            ), patch.object(
                collector,
                "_get_3d_bounding_boxes",
                return_value=[],
            ), patch.object(
                collector,
                "_get_2d_bounding_boxes",
                return_value=[],
            ), patch.object(
                collector,
                "_get_affordances",
                return_value={},
            ), patch.object(
                collector,
                "_get_measurements",
                return_value={},
            ), patch.object(
                collector,
                "_get_other_actors",
                return_value={},
            ):
                collector._save_all_data(
                    input_data,
                    control=None,
                    timestamp=0.0,
                )

            build_views.assert_called_once()
            camera_frames = build_views.call_args.kwargs["camera_frames"]
            np.testing.assert_array_equal(
                camera_frames["front"]["rgb"],
                input_data["rgb_front"][1][:, :, :3],
            )
            lidar_frame = build_views.call_args.kwargs["lidar_frame"]
            self.assertIs(lidar_frame["transform"], lidar_transform)
            self.assertIs(lidar_frame["ego_transform"], ego_transform)
            self.assertIs(lidar_frame["points"], input_data["lidar"][1])
            collect_labels.assert_called_once_with(
                ANY,
                world,
                frame_id="0000",
                route_waypoints=[route_marker],
            )
            phase_path = collector.save_path / "traffic_elements" / "0000.json"
            self.assertTrue(phase_path.exists())
            self.assertFalse(phase_path.with_suffix(".json.tmp").exists())
            final_path = collector.save_path / "traffic_element_views" / "0000.json"
            self.assertTrue(final_path.exists())
            self.assertFalse(final_path.with_suffix(".json.tmp").exists())

    def test_missing_lidar_frame_is_explicitly_unknown(self):
        collector = object.__new__(collector_module.InterfuserDataCollector)
        collector.sensor_interface = SimpleNamespace(_sensors_objects={})

        frame = collector._get_traffic_element_lidar_frame(
            {},
            SimpleNamespace(get_transform=lambda: SimpleNamespace()),
        )

        self.assertEqual(frame, {"error": "required sensor unavailable: lidar"})


if __name__ == "__main__":
    unittest.main()
