import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
        sensor = SimpleNamespace(get_transform=lambda: SimpleNamespace())
        collector.sensor_interface = SimpleNamespace(
            _sensors_objects={
                "seg_front": sensor,
                "seg_left": sensor,
                "seg_right": sensor,
            }
        )
        world = SimpleNamespace(get_actors=lambda: [])
        phase1 = {
            "schema_version": 1,
            "traffic_lights": [],
            "stop_signs": [],
            "errors": [],
        }
        view_record = {
            "schema_version": 1,
            "source_traffic_element_schema_version": 1,
            "frame_id": "0000",
            "association": {},
            "cameras": {},
            "errors": [],
        }

        with tempfile.TemporaryDirectory() as tmp:
            collector.save_path = Path(tmp)
            self._make_save_dirs(collector.save_path)
            with patch.object(
                collector_module.CarlaDataProvider,
                "get_hero_actor",
                return_value=SimpleNamespace(),
            ), patch.object(
                collector_module.CarlaDataProvider,
                "get_world",
                return_value=world,
            ), patch.object(
                collector_module,
                "collect_traffic_element_labels",
                return_value=phase1,
            ), patch.object(
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
                    self._input_data(),
                    control=None,
                    timestamp=0.0,
                )

            build_views.assert_called_once()
            final_path = collector.save_path / "traffic_element_views" / "0000.json"
            self.assertTrue(final_path.exists())
            self.assertFalse(final_path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
