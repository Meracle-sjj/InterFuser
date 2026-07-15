import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from team_code import interfuser_data_collector as collector_module


class TrafficElementCollectorTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
