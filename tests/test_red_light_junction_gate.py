import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from team_code.interfuser_controller import InterfuserController


def _config():
    return SimpleNamespace(
        turn_KP=1.25,
        turn_KI=0.75,
        turn_KD=0.3,
        turn_n=40,
        speed_KP=5.0,
        speed_KI=0.5,
        speed_KD=1.0,
        speed_n=40,
        collision_buffer=[2.5, 1.2],
        detect_threshold=0.04,
        max_speed=5,
        clip_delta=0.35,
        max_throttle=0.75,
        brake_ratio=1.1,
    )


class RedLightJunctionGateTests(unittest.TestCase):
    def setUp(self):
        os.environ.pop("INTERFUSER_CONTROL_JUNCTION_SOURCE", None)

    def tearDown(self):
        os.environ.pop("INTERFUSER_CONTROL_JUNCTION_SOURCE", None)

    def test_auxiliary_junction_prevents_false_red_light_brake(self):
        controller = InterfuserController(_config())
        waypoints = np.array([[0.0, -2.0 * (i + 1)] for i in range(10)], dtype=float)
        meta_data = np.zeros((400, 7), dtype=float)

        with patch("team_code.interfuser_controller.get_max_safe_distance", return_value=20.0):
            _, throttle, brake, meta = controller.run_step(
                speed=0.0,
                waypoints=waypoints,
                junction=0.9999,
                traffic_light_state=0.9999,
                stop_sign=1.0,
                meta_data=meta_data,
                aux_junction=0.01,
            )

        self.assertFalse(brake)
        self.assertGreater(throttle, 0.0)
        self.assertGreater(meta[4]["desired_speed"], 0.0)

    def test_auxiliary_junction_alone_does_not_trigger_red_light_brake(self):
        controller = InterfuserController(_config())
        waypoints = np.array([[0.0, -2.0 * (i + 1)] for i in range(10)], dtype=float)
        meta_data = np.zeros((400, 7), dtype=float)

        with patch("team_code.interfuser_controller.get_max_safe_distance", return_value=20.0):
            _, throttle, brake, meta = controller.run_step(
                speed=0.0,
                waypoints=waypoints,
                junction=0.01,
                traffic_light_state=0.9999,
                stop_sign=1.0,
                meta_data=meta_data,
                aux_junction=0.9999,
            )

        self.assertFalse(brake)
        self.assertGreater(throttle, 0.0)
        self.assertGreater(meta[4]["desired_speed"], 0.0)

    def test_consensus_junction_brakes_when_both_sources_match(self):
        controller = InterfuserController(_config())
        waypoints = np.array([[0.0, -2.0 * (i + 1)] for i in range(10)], dtype=float)
        meta_data = np.zeros((400, 7), dtype=float)

        with patch("team_code.interfuser_controller.get_max_safe_distance", return_value=20.0):
            _, throttle, brake, meta = controller.run_step(
                speed=0.0,
                waypoints=waypoints,
                junction=0.9999,
                traffic_light_state=0.9999,
                stop_sign=1.0,
                meta_data=meta_data,
                aux_junction=0.9999,
            )

        self.assertTrue(brake)
        self.assertEqual(throttle, 0.0)
        self.assertEqual(meta[4]["desired_speed"], 0.0)

    def test_raw_junction_source_keeps_previous_red_light_gate(self):
        os.environ["INTERFUSER_CONTROL_JUNCTION_SOURCE"] = "raw"
        controller = InterfuserController(_config())
        waypoints = np.array([[0.0, -2.0 * (i + 1)] for i in range(10)], dtype=float)
        meta_data = np.zeros((400, 7), dtype=float)

        with patch("team_code.interfuser_controller.get_max_safe_distance", return_value=20.0):
            _, throttle, brake, meta = controller.run_step(
                speed=0.0,
                waypoints=waypoints,
                junction=0.9999,
                traffic_light_state=0.9999,
                stop_sign=1.0,
                meta_data=meta_data,
                aux_junction=0.01,
            )

        self.assertTrue(brake)
        self.assertEqual(throttle, 0.0)
        self.assertEqual(meta[4]["desired_speed"], 0.0)


