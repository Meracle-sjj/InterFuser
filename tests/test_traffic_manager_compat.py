import unittest

from leaderboard.leaderboard_evaluator import _call_traffic_manager_api


class TrafficManagerCompatTests(unittest.TestCase):
    def test_uses_first_available_method_name(self):
        class FakeTrafficManager:
            def __init__(self):
                self.calls = []

            def old_name(self, value):
                self.calls.append(("old_name", value))

            def new_name(self, value):
                self.calls.append(("new_name", value))

        tm = FakeTrafficManager()

        used = _call_traffic_manager_api(tm, ("old_name", "new_name"), 12.5)

        self.assertEqual(used, "old_name")
        self.assertEqual(tm.calls, [("old_name", 12.5)])

    def test_falls_back_to_carla_0916_speed_api_name(self):
        class FakeTrafficManager:
            def __init__(self):
                self.calls = []

            def global_percentage_speed_difference(self, value):
                self.calls.append(("global_percentage_speed_difference", value))

        tm = FakeTrafficManager()

        used = _call_traffic_manager_api(
            tm,
            ("set_global_percentage_speed_difference", "global_percentage_speed_difference"),
            -15.0,
        )

        self.assertEqual(used, "global_percentage_speed_difference")
        self.assertEqual(tm.calls, [("global_percentage_speed_difference", -15.0)])

    def test_falls_back_to_carla_0916_hybrid_radius_api_name(self):
        class FakeTrafficManager:
            def __init__(self):
                self.calls = []

            def set_hybrid_physics_radius(self, value):
                self.calls.append(("set_hybrid_physics_radius", value))

        tm = FakeTrafficManager()

        used = _call_traffic_manager_api(
            tm,
            ("set_hybridphysicsmode_radius", "set_hybrid_physics_radius"),
            50.0,
        )

        self.assertEqual(used, "set_hybrid_physics_radius")
        self.assertEqual(tm.calls, [("set_hybrid_physics_radius", 50.0)])

    def test_raises_when_no_api_name_exists(self):
        with self.assertRaisesRegex(AttributeError, "missing_one, missing_two"):
            _call_traffic_manager_api(object(), ("missing_one", "missing_two"), 1)


