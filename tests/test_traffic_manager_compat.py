"""
[INPUT]: 依赖 leaderboard.leaderboard_evaluator 的地图/API 兼容层与 evaluator 清理边界，使用纯内存 fake 隔离 CARLA 进程。
[OUTPUT]: 验证 `_Opt` 地图归一化、Traffic Manager API 回退，以及同步模式退出先于 actor 回收且同一 attempt 只清理一次。
[POS]: tests 的 Leaderboard 运行时兼容回归，覆盖真实 D7 批次暴露的 CARLA 0.9.16 生命周期竞态，不启动 simulator。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import unittest
from unittest.mock import patch

from leaderboard.leaderboard_evaluator import (
    LeaderboardEvaluator,
    _call_traffic_manager_api,
    _normalized_town_name,
)


class TrafficManagerCompatTests(unittest.TestCase):
    def test_cleanup_disables_sync_before_single_actor_pool_cleanup(self):
        events = []

        class FakeTrafficManager:
            def set_synchronous_mode(self, enabled):
                events.append(("traffic_manager_sync", enabled))

        class FakeSettings:
            synchronous_mode = True
            fixed_delta_seconds = 0.05

        class FakeWorld:
            def get_settings(self):
                events.append(("world_get_settings",))
                return FakeSettings()

            def apply_settings(self, settings):
                events.append(
                    ("world_apply_settings", settings.synchronous_mode, settings.fixed_delta_seconds)
                )

        class FakeManager:
            def cleanup(self):
                events.append(("manager_cleanup",))

        class FakeWatchdog:
            _timer = None

        evaluator = LeaderboardEvaluator.__new__(LeaderboardEvaluator)
        evaluator._cleanup_complete = False
        evaluator.traffic_manager = FakeTrafficManager()
        evaluator.world = FakeWorld()
        evaluator.manager = FakeManager()
        evaluator.ego_vehicles = [object()]
        evaluator._agent_watchdog = FakeWatchdog()
        evaluator.statistics_manager = type("Stats", (), {"scenario": object()})()

        with patch(
            "leaderboard.leaderboard_evaluator.CarlaDataProvider.cleanup",
            side_effect=lambda: events.append(("actor_pool_cleanup",)),
        ) as actor_cleanup:
            evaluator._cleanup()
            evaluator._cleanup()

        self.assertEqual(
            events,
            [
                ("traffic_manager_sync", False),
                ("world_get_settings",),
                ("world_apply_settings", False, None),
                ("manager_cleanup",),
                ("actor_pool_cleanup",),
            ],
        )
        actor_cleanup.assert_called_once_with()
        self.assertEqual(evaluator.ego_vehicles, [])
        self.assertIsNone(evaluator.statistics_manager.scenario)

    def test_normalizes_opt_map_names_for_reuse_check(self):
        self.assertEqual(_normalized_town_name("Town02_Opt"), "Town02")
        self.assertEqual(_normalized_town_name("Carla/Maps/Town02_Opt"), "Town02")
        self.assertEqual(_normalized_town_name("/Game/Carla/Maps/Town02_Opt"), "Town02")
        self.assertEqual(_normalized_town_name("Town03"), "Town03")

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
