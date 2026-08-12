"""
[INPUT]: 依赖 Leaderboard RouteProgressBlockedTest，并以纯内存 `(Location, RoadOption)` 路线、地图、位置和仿真时钟隔离 CARLA 进程。
[OUTPUT]: 验证短促速度脉冲不能绕过路线进度门禁、足量净进度重置窗口且失败生成 VEHICLE_BLOCKED 证据。
[POS]: tests 的闭环终止语义回归测试，覆盖 route0 高密度拥堵暴露的速度 blocked 计时反复重置漏洞。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import unittest
from unittest.mock import patch

import py_trees

from leaderboard.scenarios.scenarioatomics.atomic_criteria import (
    RouteProgressBlockedTest,
)
from leaderboard.scenarios.route_scenario import (
    ROUTE_PROGRESS_BLOCKED_METERS,
    ROUTE_PROGRESS_BLOCKED_SECONDS,
    _build_blocked_criteria,
)
from srunner.scenariomanager.traffic_events import TrafficEventType


class FakeVector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __sub__(self, other):
        return FakeVector(self.x - other.x, self.y - other.y, self.z - other.z)

    def distance(self, other):
        delta = self - other
        return (delta.x ** 2 + delta.y ** 2 + delta.z ** 2) ** 0.5


class FakeWaypoint:
    class Transform:
        @staticmethod
        def get_forward_vector():
            return FakeVector(x=1.0)

    transform = Transform()


class FakeMap:
    @staticmethod
    def get_waypoint(_location):
        return FakeWaypoint()


class FakeActor:
    pass


class RouteProgressBlockedTests(unittest.TestCase):
    def _criterion(self):
        route = [(FakeVector(x=float(index)), object()) for index in range(101)]
        with patch(
            "leaderboard.scenarios.scenarioatomics.atomic_criteria."
            "CarlaDataProvider.get_map",
            return_value=FakeMap(),
        ):
            return RouteProgressBlockedTest(
                FakeActor(),
                route,
                min_progress_meters=18.0,
                max_time_seconds=180.0,
                terminate_on_failure=True,
            )

    def _update(self, criterion, time_seconds, x):
        with patch(
            "leaderboard.scenarios.scenarioatomics.atomic_criteria."
            "CarlaDataProvider.get_location",
            return_value=FakeVector(x=x),
        ), patch(
            "leaderboard.scenarios.scenarioatomics.atomic_criteria.GameTime.get_time",
            return_value=time_seconds,
        ):
            return criterion.update()

    def test_short_forward_pulses_still_fail_without_eighteen_meter_progress(self):
        criterion = self._criterion()

        self.assertEqual(self._update(criterion, 0.0, 0.0), py_trees.common.Status.RUNNING)
        for time_seconds, x in ((60.0, 3.0), (120.0, 6.0), (181.0, 9.0)):
            status = self._update(criterion, time_seconds, x)

        self.assertEqual(status, py_trees.common.Status.FAILURE)
        self.assertEqual(criterion.test_status, "FAILURE")
        self.assertLess(criterion.actual_value, 18.0)
        self.assertEqual(len(criterion.list_traffic_events), 1)
        event = criterion.list_traffic_events[0]
        self.assertEqual(event.get_type(), TrafficEventType.VEHICLE_BLOCKED)
        self.assertIn("route progress stagnation", event.get_message())

    def test_eighteen_meter_progress_prevents_premature_failure(self):
        criterion = self._criterion()

        status = self._update(criterion, 0.0, 0.0)
        for index in range(1, 21):
            status = self._update(criterion, index * 9.0, float(index))
        status = self._update(criterion, 181.0, 20.0)

        self.assertEqual(status, py_trees.common.Status.RUNNING)
        self.assertEqual(criterion.test_status, "RUNNING")
        self.assertEqual(criterion.list_traffic_events, [])

    def test_fails_after_a_later_full_window_without_progress(self):
        criterion = self._criterion()

        self._update(criterion, 0.0, 0.0)
        for time_seconds, x in (
            (30.0, 5.0),
            (60.0, 10.0),
            (90.0, 15.0),
            (120.0, 20.0),
            (181.0, 20.0),
            (301.0, 20.0),
        ):
            status = self._update(criterion, time_seconds, x)

        self.assertEqual(status, py_trees.common.Status.FAILURE)

    def test_route_scenario_installs_speed_and_progress_blocked_criteria(self):
        route = [(FakeVector(x=float(index)), object()) for index in range(21)]
        with patch(
            "leaderboard.scenarios.scenarioatomics.atomic_criteria."
            "CarlaDataProvider.get_map",
            return_value=FakeMap(),
        ):
            criteria = _build_blocked_criteria(FakeActor(), route)

        self.assertEqual([item.name for item in criteria], [
            "AgentBlockedTest",
            "AgentRouteProgressBlockedTest",
        ])
        progress = criteria[1]
        self.assertEqual(progress._min_progress_meters, ROUTE_PROGRESS_BLOCKED_METERS)
        self.assertEqual(progress._max_time_seconds, ROUTE_PROGRESS_BLOCKED_SECONDS)
        self.assertEqual(ROUTE_PROGRESS_BLOCKED_METERS, 18.0)
        self.assertEqual(ROUTE_PROGRESS_BLOCKED_SECONDS, 180.0)


if __name__ == "__main__":
    unittest.main()
