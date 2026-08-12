#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
[INPUT]: 依赖 ScenarioRunner Criterion、GameTime、CarlaDataProvider、`(Location, RoadOption)` 路线序列与 TrafficEvent，逐 tick 观察 ego 速度和沿冻结路线的单调进度。
[OUTPUT]: 对外提供速度连续过低和路线进度长期停滞两种 VEHICLE_BLOCKED 判据，均可终止 Leaderboard 场景并保留失败位置与进度证据。
[POS]: leaderboard.scenarios.scenarioatomics 的附加评测原子层，补足上游仅按瞬时速度判断 blocked 时会被短促强制油门反复重置的漏洞。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from collections import deque

import py_trees
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_criteria import Criterion
from srunner.scenariomanager.traffic_events import TrafficEvent, TrafficEventType


class ActorSpeedAboveThresholdTest(Criterion):
    """
    This test will fail if the actor has had its linear velocity lower than a specific value for
    a specific amount of time

    Important parameters:
    - actor: CARLA actor to be used for this test
    - speed_threshold: speed required
    - below_threshold_max_time: Maximum time (in seconds) the actor can remain under the speed threshold
    - terminate_on_failure [optional]: If True, the complete scenario will terminate upon failure of this test
    """

    def __init__(self, actor, speed_threshold, below_threshold_max_time,
                 name="ActorSpeedAboveThresholdTest", terminate_on_failure=False):
        """
        Class constructor.
        """
        super(ActorSpeedAboveThresholdTest, self).__init__(name, actor, 0, terminate_on_failure=terminate_on_failure)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._actor = actor
        self._speed_threshold = speed_threshold
        self._below_threshold_max_time = below_threshold_max_time
        self._time_last_valid_state = None

    def update(self):
        """
        Check if the actor speed is above the speed_threshold
        """
        new_status = py_trees.common.Status.RUNNING

        linear_speed = CarlaDataProvider.get_velocity(self._actor)
        if linear_speed is not None:
            if linear_speed < self._speed_threshold and self._time_last_valid_state:
                if (GameTime.get_time() - self._time_last_valid_state) > self._below_threshold_max_time:
                    # Game over. The actor has been "blocked" for too long
                    self.test_status = "FAILURE"

                    # record event
                    vehicle_location = CarlaDataProvider.get_location(self._actor)
                    
                    # Print prominent error message
                    blocked_duration = GameTime.get_time() - self._time_last_valid_state
                    print("\n" + "="*80)
                    print("❌ 严重错误：车辆因塞车被强制终止！")
                    print("="*80)
                    print(f"阻塞时长: {blocked_duration:.1f} 秒 (超过最大允许时间 {self._below_threshold_max_time:.1f} 秒)")
                    print(f"车辆速度: {linear_speed:.3f} m/s (低于阈值 {self._speed_threshold} m/s)")
                    print(f"阻塞位置: x={vehicle_location.x:.2f}, y={vehicle_location.y:.2f}, z={vehicle_location.z:.2f}")
                    print(f"游戏时间: {GameTime.get_time():.2f} 秒")
                    print("="*80 + "\n")
                    
                    blocked_event = TrafficEvent(event_type=TrafficEventType.VEHICLE_BLOCKED)
                    ActorSpeedAboveThresholdTest._set_event_message(blocked_event, vehicle_location)
                    ActorSpeedAboveThresholdTest._set_event_dict(blocked_event, vehicle_location)
                    self.list_traffic_events.append(blocked_event)
            else:
                self._time_last_valid_state = GameTime.get_time()

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE
        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    @staticmethod
    def _set_event_message(event, location):
        """
        Sets the message of the event
        """

        event.set_message('Agent got blocked at (x={}, y={}, z={})'.format(round(location.x, 3),
                                                                               round(location.y, 3),
                                                                               round(location.z, 3)))
    @staticmethod
    def _set_event_dict(event, location):
        """
        Sets the dictionary of the event
        """
        event.set_dict({
            'x': location.x,
            'y': location.y,
            'z': location.z,
          })


class RouteProgressBlockedTest(Criterion):
    """Fail when route progress stays below a fixed distance for a time window."""

    WINDOWS_SIZE = 2

    def __init__(self, actor, route, min_progress_meters, max_time_seconds,
                 name="RouteProgressBlockedTest", terminate_on_failure=False):
        super(RouteProgressBlockedTest, self).__init__(
            name, actor, 0, terminate_on_failure=terminate_on_failure
        )
        if min_progress_meters <= 0:
            raise ValueError("min_progress_meters must be positive")
        if max_time_seconds <= 0:
            raise ValueError("max_time_seconds must be positive")
        if not route:
            raise ValueError("route must not be empty")

        self._actor = actor
        self._route = route
        self._waypoints, _ = zip(*route)
        self._map = CarlaDataProvider.get_map()
        self._min_progress_meters = float(min_progress_meters)
        self._max_time_seconds = float(max_time_seconds)
        self._current_index = 0
        self._route_length = len(self._waypoints)
        self._progress_samples = deque()

        self._accum_meters = []
        previous = self._waypoints[0]
        accumulated = 0.0
        for index, waypoint in enumerate(self._waypoints):
            if index > 0:
                accumulated += waypoint.distance(previous)
            self._accum_meters.append(accumulated)
            previous = waypoint

    def _update_route_progress(self, location):
        end = min(
            self._current_index + self.WINDOWS_SIZE + 1,
            self._route_length,
        )
        for index in range(self._current_index, end):
            route_location = self._waypoints[index]
            waypoint = self._map.get_waypoint(route_location)
            forward = waypoint.transform.get_forward_vector()
            displacement = location - route_location
            dot_product = (
                displacement.x * forward.x
                + displacement.y * forward.y
                + displacement.z * forward.z
            )
            if dot_product > 0:
                self._current_index = index
        return self._accum_meters[self._current_index]

    def _record_blocked_event(self, location, elapsed, progress_delta):
        blocked_event = TrafficEvent(event_type=TrafficEventType.VEHICLE_BLOCKED)
        blocked_event.set_message(
            "Agent got blocked by route progress stagnation at "
            "(x={}, y={}, z={}); advanced {:.2f} m in {:.1f} s".format(
                round(location.x, 3),
                round(location.y, 3),
                round(location.z, 3),
                progress_delta,
                elapsed,
            )
        )
        blocked_event.set_dict({
            'x': location.x,
            'y': location.y,
            'z': location.z,
            'route_progress_delta_meters': progress_delta,
            'route_progress_window_seconds': elapsed,
            'route_progress_threshold_meters': self._min_progress_meters,
        })
        self.list_traffic_events.append(blocked_event)

    def update(self):
        new_status = py_trees.common.Status.RUNNING
        if self._terminate_on_failure and self.test_status == "FAILURE":
            return py_trees.common.Status.FAILURE

        location = CarlaDataProvider.get_location(self._actor)
        if location is None:
            return new_status

        progress = self._update_route_progress(location)
        now = GameTime.get_time()
        self._progress_samples.append((now, progress))
        cutoff = now - self._max_time_seconds
        while (
            len(self._progress_samples) > 1
            and self._progress_samples[1][0] <= cutoff
        ):
            self._progress_samples.popleft()

        oldest_time, oldest_progress = self._progress_samples[0]
        elapsed = now - oldest_time
        progress_delta = progress - oldest_progress
        if (
            elapsed >= self._max_time_seconds
            and progress_delta < self._min_progress_meters
        ):
            self.test_status = "FAILURE"
            self.actual_value = round(progress_delta, 3)
            self._record_blocked_event(location, elapsed, progress_delta)
            if self._terminate_on_failure:
                new_status = py_trees.common.Status.FAILURE
        else:
            self.test_status = "RUNNING"

        self.logger.debug(
            "%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status)
        )
        return new_status
