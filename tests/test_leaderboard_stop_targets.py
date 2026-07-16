import math
import unittest

from team_code.leaderboard_stop_targets import (
    BOUNDARY_HALF_LANE_RATIO,
    BOUNDARY_STEP_M,
    CORRIDOR_EXTENSION_M,
    GEOMETRY_SOURCE,
    ROUTE_MIN_DISTANCE_M,
    SAFETY_MARGIN_M,
    advance_to_infraction_boundary,
    boundary_from_waypoint,
    build_stop_targets,
    sample_trigger_lane_waypoints,
)


class FakeLocation:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def distance(self, other):
        return math.sqrt(
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        )


class FakeRotation:
    def __init__(self, yaw=0.0, pitch=0.0, roll=0.0):
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll


class FakeTransform:
    def __init__(self, location=None, yaw=0.0):
        self.location = location or FakeLocation()
        self.rotation = FakeRotation(yaw=yaw)

    def transform(self, location):
        yaw = math.radians(self.rotation.yaw)
        return FakeLocation(
            self.location.x
            + location.x * math.cos(yaw)
            - location.y * math.sin(yaw),
            self.location.y
            + location.x * math.sin(yaw)
            + location.y * math.cos(yaw),
            self.location.z + location.z,
        )


class FakeExtent:
    def __init__(self, x=1.0, y=1.0, z=1.0):
        self.x = x
        self.y = y
        self.z = z


class FakeTrigger:
    def __init__(self, location=None, extent=None):
        self.location = location or FakeLocation()
        self.extent = extent or FakeExtent()


class FakeWaypoint:
    def __init__(
        self,
        road_id,
        lane_id,
        x,
        y=0.0,
        yaw=0.0,
        lane_width=4.0,
        section_id=0,
        s=None,
        is_intersection=False,
    ):
        self.road_id = road_id
        self.section_id = section_id
        self.lane_id = lane_id
        self.s = float(x if s is None else s)
        self.lane_width = lane_width
        self.is_intersection = is_intersection
        self.is_junction = is_intersection
        self.transform = FakeTransform(FakeLocation(x, y, 0.0), yaw=yaw)
        self.successors = []
        self.next_distances = []

    def next(self, distance):
        self.next_distances.append(distance)
        return list(self.successors)


class FakeMap:
    def __init__(self, default_lane=None, route=None, name="Town01_Opt"):
        self.default_lane = default_lane
        self.route = list(route or [])
        self.name = name
        self.queries = []

    def get_waypoint(self, location):
        self.queries.append(location)
        if self.route:
            return min(
                self.route,
                key=lambda waypoint: waypoint.transform.location.distance(location),
            )
        return self.default_lane


class FakeTrafficLight:
    type_id = "traffic.traffic_light"

    def __init__(
        self,
        actor_id=11,
        state="Red",
        trigger_x=0.0,
        extent_x=2.0,
        yaw=0.0,
    ):
        self.id = actor_id
        self.state = state
        self._transform = FakeTransform(FakeLocation(trigger_x, 0.0, 0.0), yaw=yaw)
        self.trigger_volume = FakeTrigger(
            location=FakeLocation(),
            extent=FakeExtent(extent_x, 1.0, 2.0),
        )

    def get_transform(self):
        return self._transform

    def get_location(self):
        return self._transform.location


class FakeBoundingBox:
    def __init__(self, location_x=0.0, extent_x=2.0):
        self.location = FakeLocation(x=location_x)
        self.extent = FakeExtent(x=extent_x)


def linked_waypoints(xs, intersection_index=None):
    waypoints = [
        FakeWaypoint(
            road_id=7,
            lane_id=-1,
            x=x,
            is_intersection=index == intersection_index,
        )
        for index, x in enumerate(xs)
    ]
    for current, following in zip(waypoints, waypoints[1:]):
        current.successors = [following]
    return waypoints


def dense_route(start, end, step=0.5, intersection_start=20.0):
    count = int(round((end - start) / step))
    xs = [start + index * step for index in range(count + 1)]
    route = [
        FakeWaypoint(
            road_id=7,
            lane_id=-1,
            x=x,
            s=x,
            is_intersection=x >= intersection_start,
        )
        for x in xs
    ]
    for current, following in zip(route, route[1:]):
        current.successors = [following]
    return route


def ego_transform_at(x, yaw=0.0):
    return FakeTransform(FakeLocation(x, 0.0, 0.0), yaw=yaw)


class LeaderboardBoundaryTests(unittest.TestCase):
    def test_samples_ninety_percent_trigger_extent_at_one_meter(self):
        light = FakeTrafficLight(extent_x=2.0, yaw=0.0)
        world_map = FakeMap(default_lane=FakeWaypoint(7, -1, 0.0))

        samples = sample_trigger_lane_waypoints(light, world_map)

        self.assertEqual(
            [round(location.x, 1) for location in world_map.queries],
            [-1.8, -0.8, 0.2, 1.2],
        )
        self.assertEqual(len(samples), 1)

    def test_advances_by_half_meter_to_last_non_intersection_waypoint(self):
        start, middle, boundary, _junction = linked_waypoints(
            [0.0, 0.5, 1.0, 1.5],
            intersection_index=3,
        )

        result = advance_to_infraction_boundary(start)

        self.assertEqual(result["status"], "valid")
        self.assertIs(result["waypoint"], boundary)
        self.assertEqual(result["steps"], 2)
        self.assertEqual(start.next_distances, [BOUNDARY_STEP_M])
        self.assertEqual(middle.next_distances, [BOUNDARY_STEP_M])
        self.assertEqual(boundary.next_distances, [BOUNDARY_STEP_M])

    def test_branch_is_unknown_even_when_first_branch_matches_evaluator(self):
        start = FakeWaypoint(7, -1, 0.0)
        first = FakeWaypoint(7, -1, 0.5, is_intersection=True)
        second = FakeWaypoint(8, -1, 0.5, is_intersection=True)
        start.successors = [first, second]

        result = advance_to_infraction_boundary(start)

        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["unknown_reason"], "waypoint_branch")
        self.assertIs(result["waypoint"], start)

    def test_boundary_uses_evaluator_lane_width_ratio_and_left_right_names(self):
        waypoint = FakeWaypoint(
            road_id=7,
            lane_id=-1,
            x=10.0,
            y=2.0,
            yaw=0.0,
            lane_width=4.0,
            section_id=3,
            s=42.5,
        )

        boundary = boundary_from_waypoint(waypoint)

        self.assertEqual(BOUNDARY_HALF_LANE_RATIO, 0.4)
        self.assertEqual(
            boundary["geometry_source"],
            "scenario_runner_running_red_light_test_v1",
        )
        self.assertEqual(GEOMETRY_SOURCE, boundary["geometry_source"])
        self.assertAlmostEqual(boundary["left_endpoint"]["y"], 3.6)
        self.assertAlmostEqual(boundary["right_endpoint"]["y"], 0.4)
        self.assertEqual(
            (boundary["road_id"], boundary["section_id"], boundary["lane_id"]),
            (7, 3, -1),
        )
        self.assertEqual(boundary["s"], 42.5)

    def test_waypoint_loop_is_unknown(self):
        start = FakeWaypoint(7, -1, 0.0, s=1.0)
        following = FakeWaypoint(7, -1, 0.5, s=2.0)
        start.successors = [following]
        following.successors = [start]

        result = advance_to_infraction_boundary(start)

        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["unknown_reason"], "waypoint_loop")

    def test_missing_successor_is_unknown(self):
        result = advance_to_infraction_boundary(FakeWaypoint(7, -1, 0.0))

        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["unknown_reason"], "intersection_not_found")

    def test_initial_intersection_waypoint_is_valid_without_advancing(self):
        start = FakeWaypoint(7, -1, 0.0, is_intersection=True)

        result = advance_to_infraction_boundary(start)

        self.assertEqual(result["status"], "valid")
        self.assertIs(result["waypoint"], start)
        self.assertEqual(result["steps"], 0)
        self.assertEqual(start.next_distances, [])


class RouteTargetTests(unittest.TestCase):
    def _targets(
        self,
        ego_x=0.0,
        ego_yaw=0.0,
        trigger_x=10.0,
        actor_ids=(41,),
        bbox_location_x=0.2,
        bbox_extent_x=2.0,
    ):
        route = dense_route(0.0, 30.0)
        world_map = FakeMap(route=route)
        lights = [
            FakeTrafficLight(
                actor_id=actor_id,
                state="Red",
                trigger_x=trigger_x,
                extent_x=0.1,
            )
            for actor_id in actor_ids
        ]
        return build_stop_targets(
            lights,
            world_map,
            route,
            ego_transform_at(ego_x, yaw=ego_yaw),
            FakeBoundingBox(
                location_x=bbox_location_x,
                extent_x=bbox_extent_x,
            ),
            world_map.name,
        )

    def test_groups_owner_lights_and_uses_map_stable_target_id(self):
        targets = self._targets(actor_ids=(99, 41))

        self.assertEqual(len(targets), 1)
        target = targets[0]
        self.assertEqual(target["owner_traffic_light_actor_ids"], [41, 99])
        self.assertEqual(
            target["target_id"],
            "Town01_Opt:7:0:-1:19.5",
        )
        self.assertEqual(target["state_by_actor_id"], {"41": "Red", "99": "Red"})

    def test_signed_route_distance_and_primary_target(self):
        target = self._targets(ego_x=5.0)[0]

        self.assertAlmostEqual(target["signed_route_distance_m"], 14.5)
        self.assertAlmostEqual(target["euclidean_distance_m"], 14.5)
        self.assertTrue(target["ego_before_boundary"])
        self.assertTrue(target["primary_for_ego"])

    def test_recently_crossed_target_is_retained(self):
        target = self._targets(ego_x=25.0)[0]

        self.assertAlmostEqual(target["signed_route_distance_m"], -5.5)
        self.assertGreaterEqual(
            target["signed_route_distance_m"],
            ROUTE_MIN_DISTANCE_M,
        )
        self.assertFalse(target["ego_before_boundary"])
        self.assertFalse(target["primary_for_ego"])

    def test_target_before_post_crossing_window_is_omitted(self):
        self.assertEqual(self._targets(ego_x=30.0), [])

    def test_stop_pose_uses_vehicle_front_offset_and_safety_margin(self):
        target = self._targets(ego_x=0.0)[0]

        self.assertEqual(SAFETY_MARGIN_M, 1.0)
        self.assertAlmostEqual(target["vehicle_front_offset_m"], 2.2)
        self.assertAlmostEqual(
            target["recommended_ego_stop_pose"]["location"]["x"],
            16.3,
        )

    def test_corridor_extends_beyond_trigger_and_boundary(self):
        target = self._targets(ego_x=0.0)[0]
        xs = [
            sample["location"]["x"]
            for sample in target["stop_evidence_corridor"]["centerline"]
        ]

        self.assertEqual(CORRIDOR_EXTENSION_M, 3.0)
        self.assertAlmostEqual(xs[0], 7.0)
        self.assertAlmostEqual(xs[-1], 22.5)
        self.assertTrue(all(sample["lane_width"] == 4.0 for sample in target["stop_evidence_corridor"]["centerline"]))

    def test_reverse_ego_heading_makes_target_unknown(self):
        target = self._targets(ego_x=5.0, ego_yaw=180.0)[0]

        self.assertEqual(target["status"], "unknown")
        self.assertEqual(target["unknown_reason"], "direction_mismatch")
        self.assertFalse(target["primary_for_ego"])

    def test_empty_route_produces_no_target_or_negative(self):
        light = FakeTrafficLight(trigger_x=10.0, extent_x=0.1)

        targets = build_stop_targets(
            [light],
            FakeMap(default_lane=FakeWaypoint(7, -1, 10.0)),
            [],
            ego_transform_at(0.0),
            FakeBoundingBox(),
            "Town01_Opt",
        )

        self.assertEqual(targets, [])


if __name__ == "__main__":
    unittest.main()
