import fnmatch
import json
import math
import unittest

from team_code.traffic_element_labels import (
    collect_traffic_element_labels,
    legacy_affordances_from_labels,
    merge_legacy_affordances,
    validate_traffic_element_record,
    world_to_ego,
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


class FakeBoundingBox:
    def __init__(self, location_x=0.0, extent_x=2.0):
        self.location = FakeLocation(x=location_x)
        self.extent = FakeExtent(x=extent_x)


class FakeTrigger:
    def __init__(self, extent_x=0.1):
        self.location = FakeLocation()
        self.extent = FakeExtent(extent_x, 1.0, 2.0)
        self.rotation = FakeRotation()


class FakeWaypoint:
    def __init__(self, x, is_intersection=False):
        self.road_id = 7
        self.section_id = 0
        self.lane_id = -1
        self.s = float(x)
        self.lane_width = 4.0
        self.is_intersection = is_intersection
        self.is_junction = is_intersection
        self.transform = FakeTransform(FakeLocation(x, 0.0, 0.0))
        self.successors = []

    def next(self, distance):
        return list(self.successors)


def dense_route(start=0.0, end=30.0, step=0.5):
    count = int(round((end - start) / step))
    route = [
        FakeWaypoint(
            start + index * step,
            is_intersection=start + index * step >= 20.0,
        )
        for index in range(count + 1)
    ]
    for current, following in zip(route, route[1:]):
        current.successors = [following]
    return route


class FakeTrafficLight:
    type_id = "traffic.traffic_light"

    def __init__(self, actor_id=11, state="Red", x=10.0, affected=None):
        self.id = actor_id
        self.state = state
        self._transform = FakeTransform(FakeLocation(x, 0.0, 0.0))
        self.trigger_volume = FakeTrigger()
        self._affected = list(affected or [])

    def get_location(self):
        return self._transform.location

    def get_transform(self):
        return self._transform

    def get_affected_lane_waypoints(self):
        return list(self._affected)


class FakeStopSign:
    type_id = "traffic.stop"

    def __init__(self, actor_id=21):
        self.id = actor_id


class RecordingActorList(list):
    def __init__(self, values):
        super().__init__(values)
        self.filter_patterns = []

    def filter(self, pattern):
        self.filter_patterns.append(pattern)
        return RecordingActorList(
            actor
            for actor in self
            if fnmatch.fnmatch(str(getattr(actor, "type_id", "")), pattern)
        )


class FakeMap:
    name = "Carla/Maps/Town01_Opt"

    def __init__(self, route, missing_ego=False):
        self.route = list(route)
        self.missing_ego = missing_ego

    def get_waypoint(self, location):
        if self.missing_ego and abs(location.x) < 1e-6:
            return None
        return min(
            self.route,
            key=lambda waypoint: waypoint.transform.location.distance(location),
        )


class FakeWorld:
    def __init__(self, actors, route, missing_ego=False):
        self._actors = actors
        self._map = FakeMap(route, missing_ego=missing_ego)

    def get_actors(self):
        return self._actors

    def get_map(self):
        return self._map


class FakeHero:
    id = 1

    def __init__(self, active_light=None, x=0.0, yaw=0.0):
        self._active_light = active_light
        self._transform = FakeTransform(FakeLocation(x, 0.0, 0.0), yaw=yaw)
        self.bounding_box = FakeBoundingBox(location_x=0.2, extent_x=2.0)

    def get_location(self):
        return self._transform.location

    def get_transform(self):
        return self._transform

    def get_traffic_light(self):
        return self._active_light


class TrafficElementLabelTests(unittest.TestCase):
    def test_world_to_ego_uses_forward_right_coordinates(self):
        ego = FakeTransform(FakeLocation(10.0, 20.0, 0.0), yaw=90.0)

        relative = world_to_ego(FakeLocation(10.0, 25.0, 1.0), ego)

        self.assertAlmostEqual(relative["forward"], 5.0)
        self.assertAlmostEqual(relative["right"], 0.0)
        self.assertAlmostEqual(relative["up"], 1.0)

    def test_v2_record_contains_route_stop_target_without_stop_sign_schema(self):
        route = dense_route()
        light = FakeTrafficLight(affected=[route[20]])
        record = collect_traffic_element_labels(
            FakeHero(active_light=light),
            FakeWorld(RecordingActorList([light]), route),
            frame_id="0042",
            route_waypoints=route,
        )

        self.assertEqual(record["schema_version"], 2)
        self.assertEqual(record["frame_id"], "0042")
        self.assertEqual(record["map_name"], "Town01_Opt")
        self.assertEqual(len(record["stop_targets"]), 1)
        self.assertNotIn("stop_signs", record)
        self.assertNotIn("stop_lines", record["traffic_lights"][0])
        self.assertEqual(
            record["stop_targets"][0]["geometry_source"],
            "scenario_runner_running_red_light_test_v1",
        )

    def test_traffic_stop_actor_is_never_queried_or_serialized(self):
        route = dense_route()
        actors = RecordingActorList(
            [FakeStopSign(), FakeTrafficLight(state="Green", x=120.0)]
        )
        record = collect_traffic_element_labels(
            FakeHero(),
            FakeWorld(actors, route),
            frame_id="0000",
            route_waypoints=route,
        )

        self.assertNotIn("traffic.stop*", actors.filter_patterns)
        self.assertNotIn("stop_signs", record)
        serialized = json.dumps(record).lower()
        self.assertNotIn("traffic.stop", serialized)
        self.assertNotIn("trigger_volume_route_entry_approximation", serialized)

    def test_merge_legacy_affordances_preserves_hazards_without_stop_sign(self):
        labels = {
            "traffic_lights": [
                {"is_active_for_ego": True, "state": "Red"},
            ],
        }

        merged = merge_legacy_affordances(
            {"hazard_vehicle": True, "hazard_pedestrian": False},
            labels,
        )

        self.assertEqual(
            merged,
            {
                "traffic_light": "Red",
                "hazard_vehicle": True,
                "hazard_pedestrian": False,
            },
        )
        self.assertNotIn("stop_sign", merged)

    def test_no_active_light_affordance_is_none_without_stop_field(self):
        self.assertEqual(
            legacy_affordances_from_labels({"traffic_lights": []}),
            {"traffic_light": None},
        )

    def test_missing_ego_waypoint_keeps_frame_unknown_without_stop_negative(self):
        route = dense_route()
        record = collect_traffic_element_labels(
            FakeHero(),
            FakeWorld(RecordingActorList([]), route, missing_ego=True),
            frame_id="0000",
            route_waypoints=route,
        )

        self.assertIsNone(record["ego"]["lane"])
        self.assertEqual(record["stop_targets"], [])
        self.assertEqual(record["errors"][0]["field"], "ego_waypoint")

    def test_active_light_is_retained_outside_nearby_actor_radius(self):
        route = dense_route()
        light = FakeTrafficLight(actor_id=11, state="Red", x=120.0)
        record = collect_traffic_element_labels(
            FakeHero(active_light=light),
            FakeWorld(RecordingActorList([light]), route),
            frame_id="0000",
            route_waypoints=route,
            max_distance=10.0,
        )

        self.assertEqual(record["active_traffic_light_id"], 11)
        self.assertEqual(len(record["traffic_lights"]), 1)
        self.assertTrue(record["traffic_lights"][0]["is_active_for_ego"])

    def test_validator_rejects_legacy_stop_sign_field(self):
        record = {
            "schema_version": 2,
            "frame_id": "0000",
            "map_name": "Town01_Opt",
            "ego": {},
            "traffic_lights": [],
            "stop_targets": [],
            "errors": [],
            "stop_signs": [],
        }

        errors = validate_traffic_element_record(record)

        self.assertIn("stop_signs is forbidden", errors)


if __name__ == "__main__":
    unittest.main()
