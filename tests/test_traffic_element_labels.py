import fnmatch
import math
import unittest

from team_code.traffic_element_labels import (
    collect_traffic_element_labels,
    legacy_affordances_from_labels,
    merge_legacy_affordances,
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
            self.location.x + location.x * math.cos(yaw) - location.y * math.sin(yaw),
            self.location.y + location.x * math.sin(yaw) + location.y * math.cos(yaw),
            self.location.z + location.z,
        )


class FakeExtent:
    def __init__(self, x=1.0, y=1.0, z=1.0):
        self.x = x
        self.y = y
        self.z = z


class FakeTriggerVolume:
    def __init__(self, location=None, extent=None, yaw=0.0):
        self.location = location or FakeLocation()
        self.extent = extent or FakeExtent()
        self.rotation = FakeRotation(yaw=yaw)


class FakeWaypoint:
    def __init__(
        self,
        road_id,
        lane_id,
        x,
        y,
        yaw=0.0,
        lane_width=4.0,
        section_id=0,
        s=0.0,
    ):
        self.road_id = road_id
        self.lane_id = lane_id
        self.section_id = section_id
        self.s = s
        self.lane_width = lane_width
        self.transform = FakeTransform(FakeLocation(x, y, 0.0), yaw=yaw)
        self._next = []

    def next(self, distance):
        return list(self._next)


class BrokenNextWaypoint(FakeWaypoint):
    def next(self, distance):
        raise RuntimeError("waypoint traversal failed")


def linked_waypoints(xs, road_id=7, lane_id=-1):
    waypoints = [
        FakeWaypoint(road_id, lane_id, x, 0.0, s=float(index))
        for index, x in enumerate(xs)
    ]
    for current, following in zip(waypoints, waypoints[1:]):
        current._next = [following]
    return waypoints[0]


class FakeTrafficLight:
    type_id = "traffic.traffic_light"

    def __init__(self, actor_id, state, stop_waypoints, x=15.0, y=0.0):
        self.id = actor_id
        self.state = state
        self._stop_waypoints = stop_waypoints
        self._transform = FakeTransform(FakeLocation(x, y, 0.0), yaw=0.0)
        self.trigger_volume = FakeTriggerVolume()

    def get_location(self):
        return self._transform.location

    def get_transform(self):
        return self._transform

    def get_stop_waypoints(self):
        return list(self._stop_waypoints)

    def get_affected_lane_waypoints(self):
        return list(self._stop_waypoints)


class FakeStopSign:
    type_id = "traffic.stop"

    def __init__(self, actor_id, trigger_center, trigger_extent):
        self.id = actor_id
        self._transform = FakeTransform(FakeLocation(), yaw=0.0)
        self.trigger_volume = FakeTriggerVolume(
            location=trigger_center,
            extent=FakeExtent(trigger_extent[0], trigger_extent[1], 2.0),
        )

    def get_location(self):
        return self._transform.transform(self.trigger_volume.location)

    def get_transform(self):
        return self._transform


class FakeActorList(list):
    def filter(self, pattern):
        return FakeActorList(
            actor for actor in self if fnmatch.fnmatch(actor.type_id, pattern)
        )


class FakeMap:
    def __init__(self, hero_waypoint):
        self.hero_waypoint = hero_waypoint

    def get_waypoint(self, location, *args, **kwargs):
        return self.hero_waypoint


class FakeWorld:
    def __init__(self, actors, hero_waypoint):
        self._actors = FakeActorList(actors)
        self._map = FakeMap(hero_waypoint)

    def get_actors(self):
        return self._actors

    def get_map(self):
        return self._map


class FakeHero:
    id = 1
    type_id = "vehicle.ego"

    def __init__(self, active_light=None, x=0.0, y=0.0, yaw=0.0):
        self._active_light = active_light
        self._transform = FakeTransform(FakeLocation(x, y, 0.0), yaw=yaw)

    def get_location(self):
        return self._transform.location

    def get_transform(self):
        return self._transform

    def get_traffic_light(self):
        return self._active_light

    def is_at_traffic_light(self):
        return self._active_light is not None


class TrafficElementLabelTests(unittest.TestCase):
    def test_world_to_ego_uses_forward_right_coordinates(self):
        ego = FakeTransform(FakeLocation(10.0, 20.0, 0.0), yaw=90.0)

        relative = world_to_ego(FakeLocation(10.0, 25.0, 1.0), ego)

        self.assertAlmostEqual(relative["forward"], 5.0)
        self.assertAlmostEqual(relative["right"], 0.0)
        self.assertAlmostEqual(relative["up"], 1.0)

    def test_active_light_emits_exact_carla_stop_waypoint_geometry(self):
        stop_waypoint = FakeWaypoint(
            road_id=3,
            lane_id=-1,
            x=12.0,
            y=0.0,
            yaw=0.0,
            lane_width=4.0,
        )
        light = FakeTrafficLight(11, "Red", [stop_waypoint])
        hero = FakeHero(active_light=light)
        world = FakeWorld(
            [light],
            hero_waypoint=FakeWaypoint(3, -1, 0.0, 0.0),
        )

        labels = collect_traffic_element_labels(hero, world)

        item = labels["traffic_lights"][0]
        stop_line = item["stop_lines"][0]
        self.assertTrue(item["is_active_for_ego"])
        self.assertTrue(item["controls_ego_lane"])
        self.assertEqual(stop_line["geometry_source"], "carla_stop_waypoint")
        self.assertAlmostEqual(stop_line["longitudinal_distance"], 12.0)
        self.assertAlmostEqual(stop_line["left_endpoint"]["y"], -2.0)
        self.assertAlmostEqual(stop_line["right_endpoint"]["y"], 2.0)

    def test_stop_sign_route_trigger_intersection_sets_legacy_affordance(self):
        stop = FakeStopSign(
            actor_id=21,
            trigger_center=FakeLocation(8.0, 0.0),
            trigger_extent=(2.0, 2.0),
        )
        hero = FakeHero(active_light=None)
        world = FakeWorld(
            [stop],
            hero_waypoint=linked_waypoints([0.0, 4.0, 7.0, 9.0]),
        )

        labels = collect_traffic_element_labels(hero, world)
        affordances = legacy_affordances_from_labels(labels)

        item = labels["stop_signs"][0]
        self.assertTrue(item["affects_ego_route"])
        self.assertEqual(
            item["stop_lines"][0]["geometry_source"],
            "trigger_volume_route_entry_approximation",
        )
        self.assertAlmostEqual(
            item["stop_lines"][0]["longitudinal_distance"],
            4.0,
        )
        self.assertTrue(affordances["stop_sign"])

    def test_merge_legacy_affordances_preserves_other_hazards(self):
        labels = {
            "traffic_lights": [
                {"is_active_for_ego": True, "state": "Red"},
            ],
            "stop_signs": [
                {"affects_ego_route": True},
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
                "stop_sign": True,
                "hazard_vehicle": True,
                "hazard_pedestrian": False,
            },
        )

    def test_stop_sign_traversal_error_does_not_create_negative_label(self):
        stop = FakeStopSign(
            actor_id=21,
            trigger_center=FakeLocation(8.0, 0.0),
            trigger_extent=(2.0, 2.0),
        )
        hero = FakeHero(active_light=None)
        world = FakeWorld(
            [stop],
            hero_waypoint=BrokenNextWaypoint(7, -1, 0.0, 0.0),
        )

        labels = collect_traffic_element_labels(hero, world)

        self.assertEqual(labels["stop_signs"], [])
        self.assertEqual(labels["errors"][0]["field"], "stop_sign")
        self.assertIn("waypoint traversal failed", labels["errors"][0]["error"])

    def test_missing_ego_waypoint_does_not_create_stop_sign_negative_label(self):
        stop = FakeStopSign(
            actor_id=21,
            trigger_center=FakeLocation(8.0, 0.0),
            trigger_extent=(2.0, 2.0),
        )
        labels = collect_traffic_element_labels(
            FakeHero(active_light=None),
            FakeWorld([stop], hero_waypoint=None),
        )

        self.assertEqual(labels["stop_signs"], [])
        self.assertEqual(labels["errors"][0]["field"], "ego_waypoint")

    def test_active_light_is_retained_outside_nearby_actor_radius(self):
        light = FakeTrafficLight(
            actor_id=11,
            state="Red",
            stop_waypoints=[FakeWaypoint(3, -1, 12.0, 0.0)],
            x=120.0,
        )
        labels = collect_traffic_element_labels(
            FakeHero(active_light=light),
            FakeWorld([light], hero_waypoint=FakeWaypoint(3, -1, 0.0, 0.0)),
            max_distance=10.0,
        )

        self.assertEqual(labels["active_traffic_light_id"], 11)
        self.assertEqual(len(labels["traffic_lights"]), 1)
        self.assertTrue(labels["traffic_lights"][0]["is_active_for_ego"])


if __name__ == "__main__":
    unittest.main()
