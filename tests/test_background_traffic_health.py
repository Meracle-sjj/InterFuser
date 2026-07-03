import unittest

from leaderboard.scenarios.route_scenario import _measure_background_traffic


class FakeLocation:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def distance(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return (dx * dx + dy * dy + dz * dz) ** 0.5


class FakeVector:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y


class FakeTransform:
    def __init__(self, location):
        self.location = location


class FakeWaypoint:
    def __init__(self, location):
        self.transform = FakeTransform(location)


class FakeMap:
    def __init__(self, road_locations):
        self.road_locations = road_locations

    def get_waypoint(self, location, project_to_road=True, lane_type=None):
        return FakeWaypoint(self.road_locations.get((location.x, location.y), location))


class FakeActor:
    def __init__(self, actor_id, location, velocity, alive=True):
        self.id = actor_id
        self.is_alive = alive
        self._location = location
        self._velocity = velocity

    def get_location(self):
        return self._location

    def get_velocity(self):
        return self._velocity


class BackgroundTrafficHealthTests(unittest.TestCase):
    def test_counts_moving_and_near_road_background_actors(self):
        actors = [
            FakeActor(1, FakeLocation(3.0, 4.0), FakeVector(3.0, 4.0)),
            FakeActor(2, FakeLocation(10.0, 0.0), FakeVector(0.0, 0.1)),
            FakeActor(3, FakeLocation(99.0, 0.0), FakeVector(10.0, 0.0), alive=False),
        ]
        start_locations = {
            1: FakeLocation(0.0, 0.0),
            2: FakeLocation(10.0, 0.0),
            3: FakeLocation(0.0, 0.0),
        }
        world_map = FakeMap({
            (3.0, 4.0): FakeLocation(3.2, 4.1),
            (10.0, 0.0): FakeLocation(15.0, 0.0),
        })

        health = _measure_background_traffic(actors, world_map, start_locations)

        self.assertEqual(health["total"], 3)
        self.assertEqual(health["alive"], 2)
        self.assertEqual(health["moving"], 1)
        self.assertEqual(health["near_road"], 1)
        self.assertAlmostEqual(health["avg_speed"], 2.55)
        self.assertAlmostEqual(health["avg_distance"], 2.5)
        self.assertAlmostEqual(health["max_road_distance"], 5.0)


if __name__ == "__main__":
    unittest.main()
