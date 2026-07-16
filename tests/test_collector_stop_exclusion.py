import json
import unittest
from unittest.mock import patch

from team_code.interfuser_collector_complete import InterfuserCollectorComplete
from team_code.interfuser_data_collector import (
    AGENT_OPTIONS,
    InterfuserDataCollector,
)


class FakeVector:
    def __init__(self, x=1.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class FakeLocation(FakeVector):
    def __sub__(self, other):
        return FakeVector(self.x - other.x, self.y - other.y, self.z - other.z)

    def distance(self, other):
        delta = self - other
        return (delta.x**2 + delta.y**2 + delta.z**2) ** 0.5


class FakeRotation:
    yaw = 0.0
    pitch = 0.0
    roll = 0.0


class FakeTransform:
    rotation = FakeRotation()

    def get_forward_vector(self):
        return FakeVector()


class FakeWaypoint:
    is_junction = False
    road_id = 7
    lane_id = -1


class RecordingActors:
    def __init__(self):
        self.patterns = []

    def filter(self, pattern):
        self.patterns.append(pattern)
        return []


class FakeMap:
    def get_waypoint(self, location):
        return FakeWaypoint()


class FakeWorld:
    def __init__(self):
        self.actors = RecordingActors()
        self.map = FakeMap()

    def get_actors(self):
        return self.actors

    def get_map(self):
        return self.map


class FakeHero:
    def __init__(self):
        self.world = FakeWorld()
        self.location = FakeLocation()
        self.transform = FakeTransform()

    def get_world(self):
        return self.world

    def get_location(self):
        return self.location

    def get_transform(self):
        return self.transform

    def get_traffic_light(self):
        return None


class CollectorStopExclusionTests(unittest.TestCase):
    def test_agent_options_ignore_stop_signs(self):
        self.assertIs(AGENT_OPTIONS["ignore_stop_signs"], True)

    def test_complete_measurements_have_no_stop_fields_or_actor_query(self):
        collector = object.__new__(InterfuserCollectorComplete)
        hero = FakeHero()
        with patch.object(
            InterfuserDataCollector,
            "_get_measurements",
            return_value={"speed": 0.0},
        ):
            record = collector._get_measurements(hero, {}, None, 0.0)

        serialized = json.dumps(record).lower()
        self.assertNotIn("stop_sign", serialized)
        self.assertFalse(
            any("stop" in pattern.lower() for pattern in hero.world.actors.patterns)
        )


if __name__ == "__main__":
    unittest.main()
