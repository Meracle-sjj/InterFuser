import math
import unittest

from tools.data.profile_traffic_element_routes import score_dense_route


class RouteProfileScoringTests(unittest.TestCase):
    def test_scores_traffic_lights_and_hard_negative_points(self):
        summary = score_dense_route(
            route_points=[(0.0, 0.0), (10.0, 0.0), (200.0, 0.0)],
            traffic_lights=[(8.0, 1.0)],
            relevant_radius_m=30.0,
            nearby_radius_m=80.0,
        )

        self.assertEqual(summary["dense_route_points"], 3)
        self.assertEqual(summary["nearby_traffic_lights"], 1)
        self.assertEqual(summary["relevant_traffic_lights"], 1)
        self.assertEqual(summary["hard_negative_points"], 1)
        self.assertAlmostEqual(summary["minimum_traffic_light_distance_m"], math.sqrt(5.0))
        self.assertNotIn("stop_sign_actors", summary)

    def test_no_lights_make_all_route_points_hard_negatives(self):
        summary = score_dense_route(
            route_points=[(0.0, 0.0), (10.0, 0.0)],
            traffic_lights=[],
            relevant_radius_m=30.0,
            nearby_radius_m=80.0,
        )

        self.assertEqual(summary["nearby_traffic_lights"], 0)
        self.assertEqual(summary["hard_negative_points"], 2)
        self.assertIsNone(summary["minimum_traffic_light_distance_m"])

    def test_consecutive_repeated_route_points_are_counted_once(self):
        summary = score_dense_route(
            route_points=[(0.0, 0.0), (0.0, 0.0), (10.0, 0.0)],
            traffic_lights=[(0.0, 0.0)],
            relevant_radius_m=1.0,
            nearby_radius_m=20.0,
        )

        self.assertEqual(summary["dense_route_points"], 2)
        self.assertEqual(summary["hard_negative_points"], 1)


if __name__ == "__main__":
    unittest.main()
