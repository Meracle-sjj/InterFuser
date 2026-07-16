import copy
import unittest

from tools.data.check_leaderboard_stop_target_geometry import compare_boundaries


def _boundary(x=10.0, lane_id=-1):
    return {
        "road_id": 7,
        "section_id": 0,
        "lane_id": lane_id,
        "s": 42.0,
        "center": {"x": x, "y": 0.0, "z": 0.0},
        "left_endpoint": {"x": x, "y": -1.6, "z": 0.0},
        "right_endpoint": {"x": x, "y": 1.6, "z": 0.0},
    }


class BoundaryComparisonTests(unittest.TestCase):
    def test_exact_match_has_no_mismatches(self):
        self.assertEqual(compare_boundaries([_boundary()], [_boundary()]), [])

    def test_lane_mismatch_is_reported(self):
        expected = _boundary(lane_id=-2)
        mismatches = compare_boundaries([_boundary()], [expected])
        self.assertTrue(any("lane identity" in item for item in mismatches))

    def test_count_mismatch_is_reported(self):
        mismatches = compare_boundaries([_boundary()], [])
        self.assertTrue(any("boundary count" in item for item in mismatches))

    def test_coordinate_mismatch_respects_tolerance(self):
        expected = copy.deepcopy(_boundary())
        expected["left_endpoint"]["x"] += 0.01
        mismatches = compare_boundaries([_boundary()], [expected], tolerance_m=1e-3)
        self.assertTrue(any("left_endpoint" in item for item in mismatches))


if __name__ == "__main__":
    unittest.main()
