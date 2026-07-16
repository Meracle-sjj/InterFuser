import json
import tempfile
import unittest
from pathlib import Path

from tools.data.apply_painted_line_reviews import ReviewError, apply_reviews


TARGET_ID = "Town01_Opt:7:0:-1:20.0"


def _view(candidate_status="candidate"):
    return {
        "cameras": {
            "front": {
                "stop_targets": [
                    {
                        "target_id": TARGET_ID,
                        "painted_line": {
                            "status": candidate_status,
                            "image_segment": [[10.0, 20.0], [50.0, 20.0]],
                            "score": 0.8,
                        },
                    }
                ]
            }
        }
    }


class PaintedLineReviewTests(unittest.TestCase):
    def _fixture(self, root, status="candidate"):
        root = Path(root)
        view_path = root / "route" / "traffic_element_views" / "0000.json"
        view_path.parent.mkdir(parents=True)
        view_path.write_text(json.dumps(_view(status)), encoding="utf-8")
        return view_path

    def _manifest(self, root, entries):
        path = Path(root) / "manifest.json"
        path.write_text(json.dumps(entries), encoding="utf-8")
        return path

    def test_verified_decision_promotes_existing_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            view_path = self._fixture(tmp)
            manifest = self._manifest(
                tmp,
                [
                    {
                        "view_path": "route/traffic_element_views/0000.json",
                        "camera": "front",
                        "target_id": TARGET_ID,
                        "decision": "verified",
                    }
                ],
            )

            summary = apply_reviews(tmp, manifest)
            result = json.loads(view_path.read_text(encoding="utf-8"))

        painted = result["cameras"]["front"]["stop_targets"][0]["painted_line"]
        self.assertEqual(painted["status"], "verified")
        self.assertEqual(painted["review_source"], "manual_manifest")
        self.assertEqual(summary["verified"], 1)

    def test_rejected_decision_marks_candidate_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            view_path = self._fixture(tmp)
            manifest = self._manifest(
                tmp,
                [
                    {
                        "view_path": "route/traffic_element_views/0000.json",
                        "camera": "front",
                        "target_id": TARGET_ID,
                        "decision": "rejected",
                    }
                ],
            )

            apply_reviews(tmp, manifest)
            result = json.loads(view_path.read_text(encoding="utf-8"))

        painted = result["cameras"]["front"]["stop_targets"][0]["painted_line"]
        self.assertEqual(painted["status"], "unknown")
        self.assertEqual(painted["review_source"], "manual_manifest")

    def test_unreviewed_decision_does_not_rewrite_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            view_path = self._fixture(tmp)
            original = view_path.read_bytes()
            manifest = self._manifest(
                tmp,
                [
                    {
                        "view_path": "route/traffic_element_views/0000.json",
                        "camera": "front",
                        "target_id": TARGET_ID,
                        "decision": "unreviewed",
                    }
                ],
            )

            summary = apply_reviews(tmp, manifest)

            self.assertEqual(view_path.read_bytes(), original)
            self.assertEqual(summary["unreviewed"], 1)

    def test_missing_target_fails_before_any_file_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            view_path = self._fixture(tmp)
            original = view_path.read_bytes()
            manifest = self._manifest(
                tmp,
                [
                    {
                        "view_path": "route/traffic_element_views/0000.json",
                        "camera": "front",
                        "target_id": TARGET_ID,
                        "decision": "verified",
                    },
                    {
                        "view_path": "route/traffic_element_views/0000.json",
                        "camera": "front",
                        "target_id": "missing",
                        "decision": "verified",
                    },
                ],
            )

            with self.assertRaisesRegex(ReviewError, "target_id missing"):
                apply_reviews(tmp, manifest)

            self.assertEqual(view_path.read_bytes(), original)

    def test_unknown_candidate_cannot_be_verified_directly(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._fixture(tmp, status="unknown")
            manifest = self._manifest(
                tmp,
                [
                    {
                        "view_path": "route/traffic_element_views/0000.json",
                        "camera": "front",
                        "target_id": TARGET_ID,
                        "decision": "verified",
                    }
                ],
            )

            with self.assertRaisesRegex(ReviewError, "existing candidate"):
                apply_reviews(tmp, manifest)


if __name__ == "__main__":
    unittest.main()
