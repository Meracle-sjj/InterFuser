"""
[INPUT]: 依赖 tools.data.audit_pedestrian_hazard_visibility 的 inventory、碰撞威胁测量与三相机语义可见性审计 API。
[OUTPUT]: 提供威胁阶段覆盖、阶段划分、天气重复签名、缺失真值和 CLI 门禁的最小回归测试。
[POS]: tests 的 M1 行人特权监督契约测试，防止普通可见行人与真实碰撞威胁在数据结论中被混为一谈。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from PIL import Image

from tools.data.audit_pedestrian_hazard_visibility import (
    PedestrianVisibilityAuditError,
    audit_pedestrian_hazard_visibility,
    main,
)


class PedestrianHazardVisibilityAuditTests(unittest.TestCase):
    def _write_sequence(
        self,
        root,
        relative,
        visible_by_frame,
        hazard_frames,
        cameras=("front", "right"),
        include_hazard_field=True,
    ):
        route = Path(root) / relative
        for camera in cameras:
            (route / f"seg_{camera}").mkdir(parents=True)
            (route / f"rgb_{camera}").mkdir(parents=True)
        (route / "measurements").mkdir(parents=True)
        (route / "birdview").mkdir(parents=True)

        for frame in sorted(visible_by_frame):
            for camera in cameras:
                labels = np.zeros((4, 4), dtype=np.uint8)
                pixels = visible_by_frame[frame].get(camera, 0)
                labels.flat[:pixels] = 12
                Image.fromarray(labels).save(route / f"seg_{camera}" / f"{frame}.png")
                Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
                    route / f"rgb_{camera}" / f"{frame}.jpg"
                )
            measurement = {"should_brake": frame in hazard_frames}
            if include_hazard_field:
                measurement["is_pedestrian_present"] = (
                    [101] if frame in hazard_frames else []
                )
            (route / "measurements" / f"{frame}.json").write_text(
                json.dumps(measurement), encoding="utf-8"
            )
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
                route / "birdview" / f"{frame}.png"
            )

    def _write_inventory(self, root, sequences):
        path = Path(root) / "inventory.json"
        path.write_text(
            json.dumps(
                {
                    "total_scanned": len(sequences),
                    "pedestrian_sequences": len(sequences),
                    "sequences": [{"sequence": item} for item in sequences],
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_reports_hazard_visibility_and_phases(self):
        relative = "town01/town01_tiny_route04_w0_ClearNoon/sequence_a"
        visible = {
            "0000": {"front": 4},
            "0001": {"front": 3},
            "0002": {},
            "0003": {"right": 5},
        }
        with tempfile.TemporaryDirectory() as root:
            self._write_sequence(root, relative, visible, {"0001", "0002"})
            inventory = self._write_inventory(root, [relative])
            report = audit_pedestrian_hazard_visibility(
                root,
                inventory,
                cameras=("front", "right"),
                minimum_pixels=2,
            )

        sequence = report["sequences"][0]
        self.assertTrue(report["valid"])
        self.assertTrue(sequence["has_privileged_collision_hazard"])
        self.assertTrue(sequence["has_visible_hazard_frame"])
        self.assertEqual(sequence["hazard_intervals"], [["0001", "0002"]])
        self.assertEqual(sequence["hazard_visible_frames"], 1)
        self.assertEqual(sequence["hazard_not_visible_frames"], 1)
        self.assertEqual(sequence["visible_before_hazard_frames"], 1)
        self.assertEqual(sequence["visible_after_hazard_frames"], 1)
        self.assertEqual(sequence["hazard_frame_visibility_ratio"], 0.5)

    def test_collapses_weather_repeats_by_trajectory_visibility_signature(self):
        first = "town03/town03_tiny_route12_w0_ClearNoon/sequence_a"
        second = "town03/town03_tiny_route12_w6_MidRainyNoon/sequence_b"
        visible = {
            "0000": {"front": 3},
            "0001": {"right": 4},
        }
        with tempfile.TemporaryDirectory() as root:
            self._write_sequence(root, first, visible, {"0001"})
            self._write_sequence(root, second, visible, {"0001"})
            inventory = self._write_inventory(root, [first, second])
            report = audit_pedestrian_hazard_visibility(
                root,
                inventory,
                cameras=("front", "right"),
                minimum_pixels=2,
            )

        self.assertEqual(report["summary"]["sequence_count"], 2)
        self.assertEqual(report["summary"]["town_route_group_count"], 1)
        self.assertEqual(
            report["summary"]["trajectory_visibility_signature_count"], 1
        )
        self.assertEqual(report["signature_groups"][0]["member_count"], 2)
        self.assertEqual(
            report["signature_groups"][0]["representative_sequence"], first
        )

    def test_distinguishes_visible_pedestrian_without_collision_hazard(self):
        relative = "town05/town05_short_route24_w0_ClearNoon/sequence_a"
        visible = {"0000": {"front": 4}, "0001": {"front": 2}}
        with tempfile.TemporaryDirectory() as root:
            self._write_sequence(root, relative, visible, set())
            inventory = self._write_inventory(root, [relative])
            report = audit_pedestrian_hazard_visibility(
                root,
                inventory,
                cameras=("front", "right"),
                minimum_pixels=2,
            )
            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        root,
                        "--inventory",
                        str(inventory),
                        "--cameras",
                        "front,right",
                        "--minimum-pixels",
                        "2",
                        "--require-all-hazard",
                    ]
                )

        summary = report["summary"]
        self.assertTrue(report["valid"])
        self.assertFalse(summary["all_sequences_have_privileged_collision_hazard"])
        self.assertEqual(summary["sequences_with_privileged_collision_hazard"], 0)
        self.assertEqual(exit_code, 2)

    def test_missing_hazard_field_is_structural_failure(self):
        relative = "town04/town04_tiny_route39_w0_ClearNoon/sequence_a"
        visible = {"0000": {"front": 4}}
        with tempfile.TemporaryDirectory() as root:
            self._write_sequence(
                root,
                relative,
                visible,
                set(),
                include_hazard_field=False,
            )
            inventory = self._write_inventory(root, [relative])
            report = audit_pedestrian_hazard_visibility(
                root,
                inventory,
                cameras=("front", "right"),
                minimum_pixels=2,
            )

        self.assertFalse(report["valid"])
        self.assertTrue(
            any("missing is_pedestrian_present" in item for item in report["errors"])
        )

    def test_rejects_inventory_escape(self):
        with tempfile.TemporaryDirectory() as root:
            inventory = self._write_inventory(root, ["../escape"])
            with self.assertRaisesRegex(
                PedestrianVisibilityAuditError, "escapes dataset root"
            ):
                audit_pedestrian_hazard_visibility(root, inventory)


if __name__ == "__main__":
    unittest.main()
