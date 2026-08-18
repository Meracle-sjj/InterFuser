"""
 * [INPUT]: 依赖 tools/training/architecture_pretraining 的损失移植与契约校验。
 * [OUTPUT]: 对外提供 v9 损失数学与契约守护测试：WaypointL1 无效掩码与衰减权重、
 *   MVTL1 平衡项/几何项/速度项与空正例防护、组合权重、driving_loss_weights 契约。
 * [POS]: tests 的 v9 守护测试；不加载真实 checkpoint。
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
INTERFUSER_ROOT = REPO_ROOT / "interfuser"
for import_root in (REPO_ROOT, INTERFUSER_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tools.training.architecture_pretraining import (  # noqa: E402
    DRIVING_LOSS_WEIGHTS,
    MVTL1Loss,
    WaypointL1Loss,
    combined_driving_loss,
    load_architecture_contract,
)


class WaypointL1LossTests(unittest.TestCase):
    def test_invalid_targets_masked(self):
        output = torch.zeros(1, 10, 2)
        target = torch.zeros(1, 10, 2)
        target[0, 5, 0] = 10000.0
        loss = WaypointL1Loss()(output.clone(), target)
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_decay_weights_favor_near_horizon(self):
        output = torch.full((1, 10, 2), 3.0)
        target = torch.zeros(1, 10, 2)
        loss = WaypointL1Loss()(output.clone(), target)
        # 作者实现对加权后的逐步损失取 mean 而非 sum：3.0 * mean(weights)
        expected = 3.0 * sum(WaypointL1Loss.WEIGHTS) / len(WaypointL1Loss.WEIGHTS)
        self.assertAlmostEqual(float(loss), expected, places=5)
        self.assertAlmostEqual(sum(WaypointL1Loss.WEIGHTS), 1.0, places=6)


class MVTL1LossTests(unittest.TestCase):
    def test_all_empty_positive_guard(self):
        output = torch.zeros(2, 400, 7)
        target = torch.zeros(2, 400, 7)
        combined, speed = MVTL1Loss(1.0)(output, target)
        self.assertTrue(torch.isfinite(combined))
        self.assertTrue(torch.isfinite(speed))
        self.assertAlmostEqual(float(combined), 0.0, places=6)

    def test_occupancy_disagreement_raises_loss(self):
        target = torch.zeros(1, 400, 7)
        target[0, 0, 0] = 1.0
        output = torch.zeros(1, 400, 7)
        combined, _ = MVTL1Loss(1.0)(output, target)
        self.assertGreater(float(combined), 0.0)


class CombinedLossTests(unittest.TestCase):
    def test_weight_sum_matches_author_recipe(self):
        self.assertAlmostEqual(sum(DRIVING_LOSS_WEIGHTS.values()), 0.91, places=6)

    def test_zero_outputs_zero_targets_near_zero(self):
        outputs = (
            torch.zeros(1, 400, 7),
            torch.zeros(1, 10, 2),
            torch.zeros(1, 2),
            torch.zeros(1, 3),
            torch.zeros(1, 6),
        )
        targets = (
            None,
            torch.zeros(1, 10, 2),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, 400, 7),
            None,
            torch.zeros(1, dtype=torch.long),
        )
        total, parts = combined_driving_loss(outputs, targets)
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(all(torch.isfinite(value) for value in parts.values()))


class ContractTests(unittest.TestCase):
    def test_driving_weights_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            payload = {
                "initial_checkpoint": "nonexistent.pth",
                "training": {"driving_loss_weights": {"traffic": 0.9}},
            }
            path.write_text(json.dumps(payload))
            with self.assertRaises(Exception):
                load_architecture_contract(path)


if __name__ == "__main__":
    unittest.main()
