"""Regression: timm helpers torch.load must accept argparse.Namespace.

formal InterFuser checkpoints store `args: argparse.Namespace`. Under
PyTorch 2.6+ torch.load defaults to weights_only=True, which rejects
argparse.Namespace and raises UnpicklingError. load_state_dict must load
these trusted project checkpoints without error.
"""
import argparse
import tempfile
import unittest
from pathlib import Path

import torch

from timm.models.helpers import load_state_dict


class TestLoadStateDictNamespace(unittest.TestCase):
    def test_loads_checkpoint_containing_argparse_namespace(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "ckpt.pth.tar"
            torch.save(
                {
                    "epoch": 24,
                    "arch": "interfuser_baseline_224",
                    "state_dict": {"layer.weight": torch.zeros(1)},
                    "args": argparse.Namespace(foo="bar"),
                },
                str(ckpt),
            )
            state = load_state_dict(str(ckpt))
        self.assertIn("layer.weight", state)
        self.assertEqual(tuple(state["layer.weight"].shape), (1,))


if __name__ == "__main__":
    unittest.main()
