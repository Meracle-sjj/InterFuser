"""
[INPUT]: 依赖 interfuser_visual_swap_pair 的固定 M0 底座加载、RGB strict replacement 与状态哈希 API。
[OUTPUT]: 验证 M0-FT 原样继承底座、M0-V 仅改变 RGB alias、非 RGB 指纹一致，并拒绝错误底座元数据。
[POS]: tests 的固定底座视觉替换回归，防止重新随机初始化被误写为预训练视觉接入实验。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from tools.training import interfuser_visual_swap_pair as visual_swap


class _Embed(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone


class _FakeInterfuser(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_backbone = nn.Linear(2, 2)
        self.rgb_patch_embed = _Embed(self.rgb_backbone)
        self.other = nn.Linear(2, 1)


class InterfuserVisualSwapPairTests(unittest.TestCase):
    def test_prepares_fixed_base_pair_with_only_rgb_aliases_changed(self):
        with tempfile.TemporaryDirectory() as root_value:
            root = Path(root_value)
            base_model = _FakeInterfuser()
            with torch.no_grad():
                base_model.rgb_backbone.weight.copy_(
                    torch.tensor([[1.0, 2.0], [3.0, 4.0]])
                )
                base_model.rgb_backbone.bias.copy_(torch.tensor([0.5, -0.5]))
                base_model.other.weight.copy_(torch.tensor([[8.0, 9.0]]))
                base_model.other.bias.copy_(torch.tensor([7.0]))
            base_state = {
                key: value.detach().clone()
                for key, value in base_model.state_dict().items()
            }
            base_path = root / "m0.pth.tar"
            torch.save(
                {
                    "epoch": 26,
                    "arch": "interfuser_baseline",
                    "state_dict": base_state,
                },
                base_path,
            )

            visual_rgb = {
                "weight": torch.tensor([[4.0, 3.0], [2.0, 1.0]]),
                "bias": torch.tensor([-1.0, 1.0]),
            }
            visual_path = root / "visual.pth"
            torch.save(
                {
                    "format_version": 1,
                    "architecture": "resnet50d",
                    "source_training_config_sha256": "a" * 64,
                    "state_dict": {
                        f"backbone.{key}": value for key, value in visual_rgb.items()
                    },
                },
                visual_path,
            )
            contract = {
                "path": root / "config.json",
                "sha256": "b" * 64,
                "model": {"name": "interfuser_baseline"},
                "base_checkpoint": {
                    "architecture": "interfuser_baseline",
                    "epoch": 26,
                    "sha256": visual_swap.sha256_file(base_path),
                },
                "visual_rgb_initialization": {
                    "source": "traffic-semantic",
                    "checkpoint_sha256": visual_swap.sha256_file(visual_path),
                    "source_training_config_sha256": "a" * 64,
                },
                "pair_invariant": {
                    "expected_full_model_tensors": len(base_state),
                    "expected_unique_rgb_tensors": 2,
                    "expected_full_model_rgb_alias_tensors": 4,
                },
                "seed": 7,
                "require_clean_git": False,
                "result_root_path": root,
                "resolved": {
                    "base_checkpoint": base_path,
                    "visual_rgb_initialization": visual_path,
                },
            }
            with mock.patch.object(
                visual_swap, "load_visual_swap_contract", return_value=contract
            ), mock.patch.object(
                visual_swap, "_create_interfuser_model", side_effect=lambda _: _FakeInterfuser()
            ), mock.patch.object(
                visual_swap, "_git_output", side_effect=["", "deadbeef"]
            ), mock.patch.object(visual_swap, "REPO_ROOT", root):
                manifest = visual_swap.prepare_visual_swap_pair(
                    root / "config.json", "swap-test"
                )

            self.assertTrue(manifest["pipeline_valid"])
            self.assertTrue(manifest["invariants"]["base_state_preserved"])
            self.assertEqual(
                manifest["invariants"]["changed_full_model_alias_tensors"], 4
            )
            m0_ft = torch.load(
                manifest["variants"]["m0_ft"]["checkpoint"],
                map_location="cpu",
                weights_only=True,
            )["state_dict"]
            m0_v = torch.load(
                manifest["variants"]["m0_v"]["checkpoint"],
                map_location="cpu",
                weights_only=True,
            )["state_dict"]
            self.assertEqual(
                visual_swap.state_dict_sha256(m0_ft),
                visual_swap.state_dict_sha256(base_state),
            )
            non_rgb = lambda key: not key.startswith(visual_swap.RGB_PREFIXES)
            self.assertEqual(
                visual_swap.state_dict_sha256(m0_ft, include=non_rgb),
                visual_swap.state_dict_sha256(m0_v, include=non_rgb),
            )

    def test_rejects_base_checkpoint_epoch_drift(self):
        with tempfile.TemporaryDirectory() as root_value:
            root = Path(root_value)
            path = root / "m0.pth.tar"
            torch.save(
                {
                    "epoch": 25,
                    "arch": "interfuser_baseline",
                    "state_dict": {"weight": torch.zeros(1)},
                },
                path,
            )
            contract = {
                "base_checkpoint": {
                    "architecture": "interfuser_baseline",
                    "epoch": 26,
                }
            }
            with self.assertRaisesRegex(visual_swap.VisualSwapError, "epoch"):
                visual_swap._load_base_state(path, contract)

    def test_rejects_non_backbone_visual_export(self):
        with tempfile.TemporaryDirectory() as root_value:
            path = Path(root_value) / "bad.pth"
            torch.save(
                {
                    "format_version": 1,
                    "architecture": "resnet50d",
                    "source_training_config_sha256": "a" * 64,
                    "state_dict": {"head.weight": torch.zeros(1)},
                },
                path,
            )
            contract = {
                "visual_rgb_initialization": {
                    "source_training_config_sha256": "a" * 64
                }
            }
            with self.assertRaisesRegex(visual_swap.VisualSwapError, "non-backbone"):
                visual_swap._load_visual_rgb_state(path, contract)


if __name__ == "__main__":
    unittest.main()
