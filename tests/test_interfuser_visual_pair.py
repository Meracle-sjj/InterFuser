"""
[INPUT]: 依赖 interfuser_visual_pair 的状态哈希、ImageNet/V 骨干映射与全模型初始 checkpoint 生成 API。
[OUTPUT]: 验证 B0/V checkpoint 严格可加载、唯一变化为 RGB 共享骨干、非 RGB 张量指纹相同，并拒绝越界导出键。
[POS]: tests 的 M2→InterFuser 权重迁移回归，阻止模型结构或非视觉初始化暗中漂移进入 H1 对照。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from tools.training import interfuser_visual_pair as visual_pair


class _Embed(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone


class _FakeInterfuser(nn.Module):
    def __init__(self, rgb_state):
        super().__init__()
        self.rgb_backbone = nn.Linear(2, 2)
        self.rgb_backbone.load_state_dict(rgb_state, strict=True)
        self.rgb_patch_embed = _Embed(self.rgb_backbone)
        self.other = nn.Linear(2, 1)


class InterfuserVisualPairTests(unittest.TestCase):
    def test_state_dict_hash_is_independent_of_key_order(self):
        first = {"a": torch.tensor([1.0]), "b": torch.tensor([2])}
        second = {"b": torch.tensor([2]), "a": torch.tensor([1.0])}
        self.assertEqual(
            visual_pair.state_dict_sha256(first),
            visual_pair.state_dict_sha256(second),
        )

    def test_prepares_strict_pair_with_only_rgb_aliases_changed(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            b0_rgb = {
                "weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                "bias": torch.tensor([0.5, -0.5]),
            }
            v_rgb = {
                "weight": torch.tensor([[4.0, 3.0], [2.0, 1.0]]),
                "bias": torch.tensor([-1.0, 1.0]),
            }
            b0_source = root / "imagenet.pth"
            torch.save(
                {
                    **b0_rgb,
                    "fc.weight": torch.zeros(1),
                    "fc.bias": torch.zeros(1),
                },
                b0_source,
            )
            v_source = root / "traffic.pth"
            torch.save(
                {
                    "format_version": 1,
                    "architecture": "resnet50d",
                    "source_training_config_sha256": "a" * 64,
                    "state_dict": {f"backbone.{key}": value for key, value in v_rgb.items()},
                },
                v_source,
            )
            contract = {
                "path": root / "config.json",
                "sha256": "b" * 64,
                "model": {"name": "interfuser_baseline"},
                "b0_rgb_initialization": {
                    "source": "imagenet",
                    "checkpoint_sha256": visual_pair.sha256_file(b0_source),
                },
                "v_rgb_initialization": {
                    "source": "traffic",
                    "checkpoint_sha256": visual_pair.sha256_file(v_source),
                    "source_training_config_sha256": "a" * 64,
                },
                "pair_invariant": {
                    "expected_unique_rgb_tensors": 2,
                    "expected_full_model_rgb_alias_tensors": 4,
                },
                "seed": 7,
                "require_clean_git": False,
                "result_root_path": root,
                "resolved": {
                    "b0_rgb_initialization": b0_source,
                    "v_rgb_initialization": v_source,
                },
            }
            factory = lambda _: _FakeInterfuser(b0_rgb)
            with mock.patch.object(
                visual_pair, "load_visual_pair_contract", return_value=contract
            ), mock.patch.object(
                visual_pair, "_create_interfuser_model", side_effect=factory
            ), mock.patch.object(
                visual_pair, "_git_output", side_effect=["", "deadbeef"]
            ), mock.patch.object(visual_pair, "REPO_ROOT", root):
                manifest = visual_pair.prepare_visual_initialization_pair(
                    root / "config.json", "pair-test"
                )

            self.assertTrue(manifest["pipeline_valid"])
            self.assertEqual(manifest["invariants"]["unique_rgb_tensors"], 2)
            self.assertEqual(
                manifest["invariants"]["changed_full_model_alias_tensors"], 4
            )
            self.assertNotEqual(
                manifest["variants"]["b0"]["rgb_state_sha256"],
                manifest["variants"]["v"]["rgb_state_sha256"],
            )
            self.assertEqual(
                torch.load(
                    manifest["variants"]["v"]["checkpoint"],
                    map_location="cpu",
                    weights_only=True,
                )["variant"],
                "v",
            )

    def test_rejects_non_backbone_key_in_v_export(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "bad.pth"
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
                "v_rgb_initialization": {"source_training_config_sha256": "a" * 64}
            }
            with self.assertRaisesRegex(visual_pair.VisualPairError, "non-backbone"):
                visual_pair._load_v_rgb_state(path, contract)


if __name__ == "__main__":
    unittest.main()
