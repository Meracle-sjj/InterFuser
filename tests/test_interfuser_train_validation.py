"""
[INPUT]: 依赖 interfuser.train 的 validation tensor 归约边界与 PyTorch tensor。
[OUTPUT]: 验证单 GPU validation 为全部 loss/accuracy 张量赋值，不在首个验证 batch 触发未绑定局部变量。
[POS]: tests 的 Stage 2 单卡训练回归；隔离验证归约逻辑，不加载 CARLA 数据或启动 GPU。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import unittest
from types import SimpleNamespace

import torch

from train import _reduce_validation_tensors


class InterfuserSingleGpuValidationTest(unittest.TestCase):
    def test_all_validation_tensors_are_returned_without_distributed_reduce(self):
        tensors = tuple(torch.tensor(float(index)) for index in range(10))
        args = SimpleNamespace(distributed=False, world_size=1)

        reduced = _reduce_validation_tensors(tensors, args)

        self.assertEqual(len(reduced), len(tensors))
        for expected, actual in zip(tensors, reduced):
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
