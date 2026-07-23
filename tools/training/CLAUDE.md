# tools/training/
> L2 | 父级: ../CLAUDE.md

## 成员清单

CLAUDE.md: M2 训练工具模块地图，约束数据、模型、运行编排和迁移导出共享同一版本化契约。
semantic_pretraining.py: 提供 split 驱动的 RGB/语义数据集、CARLA 标签映射、同构 ResNet50d-FPN、strict-deterministic 交叉熵、混淆矩阵指标和骨干迁移校验。
run_semantic_pretraining.py: 消费 M2 配置执行单机训练/验证，守卫 GPU 与 Git provenance，并原子固化 checkpoint、骨干导出和 run manifest。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
