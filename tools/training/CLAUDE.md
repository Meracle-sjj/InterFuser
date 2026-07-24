# tools/training/
> L2 | 父级: ../CLAUDE.md

## 成员清单

CLAUDE.md: M2 训练工具模块地图，约束数据、模型、运行编排和迁移导出共享同一版本化契约。
semantic_pretraining.py: 提供 split 驱动的 RGB/语义数据集、CARLA 标签映射、同构 ResNet50d-FPN、可选显式类别权重的 strict-deterministic 交叉熵、混淆矩阵指标和骨干迁移校验。
run_semantic_pretraining.py: 消费 M2 配置执行单机训练/验证，守卫 GPU 与 Git provenance，并原子固化 loss 配置、last/best checkpoint、最佳骨干导出和 run manifest。
summarize_semantic_learning_curve.py: 验证 pilot run 的完整预算矩阵、嵌套 train 样本、相同完整 validation、provenance 与产物哈希，并确定性归约学习曲线。
interfuser_visual_pair.py: 冻结 ImageNet B0 与交通域 V 的 RGB 骨干差异，生成非 RGB 张量逐值相同且可被 InterFuser strict load 的全模型初始 checkpoint 对。
run_interfuser_visual_pair.py: 串行编排 B0/V 下游训练，守卫 GPU/分布式端口/进程组，并严格归约训练参数、summary、checkpoint 和资源释放 manifest。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
