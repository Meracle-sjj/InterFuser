# interfuser/timm/
> L2 | 父级: ../CLAUDE.md

## 成员清单

CLAUDE.md: 内置 timm 运行时地图，标记 InterFuser 扩展面与上游通用实现的边界。
__init__.py: 暴露 timm 版本与顶层 API。
version.py: 冻结内置 timm 版本号。
data/: 数据集、数据加载、增强与 CARLA 多传感器适配。
models/: 通用视觉模型与 InterFuser 多模态模型定义。
optim/: 优化器工厂与参数组策略。
scheduler/: 学习率调度器。
utils/: 分布式、checkpoint、日志、指标与随机种工具。
loss/: 分类与蒸馏损失。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
