<!--
[INPUT]: 依赖全模型/RGB-only插值失败证据、官方B0 RGB checkpoint、冻结普通+行人危险语义数据、inverse-sqrt类别权重与分阶段优化实现。
[OUTPUT]: 对外提供从B0驾驶骨干出发的头部warmup、低学习率语义适配probe、语义准入阈值和接回InterFuser前的停止条件。
[POS]: docs 的 M2 H1 负迁移修复第三阶段协议；让交通语义在原驾驶表征上生长，替代彼此独立训练后再做参数混合。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
-->

# InterFuser 视觉负迁移修复协议 v3：B0 锚定语义适配

| 字段 | 内容 |
| --- | --- |
| 状态 | **PROBE-FROZEN：两轮插值失败后冻结** |
| 配置 | `configs/thesis/semantic_pretraining_b0_anchored_staged_probe_v1.json` |
| 训练入口 | `tools/training/run_semantic_pretraining.py` |

## 1. 上游结论

最终B0/V的全模型和RGB-only插值均无合格alpha。RGB-only即使完整保留B0 Transformer、LiDAR和任务头，alpha=0.25的非行人最坏退化仍为191.08%；说明独立训练后的两套RGB参数存在强表征基不对齐，不能通过权重平均修复。

当前语义数据已包含7,590个基础普通样本和4,317个危险扩充样本；危险增强模型在原语义validation上还提升了多数普通类别。因此v3不继续盲目追加普通帧，而是把官方B0 RGB骨干设为源点，使语义能力在驾驶表征上增量生长。

## 2. 唯一训练方案

- 初始化：作者发布B0完整checkpoint中的 `rgb_backbone.*` 330个张量；
- 数据、类别权重、分辨率、五轮预算与原行人危险语义probe完全相同；
- epoch 1：冻结RGB骨干及其BatchNorm统计，只训练一次性FPN语义头；
- epoch 2–5：解冻骨干，头部学习率 `1e-4`，骨干学习率 `1e-5`；
- best checkpoint仍只由冻结语义validation mIoU选择。

该probe同时改变了初始化源点和适配日程，现阶段只判断可行性。若成功，后续必须以消融区分B0初始化与warmup/低学习率的贡献。

## 3. 语义准入

接回InterFuser前必须满足：validation mIoU不低于基础inverse-sqrt模型的0.46324；行人IoU不低于0.08；traffic-light IoU不低于0.15；模型严格导出330个ResNet50d张量。任一条件失败，不生成新的官方swap checkpoint。

语义准入只证明“B0驾驶骨干可以吸收交通语义”，不能证明普通驾驶能力已保留。通过后只准进行新骨干的零微调scene validation；若非行人最坏退化仍超过原V或行人收益消失，则停止，不进入Stage 2。

## 4. 后续保持机制

若v3语义通过但接回InterFuser仍退化，下一步才准入相对B0源点的L2-SP或B0教师特征蒸馏。普通场景replay必须与教师保持配合；重复相同语义标签不能单独证明保留控制相关特征。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
