<!--
[INPUT]: 依赖 Stage 2 scene validation 的混合结论、B0/V best checkpoint、语义预训练数据/类别审计与迁移学习中的权重插值、源点正则化和 replay 原理。
[OUTPUT]: 对外提供 InterFuser 视觉负迁移的分阶段修复顺序、插值 probe 门禁及教师保持/L2-SP/普通场景 replay 的准入条件。
[POS]: docs 的 M2 H1 负迁移修复协议；先用零训练成本筛选稳定性-可塑性折中，再决定是否新增训练机制。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
-->

# InterFuser 视觉负迁移修复协议 v1

| 字段 | 内容 |
| --- | --- |
| 状态 | **PROBE-FROZEN：alpha 与停止条件在运行前固定** |
| 服务问题 | Stage 2 V 在行人条件局部获益，但整体/非行人/时序退化 |
| Probe 配置 | `configs/thesis/interfuser_weight_interpolation_probe_v1.json` |
| Probe runner | `tools/evaluation/run_interfuser_weight_interpolation_probe.py` |

## 1. 根因边界

当前 V 并非只用行人数据训练：基础语义集有7,590个普通三相机样本，行人危险扩充再增加4,317个样本。危险增强后的冻结语义 validation 相比未扩充 inverse-sqrt 模型，mIoU从0.46324升至0.48675，行人IoU从0.05667升至0.12047，同时道路、车辆、交通灯和交通标志也多数改善。

因此现有证据不支持“语义预训练阶段因普通路段不足而直接遗忘”。更可能的失效点是：用语义骨干整体替换作者发布 InterFuser 的驾驶骨干时，抛弃了后者已形成的控制相关表征；三轮端到端适配只恢复部分轨迹能力，未恢复普通交通语义与时序稳定性。

普通场景 replay 仍可作为后续手段，但 replay 的教师或正则化锚点必须是官方 B0 驾驶骨干，而不只是继续重复普通道路的语义分割标签。

## 2. 第一阶段：权重空间插值

先在 Stage 2 最终 B0/V checkpoint 之间对全部浮点张量做线性插值：`Wα=(1−α)WB0+αWV`，固定 `α={0.25,0.5,0.75}`。BatchNorm `num_batches_tracked` 等非浮点缓冲复制 B0；推理时不读取 test。

该 probe 不声称形成最终方法，只判断 B0 与 V 是否存在低损失连接和可用折中。若不存在合格 alpha，直接进行额外90分钟 Stage 2 重训没有依据。

每个 alpha 必须同时满足：

- 行人条件五项核心指标至少3项优于B0；
- 非行人五项中最坏退化不超过3%；
- 整体 traffic AP/AUC/IoU 最坏退化不超过3%；
- 行人 traffic/waypoint 连续帧残差最坏退化不超过10%。

只有四项门禁全部通过才准入完整重训。alpha 选择只使用 validation；test继续冻结。

## 3. 第二阶段：教师保持或 L2-SP

若插值存在合格解，则将该alpha转移到“官方 B0 RGB backbone → 语义适配 backbone”的训练起点，并在普通场景上加入以下一种保持约束：

1. 特征蒸馏：冻结官方B0作为教师，约束普通帧的多层RGB特征；
2. L2-SP：对RGB参数施加相对官方B0源点的二次惩罚；
3. 普通场景 replay：保持现有危险样本，同时按固定比例回放普通帧，并用教师特征而非单独语义标签保护驾驶表征。

首个训练 probe 只允许选择一种保持机制，禁止同时改变采样比例、损失和解冻策略。优先顺序为特征蒸馏/L2-SP，再考虑扩大普通 replay；原因是现有普通样本已占多数，单纯加帧不能约束模型保留官方驾驶特征。

## 4. 停止条件

- 所有alpha均未通过四项门禁：停止权重插值，不进入完整训练；
- alpha仅改善聚合行人指标但少于4/8 route group支持：记为路线偏置；
- 保持机制恢复普通场景却消除行人收益：不准入test；
- 任何方案读取test或根据test选择alpha/损失权重：结果永久无效。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
