<!--
[INPUT]: 依赖B0锚定v3近门槛结果、L2-SP v4训练归约、候选/B0逐张量漂移分解与BatchNorm缓冲语义。
[OUTPUT]: 对外提供冻结B0 BatchNorm running statistics的单变量语义probe、语义/回接门禁和停止边界。
[POS]: docs 的 M2 H1 负迁移修复第五阶段协议；根据状态漂移证据保护驾驶域归一化统计，而非继续增加普通样本或正则系数。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
-->

# InterFuser 视觉负迁移修复协议 v5：冻结 B0 BatchNorm 统计

| 字段 | 内容 |
| --- | --- |
| 状态 | **PROBE-FROZEN：状态漂移分解后冻结** |
| 配置 | `configs/thesis/semantic_pretraining_b0_anchored_frozen_bn_probe_v3.json` |

## 1. 新根因证据

B0锚定v3候选相对官方B0的浮点状态漂移平方和为24,591.95，其中可训练参数仅4.90，BatchNorm running mean/variance占24,587.05。L2-SP v4只约束参数：训练末期未加权参数惩罚9.80、加权项约`9.8e-5`，无法保护BN缓冲；其best mIoU 0.47267与v3 0.47315基本相同。

因此普通场景退化更可能由小规模、低分辨率语义数据重估了官方驾驶域BN统计，而不是卷积参数发生大幅灾难性遗忘。继续提高L2-SP或添加普通帧都不直接约束该缓冲状态。

## 2. 唯一变化

v5继承v3的B0初始化、普通+危险数据、类别权重、头部warmup、`1e-5/1e-4`分层学习率、五轮预算和best-mIoU选择；不使用L2-SP。唯一变化是训练全程将RGB骨干全部BatchNorm模块保持eval模式，冻结running mean/variance与num_batches_tracked，同时允许epoch2–5的BN affine参数和卷积参数正常求梯度。

语义门禁与零微调scene保持门禁完全复用v3/v4。通过后才准入Stage 2；失败则停止该视觉预训练路线，不再扫描BN动量或正则系数。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
