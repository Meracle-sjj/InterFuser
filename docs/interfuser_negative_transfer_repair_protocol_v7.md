<!--
[INPUT]: 依赖 v5 冻结BN候选（单一门禁 3.00577% 超限）、L2-SP v4 失败证据（参数惩罚管不住 BN 缓冲）、官方 B0 checkpoint 与 v5 语义训练配方。
[OUTPUT]: 对外提供 B0 教师特征蒸馏的单变量语义 probe、语义/零微调门禁复用口径和停止边界。
[POS]: docs 的 M2 H1 负迁移修复第七阶段协议；在 v5 冻结BN之上只加特征蒸馏一个变量，让保持约束作用于参数与 BN 统计共同决定的特征层。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
-->

# InterFuser 视觉负迁移修复协议 v7：B0 教师特征蒸馏

| 字段 | 内容 |
| --- | --- |
| 状态 | **PROBE-FROZEN：钧舰审阅通过后冻结，2026-08-17** |
| 上游协议 | `docs/interfuser_negative_transfer_repair_protocol_v5.md`（唯一变更基准）、`docs/interfuser_negative_transfer_repair_protocol_v3.md` §4（蒸馏准入出处） |
| 待建配置 | `configs/thesis/semantic_pretraining_b0_anchored_distill_probe_v1.json` |

## 1. 定位与上游结论

- 教师 = **官方 B0 完整 checkpoint**（与 v3/v5 初始化同一来源；路径与 SHA-256 在 v7 配置冻结时绑定），仅训练期冻结前向，不进最终系统；
- v5 已证明：冻结 BN 统计把灾难性普通场景退化收敛为单一固定阈值门禁的边界性超限（3.00577%，超 0.00577pp）；
- v4 已证明：L2-SP 只约束可训练参数（漂移 4.89），管不住 BN 缓冲（漂移 24,579.38）；
- 因此保持约束必须作用于**参数与 BN 统计共同决定的特征层**。特征蒸馏是 v3 §4 预留的最后一道保持机制，也是本路线最后一次修复尝试。

## 2. 唯一变化（相对 v5 的单变量）

完全继承 v5：B0 `rgb_backbone.*` 330 张量初始化、普通 7,590 + 危险 4,317 数据与类别权重、头部 warmup 一轮、`1e-5/1e-4` 分层学习率、五轮预算、best-mIoU 选择、RGB 骨干全部 BN 保持 eval 模式。不使用 L2-SP。

唯一新增：训练总损失加一项对冻结 B0 的特征蒸馏损失。

- 教师与学生同批输入；教师 `eval()`、无梯度、仅前向；
- 蒸馏位置：ResNet-50d 各 stage 输出特征图（学生与教师张量同构、形状一致，**不加任何对齐/适配层**；具体张量名以实现时核对的模型结构为准，写入 manifest）；
- 损失形式：逐 stage 归一化 L2，`L_distill = Σ_l ||f_s^l − f_t^l||² / ||f_t^l||²`，总损失 `L = L_semantic + L_distill`；
- **λ 固定为 1.0**：归一化已消除特征尺度差异，不引入可调系数——v5 已冻结"不扫描正则系数"的纪律，本协议不设任何待扫超参。

## 3. 门禁（与 v5 逐字一致，不修订）

- **语义准入**：validation mIoU ≥ 0.46324；行人 IoU ≥ 0.08；traffic-light IoU ≥ 0.15；严格导出 330 个 ResNet50d 张量。任一失败，不生成官方 swap checkpoint；
- **零微调 scene 保持门禁**（固定 0.5 阈值，复用 v5 配置与工具链）：行人核心改善 ≥ 3/5；非行人核心最坏退化 ≤ 3%；整体 traffic 最坏退化 ≤ 3%；行人时序最坏退化 ≤ 10%；
- 两步全过 → 按 scene-split v2 准入三轮配对 Stage 2；
- 任一失败 → **视觉预训练修复路线关闭**，不扫描蒸馏系数、不改门禁、不换阈值；结果作为 H1 的诚实负证据进入论文；v6 校准诊断证据仅用于分层讨论。

## 4. 工程与运行

- 训练入口：`tools/training/run_semantic_pretraining.py` 新增可选蒸馏开关（默认关闭）；教师加载、特征抽取钩子、归一化 L2 与 v7 配置绑定；
- 运行时契约沿用 v3 §2：GPU1 共享容量语义、启动时 ≥ 20,000 MiB 空闲、不清理外部低占用 context；`require_clean_git`；
- 新增 unittest：教师全程无梯度且 eval 模式、蒸馏损失形状与归一化、蒸馏开关关闭时与 v5 配方字节级等价；全仓测试保持通过；
- 产物：训练 manifest（含教师 checkpoint SHA-256、逐轮语义指标、蒸馏损失曲线）、语义 best checkpoint、swap-init manifest、零微调 scene manifest、实验记录 `docs/experiment_records/2026-08-XX-m2-distill-probe-v1.md`；
- 全部改代码点执行 GEB 回环：L3 头部 → L2 成员清单 → L1 检查。

## 5. 与 v6 的关系

v6（纯诊断）与 v7 互相独立、可并行：v6 回答"刻度是否偏移"，v7 尝试"修模型本身"。v7 不以 v6 的任何结论为前置；即使 v6 证明刻度偏移存在，v7 仍必须在固定 0.5 门禁下真过线才准入 Stage 2。
