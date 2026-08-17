<!--
[INPUT]: 依赖 v5 冻结BN零微调门禁结果（非行人 occupied IoU −3.00577% 超限 0.00577pp）、scene-split v2 validation index、interfuser_offline_metrics 的 per-sample score 累积与 _threshold_curve 精确阈值搜索。
[OUTPUT]: 对外提供"排序改善、阈值偏移"假说的可证伪诊断设计与证据记录格式；不产出任何门禁放行。
[POS]: docs 的 M2 H1 负迁移修复第六阶段协议；纯测量诊断，不改变 v5 失败判定、不改变任何模型权重、不与 Stage 2 准入挂钩。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
-->

# InterFuser 视觉负迁移修复协议 v6：概率校准诊断（纯诊断版）

| 字段 | 内容 |
| --- | --- |
| 状态 | **DIAGNOSTIC-FROZEN：钧舰审阅通过后冻结，2026-08-17** |
| 上游协议 | `docs/interfuser_negative_transfer_repair_protocol_v5.md` |
| 待建配置 | `configs/thesis/interfuser_b0_anchored_frozen_bn_calibration_probe_v1.json` |

## 0. 性质声明（先读，优先级高于本文其余部分）

本协议是**纯测量诊断**。它：

- 不训练、不微调、不改变任何模型权重；
- **不推翻、不修订 v5 零微调门禁的失败判定**——非行人 occupied IoU 在固定 0.5 阈值下退化 3.00577% > 3.0% 的结论永久有效；
- 不向 Stage 2 输送任何准入结论；本诊断通过与否，Stage 2 都不因此启动；
- 唯一产出是一份回答"候选的交通占用排序能力是否改善、概率刻度是否偏移"的证据记录。

门禁规则（0.5 阈值、3%/3%/10%、行人核心 ≥3/5）一字不动。若未来要修订门禁本身，必须作为面向全部未来候选的独立协议修订另行冻结，且不得溯及 v5 结论。

## 1. 诊断问题（可证伪）

v5 结果中，非行人队列排序指标改善（AP +3.6034%、ROC-AUC +1.2999%）而固定 0.5 阈值下 occupied IoU 退化 3.00577%（绝对 0.12809 → 0.12424）。由此产生两个可证伪问题：

- **Q1（排序）**：候选的排序优势在 route-group 留出子集上是否稳定复现？
- **Q2（刻度）**：候选的最佳占用阈值是否系统性偏离 0.5？偏离是否足以解释 0.5 处的 IoU 退化？

若 Q1 为否（留出子集上 AP/AUC 优势消失），则"v5 的排序改善"本身不可靠，假说整体证伪；若 Q1 为是而 Q2 为否（最佳阈值紧贴 0.5），则"刻度偏移解释 IoU 退化"证伪。

## 2. 绑定对象（哈希锚定，禁止替换）

- B0 checkpoint：SHA-256 `15d70aa489f33b84b8cad0988aa0b8f1787ee626069d8c95c3e9e518e781a0ce`；
- v5 候选 checkpoint：SHA-256 `fc7a943f1b15f4173c79d3c090e64adfe7cd4161a1636471e90ff4d450ed8e2e`；
- 数据：scene-split v2 validation index，SHA-256 `b4b5a62ac57a310c0d53aceadba53059eefce4b149fd47c30ae754b0694403d2`（7,437 帧、8 个行人 route group、Town01/03/04/05、16 种冻结天气）；
- loader 契约与 split v2 实测一致：`lidar_y_axis_multiplier=+1`、`navigation_frame=carla0916_compass`、`missing_navigation_policy=drop`；
- test index 全程冻结，任何步骤不得读取。

## 3. 诊断流程

1. 每个模型在完整 validation 上只做**一次前向**，落盘全部 per-sample traffic scores 与 targets（含 cohort、route group 标记），文件记 SHA-256；此后全部离线分析，不再触碰模型；
2. 按 route group 做 calibration/evaluation 蛇形划分：`(town, route)` 字典序排序、种子 `20260817`；约束为两子集各覆盖 4 个 Town、行人组 4/4、帧数比落在 [0.75, 1.33]；不满足则以 20260818 起递增重试，首次满足即冻结，重试次数记入 manifest；
3. 每个模型在 calibration 子集上按 `_threshold_curve` 同口径精确搜索非行人 occupied IoU 最优阈值 `t*`；不做平滑、不做人工剔除；
4. 报告（全部为描述性证据，不构成任何门禁）：
   - calibration 与 evaluation 两子集上，0.5 与 `t*` 两个阈值下的非行人 occupied IoU、AP、AUC；
   - B0 与候选的对称 `t*` 及间距；
   - 完整 validation 上 0.5 与 `t*` 的同口径对照；
   - 结论只许表述为"校准偏移解释与现有证据一致/不一致"，禁止表述为"候选已通过保持门禁"。

## 4. 停止边界

- 禁止读取 test；禁止运行 route39 或 D7；禁止以本诊断结果为任何候选申请 Stage 2；
- 诊断完成后路线选择不变：修复路线已由 v5 判负，下一准入方向为 v7（B0 教师特征蒸馏）；v6 证据仅用于估计蒸馏必要性与撰写论文诚实分层证据。

## 5. 工程实现清单（改代码即走 GEB 回环）

1. `tools/evaluation/interfuser_offline_metrics.py`：累积器新增可选 per-sample score/target 落盘（默认关闭，由 v6 配置显式开启）；更新 L3 头部与所属 L2 成员清单；
2. 新增 `configs/thesis/interfuser_b0_anchored_frozen_bn_calibration_probe_v1.json`（status: frozen，绑定 §2 全部哈希与本协议文档路径）；
3. 阈值搜索复用 `binary_score_metrics` / `_threshold_curve`，不新写指标公式；
4. 产物：两份 score dump（含 SHA-256）、calibration manifest（种子、重试次数、划分、`t*`、全部对照数字）、实验记录 `docs/experiment_records/2026-08-XX-m2-calibration-diagnostic-v1.md`；
5. 新增 unittest：划分确定性（同种子字节一致）、约束检查、阈值搜索与既有 `binary_score_metrics` 在 0.5 处一致；全仓测试保持通过。
