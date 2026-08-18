<!--
[INPUT]: 依赖 v9 协议（FROZEN）、KEYreferences.md 文献结论、阶段一预训练 run、阶段二 Stage 2 v9 配对 run
  （only_initial_checkpoint_differs=true）与 scene validation v9 对照 manifest。
[OUTPUT]: 对外提供 v9 两阶段完整证据链：预训练曲线、配对可比性、四判据宣判（全过）、
  跨会话噪声带校准与诚实边界；test/D7 解冻准入成立。
[POS]: docs/experiment_records 的 M2 语义注入方式研究的正向终点锚点；与 v1-v8b 负证据链构成完整故事。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
-->

# M2 v9 架构内辅助语义预训练记录（四判据全过）

## 1. Provenance

| 字段 | 值 |
| --- | --- |
| 协议 | `docs/interfuser_architecture_aware_semantic_pretraining_protocol_v9.md`（FROZEN 2026-08-18，Owner 批准直接执行） |
| 阶段一 Run | `m2-v9-architecture-semantic-pretraining-v1-seed20260723-20260818-v1`（checkpoint_best SHA `2faf6068…`，best epoch 1） |
| 阶段二 Run | `m2-interfuser-official-stage2-v9-pair-seed20260814-20260818-v1`（pipeline_valid=true；`only_initial_checkpoint_differs=true`，args SHA `c203b637…` 两臂一致） |
| 场景评估 Run | `m2-interfuser-stage2-scene-validation-v9-20260818-v1`（pipeline_valid=true，errors 空，`test_accessed=false`） |
| 判定基准 | scene validation v2 冻结管线（7,437 帧 validation-only，行人 1,131 帧、8 route group） |

## 2. 阶段一：架构内预训练（官方 B0 全模型可训）

| epoch | train driving | train semantic | val driving | val mIoU |
| --- | ---: | ---: | ---: | ---: |
| 1 (**best**) | 0.115509 | 0.637072 | **0.123493** | 0.291147 |
| 2 | 0.073607 | 0.441861 | 0.124438 | 0.320661 |
| 4 | 0.058613 | 0.332025 | 0.162430 | 0.327070 |
| 5 | 0.054860 | 0.306571 | 0.142909 | 0.341491 |

选择准则（冻结）：validation 驾驶损失最低 → epoch 1。语义注入相对薄（后续 epoch 驾驶验证损失过拟合），主战场在阶段二终态。驾驶损失权重逐字移植 `interfuser/train.py`（0.5/0.2/0.05/0.05/0.1/0.01），辅助语义系数 1.0 固定。

## 3. 阶段二：标准适配对照（同会话配对）

- B0 臂：官方 checkpoint 直接适配；v9 臂：v9 预训练 checkpoint 适配；其余逐位相同（165,264 帧、三轮、bs32、lr 5e-4/骨干 2e-4、cosine、GPU1 共享容量）；
- B0 best epoch 2（best_metric 0.26707）；v9 best 见 pair manifest；
- **跨会话噪声带**：今日 B0 vs 08-17 v2 B0 在行人队列的漂移为 AP −0.39%、IoU +3.57%、ADE −3.97%、FDE −3.26%——作者级训练器的 run-to-run 方差约 ±4%，v9 的行人收益（+39%/+48%）是噪声带一个数量级之上。

## 4. 预注册四判据（全过）

| 判据 | 门槛 | 实测 | 判定 |
| --- | --- | --- | --- |
| 行人条件五项改善 | ≥ 3/5 | **5/5**（AP +39.33%、AUC +1.07%、IoU +48.16%、ADE −8.18%、FDE −9.54%） | ✓ |
| 非行人核心最坏退化 | ≤ 3% | 最差 junction F1 **−0.112%**（其余 6 项全改善：AP +0.60%、IoU +1.06%、红灯 +0.31%、ADE −12.20%、FDE −15.61%） | ✓ |
| 整体 traffic 最坏退化 | ≤ 3% | 最差 junction **−0.014%**（整体 9 项 8 改善：AP +3.25%、IoU +5.30%、ADE −11.15%、FDE −13.25%、时序双改善） | ✓ |
| 行人时序最坏退化 | ≤ 10% | traffic delta MAE **−4.75%**、waypoint delta ADE **−14.07%**（双双改善） | ✓ |

行人队列全部 9 项指标改善（另含 junction F1 +5.24%、红灯 F1 +4.99%）。工具 gate `pedestrian_core_improved=5`、`pedestrian_majority_supports_v=true`。

## 5. 结论与解冻

**v9 成立：架构内辅助语义预训练 + 标准适配，在全部四项预注册判据下优于同会话 B0 对照。** 按协议 §3：准入冻结 test 解封与 D7 闭环（`test_remains_frozen` 由评估工具标记，解冻动作需下一步显式执行）。

与 v1-v8b 的合读：语义监督的**注入位置**是决定性变量——外部头独立预训练 + zero-touch 替换（v1-v8b，五类修复假设全部失败）vs 架构内联合注入（v9，全过）。文献主线（TransFuser/LAV/ViDAR/AD-PT）与本项目实测自此互相印证。

## 6. 边界与诚实标注

- `test_accessed=false` 维持；D7 未跑；闭环 Driving Score 改善尚未测量——论文可写的当前上限是"分层 validation 全面优于配对对照"；
- 行人 route group 存在异质性（如 Town01:route004 AP +447% 而 ADE +7.6% 劣化），聚合判据成立但逐路线叙述需谨慎；
- 行人队列 junction macro-F1 达 1.00000（b0 为 0.95025）：小样本下完全分类可能，非 gate 项，记录备查；
- 阶段一 best=epoch 1（语义注入薄）仍取得全过——辅助信号的作用下界估计，加强注入（更多 epoch/更强系数）属未来工作，不在本协议内扫描；
- 今日 B0 臂与 08-17 v2 B0 臂的 ±4% 漂移为所有同类对比提供了噪声带标定。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
