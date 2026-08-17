<!--
[INPUT]: 依赖 v6 纯校准诊断协议、校准 probe 配置 d97d3ad、双模型 per-sample 分数落盘与离线阈值对照 manifest。
[OUTPUT]: 对外提供 Q1 排序改善复现与 Q2 阈值偏移的诊断结论、全部对照数字与停止边界。
[POS]: docs/experiment_records 的 M2 H1 校准诊断证据锚点；只陈述测量事实，不修订 v5 门禁失败判定、不产出 Stage 2 准入。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
-->

# M2 校准诊断 v1 记录（纯测量，无门禁）

## 1. Provenance

| 字段 | 值 |
| --- | --- |
| 协议 | `docs/interfuser_negative_transfer_repair_protocol_v6.md`（DIAGNOSTIC-FROZEN） |
| 评测代码提交 | `d97d3ad` |
| Run ID | `m2-interfuser-b0-anchored-frozen-bn-calibration-probe-v1-20260817-v1` |
| probe manifest SHA-256 | `3f062555e175e0493ec9f54a8bfe9c51952007b17b699d353c367a454693caa8` |
| calibration manifest SHA-256 | `6828cea52e191419452b0835811f216e882388f62b2d424ecf30fb4b2dcc34a3` |
| B0 scores / candidate scores | `88e6a1c6…432289` / `e57811ad…09ecfd`（400×7,437 float32） |
| targets（两模型逐字节相同） | `9802f564…dc758`；meta（两模型逐字节相同） `e702be3a…354407` |

双模型 targets 与 meta 哈希逐字节一致，从文件级证明两次前向的样本顺序与真值完全对齐。GPU1 监测与前向复用 v5 零微调链路；probe 内置门禁输出与 v5 原结果逐位一致（非行人退化 3.0057684425328106%），确认落盘路径未改变任何评测数值。

## 2. 划分

validation 共 32 个 route group。蛇形划分在种子 `20260817+18=20260835` 首次满足全部冻结约束（重试 18 次，已记入 manifest）：calibration/evaluation 各 16 组，各覆盖 Town01/03/04/05，行人组 4/4，帧数比在 [0.75, 1.33] 内。

## 3. 结果（非行人队列）

| 模型 | t* | 子集 | IoU@0.5 | IoU@t* | AP | AUC |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| B0 | 0.411355 | calibration | 0.110316 | 0.128068 | 0.126672 | 0.689027 |
| B0 | | evaluation | 0.145751 | 0.198100 | 0.210353 | 0.752116 |
| B0 | | full validation | 0.128088 | 0.161070 | 0.164874 | 0.722974 |
| 候选 | 0.388757 | calibration | 0.107578 | 0.134935 | 0.131492 | 0.698212 |
| 候选 | | evaluation | 0.140751 | 0.209769 | 0.217606 | 0.761581 |
| 候选 | | full validation | 0.124238 | 0.170046 | 0.170815 | 0.732372 |

full validation 的 IoU@0.5 与 v5 门禁原始数字（0.12809/0.12424）一致。留出 evaluation 子集差值（候选−B0）：IoU@0.5 **−0.00500**，IoU@t* **+0.01167**，AP **+0.00725**，AUC **+0.00947**。

## 4. 诊断结论

- **Q1（排序改善复现）：成立**。AP/AUC 优势在未参与阈值选择的留出子集上稳定复现；
- **Q2（阈值偏移）：成立**。候选最优阈值 0.3888 低于 B0 的 0.4114（间距 −0.0226），且各自校准后 IoU 差值由 −0.005 翻正为 +0.0117；0.5 处的 IoU 退化可被刻度偏移解释；
- 附注：B0 自身 t* 也偏离 0.5，说明 0.5 是两模型共同的固定测量点而非任一模型的最优点，v5 在 0.5 处的对比口径本身公平。

**结论表述止于"校准偏移解释与现有证据一致"。** v5 保持门禁失败判定不变，本诊断不产生 Stage 2 准入；修复路线按 v7（B0 教师特征蒸馏）执行。

## 5. 实现偏差说明

协议 §5.1 原计划由指标累积器落盘分数；实际实现位于 runner 层（`_evaluate_variant` 可选落盘 + `_write_score_dump`），因为逐样本的 cohort/route-group 元数据只存在于 runner 循环，累积器无此上下文。落盘默认关闭，未改变任何既有评测行为（全仓 286/286 通过，其中本诊断新增 11 项）。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
