<!--
[INPUT]: 依赖 v7 B0 教师特征蒸馏协议、训练实现提交 8437012、swap-init 提交 2323c72、零微调 scene 门禁提交 1046180，及三个 run 目录下的 run/initialization/probe/calibration manifest。
[OUTPUT]: 对外提供 v7 蒸馏探针的完整证据链：语义门禁 4/4 通过、零微调 scene 门禁保持三项全过但行人核心 2/5 失败的逐指标明细、蒸馏后阈值漂移消隐的校准对照，以及按 v7 协议 §3 关闭视觉预训练修复路线的裁决。
[POS]: docs/experiment_records 的 M2 H1 终止性证据锚点；只陈述测量事实与协议裁决，不启动 Stage 2、不访问 test、不再扫描任何修复系数。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
-->

# M2 v7 B0 教师特征蒸馏探针记录（路线关闭裁决）

## 1. Provenance

| 字段 | 值 |
| --- | --- |
| 协议 | `docs/interfuser_negative_transfer_repair_protocol_v7.md`（§3 失败即关闭路线，已与 Owner 冻结确认） |
| 训练实现提交 | `843701296b4577d9589e2bbcfe331565ca387e9c`（B0 教师特征蒸馏） |
| swap-init 提交 | `2323c72` |
| 零微调门禁提交 | `1046180bfaafec677c17e312da33672456517e08` |
| 语义训练 Run ID | `m2-semantic-b0-anchored-distill-probe-v1-seed20260723-20260817-v1` |
| 语义训练 run manifest SHA-256 | `8959ca585a21dc1bf3f39650cfef74b95031d6a19a3c48a0718b6635f9a22562` |
| swap-init manifest SHA-256 | `d3e157889040a6bbf7f5f741366386b85adc297d4d1eea11d828ac359a5e0e49` |
| 候选 checkpoint SHA-256 | `e078cd7f6310cffb48a46c4e48d87aac76176e17b9caa5943ca57cbea3234f4b` |
| B0 教师 checkpoint SHA-256 | `15d70aa489f33b84b8cad0988aa0b8f1787ee626069d8c95c3e9e518e781a0ce`（冻结、eval、仅前向） |
| scene probe Run ID | `m2-interfuser-b0-anchored-distill-direct-scene-probe-v1-20260817-v1` |
| scene probe manifest SHA-256 | `e9ef98d40f377ef667817a89bb27d1eadbf7514ee9b7600a6b55d7fcaf7e6b6e` |
| scene probe 配置 SHA-256 | `49f3399ab71067468d151c61fbd997fb9c3177f3f52225491640d6411b310225` |
| 校准 manifest SHA-256 | `c1cb2907d1207c11e8211378ffd2f666efa08e8f56461e38e2e566ba751677fb` |

蒸馏损失：`L = L_semantic + 1.0 × Σ_l ‖f_s − f_t‖² / ‖f_t‖²`，λ 固定 1.0，协议禁止调系数。语义类别权重与 v3/v5 完全继承。

## 2. 语义门禁（4/4 通过）

| 检查 | 实测 | 阈值 | 判定 |
| --- | ---: | ---: | --- |
| best epoch 3 mIoU | 0.46775 | ≥ 0.46324 | ✓ |
| 行人 IoU（类 5） | 0.08311 | ≥ 0.08 | ✓ |
| 交通灯 IoU（类 7） | 0.19087 | ≥ 0.15 | ✓ |
| 张量严格导出 | 330/330 唯一 RGB 张量 | strict load 全过 | ✓ |

逐轮 mIoU：ep1 0.41846 → ep2 0.45577 → **ep3 0.46775（best）** → ep4 0.46078 → ep5 0.46204。以上数字由 run manifest 内逐轮混淆矩阵重算复核一致。swap-init 不变量：整模 1132 张量中仅 330 个 RGB 别名张量变更，非 RGB 状态逐字节保持。

## 3. 零微调 scene 门禁（失败：行人核心 2/5 < 3）

四项 checks：

| check | 结果 | 实测 |
| --- | --- | --- |
| nonpedestrian_retention | ✓ | 最大核心退化 0.01959%（≤ 3% 上限） |
| overall_traffic_retention | ✓ | 最大退化 0.0%（整体五项全部改善） |
| pedestrian_temporal_retention | ✓ | 最大时序退化 0.21641% |
| **pedestrian_core** | **✗** | **5 项核心仅 2 项改善，需 ≥ 3** |

行人核心五项明细（v 相对 B0 的相对变化）：

| 指标 | B0 | v7 | 相对变化 | 判定 |
| --- | ---: | ---: | ---: | --- |
| traffic AP | 0.115265 | 0.115370 | +0.0914% | ✓ 改善 |
| traffic occupied IoU@0.5 | 0.111445 | 0.111991 | +0.4900% | ✓ 改善 |
| traffic ROC AUC | 0.829795 | 0.829399 | −0.0477% | ✗ |
| waypoint ADE | 3.844454 | 3.845561 | +0.0288%（劣化） | ✗ |
| waypoint FDE@10 | 8.879360 | 8.881880 | +0.0284%（劣化） | ✗ |

三项失败全部为发丝级差距（≤ 0.05%）。行人队列其余指标：红灯 F1 +0.1882% ✓、waypoint delta-residual ADE −0.7135% ✓、junction F1 持平、traffic delta-residual MAE +0.2164%（时序保持项内）。

非行人队列：occupied IoU +0.5423%、AUC +0.0805%、红灯 F1 +0.2778%、路口 F1 +0.0674% 均改善；ADE/FDE 退化 0.0196%/0.0055%，远在 3% 上限内。整体队列：junction F1、红灯 F1、occupied IoU、AUC、traffic delta-residual MAE、waypoint delta-residual ADE 全部改善，无任何退化。

## 4. 蒸馏后校准对照（刻度偏移消隐）

复用 v6 离线阈值对照链路（score_dump 落盘 + `calibrate_interfuser_occupancy_threshold.py`，同一蛇形划分）：

| 项 | B0 | v7 候选 | 差值 |
| --- | ---: | ---: | ---: |
| t*（calibration 子集最优阈值） | 0.411355 | 0.410096 | **−0.00126** |
| evaluation 子集 IoU@0.5 差 | — | — | +0.00054 |
| evaluation 子集 IoU@t* 差 | — | — | +0.00109 |
| evaluation 子集 AP 差 | — | — | +0.00133 |
| evaluation 子集 AUC 差 | — | — | +0.00082 |

对比 v5（t* 差 −0.0226，IoU@t* 差 +0.01167）：v7 的阈值几乎贴住 B0，v6 诊断出的"刻度偏移"在蒸馏后基本消失；但与之相伴，v5 校准后可释放的收益也同量级缩没。教师约束同时勒住了退化与收益。

## 5. 裁决

按 v7 协议 §3（Owner 已冻结批准）：**任一 scene 门禁失败 → 视觉预训练修复路线关闭**。

- 禁止再扫描任何修复系数（λ、BN 动量、L2-SP 系数、回放比例）；
- 禁止修改门禁阈值或以校准后阈值替代门禁判定；
- 本结果作为 H1（交通语义预训练可无损替换 B0 视觉骨干）的诚实负证据链终点：
  v1/v2 权重插值失败 → v3 B0 锚定接近 → v4 L2-SP 无效 → v5 冻结 BN 统计差 0.00577pp → v6 证伪刻度解释（阈值偏移存在但非根因）→ **v7 蒸馏保持项全过、行人核心收益被稀释至 2/5**。

## 6. 边界

- `test_accessed = false`，`stage2_admitted = false`（scene probe manifest 字段实测值）；
- 未读取、未触碰任何 test 划分；Stage 2 未启动；
- 本记录只陈述测量事实与协议裁决，是否将 H1 负结论写入论文由 Owner 决定。
