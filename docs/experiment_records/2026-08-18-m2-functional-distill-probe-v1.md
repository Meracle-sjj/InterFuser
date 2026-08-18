<!--
[INPUT]: 依赖 v8.1 协议（Owner 2026-08-18 批准冻结）、实现提交 388f050..38e6e70（含七项基础设施修正）、
  v7 swap-init 学生初始化与官方 B0 教师、语义语料 P0 审计（LiDAR 乘子 1.0 判定）、v7 语义验证管线。
[OUTPUT]: 对外提供 v8b BCT 功能蒸馏探针的完整证据链：五轮训练曲线、非 RGB 零变更不变量、
  语义准入失败判定与评估器保真度对照、失败解剖与协议裁决。
[POS]: docs/experiment_records 的 M2 H1 负迁移修复 v8 首个训练方向终局锚点；只陈述测量事实与协议裁决。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
-->

# M2 v8b BCT 功能蒸馏探针记录（语义准入失败裁决）

## 1. Provenance

| 字段 | 值 |
| --- | --- |
| 协议 | `docs/interfuser_negative_transfer_repair_protocol_v8_1.md`（FROZEN 2026-08-18） |
| 训练 Run ID | `m2-semantic-b0-functional-distill-probe-v1-seed20260723-20260818-v4` |
| run manifest SHA-256 | 见 `results/thesis_m2/…-v4/run_manifest.json`（status=completed, pipeline_valid=true） |
| 训练配置 SHA-256 | `3140a8f2cc5f9858…`（`semantic_pretraining_b0_functional_distill_probe_v1.json`） |
| 实现 HEAD | `38e6e70`（实现 388f050 + 七项修正 a7c6038/e8ea0a6/d7c2a07/8927f79/d9452ab/20c2db1/41b1677/38e6e70） |
| 学生初始化 | v7 swap-init 全模型（SHA `e078cd7f…`） |
| 教师 | 官方 B0 完整 checkpoint（SHA `15d70aa…`，eval、零梯度） |
| 语义头初始化 | v7 checkpoint_best（SHA `00fd0c20…`） |
| 语料 | P0 审计通过：train 161 序列 3,868 有效帧、validation 26 序列 557 帧、LiDAR 乘子判定 1.0（+1.0→11,459.74 非零格 vs −1.0→25.59，参考带 [5000,20000]）；101 帧缺导航字段按 drop 契约剔除 |
| 确定性 | v1→v4 四次 run 逐 epoch 数值逐位一致（0.685502/0.528758/…） |

## 2. 训练曲线（bs=16 帧级，五轮，无 OOM 回退）

| epoch | train loss | train task(语义) | train functional | val mIoU(模型几何) | val functional |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 (warmup，仅解码器) | 0.685502 | 0.684762 | 0.000740 | 0.279577 | 0.000711 |
| 2 | 0.528758 | 0.515219 | 0.013539 | 0.308025 | 0.008077 |
| 3 | 0.464323 | 0.438344 | 0.025979 | 0.311758 | 0.014746 |
| 4 | 0.411912 | 0.391808 | 0.020104 | 0.313754 | 0.014378 |
| 5 (best) | 0.373516 | 0.358849 | 0.014667 | 0.319480 | 0.016001 |

选择准则 validation.mean_iou（best=epoch 5）。运行时长约 40 秒/epoch（GPU1 共享容量预检通过，nice -n 19）。

## 3. 冻结纪律不变量（实测通过）

- 全模型 1,132 张量中**仅 330 个 RGB 张量变更**（`changed_tensors_all_rgb=true`）；
- `rgb_patch_embed.backbone.*` 与 `rgb_backbone.*` 为同一批张量的双重注册（`rgb_alias_key_count=660`，即 v7 时代"330 别名"机制本体），以存储身份判定归属；
- RGB BN running stats 零漂移（全程 eval）；非 RGB 张量与初始 checkpoint 逐字节一致；
- 骨干导出为 v7 容器契约（`backbone.` 前缀 + `source_training_config_sha256`），strict load 通过。

## 4. 语义准入门禁（失败）

评估管线：v8b 导出骨干 + v8b 解码器，v7 同一 160×120 validation（1,725 相机样本）。

| 检查 | 实测 | 门槛 | 判定 |
| --- | ---: | ---: | --- |
| validation mIoU | **0.26303** | ≥ 0.46324 | ✗ |
| 行人 IoU | **0.01448** | ≥ 0.08 | ✗ |
| 交通灯 IoU | **0.01017** | ≥ 0.15 | ✗ |

**评估器保真度对照**：同一管线评估 v7 自身产物复现 **0.46775 / 0.08311 / 0.19087**——与 v7 实验记录（2026-08-17-m2-distill-probe-v1 §2）逐位一致，排除评估器缺陷。

## 5. 失败解剖（观察，非门禁改写依据）

1. **语义监督被稀释**：v8b 的语义损失仅作用于每帧前摄（3,868 帧），v7 为三相机逐样本（11,907/epoch）——有效语义监督约为 v7 的 1/3，且与功能损失竞争同一骨干；
2. **解码器几何错配**：v8b 解码器在模型几何（resize 341×256 + center-crop 224，视野左右各裁 ~24%）上共同训练，门禁却在 160×120 全视野评估；训练时 val mIoU 峰值 0.3195（模型几何）→ 门禁 0.2630（v7 几何）；
3. **张力前移**：v5 在 scene 门禁差 0.00577pp、v7 行人核心差发丝级，v8b 的语义-保持张力直接在**语义准入**以宽边际爆发——BCT 形态未打破 trade-off，反而使其提前显形。

第 2 点构成 v8b 特有的实现性混叠（协议未预注册双几何解码器适配）；但第 1、3 点表明即使消除几何混叠，功能性锚定对语义学习的抑制仍在。该观察供 Owner 裁决后续方向时参考，不构成改写本判定的依据。

## 6. 裁决（协议 §4 预注册）

**语义准入任一失败 → 不生成官方 swap checkpoint，v8b 方向停止。**

- 不执行 swap-init 与零微调 scene 门禁；
- 不扫描系数、不改门禁、不换阈值；
- `test_accessed = false`、`stage2_admitted = false` 维持；
- v8c 维持降级、v8d（残差注入）维持备选；是否开启新方向（v8d、双几何语义头增补、或接受 H1 负结论收官）由 Owner 裁决；
- 本记录作为 H1 诚实负证据链 v1→v8b 的终点之一固化：参数漂移、BN 统计、刻度、线性几何、功能性输出匹配五类假设全部经预注册实验证伪或未过门禁。

## 7. 基础设施注记

- v1→v3 为基础设施无效 run（类配置归一化、帧号补零、损失设备、GRU 训练态、不变量设备比较、别名判定、导出容器七项修正），v4 为首个 pipeline_valid run；四次 run 训练数值逐位一致；
- 新工具：`audit_functional_distillation_corpus.py`（P0）、`functional_distillation.py` + `run_functional_distillation.py`（训练）、`evaluate_functional_export_semantics.py`（语义准入）；新增 15 项 unittest，全仓 320/320 通过（含 v7 浮点断言加固 places=5→delta=1e-2）；
- 期间清理过两个本方失控测试进程（约 140 核空转，ssh 客户端超时遗留），已由 Owner 手动 kill；
- 全部改代码点执行 GEB 回环：L3 头部 → L2 成员清单 → L1 检查。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
