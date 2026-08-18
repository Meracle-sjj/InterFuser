<!--
[INPUT]: 依赖 Owner 2026-08-18 两次裁决（架构假设存疑、批准 v9 直接执行）、docs/KEYreferences.md 文献结论、
  Stage 2 v2 已有 B0 臂与 scene validation 冻结管线、v8b 建成的语料审计/前向/语义头机器。
[OUTPUT]: 对外提供 v9 两阶段协议：架构内辅助语义预训练 + 标准适配的预注册设计与判定规则。
[POS]: docs 的 M2 语义注入方式研究第九阶段协议；研究问题从"zero-touch 替换"转为"监督注入位置"。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
-->

# InterFuser 架构内辅助语义预训练协议 v9（TransFuser×ViDAR 范式迁移）

| 字段 | 内容 |
| --- | --- |
| 状态 | **FROZEN：Owner 2026-08-18 批准直接执行** |
| 上游 | `docs/KEYreferences.md`（文献否定/揭示）、v8b 证据链终点、Stage 2 v2 协议 |
| 待建配置 | `configs/thesis/interfuser_architecture_semantic_pretraining_v1.json` |

## 1. 定位

v1→v8b 证明：外部头预训练 + zero-touch 替换在五类修复假设下全部失败（详见 2026-08-18-m2-functional-distill-probe-v1 §5）。文献主线（TransFuser/LAV/InterFuser-aux；ViDAR/AD-PT）一致表明：**语义监督必须在架构内部与驾驶目标联合注入，评价协议允许标准适配**。v9 检验这一迁移在本项目是否成立。

## 2. 两阶段设计

### 阶段一：架构内辅助语义预训练（v9-pretrain）
- **起点**：官方 B0 完整 checkpoint（SHA `15d70aa…`），**全模型可训**（含 transformer 与 LiDAR 编码；author 式 train 模式，BN 正常更新）；
- **数据**：v8b P0 审计过的语义语料 train 3,868 帧（驾驶目标+10 类 seg+LiDAR 完备）与 validation 557 帧；
- **损失**：`L = L_driving + 1.0 × L_semantic`
  - `L_driving = 0.5·traffic + 0.2·waypoints + 0.05·velocity + 0.05·junction + 0.1·red_light + 0.01·stop_sign`（逐字移植 `interfuser/train.py` 的 MVTL1Loss/WaypointL1Loss/CE 组合）；
  - `L_semantic` = 前摄 10 类加权 CE（v7 类权重、ignore 255），FPN 头经 layer1-4 前向钩子消费骨干 stage 特征（v8b 机器），解码器从 v7 checkpoint_best 初始化、与全模型共同训练；
  - 辅助系数固定 1.0，不扫描；
- **预算**：5 epoch、bs 16、AdamW lr 1e-4 / wd 0.01、seed 20260723；epoch 选择以 **validation 驾驶损失**最低为准（辅助任务不参与选择）；
- **阶段 sanity 门禁**：全部损失有限、验证驾驶损失下降、产物 checkpoint 可被 strict load。

### 阶段二：标准适配（Stage 2 v2 原配方）
- v9 产物 checkpoint 作为唯一初始化差异，进入与 Stage 2 v2 **完全相同**的适配（165,264 帧、三轮、单卡 batch 32、同优化器/学习率/坐标契约）；
- 对照臂 = 既有 Stage 2 v2 B0 臂（scene validation manifest `f9477b9c…`，不重跑）。

## 3. 预注册判定规则（scene validation v2，v9 臂 vs B0 臂）

| 检查 | 判据 |
| --- | --- |
| 行人条件五项（AP/AUC/IoU/ADE/FDE） | v9 改善 ≥ 3/5 |
| 非行人核心最坏退化 | ≤ 3% |
| 整体 traffic 最坏退化 | ≤ 3% |
| 行人时序最坏退化 | ≤ 10% |

全过 → "架构内语义预训练有效"成立，准入冻结 test 与 D7；任一失败 → v9 失败，证据固化，不再扫描任何系数。判定不因预训练阶段内部指标（mIoU 等）改写。

## 4. 工程与运行

- 新增 `tools/training/architecture_pretraining.py`（库：损失移植+契约+epoch 循环）与 `run_architecture_pretraining.py`（runner，资源纪律沿用 v8b：GPU1 共享容量 ≥20,000 MiB、nice -n 19、require_clean_git）；
- 复用 v8b 的 `FunctionalFrameDataset/build_base_datasets/attach_carla_transforms` 与审计绑定；新配置哈希绑定全部输入；
- Stage 2 复用现有 runner 与配置结构，仅换初始化；scene validation 复用 `configs/thesis/interfuser_stage2_scene_validation_v1.json` 的变体指向；
- 新增 unittest：WaypointL1Loss 权重与无效掩码、MVTL1Loss 平衡项与速度项、组合权重、契约校验、aux 头梯度可达全模型；全仓测试保持通过；
- 产物：预训练 run manifest（全输入 SHA、逐 epoch 驾驶/语义损失、选择记录）、Stage 2 run、scene validation 对照 manifest、实验记录 `docs/experiment_records/2026-08-18-m2-v9-architecture-semantic-pretraining-v1.md`；
- GEB 回环：L3 头部 → L2 成员清单 → L1 检查。

## 5. 边界

- `test_accessed = false` 维持；判定前不触碰 test、不跑 D7；
- 本协议修订 08-17"下游不可动"裁决为"预训练阶段全模型可训、适配阶段标准预算"（Owner 2026-08-18 授权）；
- v1-v8b 判定不被改写；论文叙事按「失败解剖（v1-v8b）+ 架构内注入验证（v9）」双章节组织。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
