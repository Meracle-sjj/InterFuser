<!--
[INPUT]: 依赖 2026-08-18 文献侦察（Owner 发起：架构假设存疑，需迁移既有研究方法）与 v1→v8b 全部实测证据。
[OUTPUT]: 对外提供文献结论的非清单式沉淀——否定了哪些想法、揭示了哪些路径、v9 设计与文献的逐条映射；供任何接手 agent 直接消费。
[POS]: docs 的方法学路标；区别于 paper/evidence_ledger.md（数值边界账本），本文件回答"为什么走这条路"。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
-->

# KEYreferences — 文献否定了什么、揭示了什么（2026-08-18 侦察沉淀）

## 0. 一句话结论

交通语义监督必须在**驾驶架构内部**注入并与驾驶目标联合训练（TransFuser/LAV 范式）；在外部头独立预训练再 zero-touch 替换骨干，是与该领域全部成功实践相反的路径——我们的 v8a/v8b 实测从反面独立证实了这一点。

## 1. 被文献否定的想法

### 否定 1：外部预训练头 + zero-touch 骨干替换
- **想法**：在 InterFuser 之外用分割头养好骨干，导出 330 张量替换进冻结的官方模型。
- **文献证据**：CARLA 排行榜家族无一采用此路径——[TransFuser (CVPR'21/PAMI'22)](https://arxiv.org/abs/2205.15997) 把 aux 语义分割+深度头挂在融合 transformer 中间特征上联合训练；[LAV (CVPR'23)](https://arxiv.org/abs/2203.11934) 在驾驶模型内嵌感知辅助任务；[InterFuser (CoRL'22)](https://proceedings.mlr.press/v205/shao23a.html) 自身的 junction/红灯/停止头就是架构内辅助监督。2025 年仍在延续（[语义一致性辅助损失](https://arxiv.org/abs/2507.17596)、[运动+语义联合学习](https://arxiv.org/abs/2502.07631)）。
- **我们的实测佐证**：v8a 线性缝合诊断——外部头骨干与 B0 的最深 stage 差距仅 1.08% 可被线性映射消解（需 ≥50%），即**结构性非线性能量差**；v8b 把功能匹配损失穿进冻结下游，语义监督被稀释至 1/3 且宽边际失败（mIoU 0.263 < 0.463）。v1-v8b 五类修复假设（参数漂移/BN/刻度/线性几何/功能输出）全部证伪。
- **推论**：失败不是调参问题，是**监督注入位置**问题。Owner 2026-08-18 的架构判断与此完全一致。

### 否定 2："冻结下游是领域标准做法"
- **想法**：为保官方驾驶能力，预训练评测期间下游 transformer 一步不许动。
- **文献证据**：[ViDAR (CVPR'24)](https://github.com/OpenDriveLab/ViDAR) 的协议是架构内预训练→**微调**；[AD-PT (NeurIPS'23)](https://neurips.cc/virtual/2023/poster/70971) 报告 pretrain→finetune 增益（+0.93 mAP/+0.98 NDS）；[Fine-Tuning Can Distort Pretrained Features (ICLR'22)](https://arxiv.org/abs/2202.10054) 连全量微调都嫌扭曲特征、主张 LP-FT 混合——**没有任何主流工作把"零适配"作为评价协议**。
- **我们的实测佐证**：这是自设的非标准严苛协议（08-17 Owner 裁定的产物），v1-v8b 的全部失败都在这条协议内发生。

### 否定 3（弱化保留）：后处理/缝合可救外部头
- **想法**：BCT/模型缝合/残差注入等后处理把外部头特征"翻译"进 B0 空间。
- **文献证据**：BCT（Shen et al. CVPR'20）确实以旧下游为约束训练新 encoder——但**训练时**约束就在场，而非事后缝合；模型缝合文献（Bansal et al.）的适用前提是两表征间存在可学的轻量变换。
- **我们的实测佐证**：v8a 证明主臂残差非线性（R=1.08%），缝合数学上够不着。v8b（BCT 的正确形态：约束在场训练）仍因语义稀释失败——说明问题比"约束时机"更深，回到否定 1 的注入位置问题。

## 2. 文献揭示的路径

### 路径 A：架构内辅助语义监督（TransFuser/LAV 范式）
语义头挂在骨干/融合特征上，与驾驶损失**联合**训练，部署时语义头可弃。这是我们 v9 处理臂的直接蓝本。
### 路径 B：架构内预训练 → 标准适配（ViDAR/AD-PT 范式）
先在完整驾驶架构上做预训练（改变初始化），再用**完全相同的标准微调预算**适配，与对照组比终态。这是我们 v9 两阶段设计的骨架——对照组（Stage 2 v2 B0 臂）已存在。
### 路径 C：LP-FT 洞识（保特征的适配节奏）
全量微调会扭曲预训练特征 → 低学习率+短预算的适配（Stage 2 的 3-epoch 设计天然符合）比激进微调更保特征。给 v9 适配阶段背书。

## 3. v9 设计 ↔ 文献映射

| v9 组件 | 文献出处 |
| --- | --- |
| 预训练：官方 B0 全模型起点，驾驶五头损失 + hook 骨干特征→FPN 辅助语义头（λ=1.0 固定） | TransFuser aux 监督 + LAV 多任务 |
| 适配：Stage 2 v2 原配方（165,264 帧三轮、单卡 batch 32、低 LR） | ViDAR pretrain→finetune + LP-FT 节奏 |
| 评价：scene validation v2 对照既有 B0 臂，预注册方向判据 | AD-PT 对照协议 |
| 论文叙事：v1-v8b 失败解剖（负）+ v9 架构内注入（正） | "how to inject" 完整故事线 |

## 4. 参考文献索引

- [TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving (PAMI 2022)](https://arxiv.org/abs/2205.15997) / [GitHub](https://github.com/autonomousvision/transfuser) / [PDF](https://www.cvlibs.net/publications/Chitta2022PAMI.pdf)
- [Safety-Enhanced Autonomous Driving Using Interpretable Sensor Fusion Transformer (InterFuser, CoRL 2022)](https://proceedings.mlr.press/v205/shao23a.html)
- [Learning from All Vehicles (LAV, CVPR 2023)](https://arxiv.org/abs/2203.11934)
- [Visual Point Cloud Forecasting enables Scalable Autonomous Driving (ViDAR, CVPR 2024)](https://cvpr.thecvf.com/virtual/2024/poster/29545) / [GitHub](https://github.com/OpenDriveLab/ViDAR)
- [AD-PT: Autonomous Driving Pre-Training (NeurIPS 2023)](https://neurips.cc/virtual/2023/poster/70971)
- [Fine-Tuning Can Distort Pretrained Features (ICLR 2022)](https://arxiv.org/abs/2202.10054)
- [Post-Training in End-to-End Autonomous Driving: A Unified View (2026)](https://arxiv.org/html/2607.08072v1)
- [Learning to Plan from Raw Pixels（语义一致性辅助损失, 2025）](https://arxiv.org/abs/2507.17596)
- [Motion and Semantic Learning in End-to-End Autonomous Driving (2025)](https://arxiv.org/abs/2502.07631)
- [Foundation models for autonomous driving: A comprehensive survey (2025)](https://www.sciencedirect.com/science/article/pii/S0952197626010870)

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
