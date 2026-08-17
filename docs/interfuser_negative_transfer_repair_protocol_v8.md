<!--
[INPUT]: 依赖 v7 蒸馏终局证据（保持全过但行人核心 2/5，路线按 v7 §3 已关闭）、Owner 2026-08-17 战略裁决（论文题目为预训练，路线必须走通）、前沿文献检索结果（workspace v8_frontier/）。
[OUTPUT]: 对外提供兼容性预训练路线的阶段协议：v8a 线性可缝合性诊断的预注册判定规则，以及 v8b/v8c/v8d 候选方向的准入顺序。
[POS]: docs 的 M2 H1 负迁移修复第八阶段协议；路线由 Owner 裁决重开，问题定义从"防漂移"转为"表征几何兼容"。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
-->

# InterFuser 视觉负迁移修复协议 v8：兼容性预训练（表征几何诊断先行）

| 字段 | 内容 |
| --- | --- |
| 状态 | **PROBE-FROZEN：钧舰授权全权执行后冻结，2026-08-17** |
| 上游协议 | `docs/interfuser_negative_transfer_repair_protocol_v7.md`（§3 路线关闭裁决）、v5/v6 证据链 |
| 待建配置 | `configs/thesis/linear_stitch_diagnostic_v1.json` |

## 1. 定位：为什么路线重开

v7 §3 关闭了"语义预训练 + 防漂移缰绳"框架（v1→v7 证明：缰绳越紧退化越小、行人收益同比例缩没）。Owner 裁决：论文题目为预训练，路线必须在**不改动下游原始方案**（B0 权重不更新、下游头不微调）的前提下走通。问题定义随之升级：

> **兼容性预训练**：如何预训练/后处理视觉骨干，使其表征几何能嵌入一个已训练、冻结的下游 transformer。

文献定位（2026-08-17 Scholar 检索，CSV 存 `workspace/v8_frontier/`）：

- **后向兼容表征学习（BCT）**：Shen et al., CVPR 2020；Zhang et al. universal BCT 2022；Hu et al., KDD 2022——新 encoder 训练时以旧下游为约束，v7 是其原始版（只对齐骨干边界，未穿过下游计算图）；
- **模型缝合**：Bansal et al. model stitching；Maiorca et al. 2024 latent translation 零样本缝合；Yu et al., NeurIPS 2025——新旧表征之间学轻量变换；
- **残差式防负迁移**：Xu et al., ICLR 2026 Residual Feature Integration——新特征 = 旧特征 + 可学残差，构造性零退化；
- **任务对齐预训练**：LAW（ICLR 2025）、PGS（NeurIPS 2025）、S4-Driver（CVPR 2025）——驾驶领域共识：pretext 须贴近下游功能。

## 2. v8a：线性可缝合性诊断（纯测量，预注册判定规则）

**研究问题**：现有最佳语义骨干（v7 蒸馏产物）的特征，离 B0 特征是否只差一个线性变换？这决定缝合方向（v8c）是否可行。

- **教师**：官方 B0 RGB 骨干（checkpoint SHA-256 `15d70aa489f33b84b8cad0988aa0b8f1787ee626069d8c95c3e9e518e781a0ce`，与 v7 蒸馏教师同源），eval、无梯度、仅前向；
- **主臂**：v7 蒸馏骨干 `results/thesis_m2/m2-semantic-b0-anchored-distill-probe-v1-seed20260723-20260817-v1/backbone_resnet50d.pth`（SHA-256 配置绑定时写入）；
- **次臂（描述性）**：v5 冻结 BN 骨干 `results/thesis_m2/m2-semantic-b0-anchored-frozen-bn-probe-v3-seed20260723-20260817-v1/backbone_resnet50d.pth`（SHA-256 配置绑定时写入）；
- **数据**：语义 validation 全集 1,725 帧（与 v7 训练同一 split manifest `ea4e5878…6de29b`），逐帧确定性 50/50 划分：`sha256(f"{seed}:{key}")` 首字节 < 0x80 进拟合半，否则进留出一半，**seed 固定 20260817**；
- **方法**：逐 stage（ResNet-50d `features_only` 全部 4 个 stage），在拟合半的全部空间位置上累积充分统计量 `G=Σxx^T`、`H=Σxy^T`，闭式岭回归 `W=(G+λI)^{-1}H`，`λ=1e-3·tr(G)/C`（无尺度可调系数）；
- **指标（留出一半，逐 stage）**：恒等相对误差 `E_id=Σ||f_s−f_t||²/Σ||f_t||²`；缝合后相对误差 `E_st=Σ||W^Tf_s−f_t||²/Σ||f_t||²`；**误差消减率 `R=1−E_st/E_id`**；缝合前后平均余弦相似度；
- **预注册判定规则**：主臂最深 stage（downstream 实际消费的特征层）留出误差消减率 **R ≥ 50%** → 线性缝合成立，v8c（冻结骨干+缝合层）获得准入；**R < 50%** → 线性变换不足以弥合几何差，v8c 降级，优先 v8b（穿过冻结下游栈的功能蒸馏）或 v8d（残差注入）；
- **边界**：本诊断不触碰 test、不产出 Stage 2 准入、不修订任何既有门禁判定；与 v6 同属纯测量。

## 3. 后续方向（各自独立协议增补后方可执行）

| 方向 | 内容 | 准入条件 |
| --- | --- | --- |
| v8b | BCT 式功能蒸馏：梯度穿过冻结完整 B0 下游栈，在语义语料上匹配 B0 最终输出（无需下游标签） | v8a 任一结果均可启动 |
| v8c | 冻结骨干 + 线性缝合层（缝合层属预训练产物，推理时冻结） | v8a 主臂 R ≥ 50% |
| v8d | 残差注入：新特征 = B0 特征 + 可学残差（Xu et al. 2026） | v8a 显示误差结构化（各 stage R 差异大）时优先 |

后续任一方向若进入训练，scene 门禁沿用 v5/v7 预注册口径（行人核心 ≥ 3/5、非行人退化 ≤ 3%、整体 ≤ 3%、行人时序 ≤ 10%），不修订。

## 4. 工程与运行

- 新工具：`tools/evaluation/linear_stitch_diagnostic.py`（库 + CLI 同文件，同 v6 校准工具风格）；配置 `configs/thesis/linear_stitch_diagnostic_v1.json` 哈希绑定全部 checkpoint 与 v7 训练配置（数据/划分字段直接复用，保证同分布）；
- 复用 `load_training_contract` / `SemanticFrameDataset` / `make_frozen_feature_teacher`，不新增数据管线；
- 运行时契约沿用 v3 §2：GPU1 共享容量、启动时 ≥ 20,000 MiB 空闲、`nice -n 19`、不清理外部 context；
- 新增 unittest：充分统计量累积正确性、岭回归闭式解、确定性划分、判定规则逻辑、清单哈希绑定；全仓测试保持通过；
- 产物：诊断 manifest（含双臂逐 stage 全部指标、划分哈希、全部输入产物 SHA-256）、实验记录 `docs/experiment_records/2026-08-17-m2-linear-stitch-diagnostic-v1.md`；
- 全部改代码点执行 GEB 回环：L3 头部 → L2 成员清单 → L1 检查。
