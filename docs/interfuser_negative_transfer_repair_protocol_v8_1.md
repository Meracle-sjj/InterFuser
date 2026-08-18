<!--
[INPUT]: 依赖 v8 协议（§3 方向准入）、v8a 诊断证据（主臂 R=1.08% 判 v8c 降级）、v7 训练配置契约与 scene 门禁实现、官方 B0 checkpoint、v7 蒸馏骨干产物。
[OUTPUT]: 对外提供 v8b BCT 式功能蒸馏的执行协议：损失形式、教师/学生/下游冻结契约、P0 预检、门禁复用口径、停止边界与工程清单。
[POS]: docs 的 M2 H1 负迁移修复 v8 首个训练方向增补；Owner 2026-08-18 批准起草，审阅冻结后方可实现。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
-->

# InterFuser 视觉负迁移修复协议 v8.1：v8b BCT 式功能蒸馏

| 字段 | 内容 |
| --- | --- |
| 状态 | **EXECUTED：2026-08-18 执行完毕，语义准入失败（mIoU 0.26303 < 0.46324），按 §4 方向停止；证据见 `docs/experiment_records/2026-08-18-m2-functional-distill-probe-v1.md`**（前状态 FROZEN 2026-08-18） |
| 上游协议 | `docs/interfuser_negative_transfer_repair_protocol_v8.md`（§3 v8b 准入）、`docs/interfuser_negative_transfer_repair_protocol_v7.md`（训练配方与门禁基准） |
| 待建配置 | `configs/thesis/semantic_pretraining_b0_functional_distill_probe_v1.json` |

## 1. 定位与上游结论

v8a 判定（R=1.08% < 50%）：v7 蒸馏已把骨干特征贴到 B0 的 99.82% 余弦邻域，剩余差距是**非线性功能响应差异**，线性缝合（v8c）在数学上够不着。v8b 直接优化下游功能行为而非中间几何：语义训练时，同一批输入同时过"学生骨干 → 冻结完整 B0 下游栈"与完整 B0，损失匹配最终输出。这是 BCT（Shen et al., CVPR 2020）在"下游不可训练"约束下的形态：B0 的下游前向即教师，无需任何下游标签。

## 2. 方法（相对 v7 的唯一变更：特征蒸馏 → 功能蒸馏）

完全继承 v7 配方：B0 源点谱系、普通 7,590 + 危险 4,317 数据与类别权重、头部 warmup 一轮、`1e-5/1e-4` 分层学习率、五轮预算、best-mIoU 选择、RGB 骨干全部 BN 保持 eval、`L_semantic` 不变。**不使用 v7 的逐 stage 特征蒸馏（两项互斥，单变量纪律）**。

新增内容：

- **学生初始化**：v7 蒸馏 best checkpoint 的 RGB 骨干（`m2-semantic-b0-anchored-distill-probe-v1-seed20260723-20260817-v1/backbone_resnet50d.pth`，SHA-256 `e078cd7f6310cffb48a46c4e48d87aac76176e17b9caa5943ca57cbea3234f4b`，配置绑定时复核）——v8a 已证它是最接近 B0 的语义骨干，功能损失直攻其残差；
- **教师**：官方 B0 完整 checkpoint（SHA-256 `15d70aa489f33b84b8cad0988aa0b8f1787ee626069d8c95c3e9e518e781a0ce`），`eval()`、`requires_grad_(False)`、仅前向；
- **学生侧下游栈**：同一官方 checkpoint 的全部非 RGB 模块（transformer、LiDAR 编码、任务头），`eval()`、`requires_grad_(False)`——梯度穿过激活（学生骨干需要梯度回传），但任何下游参数与 BN running stats 零更新；swap 装载复用现有 swap-init 不变量工具（330 张量别名映射、非 RGB 逐字节保持）；
- **输入构造**：与 scene validation 工具链同源——3 相机（front/left/right）按下游 input_size、LiDAR 按 `multi_view_input_size=(3,128,128)` 与 `lidar_y_axis_multiplier` 契约构造；教师与学生严格同批同输入；
- **损失**：`L = L_semantic + L_functional`，其中
  `L_functional = Σ_h ‖o_s^h − o_t^h‖² / ‖o_t^h‖²`（逐头归一化 L2，教师输出为锚），
  匹配头 h ∈ {traffic 网格 (B,400,7)、waypoints (B,10,2)、junction、red_light、stop_sign logits}；具体张量键以实现时核对的模型结构为准，逐头写入 manifest；
- **系数固定 1.0**：归一化消除各头尺度差，不引入任何可调系数，不扫描。

### 2.1 为什么 LiDAR 必须在场（设计依据，Owner 2026-08-18 确认）

InterFuser 是双模态融合模型：RGB 特征与 LiDAR 特征在冻结 transformer 中的交互方式，正是 v8a 定位的"功能响应"本体。若给教师/学生喂空 LiDAR，匹配的是残缺输入分布下的伪 B0 函数，学到的兼容性不可迁移到部署分布；v7（特征层对齐，不涉 LiDAR）已是不碰 LiDAR 的极限。协议内 LiDAR 流两端严格同值、其编码器权重冻结且逐字节保持官方 B0——LiDAR 不是研究变量，是恒定环境（恒温水浴）。

## 3. P0 预检（实现前强制）

语义语料逐帧 **LiDAR 可用性审计**：功能蒸馏要求每帧可构造完整模型输入。若审计发现任何 train 帧缺 LiDAR，协议**就地停止**并固化审计报告，由 Owner 裁决补采或改向——不得以空点云/复制邻近帧等静默手段绕过。审计结果（覆盖率、缺失清单哈希）写入训练 manifest。

## 4. 门禁（与 v7 §3 逐字一致，不修订）

- **语义准入**：validation mIoU ≥ 0.46324；行人 IoU ≥ 0.08；traffic-light IoU ≥ 0.15；严格导出 330 个 ResNet50d 张量。任一失败，不生成官方 swap checkpoint；
- **零微调 scene 保持门禁**（固定 0.5 阈值，复用 v5/v7 配置与工具链）：行人核心改善 ≥ 3/5；非行人核心最坏退化 ≤ 3%；整体 traffic 最坏退化 ≤ 3%；行人时序最坏退化 ≤ 10%；
- 两步全过 → 按 scene-split v2 准入三轮配对 Stage 2；
- 任一失败 → 本方向停止，结果作为 H1 诚实负证据固化；是否开启新方向（v8d 残差注入或其他）由 Owner 裁决，不扫描系数、不改门禁、不换阈值。

## 5. 工程与运行

- 训练入口：`tools/training/run_functional_distillation.py` + 库 `tools/training/functional_distillation.py`；配置块 `functional_distillation`（与 `feature_distillation` 互斥）；下游栈装载、输入构造复用 scene validation 的数据集助手；教师/学生双前向在同一 step 内完成；
  - 实现注记（2026-08-18）：由独立 runner 承载而非并入 run_semantic_pretraining.py，因其帧级批形状与双模型前向已超出原 runner 职责；语义损失经 layer1-4 前向钩子复用 v7 的 FPN 解码器，模型定义文件零改动（哈希绑定不受影响）。
- 新配置 `configs/thesis/semantic_pretraining_b0_functional_distill_probe_v1.json`：哈希绑定 v7 初始化骨干、官方 B0 checkpoint、split manifest（`ea4e5878…6de29b` 同分布）与全部训练超参（沿用 v7 数值，仅 distill 块替换）；
- 运行时契约沿用 v3 §2：GPU1 共享容量、启动时 ≥ 20,000 MiB 空闲、`nice -n 19`、不清理外部 context、`require_clean_git`；双前向显存约为 v7 的 2.5 倍，**预注册基础设施回退**：若 bs=16 CUDA OOM → bs=8 × 梯度累积 2（等价有效批量，manifest 记录），其余任何失败不回退、换新 Run ID 重跑；
- 新增 unittest：教师与下游栈全程 eval + 无参数更新、梯度可从功能损失回传到学生骨干（对 backbone 参数 grad 非 None）、逐头归一化与互斥配置校验、输入构造与 scene validation 字节级同构、确定性；全仓测试保持通过；
- 产物：训练 manifest（双 checkpoint SHA-256、逐轮语义指标、逐头功能损失曲线、LiDAR 审计、swap 不变量）、best checkpoint、swap-init manifest、零微调 scene manifest、实验记录 `docs/experiment_records/2026-08-XX-m2-functional-distill-probe-v1.md`；Run ID：`m2-semantic-b0-functional-distill-probe-v1-seed20260723-20260818-v1`；
- 全部改代码点执行 GEB 回环：L3 头部 → L2 成员清单 → L1 检查。

## 6. 边界

- `test_accessed = false` 必须保持；Stage 2 未准入前 `stage2_admitted = false`；
- 本协议不修订 v8a 及之前任何判定；v8c 维持降级，v8d 维持备选；
- 多模态判定（如失败案例目检）不属本协议范围，触发时按 HANDOFF §0 停机线交接。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
