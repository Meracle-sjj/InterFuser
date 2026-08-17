# docs/
> L2 | 父级: ../CLAUDE.md

## 成员清单

CLAUDE.md: 文档模块地图，规定论文目标与数据 schema 的依赖方向和阅读顺序。
thesis_goal_v1.md: 毕业论文研究目标基线 v1.0，冻结研究假设、两项创新边界、消融矩阵、评价证据与任务准入规则。
baseline_evaluation_protocol_v1.md: M0 基线评测校准协议，定义 P0、D7、A36、F42 四层集合和正式冻结门槛。
traffic_pretraining_dataset_v1.md: M1 交通语义数据规范，约束类别映射、采样结构、sequence 级划分、预训练准入门槛与 M2 证据驱动的补采边界。
semantic_pretraining_protocol_v1.md: M2 交通语义预训练协议，冻结同构 ResNet50d、一次性分割 head、离线指标、骨干迁移产物与优化/类别失衡诊断边界。
interfuser_visual_transfer_protocol_v1.md: M2 H1 下游迁移协议，冻结 B0/V 唯一 RGB 初始化变量、无泄漏索引、单帧/连续帧 test、D7 配对统计与 strict checkpoint 准入门槛。
interfuser_visual_swap_protocol_v1.md: M2 H1 固定历史 M0 底座的视觉替换可行性协议，区分 M0-FT/M0-V 与从头训练 B0/V，并冻结 strict-load 与非 RGB 保真门槛。
interfuser_official_visual_swap_protocol_v1.md: M2 H1 官方可驾驶底座的视觉骨干替换协议，固定零微调 route39 单变量准入、失配判定与短适配边界。
interfuser_official_modality_ablation_protocol_v1.md: M2 H1 官方底座 RGB/LiDAR 因果消融协议，以配对输出变化和真值退化区分融合器忽略视觉与视觉表征不足。
interfuser_official_modality_ablation_protocol_v2.md: M2 H1 CARLA 0.9.16 LiDAR 坐标修正审计协议，宣告空 LiDAR 的 v1 失效并冻结显式 y 轴乘数、输入密度准入与原消融阈值。
interfuser_official_stage2_protocol_v1.md: M2 H1 官方 B0/V 下游协同适配协议，冻结 compass/点云/导航字段契约、三轮同预算 pilot 与 route39 扩展门槛。
interfuser_downstream_scene_split_v2_protocol.md: M2 H1 行人场景测量修复协议，冻结只从语义预训练未使用 route group 扩充 holdout 的选择、结构验收、重训和 test 解封边界。
interfuser_official_stage2_protocol_v2.md: M2 H1 scene-split 数据测量修正版 Stage 2 协议，继承 v1 模型预算并禁止旧 checkpoint 污染新 holdout。
interfuser_negative_transfer_repair_protocol_v1.md: M2 H1 视觉负迁移修复协议，先以B0/V权重插值筛选稳定性-可塑性折中，再按证据准入官方B0教师保持、L2-SP或普通场景replay。
interfuser_negative_transfer_repair_protocol_v2.md: M2 H1 全模型插值失败后的RGB隔离协议，完整保留B0下游状态并只插值RGB alias以定位表征不兼容来源。
interfuser_negative_transfer_repair_protocol_v3.md: M2 H1 两轮插值失败后的B0锚定语义适配协议，以头部warmup和骨干低学习率让交通语义在原驾驶表征上增量生长。
interfuser_negative_transfer_repair_protocol_v4.md: M2 H1 B0锚定候选近门槛后的单系数L2-SP协议，以官方驾驶骨干参数源点显式约束普通能力保持。
interfuser_negative_transfer_repair_protocol_v5.md: M2 H1 状态漂移分解后的BN保持协议，冻结官方驾驶域running statistics并只让卷积/affine吸收语义监督。
interfuser_negative_transfer_repair_protocol_v6.md: M2 H1 纯测量校准诊断协议，证伪式检验排序改善与0.5阈值偏移假说，不修订v5失败判定且不与Stage 2准入挂钩。
interfuser_negative_transfer_repair_protocol_v7.md: M2 H1 单变量B0教师特征蒸馏协议，在v5冻结BN之上以归一化L2特征保持做本路线最后一次修复尝试。
experiment_records/: 论文实验事实记录，使用 run ID 与 SHA-256 连接 Git 配置和 results/ 原始产物。
handoff_pedestrian_visual_review.md: 历史行人单帧复核交接清单；已由连续测量帧碰撞威胁审计替代，仅保留为 584 sequence 的来源索引。
traffic_element_label_schema.md: 路线关联交通灯、虚拟停止边界及其 CARLA/Leaderboard 几何来源的 schema v2 契约。
traffic_element_image_label_schema.md: 将 schema v2 目标与 RGB、语义、深度和 LiDAR 证据对齐的 schema v3 契约。

依赖方向：`thesis_goal_v1.md` 决定需要回答的研究问题；两个 schema 文档只定义支撑视觉预训练数据的可审计事实，不能自行扩张论文目标。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
