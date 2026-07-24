# docs/
> L2 | 父级: ../CLAUDE.md

## 成员清单

CLAUDE.md: 文档模块地图，规定论文目标与数据 schema 的依赖方向和阅读顺序。
thesis_goal_v1.md: 毕业论文研究目标基线 v1.0，冻结研究假设、两项创新边界、消融矩阵、评价证据与任务准入规则。
baseline_evaluation_protocol_v1.md: M0 基线评测校准协议，定义 P0、D7、A36、F42 四层集合和正式冻结门槛。
traffic_pretraining_dataset_v1.md: M1 交通语义数据规范，约束类别映射、采样结构、sequence 级划分、预训练准入门槛与 M2 证据驱动的补采边界。
semantic_pretraining_protocol_v1.md: M2 交通语义预训练协议，冻结同构 ResNet50d、一次性分割 head、离线指标、骨干迁移产物与优化/类别失衡诊断边界。
interfuser_visual_transfer_protocol_v1.md: M2 H1 下游迁移协议，冻结 B0/V 唯一 RGB 初始化变量、无泄漏索引、单帧/连续帧 test、D7 配对统计与 strict checkpoint 准入门槛。
experiment_records/: 论文实验事实记录，使用 run ID 与 SHA-256 连接 Git 配置和 results/ 原始产物。
traffic_element_label_schema.md: 路线关联交通灯、虚拟停止边界及其 CARLA/Leaderboard 几何来源的 schema v2 契约。
traffic_element_image_label_schema.md: 将 schema v2 目标与 RGB、语义、深度和 LiDAR 证据对齐的 schema v3 契约。

依赖方向：`thesis_goal_v1.md` 决定需要回答的研究问题；两个 schema 文档只定义支撑视觉预训练数据的可审计事实，不能自行扩张论文目标。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
