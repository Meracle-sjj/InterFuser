# docs/
> L2 | 父级: ../CLAUDE.md

## 成员清单

CLAUDE.md: 文档模块地图，规定论文目标与数据 schema 的依赖方向和阅读顺序。
thesis_goal_v1.md: 毕业论文研究目标基线 v1.0，冻结研究假设、两项创新边界、消融矩阵、评价证据与任务准入规则。
traffic_element_label_schema.md: 路线关联交通灯、虚拟停止边界及其 CARLA/Leaderboard 几何来源的 schema v2 契约。
traffic_element_image_label_schema.md: 将 schema v2 目标与 RGB、语义、深度和 LiDAR 证据对齐的 schema v3 契约。

依赖方向：`thesis_goal_v1.md` 决定需要回答的研究问题；两个 schema 文档只定义支撑视觉预训练数据的可审计事实，不能自行扩张论文目标。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
