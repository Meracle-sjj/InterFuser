# tools/
> L2 | 父级: ../CLAUDE.md

## 成员清单

CLAUDE.md: 工具模块地图，区分路线生成工具与数据审计子模块的职责边界。
data/: 数据生成、审计、转换和复核工具；所有论文数据结论必须由可重复命令产生。
evaluation/: 论文闭环评测的预检、执行与汇总工具；运行前必须消费版本化评测配置。
generate_intersection_routes.py: 从地图与路口结构生成路线定义，供场景数据构造使用。
generate_scenarios.py: 生成 Scenario Runner 场景配置，使路线评测具备可触发事件。
interpolate_route.py: 将稀疏路线控制点插值为稠密轨迹，供路线分析与采样消费。
sample_junctions.py: 从 CARLA 地图抽取路口候选，支撑路线与场景生成。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
