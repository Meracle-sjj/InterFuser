# leaderboard/leaderboard/scenarios/
> L2 | 父级: ../CLAUDE.md

## 成员清单

__init__.py: Leaderboard 路线场景包标识。
background_activity.py: 动态背景交通场景，按 Town 配置参与者数量并清理拥堵车辆。
master_scenario.py: 通用 MasterScenario 装配器，面向非路线专用评测入口。
route_scenario.py: 论文闭环主路线场景，装配动态事件、背景交通、路线完成度与速度/路线进度双重 blocked 终止判据。
scenario_manager.py: Leaderboard 同步执行循环，连接 agent、行为树、watchdog 与结果输出。
scenarioatomics/: Leaderboard 特有的原子评测判据，补充 ScenarioRunner 通用 criteria。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
