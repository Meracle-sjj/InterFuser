# leaderboard/leaderboard/
> L2 | 父级: ../CLAUDE.md

## 成员清单

__init__.py: Leaderboard Python 包标识。
autoagents/: agent 抽象、传感器封装与输入合法性边界。
envs/: 传感器数据接口与伪传感器运行环境。
leaderboard_evaluator.py: 单路线闭环评测入口，负责 CARLA 同步执行、统计落盘与幂等资源回收。
scenarios/: 路线场景构造、背景交通、行为树执行与结果分析。
utils/: 路线解析、索引、统计和结果序列化公共能力。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
