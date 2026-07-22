# leaderboard/
> L2 | 父级: ../CLAUDE.md

## 成员清单

.pylintrc: Leaderboard Python 静态检查配置，约束历史上游代码风格。
CHANGELOG.md: Leaderboard 运行行为与兼容性变更历史。
LICENSE: CARLA Leaderboard 源码许可证。
README.md: Leaderboard 安装、agent 接口与评测使用说明。
data/: 冻结路线 XML 与场景 JSON，向 evaluator 提供闭环评测输入。
leaderboard/: Python 评测包，编排 agent、scenario、统计与运行时生命周期。
requirements.txt: Leaderboard Python 依赖基线。
scripts/: 上游评测启动脚本；论文运行由 tools/evaluation 的配置驱动 runner 接管。
team_code/: InterFuser agent 与模型配置适配层，是闭环模型加载入口。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
