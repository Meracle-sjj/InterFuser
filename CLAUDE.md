# InterFuser - 面向毕业论文的多模态端到端自动驾驶研究平台

Python 3.10.19 + PyTorch 2.10.0.dev20251104+cu128 + torchvision 0.25.0.dev20251104+cu128 + timm 0.4.13 + CARLA 0.9.16 + Leaderboard 1.0

<directory>
assets/ - 上游论文、演示与说明文档使用的静态资源
data_collection/ - CARLA 多实例采集任务的配置与批处理生成
dataset/ - 数据集索引与采集目录初始化工具
docs/ - 论文目标、标签 schema 与数据契约
interfuser/ - 模型、训练入口和本地 timm 实现
leaderboard/ - Leaderboard 评测入口、路线数据与 agent/collector
scenario_runner/ - CARLA Scenario Runner 场景和违规判定逻辑
tests/ - 数据标签、投影、导出与运行兼容性回归测试
tools/ - 数据采集、审计、可视化和离线转换工具
data/ - 本机生成的数据产物，不进入 Git
results/ - 评测结果与运行产物，不进入 Git
</directory>

<config>
docs/thesis_goal_v1.md - 毕业论文目标基线，所有研究与实现任务的最高优先级约束
README.md - InterFuser 上游安装、数据生成、训练与评测说明
requirements.txt - 上游 Python 依赖基线；实际实验以固定 conda 环境为准
setup_carla.sh - 上游 CARLA 安装脚本；当前服务器实际运行 CARLA 0.9.16
.gitignore - 隔离数据、结果、CARLA 运行时和 Python 缓存
</config>

进入任何实现模块前，先阅读 `docs/thesis_goal_v1.md` 并确认任务能直接形成论文交付物。历史实现与目标基线冲突时，以目标基线为准。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
