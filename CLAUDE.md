# InterFuser - 面向毕业论文的多模态端到端自动驾驶研究平台

Python 3.10.19 + PyTorch 2.10.0.dev20251104+cu128 + torchvision 0.25.0.dev20251104+cu128 + timm 0.4.13 + CARLA 0.9.16 + Leaderboard 1.0

<directory>
assets/ - 上游论文、演示与说明文档使用的静态资源
configs/ - 论文实验、类别映射和可复现运行参数的机器可读契约
data_collection/ - CARLA 多实例采集任务的配置与批处理生成
dataset/ - 数据集索引与采集目录初始化工具
docs/ - 论文目标、标签 schema 与数据契约
interfuser/ - 模型、训练入口和本地 timm 实现
leaderboard/ - Leaderboard 评测入口、路线数据与 agent/collector
scenario_runner/ - CARLA Scenario Runner 场景和违规判定逻辑
tests/ - 数据标签、投影、导出与运行兼容性回归测试
tools/ - 数据采集、审计、闭环评测、视觉预训练与离线转换工具
data/ - 本机生成的数据产物，不进入 Git
results/ - 评测结果与运行产物，不进入 Git
</directory>

<config>
docs/thesis_goal_v1.md - 毕业论文目标基线，所有研究与实现任务的最高优先级约束
configs/thesis/baseline_eval_v1.json - M0 基线评测路线、哈希、环境和随机种子配置
configs/thesis/semantic_classes_v1.json - M1 CARLA 原始标签到预训练类别的唯一映射与 readiness 门槛
configs/thesis/semantic_pretraining_smoke_v1.json - M2 首次训练链路的模型、数据、初始化、预算与 GPU provenance 契约
configs/thesis/semantic_pretraining_pilot_v1.json - M2 数据量学习曲线的嵌套训练样本、完整验证集与统一预算契约
configs/thesis/semantic_pretraining_optimization_probe_v1.json - M2 全量数据多 epoch 优化充分性与最佳 checkpoint 选择契约
configs/thesis/semantic_pretraining_class_weight_probe_v1.json - M2 类别失衡诊断契约，冻结 inverse-sqrt 像素频率权重与来源 run
configs/thesis/semantic_pretraining_pedestrian_hazard_probe_v1.json - M2 行人碰撞威胁 train-only 扩充 probe，冻结原验证口径、专项 holdout 与五轮预算
configs/thesis/interfuser_downstream_split_v1.json - M2 H1 下游无泄漏 Town+route 全量索引投影契约
configs/thesis/interfuser_visual_initialization_v1.json - M2 H1 B0/V 单变量视觉初始化与代码/权重哈希契约
configs/thesis/interfuser_visual_initialization_pedestrian_hazard_v1.json - M2 H1 行人危险骨干的 B0/V 单变量严格迁移契约
configs/thesis/interfuser_visual_swap_initialization_v1.json - M2 H1 固定历史 M0 底座的 M0-FT/M0-V 单 RGB 骨干替换契约
configs/thesis/interfuser_visual_swap_route39_m0_ft_v1.json - M2 H1 固定底座 M0-FT 的 route39 seed0 零微调闭环诊断配置
configs/thesis/interfuser_visual_swap_route39_m0_v_v1.json - M2 H1 语义 RGB 替换 M0-V 的 route39 seed0 零微调闭环诊断配置
configs/thesis/interfuser_visual_swap_m0_ft_gpu1_v1.json - M2 H1 M0-FT D7-minus-route39 资源迁移配置，固定 GPU1 同卡 agent/CARLA 与阈值内受控共享
configs/thesis/interfuser_visual_swap_m0_v_gpu1_v1.json - M2 H1 M0-V D7-minus-route39 配对资源迁移配置，与 M0-FT 共用 GPU1 和相同共享门禁
configs/thesis/interfuser_visual_pair_smoke_v1.json - M2 H1 B0/V 配对下游训练链路 smoke 预算、资源和产物契约
configs/thesis/interfuser_visual_pair_pedestrian_hazard_smoke_v1.json - M2 H1 行人危险骨干 B0/V 配对下游 smoke 契约
configs/thesis/interfuser_visual_pair_formal_v1.json - M2 H1 B0/V 正式下游训练预算、无泄漏三分索引和资源契约
configs/thesis/interfuser_visual_pair_test_v1.json - M2 H1 冻结 test 的预注册指标、formal 产物准入与单 GPU 资源契约
configs/thesis/interfuser_visual_d7_build_v1.json - M2 H1 D7 配置冻结契约，在 formal/test 有效后生成哈希绑定的 B0/V 闭环配置
README.md - InterFuser 上游安装、数据生成、训练与评测说明
requirements.txt - 上游 Python 依赖基线；实际实验以固定 conda 环境为准
setup_carla.sh - 上游 CARLA 安装脚本；当前服务器实际运行 CARLA 0.9.16
.gitignore - 隔离数据、结果、CARLA 运行时和 Python 缓存
</config>

进入任何实现模块前，先阅读 `docs/thesis_goal_v1.md` 并确认任务能直接形成论文交付物。历史实现与目标基线冲突时，以目标基线为准。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
