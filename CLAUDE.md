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
configs/thesis/interfuser_downstream_split_v2.json - M2 H1 行人场景分层下游契约，只从语义预训练未使用的 route group 确定性补足双 holdout
configs/thesis/interfuser_visual_initialization_v1.json - M2 H1 B0/V 单变量视觉初始化与代码/权重哈希契约
configs/thesis/interfuser_visual_initialization_pedestrian_hazard_v1.json - M2 H1 行人危险骨干的 B0/V 单变量严格迁移契约
configs/thesis/interfuser_visual_swap_initialization_v1.json - M2 H1 固定历史 M0 底座的 M0-FT/M0-V 单 RGB 骨干替换契约
configs/thesis/interfuser_visual_swap_route39_m0_ft_v1.json - M2 H1 固定底座 M0-FT 的 route39 seed0 零微调闭环诊断配置
configs/thesis/interfuser_visual_swap_route39_m0_v_v1.json - M2 H1 语义 RGB 替换 M0-V 的 route39 seed0 零微调闭环诊断配置
configs/thesis/interfuser_visual_swap_m0_ft_gpu1_v1.json - M2 H1 M0-FT D7-minus-route39 首次 GPU1 同卡迁移与阈值内受控共享配置
configs/thesis/interfuser_visual_swap_m0_v_gpu1_v1.json - M2 H1 M0-V D7-minus-route39 首次 GPU1 配对迁移配置
configs/thesis/interfuser_visual_swap_m0_ft_gpu1_v2.json - M2 H1 M0-FT GPU1 清理恢复配置，显式使用 300 秒显存释放窗口
configs/thesis/interfuser_visual_swap_m0_v_gpu1_v2.json - M2 H1 M0-V GPU1 清理恢复配对配置，与 M0-FT v2 保持相同资源口径
configs/thesis/interfuser_visual_swap_m0_ft_gpu1_v3.json - M2 H1 M0-FT GPU1 编排恢复配置，机器级固化 D7-minus-route39 的路线集合与顺序
configs/thesis/interfuser_visual_swap_m0_v_gpu1_v3.json - M2 H1 M0-V GPU1 编排恢复配对配置，与 M0-FT v3 共享严格 18-task 口径
configs/thesis/interfuser_visual_swap_m0_ft_gpu1_v4.json - M2 H1 M0-FT 路线进度判停恢复配置，在统一新 evaluator 下固化完整 D7 的 21-task 口径
configs/thesis/interfuser_visual_swap_m0_v_gpu1_v4.json - M2 H1 M0-V 路线进度判停恢复配对配置，与 M0-FT v4 共享代码、资源和完整 D7 顺序
configs/thesis/interfuser_visual_swap_m0_ft_gpu1_v5.json - M2 H1 M0-FT GPU1 共享容量恢复配置，冻结 12 GiB 空闲门槛与本项目 GPU PID 清理口径
configs/thesis/interfuser_visual_swap_m0_v_gpu1_v5.json - M2 H1 M0-V GPU1 共享容量配对配置，与 M0-FT v5 共享完整 D7 和资源所有权语义
configs/thesis/interfuser_visual_pair_smoke_v1.json - M2 H1 B0/V 配对下游训练链路 smoke 预算、资源和产物契约
configs/thesis/interfuser_visual_pair_pedestrian_hazard_smoke_v1.json - M2 H1 行人危险骨干 B0/V 配对下游 smoke 契约
configs/thesis/interfuser_visual_pair_formal_v1.json - M2 H1 B0/V 正式下游训练预算、无泄漏三分索引和资源契约
configs/thesis/interfuser_visual_pair_test_v1.json - M2 H1 冻结 test 的预注册指标、formal 产物准入与单 GPU 资源契约
configs/thesis/interfuser_visual_d7_build_v1.json - M2 H1 D7 配置冻结契约，在 formal/test 有效后生成哈希绑定的 B0/V 闭环配置
configs/thesis/interfuser_official_stage2_pilot_v1.json - M2 H1 官方底座 Stage 2 三轮配对协同适配契约，冻结 CARLA 0.9.16 坐标、有效帧和 GPU1 单卡全局 batch
configs/thesis/interfuser_official_stage2_pilot_scene_split_v2.json - M2 H1 scene-split v2 上的官方 B0/V 三轮重训契约，保持模型预算不变并扩大无泄漏行人 holdout
configs/thesis/interfuser_stage2_scene_validation_v1.json - M2 H1 Stage 2 best checkpoint 的 validation-only 整体/行人/路线组配对离线评估契约
configs/thesis/interfuser_weight_interpolation_probe_v1.json - M2 H1 B0/V 最终权重插值的负迁移修复筛选与普通场景保持门禁契约
configs/thesis/interfuser_weight_interpolation_rgb_probe_v2.json - M2 H1 仅插值RGB并保留B0下游全状态的负迁移定位契约
configs/thesis/semantic_pretraining_b0_anchored_staged_probe_v1.json - M2 H1 以官方B0 RGB为源点、头部warmup后低学习率解冻的受约束语义适配契约
configs/thesis/interfuser_official_visual_swap_initialization_b0_anchored_v1.json - M2 H1 将B0锚定语义骨干严格替换回作者发布底座的单RGB初始化契约
configs/thesis/interfuser_b0_anchored_direct_scene_probe_v1.json - M2 H1 B0锚定语义骨干零微调回接后的整体/行人/非行人保持门禁契约
configs/thesis/semantic_pretraining_b0_anchored_l2sp_probe_v2.json - M2 H1 在B0锚定分阶段语义适配上增加单系数源点保持的L2-SP契约
configs/thesis/semantic_pretraining_b0_anchored_frozen_bn_probe_v3.json - M2 H1 冻结官方B0归一化缓冲、只适配卷积/affine参数的语义保持契约
configs/thesis/interfuser_official_visual_swap_initialization_b0_anchored_frozen_bn_v2.json - M2 H1 将冻结BN的B0锚定语义骨干严格回接作者底座的初始化契约
README.md - InterFuser 上游安装、数据生成、训练与评测说明
requirements.txt - 上游 Python 依赖基线；实际实验以固定 conda 环境为准
setup_carla.sh - 上游 CARLA 安装脚本；当前服务器实际运行 CARLA 0.9.16
.gitignore - 隔离数据、结果、CARLA 运行时和 Python 缓存
</config>

进入任何实现模块前，先阅读 `docs/thesis_goal_v1.md` 并确认任务能直接形成论文交付物。历史实现与目标基线冲突时，以目标基线为准。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
