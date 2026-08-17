# tools/training/
> L2 | 父级: ../CLAUDE.md

## 成员清单

CLAUDE.md: M2 训练工具模块地图，约束数据、模型、运行编排和迁移导出共享同一版本化契约。
semantic_pretraining.py: 提供 split 驱动的 RGB/语义数据集、CARLA 标签映射、同构 ResNet50d-FPN、ImageNet/InterFuser B0骨干来源、分阶段低学习率与可选L2-SP源点保持契约、类别权重损失、混淆指标和严格迁移兼容性。
run_semantic_pretraining.py: 消费 M2 配置执行单机训练/验证，守卫 GPU/Git provenance，支持冻结骨干BN的头部warmup、分层学习率与L2-SP惩罚归约，并原子固化 checkpoint、骨干导出和运行事实。
evaluate_semantic_hazard_holdout.py: 将训练不可见的 route-group 行人危险 holdout 投影到冻结语义模型，对不同 checkpoint 输出同口径 mIoU、macro-F1 与逐类指标。
summarize_semantic_hazard_holdout.py: 严格校验 baseline/augmented 专项样本、类别权重和真值边界相同后，归约全局与逐类配对差值。
summarize_semantic_learning_curve.py: 验证 pilot run 的完整预算矩阵、嵌套 train 样本、相同完整 validation、provenance 与产物哈希，并确定性归约学习曲线。
interfuser_visual_pair.py: 冻结 ImageNet B0 与交通域 V 的 RGB 骨干差异，生成非 RGB 张量逐值相同且可被 InterFuser strict load 的全模型初始 checkpoint 对。
interfuser_visual_swap_pair.py: strict load 历史或官方固定底座，显式校验 checkpoint 原始架构别名并生成可命名的对照/视觉替换 checkpoint 对，证明底座全状态保真且唯一变量是交通域 RGB 骨干替换。
interfuser_pair_contract.py: 集中验证配对训练的 split/初始化哈希、传感器坐标、有效帧、预算、单/多 GPU 与资源策略，向 runner 暴露已解析契约。
run_interfuser_visual_pair.py: 串行编排 B0/V smoke/pilot/formal 下游训练，冻结初始化别名、传感器坐标、有效帧、单/多 GPU 与独占/共享容量资源策略，并严格归约参数、summary、checkpoint 和释放 manifest。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
