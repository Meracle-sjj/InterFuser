# configs/thesis/
> L2 | 父级: ../CLAUDE.md

## 成员清单

CLAUDE.md: 论文配置子模块地图，规定配置文件与研究里程碑的对应关系。
baseline_eval_v1.json: M0 评测校准配置，固定模型和输入哈希、开发/主评测路线、随机种子及运行环境变量。
semantic_classes_v1.json: M1 语义类别配置，唯一映射 CARLA 0.9.16 source tag，并给出 pilot 数据准入阈值。
semantic_split_v1.json: M1 sequence 划分配置，固定 Town+route 原子分组、确定性种子、目标比例与每个 split 的类别/Town 准入门槛。
semantic_pretraining_smoke_v1.json: M2 首次训练链路配置，固定同构 ResNet50d、ImageNet 权重哈希、FPN head、确定性小样本预算与 GPU 6 门禁。
semantic_pretraining_pilot_v1.json: M2 数据量学习曲线配置，固定嵌套 25%/50%/100% train 样本数、完整 validation、单 epoch 预算与统一初始化/GPU 契约。
semantic_pretraining_optimization_probe_v1.json: M2 优化充分性诊断配置，固定 100% train、完整 validation、5 epoch 无类别权重预算和 best-mIoU checkpoint 选择。
semantic_pretraining_class_weight_probe_v1.json: M2 类别失衡诊断配置，仅在同一五轮全量预算中加入由无权重 train 像素支持派生的 inverse-sqrt 显式权重。
interfuser_downstream_split_v1.json: M2 H1 下游划分配置，将 M1 冻结 Town+route holdout 扩展到全量 dataset_index，未见 route group 仅进 train。
interfuser_visual_initialization_v1.json: M2 H1 初始化配置，冻结 InterFuser/ResNet 代码哈希、ImageNet B0 与交通域 V 权重及唯一 RGB 变量不变式。
interfuser_visual_pair_smoke_v1.json: M2 H1 配对训练 smoke 配置，冻结无泄漏索引/初始 checkpoint 哈希、各 2 sequence 确定性抽样、相同单 epoch 预算与 GPU 6/7。
interfuser_visual_pair_formal_v1.json: M2 H1 配对正式训练配置，复用上游 2 GPU×batch 16×25 epoch 配方并强制绑定无泄漏 train/validation/test 索引。
interfuser_visual_pair_test_v1.json: M2 H1 冻结 test 预注册配置，在 formal B0/V 完整归约前阻止 test 读取，并冻结单帧任务指标、5,462 个连续帧对和 GPU 6 串行资源。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
