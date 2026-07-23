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

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
