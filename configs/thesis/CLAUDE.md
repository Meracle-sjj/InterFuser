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
semantic_pretraining_pedestrian_hazard_probe_v1.json: M2 行人危险数据扩充 probe，保持原 validation、五轮预算与 inverse-sqrt 权重，唯一增加经碰撞真值审计的 train sequence。
interfuser_downstream_split_v1.json: M2 H1 下游划分配置，将 M1 冻结 Town+route holdout 扩展到全量 dataset_index，未见 route group 仅进 train。
interfuser_downstream_split_v2.json: M2 H1 行人分层下游配置，锁定 M1 已分配 route group，只从语义预训练未使用组确定性补足 train/validation/test=`45/8/8` 行人路线覆盖。
interfuser_visual_initialization_v1.json: M2 H1 初始化配置，冻结 InterFuser/ResNet 代码哈希、ImageNet B0 与交通域 V 权重及唯一 RGB 变量不变式。
interfuser_visual_initialization_pedestrian_hazard_v1.json: M2 H1 行人危险骨干迁移配置，以新 best-epoch5 导出替换 V RGB 骨干，保持 B0 与非 RGB 张量不变式。
interfuser_visual_swap_initialization_v1.json: M2 H1 固定底座替换配置，以历史 M0 epoch26 为共同全模型状态，仅允许行人危险语义 ResNet50d 覆盖 M0-V 的 RGB 骨干。
interfuser_official_visual_swap_initialization_v1.json: M2 H1 官方底座替换配置，以作者发布 epoch34 checkpoint 为共同全模型状态，仅允许交通语义 ResNet50d 覆盖 official_b0_v 的 RGB 骨干。
interfuser_official_modality_ablation_v1.json: M2 H1 官方底座模态依赖配置，冻结 validation 全量 4,613 帧、B0/B0-V checkpoint、四种单模态干预、GPU1 资源门槛和 RGB/LiDAR 效应量阈值。
interfuser_official_modality_ablation_v2.json: M2 H1 CARLA 0.9.16 LiDAR 坐标修正版模态配置，保持 v1 干预不变，显式取消重复 y 轴反号并以输入密度门槛阻止空 LiDAR 充当对照。
interfuser_official_stage2_pilot_v1.json: M2 H1 官方 B0/V 的 Stage 2 三轮全模型协同适配配置，冻结 compass 导航 frame、LiDAR `+1`、缺失导航帧剔除和 GPU1 单卡全局 batch 32。
interfuser_official_stage2_pilot_scene_split_v2.json: M2 H1 scene-split v2 配对重训配置，继承官方 B0/V 三轮预算并绑定 45/8/8 行人路线分层后的 train/validation 索引。
interfuser_stage2_scene_validation_v1.json: M2 H1 scene-split v2 的 validation-only 配对评估配置，哈希绑定 B0/V best checkpoint 并冻结整体、行人条件、非行人与8个路线组的任务/连续帧指标。
interfuser_visual_swap_route39_m0_ft_v1.json: M2 H1 M0-FT 零微调诊断配置，绑定固定底座保真 checkpoint，并只准运行 route39 seed0 的闭环包装校准。
interfuser_visual_swap_route39_m0_v_v1.json: M2 H1 M0-V 零微调诊断配置，绑定语义 RGB 替换 checkpoint，以 route39 seed0 检测旧融合头的特征分布失配。
interfuser_visual_swap_m0_ft_gpu1_v1.json: M2 H1 M0-FT D7-minus-route39 GPU1 首次资源迁移配置，将 agent/CARLA 固定到同一 RTX 5090，并显式准入 1 GiB 门槛内的既有 compute context。
interfuser_visual_swap_m0_v_gpu1_v1.json: M2 H1 M0-V D7-minus-route39 GPU1 首次配对配置，复用与 M0-FT 完全相同的同卡资源、受控共享策略与默认清理窗口。
interfuser_visual_swap_m0_ft_gpu1_v2.json: M2 H1 M0-FT D7-minus-route39 GPU1 清理恢复配置，保持 v1 模型和评测变量，仅将显存释放窗口显式延长至 300 秒。
interfuser_visual_swap_m0_v_gpu1_v2.json: M2 H1 M0-V D7-minus-route39 GPU1 清理恢复配对配置，与 M0-FT v2 使用相同的 300 秒释放窗口和资源门禁。
interfuser_visual_swap_m0_ft_gpu1_v3.json: M2 H1 M0-FT D7-minus-route39 GPU1 编排恢复配置，保持 v2 运行口径并把 `[18,6,12,30,36,0]` 固化为专用 route set。
interfuser_visual_swap_m0_v_gpu1_v3.json: M2 H1 M0-V D7-minus-route39 GPU1 编排恢复配对配置，与 M0-FT v3 共享专用 18-task 路线集合、资源门禁和清理窗口。
interfuser_visual_swap_m0_ft_gpu1_v4.json: M2 H1 M0-FT 路线进度判停恢复配置，冻结 180 秒/18 米进度停滞准则、GPU1 资源口径与 `[18,6,12,30,36,0,39]` 完整 D7 顺序。
interfuser_visual_swap_m0_v_gpu1_v4.json: M2 H1 M0-V 路线进度判停恢复配对配置，除 checkpoint/variant 外与 M0-FT v4 完全一致并绑定相同 evaluator 哈希。
interfuser_visual_swap_m0_ft_gpu1_v5.json: M2 H1 M0-FT GPU1 共享容量恢复配置，保留 v4 评测语义并以至少 12 GiB 空闲启动、本项目 GPU PID/端口/进程组退出完成清理判定。
interfuser_visual_swap_m0_v_gpu1_v5.json: M2 H1 M0-V GPU1 共享容量恢复配对配置，除 checkpoint/variant 外与 M0-FT v5 完全一致，外部 compute owner 不再触发清理误判。
interfuser_visual_pair_smoke_v1.json: M2 H1 配对训练 smoke 配置，冻结无泄漏索引/初始 checkpoint 哈希、各 2 sequence 确定性抽样、相同单 epoch 预算与 GPU 6/7。
interfuser_visual_pair_pedestrian_hazard_smoke_v1.json: M2 H1 行人危险骨干下游 smoke 配置，复用相同 2+2 sequence、单 epoch 和 GPU 6/7，只替换 V 初始权重。
interfuser_visual_pair_formal_v1.json: M2 H1 配对正式训练配置，复用上游 2 GPU×batch 16×25 epoch 配方并强制绑定无泄漏 train/validation/test 索引。
interfuser_visual_pair_test_v1.json: M2 H1 冻结 test 预注册配置，在 formal B0/V 完整归约前阻止 test 读取，并冻结单帧任务指标、5,462 个连续帧对和 GPU 6 串行资源。
interfuser_visual_d7_build_v1.json: M2 H1 D7 配置生成契约，在 test 结果前冻结 M0 模板、B0/V/pair Run ID、路线/seed 顺序和最终配置输出路径。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
