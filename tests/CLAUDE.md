# tests/
> L2 | 父级: ../CLAUDE.md

## 成员清单

CLAUDE.md: 回归测试模块地图，确保标签来源、投影、采集、审计与运行兼容性可独立验证。
test_apply_painted_line_reviews.py: 验证人工 painted-line 决策的路径约束、状态迁移和原子写入。
test_audit_semantic_pretraining_data.py: 验证语义类别映射、dataset_index 分层抽样、RGB/mask 对齐统计和 pilot readiness 失败原因。
test_audit_pedestrian_hazard_visibility.py: 验证行人碰撞威胁阶段、RGB 可见覆盖、天气重复签名、普通可见行人区分与缺失特权真值门禁。
test_audit_traffic_element_labels.py: 验证 schema v2 审计对合法记录、禁用字段和来源错误的处理。
test_audit_traffic_element_views.py: 验证 schema v3 的相机、LiDAR、投影、可见性和帧对齐审计。
test_background_traffic_health.py: 验证背景交通健康度统计在 CARLA 参与者与道路投影上的行为。
test_check_leaderboard_stop_target_geometry.py: 验证停止边界独立几何比较的数量、车道和坐标容差。
test_collector_stop_exclusion.py: 验证采集器只保存目标窗口并排除错误停止目标。
test_export_stop_boundary_labels.py: 验证虚拟边界 mask/manifest 导出且不改写源 RGB。
test_evaluation_runtime_resources.py: 验证 runner 默认拒绝外来 GPU compute owner、显式共享仍受总显存阈值限制，并能回收包装进程已退出但子进程仍存活的完整 POSIX 进程组。
test_leaderboard_stop_targets.py: 验证 Leaderboard 红灯触发体到虚拟边界和路线距离的几何构造。
test_profile_traffic_element_routes.py: 验证路线交通灯覆盖、hard-negative 点和距离评分。
test_preflight_thesis_baseline.py: 验证 M0 静态预检对哈希、路线分区、场景事件和地图排除的约束。
test_recompute_painted_line_status.py: 验证实体标线 dry-run 的图像证据、深度和只读约束。
test_red_light_junction_gate.py: 验证控制器路口概率来源对红灯制动门控的历史兼容行为。
test_route_progress_blocked.py: 验证 180 秒内不足 18 米的路线进度停滞会生成 VEHICLE_BLOCKED，且短促强制油门不能重置判定。
test_render_traffic_element_overlays.py: 验证复核 overlay 的目标选择、颜色语义与 manifest 输出。
test_run_thesis_baseline.py: 验证 D7 runner 的路线拆分、CARLA 原生 RPC 崩溃隔离、晚发 CARLA 退出分类、结果解析、独占/共享资源门禁、可配置显存释放等待和 pipeline-invalid 立即终止。
test_semantic_split_and_review.py: 验证 M1 split 的 Town+route 原子性、三组核心类别覆盖、内容哈希、RGB/mask 尺寸门禁与人工复核证据渲染。
test_semantic_pretraining.py: 验证 M2 配置哈希、数据边界、确定性样本、CARLA 标签映射、加权损失、B0 RGB prefix加载、分阶段骨干日程、离线指标、ResNet50d前向和严格迁移兼容性。
test_interfuser_downstream_indexes.py: 验证 v1 M1 holdout 全量投影、v2 行人真值分层扩充、semantic-train 隔离、确定性/census 门禁、索引哈希与 CarlaMVDetDataset 显式 index 选择。
test_interfuser_visual_pair.py: 验证 B0/V 初始 checkpoint 仅改变 RGB 共享骨干 alias、非 RGB 状态哈希相同且全模型 strict load。
test_interfuser_visual_swap_pair.py: 验证对照分支原样继承历史/官方底座、视觉分支仅替换 RGB alias、非 RGB 状态相同，并覆盖 checkpoint 架构别名与错误元数据拒绝。
test_interfuser_train_validation.py: 验证单 GPU validation 对全部 loss/accuracy 张量执行恒等归约，阻止仅分布式路径赋值造成 epoch 后崩溃。
test_calibration_diagnostic.py: 验证 v6 校准诊断的 route group 蛇形划分确定性与约束、阈值搜索与冻结指标口径一致、分数落盘回读与端到端合成诊断。
test_run_interfuser_visual_pair.py: 验证 B0/V 训练命令共享预算与坐标、单 GPU pilot 表达、smoke 索引确定性、formal test index 强制绑定、args 差异归一和 summary 完整性门禁。
test_interfuser_offline_metrics.py: 验证冻结 test 的 traffic AP/AUC/IoU、逐类混淆、正确 stop-sign head、waypoint 精确分母、目标条件连续帧残差与缺失支持门禁。
test_run_interfuser_visual_test.py: 验证预注册 test 契约、索引哈希/相邻帧计数漂移、formal 未完成阻断、B0/V checkpoint/schema 准入和固定方向差值。
test_run_interfuser_modality_ablation.py: 验证 CARLA 采集版本的 LiDAR y 轴、compass 导航 frame、缺失导航帧过滤、RGB/LiDAR 单变量干预与 weak/material 阈值方向。
test_run_interfuser_scene_validation.py: 验证 Stage 2 scene validation 的 completed pair/validation/checkpoint 哈希绑定、test index 拒绝与行人核心指标多数票方向。
test_run_interfuser_weight_interpolation_probe.py: 验证浮点权重插值、B0离散缓冲保留、schema漂移拒绝及行人收益与普通/整体/时序保持联合门禁。
test_summarize_semantic_learning_curve.py: 验证 M2 pilot 汇总拒绝缺失预算、非嵌套 train、validation 漂移、pipeline/provenance 异常与产物哈希漂移。
test_summarize_thesis_baseline.py: 验证 M0 汇总器拒绝缺失、重复、基础设施失败和未授权输入漂移，区分评测语义与共享 GPU 资源策略后确定性归约完整路线×种子矩阵。
test_summarize_interfuser_visual_d7.py: 验证 M2 H1 汇总器要求 B0/V 各 21/21、固定 attempt 顺序、checkpoint-only 差异，并按冻结 offline+D7 规则生成结论。
test_analyze_interfuser_visual_d7_failures.py: 验证配对 D7 的状态转移、首碰撞事件与起步转向/车道偏移连续控制帧归因。
test_build_pedestrian_hazard_training_manifest.py: 验证行人危险 train-only 扩充、原 validation/test 不变、Town+route holdout 隔离与特权真值准入。
test_evaluate_semantic_hazard_holdout.py: 验证行人危险 validation/test 样本归约、训练契约不可变性与 route-group 泄漏拒绝。
test_summarize_semantic_hazard_holdout.py: 验证行人危险 baseline/augmented 样本 key 一致、逐类差值与 loss/特权边界漂移拒绝。
test_run_interfuser_visual_d7_pair.py: 验证 M2 H1 D7 父级 runner 的 test 准入、B0→V 串行顺序、42/42 完成门槛、失败即停和目录幂等。
test_build_interfuser_visual_d7_configs.py: 验证 D7 build 契约在 test 前只允许 preflight，并在 test 有效后确定性生成三份 checkpoint/provenance 绑定配置且禁止覆盖。
test_traffic_element_collector.py: 验证采集器建立并保存多传感器、交通标签和测量目录。
test_traffic_element_labels.py: 验证 schema v2 标签、坐标变换、旧 affordance 合并和记录校验。
test_traffic_element_projection.py: 验证世界/传感器投影、深度解码、证据关联与标线候选。
test_traffic_manager_compat.py: 验证 CARLA 0.9.16 Traffic Manager API、`_Opt` 地图名称兼容层，以及同步退出先于 actor 回收的幂等清理顺序。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
