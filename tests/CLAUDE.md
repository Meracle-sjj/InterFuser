# tests/
> L2 | 父级: ../CLAUDE.md

## 成员清单

CLAUDE.md: 回归测试模块地图，确保标签来源、投影、采集、审计与运行兼容性可独立验证。
test_apply_painted_line_reviews.py: 验证人工 painted-line 决策的路径约束、状态迁移和原子写入。
test_audit_semantic_pretraining_data.py: 验证语义类别映射、dataset_index 分层抽样、RGB/mask 对齐统计和 pilot readiness 失败原因。
test_audit_traffic_element_labels.py: 验证 schema v2 审计对合法记录、禁用字段和来源错误的处理。
test_audit_traffic_element_views.py: 验证 schema v3 的相机、LiDAR、投影、可见性和帧对齐审计。
test_background_traffic_health.py: 验证背景交通健康度统计在 CARLA 参与者与道路投影上的行为。
test_check_leaderboard_stop_target_geometry.py: 验证停止边界独立几何比较的数量、车道和坐标容差。
test_collector_stop_exclusion.py: 验证采集器只保存目标窗口并排除错误停止目标。
test_export_stop_boundary_labels.py: 验证虚拟边界 mask/manifest 导出且不改写源 RGB。
test_evaluation_runtime_resources.py: 验证 runner 拒绝外来 GPU 计算 owner，并能回收包装进程已退出但子进程仍存活的完整 POSIX 进程组。
test_leaderboard_stop_targets.py: 验证 Leaderboard 红灯触发体到虚拟边界和路线距离的几何构造。
test_profile_traffic_element_routes.py: 验证路线交通灯覆盖、hard-negative 点和距离评分。
test_preflight_thesis_baseline.py: 验证 M0 静态预检对哈希、路线分区、场景事件和地图排除的约束。
test_recompute_painted_line_status.py: 验证实体标线 dry-run 的图像证据、深度和只读约束。
test_red_light_junction_gate.py: 验证控制器路口概率来源对红灯制动门控的历史兼容行为。
test_render_traffic_element_overlays.py: 验证复核 overlay 的目标选择、颜色语义与 manifest 输出。
test_run_thesis_baseline.py: 验证 D7 runner 的路线拆分、CARLA 原生 RPC 崩溃隔离、晚发 CARLA 退出分类、结果解析、资源门禁和 pipeline-invalid 立即终止。
test_semantic_split_and_review.py: 验证 M1 split 的 Town+route 原子性、三组核心类别覆盖、内容哈希、RGB/mask 尺寸门禁与人工复核证据渲染。
test_semantic_pretraining.py: 验证 M2 配置哈希、smoke/pilot/optimization 数据边界、确定性样本、CARLA 标签映射、无权重/加权确定性损失、离线指标、ResNet50d 前向和骨干严格迁移兼容性。
test_interfuser_downstream_indexes.py: 验证 M1 holdout route group 的全量投影、未见组只进 train、索引哈希/覆盖门禁与 CarlaMVDetDataset 显式 index 选择。
test_interfuser_visual_pair.py: 验证 B0/V 初始 checkpoint 仅改变 RGB 共享骨干 alias、非 RGB 状态哈希相同且全模型 strict load。
test_run_interfuser_visual_pair.py: 验证 B0/V 训练命令共享预算、smoke 索引确定性、formal test index 强制绑定、args 允许差异归一和 summary 完整性门禁。
test_summarize_semantic_learning_curve.py: 验证 M2 pilot 汇总拒绝缺失预算、非嵌套 train、validation 漂移、pipeline/provenance 异常与产物哈希漂移。
test_summarize_thesis_baseline.py: 验证 M0 汇总器拒绝缺失、重复、基础设施失败和未授权输入漂移，并按冻结口径确定性归约完整路线×种子矩阵。
test_traffic_element_collector.py: 验证采集器建立并保存多传感器、交通标签和测量目录。
test_traffic_element_labels.py: 验证 schema v2 标签、坐标变换、旧 affordance 合并和记录校验。
test_traffic_element_projection.py: 验证世界/传感器投影、深度解码、证据关联与标线候选。
test_traffic_manager_compat.py: 验证 CARLA 0.9.16 Traffic Manager API、`_Opt` 地图名称兼容层，以及同步退出先于 actor 回收的幂等清理顺序。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
