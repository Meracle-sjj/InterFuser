# tools/data/
> L2 | 父级: ../CLAUDE.md

## 成员清单

CLAUDE.md: 数据工具模块地图，维护采集、标签、审计、复核和导出的单向数据流。
apply_painted_line_reviews.py: 将显式人工复核 manifest 应用到 evidence schema，只改变可见实体标线状态。
audit_semantic_pretraining_data.py: 从文件系统或 dataset_index 可复现抽样 RGB/语义帧对，统计类别覆盖、结构完整性并判定 M1 pilot readiness。
audit_traffic_element_labels.py: 校验 traffic-element schema v2 的结构、来源和路线停止目标覆盖。
audit_traffic_element_views.py: 校验 evidence schema v3 的 RGB、深度、语义、LiDAR 对齐和可见性证据。
batch_merge_data.py: 将多相机图像与测量/演员信息合并为旧训练加载器消费的完整样本。
batch_mv_data.py: 在旧 weather 数据目录间迁移 route 产物，属于离线数据维护工具。
batch_preload.py: 为旧数据集预计算局部 waypoint 序列，减少训练时重复几何计算。
batch_recollect_blocked_data.py: 根据阻塞片段重排旧采集文件编号，用于问题数据重采集后的修复。
batch_rm_haze_data.py: 删除旧采集流程标记的 haze/阻塞片段，不参与新 schema 审计。
batch_stat_blocked_data.py: 统计旧数据集中无合理制动原因的长时间停车片段。
batch_stat_data.py: 对旧 measurements_full 数据抽样统计驾驶状态与交通要素字段。
check_leaderboard_stop_target_geometry.py: 独立复现 Leaderboard 红灯边界并与采集标签比较几何一致性。
export_stop_boundary_labels.py: 从 schema v2/v3 导出虚拟停止边界 mask 和 manifest，不修改 RGB。
profile_traffic_element_routes.py: 分析稠密路线与交通灯的空间覆盖，为采样路线选择提供依据。
recompute_painted_line_status.py: 只读重算可见实体道路标线候选，不生成虚拟边界标签。
render_traffic_element_overlays.py: 将标签和证据渲染为人工复核图与 review manifest。
run_traffic_element_small_batch.sh: 编排小批量 CARLA 采集、重试、审计和结果隔离。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
