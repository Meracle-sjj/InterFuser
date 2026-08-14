# 官方 InterFuser RGB/LiDAR 模态依赖审计协议 v2.0

| 字段 | 内容 |
| --- | --- |
| 状态 | **FROZEN：修正 CARLA 0.9.16 LiDAR 采集坐标后重跑** |
| 配置 | `configs/thesis/interfuser_official_modality_ablation_v2.json` |
| 运行器 | `tools/evaluation/run_interfuser_modality_ablation.py` |
| v1 处置 | LiDAR 输入密度失败，只保留为坐标链故障证据，不进入模态依赖结论 |

## 1. v1 为何失效

v1 读取 4,613 帧后发现：原始 LiDAR 平均每帧 `14,574` 个点，但官方 loader 的固定 `y *= -1` 与当前 CARLA 0.9.16 采集坐标叠加后，平均只生成 `25.51/150,528` 个非零 tensor cell；23 帧甚至全零。抽样原始点云的前向区域主要位于负 y，额外反号后落到 histogram 接受区间 `y∈[-28,0]` 之外。

因此 v1 中 `lidar_zero` 几乎没有改变输入，它不能作为 RGB 效应量分母。该失败属于输入坐标契约错误，不是模型证据；v1 manifest 不覆盖、不删除。

## 2. 唯一修正

v2 保留 v1 的 4,613 帧 validation、两套 checkpoint、五个 condition、batch/seed、任务指标、配对输出指标和判断阈值。唯一数据变化是将 `lidar_y_axis_multiplier` 从历史默认 `-1.0` 显式设为 `+1.0`，即不再对当前采集点云重复反号。

`CarlaMVDetDataset` 的默认值仍为 `-1.0`，避免静默改写历史训练语义；只有哈希绑定且显式声明 CARLA 0.9.16 坐标约定的配置才使用 `+1.0`。

## 3. 密度准入

正式前向完成前必须同时满足：

- 平均每帧 LiDAR 非零 cell 不少于 `5,000`；
- 全零 LiDAR 样本比例不高于 `1%`；
- manifest 记录乘数、非零 cell 均值/比例和全零样本数。

任一门槛失败时整个运行 `pipeline_invalid`，不得生成 RGB/LiDAR 依赖结论。

## 4. 干预、指标与判定

沿用 v1：`normal`、四路 `rgb_mean_fill`、`11×11 rgb_blur`、batch 内四视图一致的 `rgb_shuffle`、`lidar_zero`。对每个 condition 同时计算真值任务指标及相对正常输出的 traffic、waypoint、融合 feature 和分类 head 配对变化。

- weak：强 RGB 干预相对 LiDAR 的最大输出比 `<0.25`，且 waypoint ADE 恶化 `<5%`；
- material：任一输出比 `≥0.50`，或 waypoint ADE 恶化 `≥10%`；
- 其余为 mixed。

交通 occupancy 与分类 head 必须单列，不能用 waypoint 一项代表全部视觉作用。validation 离线结果不得外推为闭环 Driving Score。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
