# M1 交通语义预训练数据规范 v1.0

| 字段 | 内容 |
| --- | --- |
| 状态 | **PILOT-READY：索引分层抽样已通过，split 与人工对齐复核待冻结** |
| 生效日期 | 2026-07-22 |
| 服务假设 | H1：交通域 ResNet-50 预训练改善语义质量、时序稳定性与闭环表现 |
| 类别配置 | `configs/thesis/semantic_classes_v1.json` |
| 审计入口 | `tools/data/audit_semantic_pretraining_data.py` |

## 1. 数据单元

一个逻辑帧由同一 route sequence、同一 frame ID 下的相机 RGB 与原始 CARLA uint8 语义标签组成。v1 使用 `front、left、right` 三个相机；rear 没有对应语义传感器，不进入首轮分割预训练。

训练 RGB 必须是未烧入 overlay、文字、边界或人工标记的原始图像。语义标签来自 CARLA 0.9.16 `CityObjectLabel`，标签变换只允许通过版本化类别配置完成。

## 2. v1 类别

| train_id | 类别 | CARLA source tags | 作用 |
| ---: | --- | --- | --- |
| 0 | background | NONE、Buildings、Poles、Vegetation、Terrain、Sky、Static、Dynamic、Other、Water、Ground、Bridge、RailTrack | 保留完整像素监督但不作为论文重点类别 |
| 1 | road | Roads | 可行驶区域上下文 |
| 2 | sidewalk | Sidewalks | 道路边缘与行人活动区域 |
| 3 | road_line | RoadLines | 道路结构与车道边界 |
| 4 | vehicle | Car、Truck、Bus、Train | 动态交通主体 |
| 5 | pedestrian | Pedestrians | 关键弱势交通参与者 |
| 6 | rider | Rider、Motorcycle、Bicycle | 骑行者及其载具 |
| 7 | traffic_light | TrafficLight | 信号设施，不区分是否控制本车 |
| 8 | traffic_sign | TrafficSigns | 道路交通设施 |
| 9 | barrier | Walls、Fences、GuardRail | 道路边界与防护结构 |

`255/Any` 是 ignore 标签。所有 0-28 源标签必须且只能映射到一个训练类别；新增或合并类别必须升级配置版本。

## 3. 当前数据审计结论

审计对象：

`data/traffic_element_small_batch_leaderboard/20260716_leaderboard_stop_targets_camera_fix_accepted`

当前只有 Town03 与 Town04 两个 route sequence，共 523 个逻辑帧、1,569 张语义图、188,280,000 个像素。关键原始标签统计：

| 标签 | 像素 | 占比 | 出现 mask 数 |
| --- | ---: | ---: | ---: |
| TrafficLight | 267,513 | 0.1421% | 1,126 |
| TrafficSigns | 9,010 | 0.0048% | 299 |
| Car | 27,889 | 0.0148% | 436 |
| Truck | 572 | 0.0003% | 211 |
| Pedestrians | 0 | 0 | 0 |
| Rider | 10 | 0.000005% | 4 |
| Bus / Motorcycle / Bicycle | 0 | 0 | 0 |

结论：该批数据是停车点附近的路口标签验证集，交通灯占比很高，但车辆像素极少且完全缺少行人、摩托车和自行车。它不能直接承担交通语义 ResNet-50 预训练，只能作为采集管线 smoke test 与路口样本来源。

### 3.1 Dataset index 分层 pilot

对 `/data1/shijj/interfuser_data/dataset_index.txt` 使用审计器提交 `0fac3e6`，以 sample seed `20260722` 按 Town×weather stratum 最多抽 3 个完整 sequence，固定 `front、left、right` 三相机并启用 `--require-ready`。报告位于：

`results/thesis_m1/semantic_index_pilot_town_weather_n3_seed20260722_20260722T0847Z.json`

冻结 provenance：dataset index SHA-256 为 `56c83f46a1010ee43021bbc2f97cafde9b3c2771522088ae938a2df2ea477ff1`，类别配置 SHA-256 为 `796222592efb68407a32bfdf9a03907b4631cf3f30d0d6a81528010e50f0b612`，报告 SHA-256 为 `16162c5167cbe994c8e22ca5224069f913989c6ca583dbec234ce18285de2614`。

该 pilot 从 9,968 个 index sequence 中选择 176 个，覆盖 60 个分层、4 个 Town、3,619 个逻辑帧与 10,857 张语义 mask；报告为 `valid=true`、`ready=true`、结构错误 0，全部核心类别通过 qualified-mask 与 sequence 覆盖门槛。Town04 weather 15 和 weather 20 各仅有 1 个候选，因此审计器按 `min(3, available)` 全部纳入，最终数量不是理论值 180；这是冻结的数据覆盖边界，不得隐去。

本次结果证明可复现抽样达到 pilot 准入，不代表已全量扫描 9,968 个 sequence，也不替代 sequence 级 split manifest 与人工 RGB/mask 对齐复核。

## 4. Pilot readiness 门槛

机器配置定义的门槛不是最终训练规模，而是“值得启动第一次预训练”的最低条件：

- 至少 6 个独立 route sequence、3 个 Town、2,000 个逻辑帧；
- RGB 与 semantic mask 在每个相机、每个 frame ID 上完全对齐；
- 不允许出现未映射的原始标签；
- 每个核心类别达到最小有效 mask 数，并覆盖多个 sequence；
- 有效 mask 只有在该类别像素超过类别配置的阈值时才计数，避免把 1-2 个噪声像素当成类别覆盖。

审计命令默认输出事实，即使未达到门槛也返回报告；加入 `--require-ready` 后，未达到门槛必须以非零状态退出。

## 5. 采样结构

后续采集必须由四个互补桶组成：

1. **普通道路**：车辆与道路结构为主，不能要求附近存在停止边界；
2. **弱势参与者**：显式增加行人、骑行者、摩托车和自行车可见帧；
3. **信号路口**：使用停止边界作为采样锚点，覆盖红、黄、绿和不同可见距离；
4. **hard negatives**：图中有无关交通灯、远距离设施、遮挡或无交通要素的场景。

采样器不得以 `relevant_to_ego` 作为视觉预训练的必要标签。交通灯状态可以保留为后续 ROI 辅助任务，但状态头不阻塞语义分割主任务。

## 6. 数据划分与泄漏防护

禁止按单帧随机划分。最小划分单元是完整 route sequence；同一连续序列的所有相机和帧必须属于同一个 split。

pilot 阶段先收集至少三个 Town，再根据实际类别覆盖冻结 train/validation/test Town 与 route。正式划分 manifest 必须记录：

- sequence 相对路径、Town、route ID、天气与采集版本；
- 逻辑帧数、三相机 mask 数和各训练类别有效 mask 数；
- split 归属与分配理由；
- RGB、mask 和 manifest 的完整性校验值。

在类别覆盖统计出来前不凭空固定 Town 划分；但一旦开始报告验证指标，split 不得随结果调整。

## 7. 训练接口边界

v1 主任务是 10 类交通语义分割。分割 head 只服务预训练，训练结束后丢弃；迁移到 InterFuser 的唯一产物是与其视觉分支同构的 ResNet-50 骨干权重。

第一次下游消融只允许改变视觉骨干初始化。冻结层数、下游学习率和预训练损失权重需写入实验配置，但不能同时更换规划器、控制器或点云分支。

## 8. M1 完成门槛

M1 只有满足以下条件才能完成：

1. 类别配置通过 schema 校验且覆盖全部 CARLA 标签；
2. 数据审计器有单元测试并能稳定扫描真实数据；
3. pilot 数据通过 `--require-ready`；
4. split manifest 证明没有 sequence 级泄漏；
5. 每个核心类别至少人工检查一组 RGB/mask 对齐样本；
6. 固化数据版本、审计 JSON 和采集配置后，才能进入 M2 预训练。

当前下一步是基于完整 sequence 冻结无泄漏 split manifest，并为每个核心类别人工复核一组 RGB/mask 对齐样本；若后续需要补齐 Town04 weather 15/20，应进行最小定向补采，而不是盲目全量扫描或扩大停车点标签复杂度。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
