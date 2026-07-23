# M1 交通语义预训练数据规范 v1.0

| 字段 | 内容 |
| --- | --- |
| 状态 | **FROZEN：数据 v1、无泄漏 split 与人工对齐证据已冻结，允许进入 M2** |
| 生效日期 | 2026-07-23 |
| 服务假设 | H1：交通域 ResNet-50 预训练改善语义质量、时序稳定性与闭环表现 |
| 类别配置 | `configs/thesis/semantic_classes_v1.json` |
| 划分配置 | `configs/thesis/semantic_split_v1.json` |
| 审计入口 | `tools/data/audit_semantic_pretraining_data.py` |
| 划分入口 | `tools/data/build_semantic_split_manifest.py` |

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

本次结果证明可复现抽样达到 pilot 准入，不代表已全量扫描 9,968 个 sequence。该冻结抽样随后作为数据 v1 的明确边界进入 route-group split 与人工 RGB/mask 对齐复核。

### 3.2 数据 v1 split 与对齐复核

split manifest 位于：

`results/thesis_m1/semantic_split_v1_town_route_seed20260723_20260723T020939Z.json`

它消费 3.1 节 pilot 的完整 176 个 sequence，以 assignment seed `20260723` 按 `Town + route_id` 原子分组；同一路线在不同天气下的 sequence 不得跨 split。153 个 route group 被固定分为 train/validation/test 的 `102/25/26` 组，对应 `123/26/27` 个 sequence 与 `2,530/575/514` 个逻辑帧。三组 sequence 比例为 `69.8864%/14.7727%/15.3409%`，四个 Town 均在每组出现。

每组都至少覆盖一个符合像素门槛的核心类别 sequence；其中稀缺的 pedestrian 在 train/validation/test 分别覆盖 `8/2/2` 个 sequence，避免随机比例划分造成 test 类别空洞。manifest 对每个 sequence 记录采集目录、Town、route、天气、三相机帧数、各类别像素与 qualified-mask 数，并分别计算 RGB/semantic 内容 SHA-256。泄漏检查为 sequence overlap `0`、route-group overlap `0`、全部选中 sequence 恰好分配一次。

冻结哈希：split 配置 SHA-256 为 `fe0641ed9979bd723d30c01d78aaf657573c610b7924dac7dbbf680cd0123936`，manifest SHA-256 为 `81d9403a5ecd39fb5c4cf7ac8ffd3c79505a9583d683a960cf5a02b6da1fe8f2`；相同输入复算得到字节一致 JSON。

人工复核报告位于 `results/thesis_m1/semantic_alignment_review_20260723T021026Z.json`，SHA-256 为 `d778f48894c8de122a36a86f662728297a06931f91d78efa4975be9dee3b7ee5`。审阅者逐一比较 road、sidewalk、road_line、vehicle、pedestrian、rider、traffic_light、traffic_sign、barrier 的 RGB、全语义着色和单类高亮三联图，9/9 均为 `accepted`；未发现相机错位、帧错配、整体偏移或类别反转。

### 3.3 M2 训练反馈与数据充分性边界

无类别权重的五轮全量 probe 使 road_line 从零 IoU 提升到 `0.230667`，但 pedestrian、rider、traffic_light、traffic_sign 仍为 0，证明优化预算与像素失衡是两个独立问题。保持数据与预算不变、仅引入 inverse-sqrt frequency 权重后，best validation mIoU 从 `0.395259` 提升到 `0.463243`，上述四类 IoU 分别为 `0.056668/0.258753/0.156095/0.176434`。

因此当前 176 个 sequence、3,619 个逻辑帧的 M1 数据 v1 已足以训练出覆盖十类的 M2 v1 骨干候选，不启动新一轮补采。这不等于数据已足以支撑最终 H1：pedestrian 仅有 12 个 qualified sequence，validation/test 各 2 个，仍然是高方差边界。只有下游 V 组、多 seed 或错误分层显示该类不稳定时，才按第 5 节弱势参与者桶做最小定向补采；停止边界不是补采必要条件。

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

禁止按单帧随机划分。最小划分单元是完整 route sequence；数据 v1 进一步以 `Town + route_id` 为原子组，使同一路线不同天气下的所有 sequence、相机和帧属于同一个 split。

pilot 阶段先收集至少三个 Town，再根据实际类别覆盖冻结 train/validation/test Town 与 route。正式划分 manifest 必须记录：

- sequence 相对路径、Town、route ID、天气与采集版本；
- 逻辑帧数、三相机 mask 数和各训练类别有效 mask 数；
- split 归属与分配理由；
- RGB、mask 和 manifest 的完整性校验值。

数据 v1 不按 Town 整体留出，因为只有四个 Town 且 pedestrian 仅覆盖 12 个 sequence；强行 Town holdout 会让类别可检验性服从地图数量。划分器先为 train/validation/test 分别播种全部核心类别，再确定性平衡 route group、sequence、逻辑帧和类别覆盖；一旦开始报告验证指标，当前 manifest 不得随结果调整。

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

上述六项门槛已全部满足，M1 数据 v1 冻结完成。下一步进入 M2：先冻结 10 类语义分割的训练配置、ResNet-50 初始化来源、训练预算与离线评价口径，再运行最小可复现预训练。Town04 weather 15/20 的候选不足仍是已知覆盖边界；只有训练或分层评价证明其形成实际缺口时才最小定向补采，不盲目全量扫描，也不扩大停车点标签复杂度。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
