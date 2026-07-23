# M1 交通语义 split 与人工对齐冻结记录

| 字段 | 内容 |
| --- | --- |
| 工具提交 | 本记录同提交引入 split 构建器、复核渲染器与回归测试 |
| Pilot 来源 | `results/thesis_m1/semantic_index_pilot_town_weather_n3_seed20260722_20260722T0847Z.json` |
| Pilot SHA-256 | `16162c5167cbe994c8e22ca5224069f913989c6ca583dbec234ce18285de2614` |
| Split 配置 | `configs/thesis/semantic_split_v1.json` |
| Split 配置 SHA-256 | `fe0641ed9979bd723d30c01d78aaf657573c610b7924dac7dbbf680cd0123936` |
| Split manifest | `results/thesis_m1/semantic_split_v1_town_route_seed20260723_20260723T020939Z.json` |
| Split manifest SHA-256 | `81d9403a5ecd39fb5c4cf7ac8ffd3c79505a9583d683a960cf5a02b6da1fe8f2` |
| 人工复核报告 | `results/thesis_m1/semantic_alignment_review_20260723T021026Z.json` |
| 人工复核 SHA-256 | `d778f48894c8de122a36a86f662728297a06931f91d78efa4975be9dee3b7ee5` |
| 结论 | **M1 数据 v1 已冻结：无 route-group 泄漏，九类人工对齐 9/9 通过，允许进入 M2** |

## 1. 划分原则与结果

数据 v1 严格消费已冻结 pilot 的 176 个 sequence、3,619 个逻辑帧和 `front/left/right` 三相机，不扩大到 dataset index 的 9,968 个 sequence。分配种子固定为 `20260723`，最小分配单元为 `Town + route_id`；同一路线即使在多个天气下出现也只能进入一个 split，防止相同道路几何跨训练与评价集合。

| Split | Route groups | Sequences | 逻辑帧 | Sequence 比例 | Town 数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 102 | 123 | 2,530 | 69.8864% | 4 |
| validation | 25 | 26 | 575 | 14.7727% | 4 |
| test | 26 | 27 | 514 | 15.3409% | 4 |

manifest 报告 `valid=true`、结构错误 0、sequence overlap 0、route-group overlap 0，且 176 个选中 sequence 均恰好分配一次。每个 sequence 都记录采集目录、Town、route ID、天气、三相机帧数、类别像素、qualified-mask 数和独立的 RGB/semantic 内容 SHA-256。使用相同输入复算后与正式 manifest 字节一致。

新增 split/review 与既有 semantic audit 定向回归为 `13/13` 通过；加入仓库、InterFuser、Leaderboard、Scenario Runner 与 CARLA PythonAPI 的完整 `PYTHONPATH` 后，全量 unittest 为 `163/163` 通过。

## 2. 三组核心类别覆盖

下表统计每个 split 中至少一张 mask 达到类别像素门槛的 sequence 数。划分器先以稀缺类别播种每个 split，再确定性平衡 route group、sequence、逻辑帧与类别分布；这避免了朴素比例算法曾出现的 test pedestrian=0。

| 类别 | train | validation | test |
| --- | ---: | ---: | ---: |
| road | 123 | 26 | 27 |
| sidewalk | 105 | 21 | 22 |
| road_line | 123 | 26 | 27 |
| vehicle | 123 | 26 | 27 |
| pedestrian | 8 | 2 | 2 |
| rider | 94 | 17 | 17 |
| traffic_light | 103 | 20 | 21 |
| traffic_sign | 89 | 14 | 14 |
| barrier | 123 | 26 | 27 |

## 3. 人工 RGB/mask 对齐复核

复核器对每个核心类别选择像素最多的合格候选，生成 RGB、10 类语义着色、目标类别红色高亮三联图。人工逐图检查实体轮廓、道路边界、相机视角和 frame ID，结论如下：

| 类别 | Split / camera / frame | 可见证据 | 结论 |
| --- | --- | --- | --- |
| road | train / front / 0008 | 高亮跟随可行驶路面并排除车辆与路侧实体 | accepted |
| sidewalk | train / right / 0012 | 高亮跟随抬高的人行区域并保持道路边界 | accepted |
| road_line | train / front / 0012 | 车道线与斑马线轮廓均与 RGB 一致 | accepted |
| vehicle | train / left / 0005 | 公交车车身与车轮区域轮廓一致 | accepted |
| pedestrian | train / front / 0100 | 两名可见行人的轮廓与 RGB 一致 | accepted |
| rider | test / left / 0008 | 夜雨场景中的摩托车与骑行者轮廓一致 | accepted |
| traffic_light | train / right / 0007 | 右侧近距离信号灯外壳轮廓一致 | accepted |
| traffic_sign | validation / front / 0002 | 限速标志牌面与立杆轮廓一致 | accepted |
| barrier | validation / right / 0019 | 路侧墙体高亮与人行道分界一致 | accepted |

九项均未发现整图偏移、相机串线、frame ID 错配或类别反转。原始 RGB、semantic mask、三联图路径及各自 SHA-256 均写入复核报告；三联图只作审计证据，不进入训练数据。

## 4. 证据边界与下一阶段

本记录冻结的是从 dataset index 确定性分层抽取的 pilot 数据 v1，不代表对 9,968 个 sequence 的全量质量背书。Town04 weather 15/20 各只有一个候选的事实继续保留；当前三组核心类别可检验性已经满足，因此不据此盲目补采。

M1 的类别 schema、真实审计、pilot readiness、无泄漏 split、内容哈希与人工对齐证据已经闭合。下一阶段只允许进入 M2 的最小语义分割预训练：先冻结训练配置、初始化来源、预算与离线指标，不同时修改 InterFuser 规划器、控制器或点云分支。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
