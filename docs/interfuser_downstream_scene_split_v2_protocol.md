<!--
[INPUT]: 依赖冻结 M1 语义 split、全量 InterFuser dataset_index、CARLA measurements 行人真值、官方 B0/V 初始化与 Stage 2 配对训练协议。
[OUTPUT]: 对外提供下游 scene-stratified v2 的无泄漏选择、结构验收、重训和冻结评价门禁。
[POS]: docs 的 M2 H1 数据测量修复协议；只修复 holdout 对稀缺行人事件的可检验性，不把 split 变化冒充模型改进。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
-->

# InterFuser 下游行人分层 Split v2 协议

| 字段 | 内容 |
| --- | --- |
| 状态 | **FROZEN：选择规则和验收阈值在读取 B0/V 新结果前固定** |
| 服务假设 | H1：交通域视觉预训练改善关键交通要素的识别质量与连续帧稳定性 |
| 配置 | `configs/thesis/interfuser_downstream_split_v2.json` |
| 生成器 | `tools/data/build_interfuser_downstream_indexes.py` |
| 替代范围 | 只替代 Stage 2 后续重训使用的 downstream split；v1 产物永久保留 |

## 1. 修复对象

v1 将 M1 小样本 split 未覆盖的 route group 全部送入 train。全量数据实际含 61 个带有效行人真值的 Town+route 组，其中 59/1/1 分别落入 train/validation/test；验证与测试各只有一个独立行人道路几何，无法支撑行人场景比较。

v2 修复的是测量能力，不是模型能力。任何模型分数都不得参与 route group 选择；v1 与 v2 的 validation/test 数值不能直接横向解释为模型进步。

## 2. 真值和原子单元

- split 原子仍为完整 `Town+route`，同一路线的所有天气和连续帧只能进入同一 split；
- 行人场景只由 `measurements.is_pedestrian_present` 非空判定，不读取 RGB、模型预测或后验驾驶结果；
- 只有同时具备 `command/x_command/y_command/future_waypoints/theta` 的帧进入有效 census，与 Stage 2 的 `missing_navigation_policy=drop` 同构；
- 数据 index、M1 manifest、scene census 和最终三个 index 都以 SHA-256 固定。

## 3. 预训练泄漏边界

M1 semantic manifest 已分配的 route group 保持原归属：semantic train 永远留在 downstream train，semantic validation/test 保持对应 holdout。v2 只能从 M1 未选择过、且包含有效行人帧的 route group 中补充 validation/test。

生成后必须同时满足：Town+route 交集为零、semantic-train 与两个 holdout 交集为零、每个源 sequence 恰好分配一次。任一条件失败即拒绝生成。

## 4. 冻结选择规则

全量 census 固定为 181,152 个有效帧、6,143 个剔除帧、10,988 个行人帧、675 个行人 sequence、61 个行人 route group。目标行人 route group 数固定为 train/validation/test=`45/8/8`。

选择器使用种子 `20260816`：先保证两个 holdout 都覆盖 Town01/Town03/Town04/Town05，再按目标行人帧份额确定性补足组数；哈希只用于同分决胜。所有剩余未分配 route group 进入 train。

两个 holdout 各自还必须满足：至少 1,000 个行人帧、60 个行人 sequence、12 种行人天气；两组行人帧数之比不得超过 1.5。

## 5. Split 结构验收

生成 v2 后、启动训练前必须完成：

1. 单元测试覆盖 v1 兼容、v2 确定性、census 漂移拒绝和 semantic-train 隔离；
2. 连续运行生成器两次得到相同 index/manifest 哈希；
3. 三个 index 的声明帧和 loader 有效帧与 manifest 一致；
4. validation/test 的行人 route group、Town、天气、帧和 sequence 均通过冻结阈值；
5. 抽取每个 holdout 至少一个连续行人 sequence 做源 JSON 与 RGB/语义对齐复核；视觉复核只验证标签可信度，不改变选择结果。

通过以上门禁只能声明“评价集合修复有效”，不能声明模型改善。

## 6. 模型验证

当前 B0/V checkpoint 已训练见过 v2 新增 holdout 中的大部分 route group，因此禁止复用。必须从官方 B0 与 official V 初始化重新开始同预算配对 Stage 2，且除初始化 checkpoint 外训练参数完全相同。

验证分两层：

- validation 用于判断三轮适配是否存在可解释信号；报告整体 traffic AP/AUC/IoU、waypoint ADE/FDE，以及行人场景分层指标；
- test 在训练和 checkpoint 选择完成后一次性解封；以 route group 为统计单元报告 B0/V 配对差和 bootstrap 区间，不把连续帧当成独立样本。

若整体指标与行人分层指标方向冲突，结论记为混合；若行人分层仍无收益，不得用新增样本量替预训练效果辩护。只有离线信号通过后才准入 route39 与 D7。

## 7. 停止条件

- scene census 与冻结总数不同；
- semantic train route group 进入任一 holdout；
- holdout 少于 8 个行人 route group 或缺少任一冻结 Town；
- B0/V 除初始化外存在配置差异；
- test 在训练完成前被读取。

任一条件触发时停止模型实验，修复数据或编排基础设施，不降低阈值继续运行。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
