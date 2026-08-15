# M2 官方 B0/V Stage 2 数据契约预检记录

| 字段 | 内容 |
| --- | --- |
| 日期 | 2026-08-15 |
| 状态 | **PREFLIGHT-VALID：允许提交代码并启动三轮配对 pilot** |
| 前序 Git | `6a06f1080bc2ebb423b2f09b43f42db0282ef404` |
| Pilot 配置 | `configs/thesis/interfuser_official_stage2_pilot_v1.json` |
| 配置 SHA-256 | `726c77967177680e65e939be9f7286a8b40fba4181a04c65868ab670e8b12f01` |

## 1. 发现的训练数据根因

此前自训练链路同时存在两项坐标错误，不能用于判断交通域视觉预训练是否有效：

1. 数据文件中的 LiDAR y 已主要为负值，训练代码再次乘 `-1` 后绝大多数点落出 `y∈[-28,0]` 的直方图接收区；
2. CARLA 0.9.16 采集数据的 `measurements.theta` 是 compass，现有 `R(pi/2-theta)^T` 把未来轨迹的前进方向旋到模型 x 轴，而官方 InterFuser waypoint/controller 契约要求前向为负 y。

这解释了“自己训练的模型闭环很差”为什么不能归因于 InterFuser 架构或视觉预训练无效：下游训练监督本身没有与官方推理坐标同构。

## 2. 连续轨迹与字段审计

对冻结 split 的全部 187,295 帧 measurement JSON 做确定性扫描。使用 `R(theta)^T` 后：

- train：96.89% 的未来点位于前向负 y，横向绝对值中位数 0.077 m；
- validation：98.33%，横向绝对值中位数 0.105 m；
- test：97.93%，横向绝对值中位数 0.084 m。

缺少 `command` 与 `x_command/y_command` 的帧分别为 train 5,640、validation 224、test 279；两类缺失在本批数据中重合。Stage 2 不再用默认 LANE_FOLLOW 与 ego position 构造伪标签，而是按相同规则从 B0/V 两组剔除。

有效样本门禁固定为 train `171,256`、validation `4,389`。训练入口在 dataset 初始化后逐 split 比对实际长度，任何数据漂移立即失败。

## 3. 真实样本自证

使用 pilot 的完整 dataset 参数构造 train/validation，并通过正式 loader 读取首个 validation 样本：

- train：171,256 retained / 5,640 dropped；
- validation：4,389 retained / 224 dropped；
- LiDAR histogram 非零栅格：10,937；
- target point：`[0.0248, -6.0823]`；
- horizon-2 waypoint：`[0.0248, -4.0826]`。

点云密度和导航几何同时符合官方模型输入方向。相关定向测试 14/14、完整测试 262/262 通过。

## 4. 结论边界

此前 v2 模态审计仍可证明“在同一错误导航条件下改变 RGB 会显著改变模型输出”，route39 的在线 RGB mean-fill 退化也仍是有效闭环因果证据；但 v2 离线 waypoint 真值指标及其 traffic/task 性能不得作为 Stage 2 选模依据。

本记录只证明新 pilot 的数据与执行链路已准入，不证明 V 优于 B0。三轮结果完成后必须先比较配对收敛，再用同一修正坐标的独立 evaluator 和 route39 连续控制帧决定是否扩大预算。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
