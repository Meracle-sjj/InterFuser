# M2 官方 InterFuser 底座视觉骨干替换协议 v1.0

| 字段 | 内容 |
| --- | --- |
| 状态 | **FROZEN-FEASIBILITY：只验证官方可驾驶底座上的视觉替换与零微调闭环准入** |
| 服务假设 | H1：交通域 ResNet-50 预训练能否在保留官方 InterFuser 驾驶能力的前提下改善视觉表征 |
| 配置 | `configs/thesis/interfuser_official_visual_swap_initialization_v1.json` |
| 生成器 | `tools/training/interfuser_visual_swap_pair.py` |

## 1. 问题与边界

历史 M0-FT/M0-V 使用本项目自训 epoch26 checkpoint 作底座。严格闭环 A/B 已证明该底座在 route39 的 600 帧 RC 仅为 `4.3470%`，而作者发布权重在同一 CARLA 0.9.16 兼容条件下为 `61.8863%`。因此旧底座上的视觉替换无法回答“是否保留已有驾驶能力”。

本协议改用作者发布 checkpoint（SHA-256 `39548dcf…e30b2`）作为唯一底座。不新增网络、规划器、控制器或行人专用分支；预训练分割 head 继续丢弃，推理时仍仅迁移同构 RGB ResNet50d 骨干。

## 2. 严格单变量

- `official_b0`：作者发布 checkpoint 的 1,132 个 state tensor 原样保留；
- `official_b0_v`：先 strict load 同一官方 checkpoint，再仅用已完成的十类交通语义 ResNet50d 覆盖 `rgb_backbone`；
- 两组只允许 330 个唯一 RGB tensor（全模型 alias 共 660 key）不同；LiDAR、Transformer、embedding、waypoint/traffic/junction/light/stop heads 和所有非 RGB buffer 必须逐值相同。

官方 checkpoint 元数据保留原始 `arch=mvt_baseline`，而当前等价注册入口为 `interfuser_baseline`。配置必须同时记录两者，不允许为了通过加载而改写原 checkpoint。

## 3. 准入顺序

1. 生成器证明官方 state strict load、对照分支全状态保真、视觉分支只改 RGB alias，且两个导出 checkpoint 可再次 strict load。
2. 用相同输入做离线前向 smoke，拒绝 shape 漂移、非有限输出或对照分支与原官方状态不一致。
3. 复用已验证的 CARLA 0.9.16 官方 agent/controller、GNSS 第 0 轴取反和正确 Lincoln 蓝图，在 route39/seed0 运行 600 帧零微调诊断。
4. 若 `official_b0_v` 相对 `official_b0` 出现起步早期失控或 RC 灾难性下降，则判定为特征分布失配，停止 D7，转入相同预算的短适配。
5. 只有 route39/seed0 不灾难退化时，才扩展 route39 三 seed；三 seed 通过后才允许完整 D7。

## 4. 短适配边界

短适配不是重训官方 InterFuser。首轮冻结 LiDAR、Transformer、任务 heads 和控制器，只允许视觉骨干末端与现有 RGB 投影/融合接口使用低学习率恢复对齐。对照组必须使用完全相同的数据、seed、optimizer、epoch 和 checkpoint 选择口径。

若零微调已保持闭环能力，不得为追求更好单路线分数而事后启动微调。

## 5. 结论边界

本阶段只回答“交通域视觉骨干能否接入作者发布的可驾驶 InterFuser 而不破坏基础闭环”。单路线不得宣称 Driving Score 改善，语义离线改善也不得代替多路线闭环证据。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
