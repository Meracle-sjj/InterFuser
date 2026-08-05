# M2 固定 M0 底座视觉骨干替换协议 v1.0

| 字段 | 内容 |
| --- | --- |
| 状态 | **FROZEN-FEASIBILITY：只准生成与验证初始化 checkpoint，不直接形成 H1 效果结论** |
| 服务假设 | H1：交通域 ResNet-50 能否作为可插拔视觉骨干改善已经具备驾驶能力的 InterFuser |
| 配置 | `configs/thesis/interfuser_visual_swap_initialization_v1.json` |
| 生成器 | `tools/training/interfuser_visual_swap_pair.py` |

## 1. 研究边界

本协议补充而不改写 `interfuser_visual_transfer_protocol_v1.md`。旧 B0/V 实验回答“不同 RGB 初始化是否影响从头下游训练”；本协议回答“在同一个已训练 InterFuser 底座上替换 RGB 骨干是否可行”。两种实验不得共用 B0 名称或混写结论。

固定底座是服务器 2026 年 1 月训练得到的 `interfuser_baseline` epoch 26 checkpoint。它是本项目历史模型，不是 InterFuser 作者发布的官方权重。配置必须冻结 checkpoint 路径、SHA-256、架构与 epoch，并在生成前 strict load 到当前模型定义。

## 2. 配对变量

- **M0-FT**：完整继承固定 M0 的 1,132 个 state tensor，RGB 不做替换；
- **M0-V**：先完整继承同一 M0，再用行人碰撞威胁增强语义预训练的同构 ResNet50d strict 替换 `rgb_backbone`；
- 两者初始化时只有 330 个唯一 RGB tensor 不同；由于 `rgb_patch_embed.backbone` 是同一模块的 state alias，全模型表现为 660 个 RGB key 不同；
- LiDAR backbone、Transformer、位置/视角 embedding、waypoint/traffic/junction/light/stop heads 与所有非 RGB buffer 必须逐值相同。

语义分割 decoder 和训练特权标签不进入 InterFuser；推理输入仍只有多视角 RGB 与 LiDAR，模型结构、输出和控制器不变。

## 3. feasibility 完成门槛

生成器必须同时证明：

1. 当前 `interfuser_baseline` 对固定 M0 state 执行 strict load 成功；
2. M0-FT 全状态哈希与固定 M0 checkpoint 的 `state_dict` 相同；
3. 交通域导出只包含 `backbone.*`，并对 330 个 RGB tensor strict load；
4. M0-FT/M0-V 只有两个合法 RGB prefix 下的 660 个 alias key 变化；
5. 非 RGB state 哈希完全相同；
6. 两个导出的全模型 checkpoint 都能 strict load 回当前模型；
7. 产物 manifest 固化配置、Git、来源 checkpoint 与状态哈希。

达到这些门槛只说明“视觉骨干已经按预期接入同一 M0”，不说明零微调或微调后的驾驶效果改善。

## 4. 后续准入顺序

1. 对 M0-FT/M0-V 做相同输入的前向 smoke，拒绝缺失、shape 漂移和非有限输出；
2. 零微调 route39 单 seed 只作为特征分布失配诊断，不写入正式效果表；
3. 若 M0-V 出现起步即失控，先采用冻结 RGB/低学习率的短微调恢复，不直接进入 D7；
4. 正式微调必须为 M0-FT/M0-V 使用完全相同的数据、seed、optimizer、epoch、增强和 checkpoint 选择口径；
5. route39 三 seed 连续控制帧通过后，才允许生成完整 D7 配置；
6. 正式结论同时报告离线任务、连续帧稳定性与闭环 DS/RC/IS，不能只报告最好路线。

## 5. 与论文矩阵的关系

本协议产物是 H1 的低成本可行性证据。若固定底座微调恢复并稳定优于 M0-FT，可将其作为 V 组实现；若只在从头训练配方有效，则保留旧 B0/V 定义并如实区分两种迁移范式。任何一种进入最终四组消融前，都必须先解决本地基线闭环复现问题。

## 6. route39 seed0 零微调诊断

首个闭环准入固定为 M0-FT 后 M0-V，分别使用：

- `configs/thesis/interfuser_visual_swap_route39_m0_ft_v1.json`，Run ID `m2-interfuser-m0-ft-route39-seed0-zero-ft-20260805-v1`；
- `configs/thesis/interfuser_visual_swap_route39_m0_v_v1.json`，Run ID `m2-interfuser-m0-v-route39-seed0-zero-ft-20260805-v1`。

两次运行都只允许 `route_id=39`、`seed=0`，复用同一 agent、控制器、路线、场景、背景交通、GPU/端口和超时。M0-FT 用于证明新 checkpoint 包装未改变历史底座行为；只有 M0-FT `pipeline_valid` 后才允许启动 M0-V。

本阶段将“灾难性特征失配”预定义为 M0-V 在前 10 个仿真秒发生 layout collision/vehicle blocked，或在前 50 个运动控制帧内首次达到 `abs(lane_offset) >= 1 m`，同时 M0-FT 未发生同类事件。连续控制归约还必须报告前 40 个运动帧的平均绝对转向、最大绝对转向、平均/最大绝对车道偏移、首次 1 m 偏移 step 与首碰撞时间。

若 M0-FT 本身不满足包装校准，停止并修复运行/加载链；若只有 M0-V 命中灾难性失配，进入相同预算的配对短微调；若二者均未命中，则进入 route39 三 seed 稳定性复核。单 seed 结果不得用于宣称视觉收益。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
