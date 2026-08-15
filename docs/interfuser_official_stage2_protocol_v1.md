# M2 官方 InterFuser 视觉迁移 Stage 2 协同适配协议 v1.0

| 字段 | 内容 |
| --- | --- |
| 状态 | **PILOT-FROZEN：数据契约与三轮配对预算已冻结，等待运行结果** |
| 服务假设 | H1：交通域 ResNet-50 经下游协同适配后优于官方原视觉初始化 |
| 初始化 | 作者发布 official B0 与仅替换 RGB backbone 的 official V |
| 配置 | `configs/thesis/interfuser_official_stage2_pilot_v1.json` |
| 编排器 | `tools/training/run_interfuser_visual_pair.py` |

## 1. 为什么必须有 Stage 2

零微调替换只证明新 RGB backbone 能 strict-load，不能证明原 Transformer、投影层和任务头理解新的特征分布。此前因果消融已证明 official B0 实质依赖 RGB；official V 零微调却使 traffic AP 与 junction macro-F1 明显下降，说明问题是下游表征没有协同适配，而不是视觉分支无效。

本 Stage 2 从同一个作者发布全模型状态出发，对 B0 与 V 执行完全相同的端到端短微调。两组唯一初始差异仍是 330 个 RGB backbone 张量；LiDAR backbone、Transformer、任务头、数据、增强、优化器、学习率、batch、epoch 和 seed 全部相同。

## 2. 训练前数据契约

CARLA 0.9.16 数据不得沿用未经声明的上游坐标假设。pilot 固定：

- 保存的 LiDAR 已处于直方图接收的负 y 半平面，使用 `lidar_y_axis_multiplier=+1.0`；
- `measurements.theta` 是 CARLA compass，target point、future waypoint 和 trajectory heatmap 共用 `R(theta)^T`；
- 缺少 `command/x_command/y_command/future_waypoints/theta` 的帧确定性剔除，不再用 LANE_FOLLOW 与 ego position 伪造监督；
- train 从 176,896 帧保留 171,256 帧，validation 从 4,613 帧保留 4,389 帧，实际数量漂移立即终止训练。

连续轨迹审计给出坐标自证：`R(theta)^T` 在 validation 上使 98.33% 的未来轨迹处于模型约定的前向负 y，横向绝对值中位数 0.105 m；旧 `R(pi/2-theta)^T` 会把前进方向旋到 x 轴。此前使用错误 target/waypoint 坐标得到的 waypoint 真值指标不再作为 Stage 2 证据。

## 3. Pilot 单变量预算

执行顺序固定为 B0 后 V，使用 GPU1 单卡串行训练。每卡 batch 32 保持既有两卡 `16×2` 的全局 batch 32；训练 3 epochs、warmup 1 epoch，AdamW/cosine、主学习率 `5e-4`、RGB backbone 学习率 `2e-4`、weight decay `0.05`、scale `0.9..1.1`、color jitter `0.1`。`train.py` 按全局 batch 做线性学习率缩放，因此与原配方的有效学习率一致。

GPU 使用 `shared_capacity` 门禁：启动前 GPU1 至少保留 24,000 MiB 空闲；清理只等待本 runner 进程组拥有的 CUDA PID，不把外部低占用 context 当作失败。

pilot 采用全模型端到端协同适配，不在首轮增加额外网络或两套不同解冻策略。若 V 出现明显震荡而 B0 正常，才以验证证据准入“短对齐 warmup → 全量解冻”的下一版，不能事后只给 V 特殊预算。

## 4. 完成与后续准入

pilot 完成必须满足：

1. B0/V 都生成 3 行有限 summary、best/last checkpoint，state schema 与各自初始化完全一致；
2. 归一化训练参数哈希相同，唯一差异是 initial checkpoint；
3. 两组有效样本数、坐标参数和资源策略写入 args/manifest；
4. 先比较验证 loss、`l1_error` 和收敛曲线，再用修正坐标的独立离线 evaluator 比较 traffic、waypoint 与分类头；
5. 只有 V 出现可解释的验证信号，才进行 route39 三 seed 连续控制帧复核；通过后才扩展 25 epoch 与完整 D7。

三轮 pilot 只回答“Stage 2 是否值得扩大预算”，不直接形成 H1 最终结论。不得用训练 loss 代替冻结 test 或闭环 Driving Score。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
