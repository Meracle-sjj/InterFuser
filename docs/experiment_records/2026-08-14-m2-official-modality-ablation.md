<!--
[INPUT]: 依赖官方 B0/B0-V validation 模态消融 manifest、CARLA 0.9.16 LiDAR 坐标密度审计及 route39 单 seed 闭环输入置零日志。
[OUTPUT]: 对外提供 RGB/LiDAR 配对效应量、v1 空 LiDAR 失效处置、v2 离线结论与闭环 RGB 依赖证据。
[POS]: experiment_records 的 M2 H1 模态利用率事实记录，约束视觉预训练论断必须区分“模型使用 RGB”与“更换视觉骨干带来有效收益”。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
-->

# M2 官方 InterFuser RGB/LiDAR 模态依赖审计

| 字段 | 内容 |
| --- | --- |
| 日期 | 2026-08-14 |
| v2 运行 Git | `c95f31cc747511a06464196bfe977b5ce3eccc84` |
| v2 配置 SHA-256 | `ed351b806d916e00a3cffad1767bb0cb0e32570bc2d227a0ce8f1995e0a6a2d2` |
| v2 manifest SHA-256 | `cc6a20892c710c72bf5ec5ea92d20ff4fbc7cbadb711571e6261333ed06a393e` |
| 样本 | validation 全量 4,613 帧，Town01/03/04/05 |
| 结论 | **官方模型实质依赖 RGB；但交通语义骨干零微调替换没有把这种依赖转化为更好的官方任务表征** |

## 1. v1 空 LiDAR 失效

v1 完成后得到异常的 RGB/LiDAR waypoint 输出效应比 `658.85×`。输入审计发现原始点云平均每帧 `14,574.02` 个点，但历史 loader 固定执行 `y *= -1` 后，平均只有 `25.51/150,528` 个非零 tensor cell，23/4,613 帧全零。抽样原始点云前向部分位于负 y，而 histogram 只接受 `y∈[-28,0]`；重复反号把绝大多数点裁掉。

v1 manifest SHA-256 为 `3f85dc58354ef9845cbff8ac8a4235f7927eb9226336d4c91fd385d87a166f02`。该 run 保留为坐标链故障证据，不进入模态依赖结论，也不覆盖。

## 2. v2 LiDAR 准入

v2 只把当前 CARLA 0.9.16 采集数据的 `lidar_y_axis_multiplier` 显式设为 `+1.0`，其余 validation、checkpoint、condition、batch、seed 和判定阈值保持不变。修正后两组输入密度相同：

- 平均非零 cell：`11,388.04/150,528`；
- 非零比例：`7.5654%`；
- 全零样本：`0/4,613`。

密度门槛通过。`CarlaMVDetDataset` 的默认乘数仍为历史 `-1.0`，避免静默改变旧训练；新坐标只由哈希绑定配置显式启用。

## 3. 官方 B0 离线因果结果

| condition | waypoint ADE | traffic AP | traffic IoU | junction macro-F1 | waypoint 输出位移 | traffic 概率变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| normal | 8.2574 | 0.1498 | 0.1030 | 0.9363 | — | — |
| rgb_mean_fill | 8.1342 | 0.0529 | 0.0281 | 0.6179 | 1.0352 m | 0.02386 |
| rgb_blur | 7.9734 | 0.0629 | 0.0503 | 0.6747 | 1.0347 m | 0.02917 |
| rgb_shuffle | 8.2341 | 0.0969 | 0.0660 | 0.8117 | 0.7184 m | 0.01176 |
| lidar_zero | 8.2896 | 0.3404 | 0.1536 | 0.9443 | 0.1751 m | 0.02919 |

强 RGB 干预相对 LiDAR 清空的最大 waypoint 输出效应比为 `5.9135×`，traffic 输出效应比为 `0.8174×`，预注册判定为 `material_rgb_dependency`。RGB 均值填充还使 junction 预测翻转 `29.48%`，traffic AP 相对 normal 下降 `64.67%`。

因此“官方 InterFuser 基本不使用 RGB”被否定。RGB 对 waypoint decoder、traffic grid 与 junction head 都产生实质因果影响。

但 waypoint 真值 ADE 在破坏 RGB 后未恶化，模糊输入反而降低 ADE `3.44%`。这说明当前 validation 上的 RGB 引发大量轨迹变化，却没有形成对应的轨迹精度收益；target point、command/measurement 或数据分布可能主导该指标。

## 4. B0-V 零微调替换

在稠密 LiDAR normal 条件下，`official_b0_v` 相对 `official_b0`：

- waypoint ADE：`8.2574 → 7.9864`，降低 `3.28%`；
- traffic AP：`0.1498 → 0.0621`，下降 `58.56%`；
- junction macro-F1：`0.9363 → 0.5935`，下降 `36.62%`；
- red-light macro-F1：`0.4832 → 0.5449`，提高 `12.77%`。

B0-V 仍判为 `material_rgb_dependency`，但其 RGB/LiDAR waypoint 效应比降至 `2.5098×`，traffic 效应比降至 `0.3958×`。直接替换交通语义 backbone 改变了模态平衡，并严重损失作者任务头需要的 traffic/junction 表征；单独的 waypoint 改善不足以证明视觉迁移有效。

## 5. route39 闭环置零

闭环固定作者干净 agent/controller、官方 B0 checkpoint、GNSS 第 0 轴修正、正确 Lincoln 蓝图、200 辆背景车、route39 seed0 与 600 帧上限，只在模型 forward 前改变一个输入。

| condition | RC@600 | 位移 | 事件/连续帧结论 |
| --- | ---: | ---: | --- |
| normal（既有配对） | 61.8863% | 132.43 m | 车道内稳定推进 |
| lidar_zero | 61.8863% | 132.42 m | 与 normal 等 RC；当前在线 LiDAR 坐标链本就疑似接近空输入 |
| rgb_mean_fill | 32.0518% | 74.21 m | 约 step 300 后右侧收敛，触发 OutsideRouteLanesTest；step 400 后约 0.04 m/s 贴护栏停滞 |

RGB 均值填充连续帧 `280/300/320/400/599` 显示车辆逐渐靠近右侧护栏并停住，不是交通堵塞。LiDAR 置零第一次 0 帧 CARLA SIGSEGV，原始失败保留；换端口重试完成 600 帧，才进入模型比较。

临时闭环包装器 SHA-256：Python `8e6044fe141659badba68252786f60b423fef2f6600943ebb8521c4be833c0f1`，shell `a8fdff44500641314bdd6e707026908195effe107a0e643b8f2107317153ed7e`。RGB/LiDAR evaluator log SHA-256 分别为 `d8673b9693ef2b02827dbc832e5ab65a738a05e91e59c5aefef3c96a087c00f7` / `2d0d01bcd3d13e5fee5f59532487bec46d113a1d2d8a9e877c37b37ac3a9cc71`。

## 6. 结论边界

1. 可以写：官方 InterFuser 对 RGB 有实质依赖，尤其是 traffic occupancy、junction 与 route39 闭环轨迹。
2. 不能写：当前交通语义预训练 backbone 已改善整体驾驶；它在官方任务头的 traffic/junction 离线指标上明显退化。
3. 不能写：官方 InterFuser 天生忽略 LiDAR。当前 CARLA 0.9.16 离线/在线坐标约定与作者训练分布仍未完全闭合；online `lidar_zero` 等效只证明当前迁移栈未有效利用该输入。
4. 下一门槛不是立即扩大 D7，而是先把 live agent 的 LiDAR histogram 密度和朝向记录出来，并以正常/轴修正/置零三条件验证官方 checkpoint 的在线 LiDAR 契约。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
