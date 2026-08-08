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

## 7. route39 seeds 1–2 稳定性续跑

seed0 两组均为 `pipeline_valid`，M0-FT/M0-V 首次达到 1 m 偏移分别为 step 694/686，M0-V 未命中第 6 节的灾难性失配，因此本轮不启动短微调，冻结续跑：

- M0-FT Run ID `m2-interfuser-m0-ft-route39-seeds1-2-zero-ft-20260805-v1`；
- M0-V Run ID `m2-interfuser-m0-v-route39-seeds1-2-zero-ft-20260805-v1`；
- 两组都只运行 `route_id=39`、`seeds=[1,2]`，并与 seed0 结果按 variant 合并为三 seed；
- 顺序固定为 M0-FT seeds 1→2，再 M0-V seeds 1→2；任一 attempt `pipeline_invalid` 立即停止，不以零分代替。

三 seed 归约先逐 seed 报告 DS/RC/IS、前 40 个运动帧统计、首次 1 m 偏移 step、首碰撞与 blocked，再报告 route39 内三个 seed 的均值和范围。若 M0-V 三个 seed 都未出现早期失控，且 DS/RC 差异不呈灾难性一致下降，则进入完整 D7；若出现可重复早期失控或三个 seed 均明显劣于 M0-FT，则先进入配对短微调。route39 的方向性优势仍不能替代 D7 总体结论。

## 8. D7-minus-route39 零微调续跑

route39 三 seed 结果记录于 `docs/experiment_records/2026-08-05-m2-m0-visual-swap-route39-zero-ft.md`：6/6 pipeline-valid，M0-V 未出现早期失控，准入 D7 剩余路线。为避免重复消耗已完成的 route39，冻结：

- 路线顺序 `[18,6,12,30,36,0]`，每条依次 seeds `[0,1,2]`，每个 variant 18 个 attempt；
- M0-FT Run ID `m2-interfuser-m0-ft-d7-minus39-seeds0-2-zero-ft-20260805-v1`；
- M0-V Run ID `m2-interfuser-m0-v-d7-minus39-seeds0-2-zero-ft-20260805-v1`；
- 执行顺序固定为 M0-FT 全部 18 个 attempt 后 M0-V 全部 18 个 attempt；任一 pipeline-invalid 立即停止；
- 最终按 variant 合并本轮 18 个 attempt 与 route39 三 seed，形成 21 个 route×seed 的完整 D7，不重跑、不以零填充。

完整 D7 主指标为先 route 内三 seed 均值、再七 route 宏平均的 DS；同时报告 RC/IS、21 个配对差值、路线级连续帧失败与资源释放。零微调 D7 只回答“固定 M0 底座上直接替换视觉骨干”的收益；若结果混合或下降，再预注册相同预算的 M0-FT/M0-V 短微调，不得用后验调参改写本轮事实。

## 9. M0-FT v1 晚发崩溃与重新准入

M0-FT 首批 `m2-interfuser-m0-ft-d7-minus39-seeds0-2-zero-ft-20260805-v1` 在第 11 个 attempt `route_30_seed_1` fail-fast。该 attempt 已完成闭环、写出 Leaderboard 记录，但 CARLA 在评测器清理前收到 SIGSEGV，随后评测器等待已退出的 simulator 120 秒并以 SIGABRT 退出。这是基础设施生命周期失败，不改写为模型零分，也不把已落盘分数改判为 pipeline-valid。

v1 原目录及 10 个有效前缀永久保留，但整批不进入 D7 聚合，禁止 resume、覆盖或与后续小批次拼接。重新准入顺序冻结为：

1. 先以新 Run ID `m2-interfuser-m0-ft-route30-seed1-lifecycle-smoke-20260808-v1` 只复现 `route30 / seed1`；
2. 只有 smoke `pipeline_valid=true`、评测器正常退出、CARLA 由 runner 回收，且 2155/2255 与 GPU 6/7 全部释放，才允许启动新 Run ID `m2-interfuser-m0-ft-d7-minus39-seeds0-2-zero-ft-20260808-v2`；
3. v2 必须从 18 个 attempt 完整重跑，严格复用 v1 的配置、checkpoint、路线/seed 顺序、代码哈希和资源约束；
4. 只有 v2 的 18/18 attempt 全部 pipeline-valid，才允许启动尚未运行的 M0-V 批次。

GPU 6/7 被外部作业占用时，smoke 和 v2 都必须等待原冻结资源释放；不得临时改占其他 GPU，也不得终止他人进程。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
