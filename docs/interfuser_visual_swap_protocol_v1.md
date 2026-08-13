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

## 10. GPU1 同卡资源迁移

2026-08-09 复核显示 GPU 6/7 被外部作业各占用约 30.6 GiB，而同型号 RTX 5090 GPU1 仅使用 709 MiB、连续短时采样利用率为 0%。经用户确认，准许将未启动的恢复链路迁移到 GPU1，但不改写 GPU6/7 上的历史 run 事实。

资源变更只包含 `agent_cuda_visible_device: 6→1` 和 `carla_graphics_adapter: 7→1`；GPU 型号、checkpoint、agent、controller、路线、场景、背景交通、seed、端口、超时和聚合口径保持不变。历史峰值为 agent 936 MiB 与 CARLA 5,825 MiB，合计显著低于 GPU1 的 32,607 MiB；M0-FT/M0-V 必须都使用同一 GPU1 配置，禁止只为一组改卡。

机器可读契约为：

- M0-FT：`configs/thesis/interfuser_visual_swap_m0_ft_gpu1_v1.json`；
- M0-V：`configs/thesis/interfuser_visual_swap_m0_v_gpu1_v1.json`。

执行顺序与新 Run ID 冻结为：

1. `m2-interfuser-m0-ft-route30-seed1-lifecycle-smoke-gpu1-20260809-v1`；
2. smoke 通过后运行 `m2-interfuser-m0-ft-d7-minus39-seeds0-2-zero-ft-gpu1-20260809-v1`；
3. M0-FT 18/18 pipeline-valid 后运行 `m2-interfuser-m0-v-d7-minus39-seeds0-2-zero-ft-gpu1-20260809-v1`。

GPU1 启动前必须连续三次低于 1,024 MiB 且 2155/2255 无监听。GPU 资源守卫默认仍拒绝任何已有 compute owner；只有本节的 M0-FT/M0-V GPU1 配置显式设置 `allow_existing_compute_processes_below_threshold=true`，允许总显存未越过 1,024 MiB 的既有 context。任一 attempt pipeline-invalid 仍 fail-fast；smoke 还必须证明 evaluator exit 0、CARLA 由 runner 回收且 GPU1 回落到同一启动门槛以下，才允许完整批次。

## 11. GPU1 v1 清理超时与完整 v2 重跑

GPU1 生命周期 smoke `m2-interfuser-m0-ft-route30-seed1-lifecycle-smoke-gpu1-20260809-v1` 已满足第 10 节准入。随后 M0-FT 正式批次 `m2-interfuser-m0-ft-d7-minus39-seeds0-2-zero-ft-gpu1-20260809-v1` 记录 14/18 个 attempt，其中前 13 个 pipeline-valid，第 14 个 `route_36_seed_1` 已由 evaluator 正常写出单路线结果，但 CARLA 清理后的显存未在默认 60 秒内回落到 1,024 MiB，因 `cleanup_error` 判为 pipeline-invalid 并 fail-fast。

该失败属于退出阶段资源释放超时，不将已落盘驾驶分数改写为有效，也不把 13 个有效前缀与后续运行拼接。v1 目录永久保留并整体排除于 D7 聚合。恢复配置由 v1 原样复制，唯一新增 `runtime.gpu_release_timeout_seconds=300`：

- M0-FT：`configs/thesis/interfuser_visual_swap_m0_ft_gpu1_v2.json`，SHA-256 `075646daf06e552284298bcf9500f0df9fa3a25a2d4fc4fbe7c1db8853156e36`；
- M0-V：`configs/thesis/interfuser_visual_swap_m0_v_gpu1_v2.json`，SHA-256 `7fd73ad8c133dc7dcc1dfe7735849607793aa4ca3c368645145b1b2e437a2d7e`。

重新准入顺序冻结为：

1. 先运行 `m2-interfuser-m0-ft-route36-seed1-cleanup-smoke-gpu1-20260810-v1`，只复现 `route36 / seed1`；
2. smoke 必须 evaluator exit 0、结果有效、CARLA 由 runner 回收、GPU1 在 300 秒内回落至门槛以下；
3. 通过后从头运行 `m2-interfuser-m0-ft-d7-minus39-seeds0-2-zero-ft-gpu1-20260810-v2` 的完整 18 个 attempt；
4. M0-FT 18/18 pipeline-valid 后才运行 `m2-interfuser-m0-v-d7-minus39-seeds0-2-zero-ft-gpu1-20260810-v2`；
5. 任一新 attempt pipeline-invalid 仍立即停止并保留原始证据，不原地覆盖、不自动改写模型结果、不跨 run 拼接。

## 12. v2 路线作用域漂移与 v3 机器化恢复

route36/seed1 清理恢复 smoke `m2-interfuser-m0-ft-route36-seed1-cleanup-smoke-gpu1-20260810-v1` 已通过：evaluator exit 0、`pipeline_valid=true`、CARLA 由 runner 回收、GPU1 释放等待 0.315 秒。随后自动门控启动 M0-FT v2 时遗漏了第 8 节冻结的显式路线列表，runner 因而使用 `development_d7` 默认集合，计划了包含 route39 的 21 个 attempt，并首先执行 `route_00_seed_0`。这是实验编排作用域漂移，不是模型或配置比较变量。

该 route0 attempt 持续生成 16,438 个控制帧，在 5,400 秒外部超时后被 runner 终止；没有 CARLA 提前退出或 GPU 清理错误，但 Leaderboard 未写出单路线记录，因此 pipeline-invalid。v2 run 及 1.4 GiB 连续帧证据永久保留，整体排除于 D7 聚合。

为消除对 CLI 手写列表的依赖，v3 配置在 v2 基础上唯一增加 `route_sets.d7_minus_route39=[18,6,12,30,36,0]`：

- M0-FT：`configs/thesis/interfuser_visual_swap_m0_ft_gpu1_v3.json`，SHA-256 `efd023a34f88c72988e1fe713d9b30ae73d648c491b176e6cfcc5265e711afd6`；
- M0-V：`configs/thesis/interfuser_visual_swap_m0_v_gpu1_v3.json`，SHA-256 `e2654ddb7c42ae29b5fd237b3e0de55820148b7857873914fd6495a993722f7b`。

重新准入顺序冻结为：

1. 先以 `m2-interfuser-m0-ft-route0-seed0-timeout-smoke-gpu1-20260811-v1` 单独重跑 route0/seed0，并保持 5,400 秒外部超时不变；
2. smoke pipeline-valid 后，以 `--route-set d7_minus_route39` 启动 `m2-interfuser-m0-ft-d7-minus39-seeds0-2-zero-ft-gpu1-20260811-v3`；
3. 正式 run plan 必须恰有 18 个 attempt，路线顺序严格为 `[18,6,12,30,36,0]`，每条 seeds `[0,1,2]`；
4. M0-FT 18/18 pipeline-valid 后，以相同 route set 启动 `m2-interfuser-m0-v-d7-minus39-seeds0-2-zero-ft-gpu1-20260811-v3`；
5. route0 smoke 若再次超时则停止，不把超时放宽或强行归为模型零分，先归因连续控制帧中的长期停车与 Leaderboard blocked 判定缺失。

## 13. v3 route0 重复超时与 v4 路线进度判停

v3 准入 smoke `m2-interfuser-m0-ft-route0-seed0-timeout-smoke-gpu1-20260811-v1` 再次运行 5,751.222 秒，记录 16,189 个连续控制帧，最终由 5,400 秒外部预算终止，evaluator exit 124 且没有单路线结果。退出后 GPU1 在 300 秒内仍为 2,153 MiB，故 cleanup 同时无效；该晚释放事实不改变“第二次 route0 已先超时”的主因。v3 smoke 与 v2 误启动 run 均永久保留且不进入聚合。

两次独立 route0/seed0 的连续帧和控制记录给出一致证据：道路中存在真实车辆排队，不是凭单帧臆测出的虚假障碍；16,438/16,189 帧中速度低于 0.1 m/s 的比例分别为 86.8%/87.4%，`d_0=0` 的比例为 48.0%/49.5%。控制器在 `stop_steps>1200` 后分别触发 10/11 次、每次 12 帧的 0.8 油门强制前进。该短促脉冲会让上游“速度连续低于 0.1 m/s 达 180 秒”的 blocked 计时反复复位，却不能形成足够路线进度，因而让已确定失败的路线悬挂到外部超时。

v4 不修改模型、checkpoint、控制器、交通量或外部超时，只在 Leaderboard RouteScenario 中并列增加 `RouteProgressBlockedTest`：以插值 route 的累积弧长度量进度，若任意连续 180 个仿真秒内净前进不足 18 米，记录包含窗口、阈值和位置的 `VEHICLE_BLOCKED` 事件并终止场景。18 米/180 秒等价于原速度判据 0.1 m/s 的窗口平均下界，但不会被数帧高速度或原地挪动清零；原速度判据仍保留，用于捕获真正连续静止。

机器可读 v4 契约为：

- M0-FT：`configs/thesis/interfuser_visual_swap_m0_ft_gpu1_v4.json`，SHA-256 `f348732884471d08db018fb3f01f1405470afe93a07c9568d1800c58c3358167`；
- M0-V：`configs/thesis/interfuser_visual_swap_m0_v_gpu1_v4.json`，SHA-256 `7d95197c9620a7ba31d24c8cbbd17c6e79bb472d71d609ca74022cf379412611`；
- runtime code anchor：`d6bf596a403b0765943044b8a913302f9f8031a1`；
- 判据源码 SHA-256：`efb4f1fb49feba24dfcab3a4edcbe4d92c8c22cbfcac4d8c476dab7d2fdcc604`；
- RouteScenario SHA-256：`1ac7cb505e5e332a2cd6eff443841d076457a59b964689c681a68d85fc9685ce`。

判停语义已经改变，因此不再把旧 evaluator 下的 route39 三 seed 与新结果拼接。v4 在同一代码锚点下完整重跑 `[18,6,12,30,36,0,39] × [0,1,2]`：

1. 先运行 `m2-interfuser-m0-ft-route0-seed0-progress-blocked-smoke-gpu1-20260812-v1`；
2. smoke 必须在外部超时前产生单路线结果、`pipeline_valid=true`，且若 route 未完成，须包含路线进度停滞的 `VEHICLE_BLOCKED` 证据；
3. 通过后运行 `m2-interfuser-m0-ft-d7-seeds0-2-zero-ft-progress-blocked-gpu1-20260812-v4` 的完整 21 个 attempt；
4. M0-FT 21/21 pipeline-valid 后运行配对的 `m2-interfuser-m0-v-d7-seeds0-2-zero-ft-progress-blocked-gpu1-20260812-v4`；
5. 任一 attempt pipeline-invalid 继续 fail-fast；blocked 是有效闭环失败而非基础设施失败，必须进入 DS/RC/IS 聚合，不能以超时或缺失结果静默丢弃。

## 14. v4 共享卡清理误判与 v5 容量式资源所有权

v4 route0 smoke `m2-interfuser-m0-ft-route0-seed0-progress-blocked-smoke-gpu1-20260812-v1` 完成 1/1 pipeline-valid：427.55 个仿真秒时记录路线进度 `17.22 m / 180.0 s`，生成明确的 route-progress `VEHICLE_BLOCKED`，evaluator exit 0，证明第 13 节判停修复有效。

随后 M0-FT v4 完整 D7 运行记录 8/21 attempt：前 7 个 pipeline-valid；第 8 个 route12/seed1 的 evaluator exit 0、Leaderboard 结果有效、CARLA 由 runner 回收，但外部用户进程在该 attempt 运行中加入 GPU1，导致整卡清理后仍使用 2,491 MiB。v4 的“整卡必须回到 1,024 MiB 以下”无法区分外部共享 owner 与本项目残留，因而把一个有效驾驶结果误判为 cleanup failure。该批次永久保留，整体不进入正式聚合。

v5 保留 checkpoint、模型、交通、路线进度判据与完整 D7，只替换资源生命周期契约：

- 启动判据为 GPU1 至少剩余 12,288 MiB，而非整卡低于 1,024 MiB；历史本项目峰值 7,268 MiB，保留约 5 GiB 额外余量；
- 外部 compute owner 被允许存在，其显存只记录为 before/peak/after provenance；
- 清理成败只绑定 runner 创建的 evaluator/CARLA POSIX 进程组、这些进程组下捕获到的 GPU PID，以及 2155/2255 端口；
- 外部任务在 attempt 中途启动或调整显存，不再被归为本项目泄漏；本项目 GPU PID 未释放仍在 300 秒后 fail-fast；
- 汇总器严格比较模型与评测 runtime，资源调度字段单独留作 provenance，不让“GPU 怎么分配”伪装成“驾驶语义改变”。

机器可读 v5 契约为：

- M0-FT：`configs/thesis/interfuser_visual_swap_m0_ft_gpu1_v5.json`，SHA-256 `e12f0d9de71ae44f882d9ae4f39aec99881bd30dc746ef712785efbdbbbce52c`；
- M0-V：`configs/thesis/interfuser_visual_swap_m0_v_gpu1_v5.json`，SHA-256 `e0e20158c9f4be8d921127a4196b96dd69708615d5f0f4e2aa243a23b719b8c8`；
- runtime code anchor：`a718da465497933c151af43d071b7541443682cc`；
- 资源策略：`shared_capacity`，`gpu_minimum_free_memory_mb=12288`。

route0 v4 smoke 已在相同模型/evaluator 下证明判停，因此 v5 不重复消费该 smoke。执行顺序冻结为：

1. 从头运行 `m2-interfuser-m0-ft-d7-seeds0-2-zero-ft-progress-blocked-shared-gpu1-20260813-v5` 的完整 21 个 attempt；
2. M0-FT 21/21 pipeline-valid 后运行 `m2-interfuser-m0-v-d7-seeds0-2-zero-ft-progress-blocked-shared-gpu1-20260813-v5`；
3. 每个 attempt 启动前重新检查 12 GiB 空闲；不足时该 attempt 不启动且 run fail-fast，禁止用 OOM 换取吞吐；
4. v4 的 7 个有效前缀不与 v5 拼接，v5 两个 variant 都从头使用同一资源规则。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
