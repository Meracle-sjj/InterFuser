# M2 M0 视觉替换 D7 GPU1 资源迁移记录

| 字段 | 内容 |
| --- | --- |
| 状态 | **FROZEN/RUNNING：v4 判停 smoke 已通过，整卡 1 GiB 清理误判已修复，v5 共享容量完整重跑** |
| 原资源 | agent GPU6 / CARLA GPU7，已被外部作业各占用约 30.6 GiB |
| 新资源 | agent GPU1 / CARLA GPU1，RTX 5090 32,607 MiB |
| M0-FT 配置 | `configs/thesis/interfuser_visual_swap_m0_ft_gpu1_v1.json` / `2be1499a8ed9960934a0cd5131d17885a777e112156474244a2daa8eedf64447` |
| M0-V 配置 | `configs/thesis/interfuser_visual_swap_m0_v_gpu1_v1.json` / `814a77df6b8f7553df564729ee13fa085ba15563068ad7a38104a996164b4506` |
| M0-FT 恢复配置 | `configs/thesis/interfuser_visual_swap_m0_ft_gpu1_v2.json` / `075646daf06e552284298bcf9500f0df9fa3a25a2d4fc4fbe7c1db8853156e36` |
| M0-V 恢复配置 | `configs/thesis/interfuser_visual_swap_m0_v_gpu1_v2.json` / `7fd73ad8c133dc7dcc1dfe7735849607793aa4ca3c368645145b1b2e437a2d7e` |
| M0-FT 编排恢复配置 | `configs/thesis/interfuser_visual_swap_m0_ft_gpu1_v3.json` / `efd023a34f88c72988e1fe713d9b30ae73d648c491b176e6cfcc5265e711afd6` |
| M0-V 编排恢复配置 | `configs/thesis/interfuser_visual_swap_m0_v_gpu1_v3.json` / `e2654ddb7c42ae29b5fd237b3e0de55820148b7857873914fd6495a993722f7b` |
| M0-FT 判停恢复配置 | `configs/thesis/interfuser_visual_swap_m0_ft_gpu1_v4.json` / `f348732884471d08db018fb3f01f1405470afe93a07c9568d1800c58c3358167` |
| M0-V 判停恢复配置 | `configs/thesis/interfuser_visual_swap_m0_v_gpu1_v4.json` / `7d95197c9620a7ba31d24c8cbbd17c6e79bb472d71d609ca74022cf379412611` |
| M0-FT 共享恢复配置 | `configs/thesis/interfuser_visual_swap_m0_ft_gpu1_v5.json` / `e12f0d9de71ae44f882d9ae4f39aec99881bd30dc746ef712785efbdbbbce52c` |
| M0-V 共享恢复配置 | `configs/thesis/interfuser_visual_swap_m0_v_gpu1_v5.json` / `e0e20158c9f4be8d921127a4196b96dd69708615d5f0f4e2aa243a23b719b8c8` |

## 1. 迁移原因与边界

2026-08-09 21:20 +08:00 资源复核显示 GPU 6/7 只剩约 1.4 GiB 且接近满负载，原门控程序已等待约 24 小时仍不具备启动条件。GPU1 只使用 709 MiB，8 次连续秒级采样的 SM 利用率均为 0%，可用显存 31,401 MiB。原 GPU6/7 等待器 PID `3596597` 已在用户确认后正常终止，没有评测 attempt 被创建。

本迁移只改变物理 GPU 索引和“双卡→同卡”布置。两张物理卡均为 RTX 5090；checkpoint、模型、运行代码、背景交通和聚合参数不变。GPU1 当前存在一个占 674 MiB 但短时利用率为 0% 的历史 compute context；本项目不终止该进程，而是以配置级显式例外允许总显存低于 1,024 MiB 时共享 GPU1。

## 2. 容量与可比性

GPU6/7 历史 attempt 记录的 agent 峰值为 936 MiB，CARLA 峰值为 5,825 MiB，合计 6,761 MiB，只占 GPU1 容量约 20.7%。M0-FT 与 M0-V 都必须使用新配置，因此 GPU 布置不成为 variant 间变量。系统时间可能受同卡竞争影响，但论文主指标来自仿真进度和 Leaderboard 事件；仍必须报告资源峰值与系统/仿真时间比。

## 3. 执行顺序

1. smoke Run ID：`m2-interfuser-m0-ft-route30-seed1-lifecycle-smoke-gpu1-20260809-v1`；
2. M0-FT Run ID：`m2-interfuser-m0-ft-d7-minus39-seeds0-2-zero-ft-gpu1-20260809-v1`；
3. M0-V Run ID：`m2-interfuser-m0-v-d7-minus39-seeds0-2-zero-ft-gpu1-20260809-v1`。

smoke 必须同时满足 `pipeline_valid=true`、evaluator exit 0、`carla_exited_before_cleanup=false`、2155/2255 释放、GPU1 回落到 1,024 MiB 以下。只有 smoke 通过才运行 M0-FT 18 个 attempt，且只有 M0-FT 18/18 pipeline-valid 才运行 M0-V。

## 4. 首次启动门禁拒绝

GPU1 配置提交 `ed5746b` 后，门控于 2026-08-09 21:30 +08:00 尝试启动 smoke。runner 在创建 Run 目录和启动 CARLA 之前拒绝：`GPU 1 has active compute processes: 3983756 (..., 674 MiB)`。因此本次没有 attempt、没有驾驶结果，也不构成 pipeline-invalid。

该拒绝证明 runner 的默认独占门禁在正常工作。后续不删除或全局绕过该门禁，而是按第 5 节增加配置级、仍受总显存阈值约束的显式例外。

## 5. 阈值内受控共享

用户确认 674 MiB、0% 短时利用率不应单独阻止本轮。资源守卫因此增加默认为 false 的 `allow_existing_compute_processes_below_threshold` 开关：其他配置仍拒绝任何 compute owner；仅两个 GPU1 配置显式开启，且总显存一旦高于 1,024 MiB 仍立即拒绝启动。

本例外不改写首次拒绝事实，也不将开关扩散到其他实验。smoke 将记录 GPU1 的 709 MiB 启动基线、运行峰值和回落时间；若同卡布置导致 OOM、超时或无法回落，仍按 pipeline-invalid fail-fast。

## 6. GPU1 smoke 与 M0-FT v1 实际结果

生命周期 smoke `m2-interfuser-m0-ft-route30-seed1-lifecycle-smoke-gpu1-20260809-v1` 完成 1/1 pipeline-valid：evaluator exit 0、CARLA 由 runner 回收、`carla_exited_before_cleanup=false`，证明 GPU1 同卡启动链路可用。

正式 M0-FT v1 `m2-interfuser-m0-ft-d7-minus39-seeds0-2-zero-ft-gpu1-20260809-v1` 于 2026-08-09 21:48 至 2026-08-10 01:57 记录 14/18 个 attempt：13 个 pipeline-valid，`route_36_seed_1` pipeline-invalid，剩余 `route_36_seed_2` 与 route0 三 seed 未启动。M0-V 因准入条件未满足而未启动。

## 7. route36 seed1 清理阶段超时

失败 attempt 的 evaluator exit 0，Leaderboard 单路线结果结构有效；CARLA 不是评测中提前退出，runner 记录 `carla_exited_before_cleanup=false`、`carla_exit_code=-15`。CARLA 关闭日志含退出阶段 `Signal 11`，随后 GPU1 在默认 60 秒清理窗口结束时仍使用 8,020 MiB，高于 1,024 MiB，最终错误为 `cleanup failed: GPUs did not become available`。

因此该 attempt 的驾驶结果不改判为零分，也不进入聚合；整个 v1 批次及其 13 个有效前缀永久保留为失败审计证据，不与新批次拼接。该事实支持“CARLA 退出/资源释放竞态”，不支持“模型在 route36 崩溃”。

## 8. 300 秒清理窗口与 v2 准入

恢复方案不改变 checkpoint、模型代码、路线/seed 顺序、背景交通、端口、GPU、显存门槛或聚合口径。runner 新增配置化显存释放等待，两个独立 v2 配置只增加 `gpu_release_timeout_seconds=300`，避免覆盖已冻结 v1 配置。

执行顺序冻结为：

1. `m2-interfuser-m0-ft-route36-seed1-cleanup-smoke-gpu1-20260810-v1`；
2. smoke 通过后完整运行 `m2-interfuser-m0-ft-d7-minus39-seeds0-2-zero-ft-gpu1-20260810-v2`；
3. M0-FT 18/18 有效后运行 `m2-interfuser-m0-v-d7-minus39-seeds0-2-zero-ft-gpu1-20260810-v2`。

任一新 attempt 仍按 pipeline-invalid fail-fast。失败 run 只允许生成新 Run ID 重新准入，不覆盖、不 resume、不把有效前缀跨批拼接。

## 9. route36 恢复 smoke 结果

`m2-interfuser-m0-ft-route36-seed1-cleanup-smoke-gpu1-20260810-v1` 于 2026-08-10 17:15–17:44 完成 1/1 pipeline-valid。该 attempt evaluator exit 0、CARLA exit -15 且 `carla_exited_before_cleanup=false`，GPU1 释放等待 0.315 秒，无 cleanup error；Leaderboard 状态为 `Failed - Agent got blocked`，DS 19.1401。300 秒窗口恢复了已知失败点，但实际本次无需消耗扩展窗口。

## 10. M0-FT v2 路线作用域漂移

自动门控在 smoke 通过后启动 `m2-interfuser-m0-ft-d7-minus39-seeds0-2-zero-ft-gpu1-20260810-v2`，但命令遗漏显式 route IDs，runner 按 `development_d7` 生成 21 个 attempt 并从 route0/seed0 开始，而非冻结的 18-task D7-minus-route39 顺序。这是编排脚本错误，v2 不具备正式实验准入资格。

route0/seed0 持续运行 5,449.272 秒并生成 16,438 个连续控制帧，最后 100 帧速度接近 0、98% 帧制动，但没有形成 Leaderboard 单路线记录；runner 在 5,400 秒外部预算处终止 evaluator，记录 exit 124、pipeline-invalid。CARLA 由 runner 回收，GPU 释放等待 0.267 秒，没有复现清理崩溃。门控随后 fail-fast，M0-V 未启动。

v2 原始目录和约 1.4 GiB sensor/control 证据永久保留，不覆盖、不 resume、不进入 D7 聚合。

## 11. v3 路线集合固化与 route0 准入

v3 配置保持 v2 的 checkpoint、代码、GPU1、300 秒清理窗口、5,400 秒外部超时和全部评测变量，只新增机器可读 `d7_minus_route39=[18,6,12,30,36,0]`。正式命令必须通过 `--route-set d7_minus_route39` 消费该集合，run plan 必须在启动前验证为 18 个 attempt。

先运行 `m2-interfuser-m0-ft-route0-seed0-timeout-smoke-gpu1-20260811-v1`。只有它 pipeline-valid，才自动启动 M0-FT v3 完整 18-task；M0-FT 18/18 通过后才启动 M0-V v3。route0 若再次超时则停止归因，不放宽预算以追逐有效结果。

## 12. v3 route0 复现、根因与 v4 恢复

v3 route0 smoke 于 2026-08-11 再次命中 5,400 秒外部超时：运行 5,751.222 秒、evaluator exit 124、无单路线记录、`pipeline_valid=false`。该 run 有 16,189 帧；v2 的独立同路线 run 有 16,438 帧。两次低速帧比例为 87.4%/86.8%，`d_0=0` 比例为 49.5%/48.0%，连续图像均显示真实交通排队。控制器为了脱困分别执行 11/10 次短促强制前进，恰好反复清空只看速度的 180 秒 blocked 计时，因此“长期停车”实质是评测终止条件被已有脱困策略绕过，而不是路线仍在持续推进。

恢复提交 `d6bf596a403b0765943044b8a913302f9f8031a1` 增加沿插值 route 累积弧长的独立判据：连续 180 个仿真秒净进度不足 18 米时产生 `VEHICLE_BLOCKED` 并终止。控制器、模型和交通配置均未改变；单元测试覆盖短促脉冲仍失败、达到阈值不误杀、后续窗口可再判停，以及 RouteScenario 同时安装速度/进度两套判据。

由于 evaluator 输入哈希已变化，本轮弃用“新 18-task + 旧 route39”的拼接方案，改为 M0-FT/M0-V 都在 v4 下完整重跑 21 个 attempt。准入顺序为 route0/seed0 v4 smoke → M0-FT 完整 D7 → M0-V 完整 D7；当前 GPU1 使用约 10 GiB，高于 1,024 MiB 门槛，因此只等待，不终止外部进程、不抢占启动。

## 13. v4 判停 smoke 通过与共享清理误判

GPU1 于 2026-08-13 06:23 +08:00 达到原门槛，v4 route0 smoke 随即启动并于 07:08 完成。它在连续 180 仿真秒仅前进 17.22 米时生成 route-progress blocked 事件，1/1 pipeline-valid，证明长期停车已从外部超时恢复为可归约驾驶失败。

M0-FT v4 于 07:09 启动，前 7 个 attempt 全部 pipeline-valid。route12/seed1 于 08:21 开始、08:59 evaluator 正常结束，Leaderboard 状态 `Failed - Agent timed out`、DS 28.6038；但外部训练进程于 08:33 加入 GPU1，整卡清理后为 2,491 MiB，旧 1,024 MiB 绝对门槛遂把该 attempt 判为 pipeline-invalid。该事实不是显存不足、模型失败或 CARLA 残留，而是共享资源所有权未建模；v4 8 个记录及 7 个有效前缀均不拼接。

提交 `a718da465497933c151af43d071b7541443682cc` 引入 v5 共享容量策略：启动至少保留 12 GiB；退出只等待 runner 所属 GPU PID、进程组和端口，外部 GPU 进程只记录不阻断。GPU1 在实现时仍有约 27 GiB 空闲，显著覆盖本项目历史 7.3 GiB 峰值，因此继续使用 GPU1 优于迁往已占 29–32 GiB 的 GPU2–7 或承载长期 CARLA 的 GPU0。

v5 从头运行 M0-FT 21-task，通过后运行配对 M0-V 21-task；两者都使用相同 shared-capacity 配置。v4 route0 smoke 作为判停准入证据保留，但 v4 正式前缀不进入 v5 D7 聚合。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
