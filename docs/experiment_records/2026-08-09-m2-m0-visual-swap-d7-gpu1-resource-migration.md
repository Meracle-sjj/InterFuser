# M2 M0 视觉替换 D7 GPU1 资源迁移记录

| 字段 | 内容 |
| --- | --- |
| 状态 | **FROZEN/READY：v1 清理超时已审计，准备 route36/seed1 恢复 smoke 与完整 v2 重跑** |
| 原资源 | agent GPU6 / CARLA GPU7，已被外部作业各占用约 30.6 GiB |
| 新资源 | agent GPU1 / CARLA GPU1，RTX 5090 32,607 MiB |
| M0-FT 配置 | `configs/thesis/interfuser_visual_swap_m0_ft_gpu1_v1.json` / `2be1499a8ed9960934a0cd5131d17885a777e112156474244a2daa8eedf64447` |
| M0-V 配置 | `configs/thesis/interfuser_visual_swap_m0_v_gpu1_v1.json` / `814a77df6b8f7553df564729ee13fa085ba15563068ad7a38104a996164b4506` |
| M0-FT 恢复配置 | `configs/thesis/interfuser_visual_swap_m0_ft_gpu1_v2.json` / `075646daf06e552284298bcf9500f0df9fa3a25a2d4fc4fbe7c1db8853156e36` |
| M0-V 恢复配置 | `configs/thesis/interfuser_visual_swap_m0_v_gpu1_v2.json` / `7fd73ad8c133dc7dcc1dfe7735849607793aa4ca3c368645145b1b2e437a2d7e` |

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

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
