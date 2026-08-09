# M2 M0 视觉替换 D7 GPU1 资源迁移记录

| 字段 | 内容 |
| --- | --- |
| 状态 | **FROZEN/QUEUED：先运行 route30 seed1 生命周期 smoke** |
| 原资源 | agent GPU6 / CARLA GPU7，已被外部作业各占用约 30.6 GiB |
| 新资源 | agent GPU1 / CARLA GPU1，RTX 5090 32,607 MiB |
| M0-FT 配置 | `configs/thesis/interfuser_visual_swap_m0_ft_gpu1_v1.json` / `05133921052a39711806785ed165f6cdaea92f7421cacb4083cf36fe1c006cd1` |
| M0-V 配置 | `configs/thesis/interfuser_visual_swap_m0_v_gpu1_v1.json` / `a1ff635a303027f321ace399a527d85528e59664b7bc1b9eac352b6e0c20afce` |

## 1. 迁移原因与边界

2026-08-09 21:20 +08:00 资源复核显示 GPU 6/7 只剩约 1.4 GiB 且接近满负载，原门控程序已等待约 24 小时仍不具备启动条件。GPU1 只使用 709 MiB，8 次连续秒级采样的 SM 利用率均为 0%，可用显存 31,401 MiB。原 GPU6/7 等待器 PID `3596597` 已在用户确认后正常终止，没有评测 attempt 被创建。

本迁移只改变物理 GPU 索引和“双卡→同卡”布置。两张物理卡均为 RTX 5090；checkpoint、模型、运行代码、背景交通和聚合参数不变。GPU1 当前存在一个占 674 MiB 但短时利用率为 0% 的历史 compute context；本项目不终止该进程，仅在总显存低于 1,024 MiB 时准入。

## 2. 容量与可比性

GPU6/7 历史 attempt 记录的 agent 峰值为 936 MiB，CARLA 峰值为 5,825 MiB，合计 6,761 MiB，只占 GPU1 容量约 20.7%。M0-FT 与 M0-V 都必须使用新配置，因此 GPU 布置不成为 variant 间变量。系统时间可能受同卡竞争影响，但论文主指标来自仿真进度和 Leaderboard 事件；仍必须报告资源峰值与系统/仿真时间比。

## 3. 执行顺序

1. smoke Run ID：`m2-interfuser-m0-ft-route30-seed1-lifecycle-smoke-gpu1-20260809-v1`；
2. M0-FT Run ID：`m2-interfuser-m0-ft-d7-minus39-seeds0-2-zero-ft-gpu1-20260809-v1`；
3. M0-V Run ID：`m2-interfuser-m0-v-d7-minus39-seeds0-2-zero-ft-gpu1-20260809-v1`。

smoke 必须同时满足 `pipeline_valid=true`、evaluator exit 0、`carla_exited_before_cleanup=false`、2155/2255 释放、GPU1 回落到 1,024 MiB 以下。只有 smoke 通过才运行 M0-FT 18 个 attempt，且只有 M0-FT 18/18 pipeline-valid 才运行 M0-V。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
