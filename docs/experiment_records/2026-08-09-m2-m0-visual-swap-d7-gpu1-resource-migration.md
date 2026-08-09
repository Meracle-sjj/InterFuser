# M2 M0 视觉替换 D7 GPU1 资源迁移记录

| 字段 | 内容 |
| --- | --- |
| 状态 | **FROZEN/READY：已显式准入阈值内共享，先运行 smoke** |
| 原资源 | agent GPU6 / CARLA GPU7，已被外部作业各占用约 30.6 GiB |
| 新资源 | agent GPU1 / CARLA GPU1，RTX 5090 32,607 MiB |
| M0-FT 配置 | `configs/thesis/interfuser_visual_swap_m0_ft_gpu1_v1.json` / `2be1499a8ed9960934a0cd5131d17885a777e112156474244a2daa8eedf64447` |
| M0-V 配置 | `configs/thesis/interfuser_visual_swap_m0_v_gpu1_v1.json` / `814a77df6b8f7553df564729ee13fa085ba15563068ad7a38104a996164b4506` |

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

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
