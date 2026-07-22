# M0 D7 seeds1/2 v1 CARLA 晚发崩溃记录

| 字段 | 内容 |
| --- | --- |
| Run ID | `b0-d7-seeds1-2-20260722-v1` |
| 运行代码 | `380e61ca5c0e03af09d6af3ed79c051c1b33234c` |
| Runner SHA-256 | `003173a345c3f899e8b3034e48098dde87b9d801551f94a93724838bf0fc45d3` |
| Manifest SHA-256 | `d6f7b5dcfd808e7278339fd84d90d12bcf488ae2d1873cf54487807878141068` |
| 原始结果 | `results/thesis_m0/b0-d7-seeds1-2-20260722-v1/` |
| 结论 | **批次无效并已 fail-fast；不得进入 D7 三种子聚合** |

## 1. 有效前缀与停止边界

计划包含路线 `18、6、12、30、36、39、0`，每条路线依次执行 seed1、seed2，共 14 个 attempt。runner 在第 4 个 attempt 后停止，manifest 为 `recorded=4`、`planned=14`、`pipeline_valid=3`、`pipeline_invalid=1`。

| Attempt | DS | RC | IS | Leaderboard status | Pipeline |
| --- | ---: | ---: | ---: | --- | --- |
| `route_18_seed_1` | 1.242318 | 3.003681% | 0.413599 | `Failed - Agent deviated from the route` | valid |
| `route_18_seed_2` | 0.817611 | 3.003681% | 0.272203 | `Failed - Agent deviated from the route` | valid |
| `route_06_seed_1` | 35.415204 | 61.218967% | 0.578501 | `Failed - Agent timed out` | valid |
| `route_06_seed_2` | 70.000000 | 100.000000% | 0.700000 | `Completed` | **invalid** |

前三个 attempt 是结构完整的驾驶结果，但该批次未覆盖冻结集合，不能单独进入种子间统计。route6 seed2 的分数只证明 Leaderboard 曾写出完整记录，不能覆盖其运行时基础设施失败。

## 2. 根因与证据边界

route6 seed2 完成路线并注册统计后，CARLA 进程收到 SIGSEGV，`carla.log` 记录 `Signal 11 caught`，最终 `carla_exit_code=139`。evaluator 随后在销毁阶段等待已崩溃的 simulator 120 秒，未捕获的 `carla::client::TimeoutException` 导致 SIGABRT，`process_exit_code=-6`。

这不是模型失败，也不是端口、GPU 或进程组泄漏。launcher 已退出，2155/2255 无监听，GPU 6/7 回到 81/45 MiB 且没有 compute owner。服务器上 world-port 2000 的长期 CARLA 属于其他作业，不在本次清理范围。

## 3. 工具修正与重新准入

runner 保持 pipeline-invalid 立即终止，不尝试把已有分数改判为有效。后续版本额外记录 CARLA 是否在 runner 清理前退出，并把 CARLA/evaluator 退出码写成明确 failure reason，避免 `error=null` 隐去基础设施故障链。

v1 原目录永久保留且禁止 resume 或覆盖。只有相关测试与完整 unittest 通过、代码提交推送、2155/2255 和 GPU 6/7 再次空闲后，才允许以新 Run ID 从 14 个 attempt 完整重跑；不得拼接 v1 的三个有效前缀。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
