# M0 D7 seeds1/2 v2 清理竞态记录

| 字段 | 内容 |
| --- | --- |
| Run ID | `b0-d7-seeds1-2-20260722-v2` |
| 运行代码 | `e3f6e9b35004504ba3a8d54dbbffb7b128d67f55` |
| Runner SHA-256 | `7939310f396ef19d6b521220d166c7e7416965a73f3e8ea6d43c6f24c151b347` |
| Manifest SHA-256 | `54230ed417642c2532b6c66035e6c688ada429e4e1f29dbcc018db3e8c265c88` |
| 原始结果 | `results/thesis_m0/b0-d7-seeds1-2-20260722-v2/` |
| 结论 | **批次无效并已 fail-fast；清理顺序修复通过前不得重跑完整 seeds1/2** |

## 1. 有效前缀与停止边界

计划包含路线 `18、6、12、30、36、39、0`，每条路线依次执行 seed1、seed2，共 14 个 attempt。runner 在第 3 个 attempt 后停止，manifest 为 `recorded=3`、`planned=14`、`pipeline_valid=2`、`pipeline_invalid=1`。

| Attempt | DS | RC | IS | Leaderboard status | Pipeline |
| --- | ---: | ---: | ---: | --- | --- |
| `route_18_seed_1` | 1.242318 | 2.969518% | 0.418357 | `Failed - Agent deviated from the route` | valid |
| `route_18_seed_2` | 1.242318 | 2.969518% | 0.418357 | `Failed - Agent deviated from the route` | valid |
| `route_06_seed_1` | 70.000000 | 100.000000% | 0.700000 | `Completed` | **invalid** |

前两个 attempt 只构成未完成批次的有效前缀，不进入种子聚合。route6 seed1 的分数已写入 Leaderboard JSON，但 CARLA/evaluator 未正常退出，因此不能改判为有效驾驶结果。

## 2. 根因定位

route6 seed1 在注册路线统计后，CARLA 再次以 SIGSEGV 退出：`carla_exit_code=139`、`carla_exited_before_cleanup=true`；evaluator 在销毁阶段等待已死 simulator 120 秒后以 `process_exit_code=-6` 退出。该失败与 v1 的 route6 seed2 同构，证明上一提交只补全了失败分类，没有消除 CARLA 0.9.16 清理竞态。

代码路径显示，完成路线后 evaluator 先调用 `scenario.remove_all_actors()`，在 Traffic Manager 与 world 仍处于同步模式时逐个销毁约 100 个 Town02 背景车辆；随后 `_cleanup()` 又通过 `CarlaDataProvider` 回收 actor，并在 `finally`/析构路径重复执行。故障只发生在统计落盘后的清理窗口，属于基础设施生命周期失败，不属于模型失败。

## 3. 修复与重新准入

修复将清理顺序固定为 Traffic Manager 退出同步模式、world 退出同步模式、ScenarioManager 清理、CarlaDataProvider 单次批量回收；删除完成路径的逐个 actor 销毁和 ego 重复销毁，并使同一 attempt 的 cleanup 幂等。清理顺序定向测试为 6/6 通过；带完整 CARLA、Leaderboard、Scenario Runner `PYTHONPATH` 的完整 unittest 为 152/152 通过。

修复提交 `1d91686` 推送后，首个 smoke Run ID `b0-d7-route6-seeds1-2-cleanup-smoke-20260722-v1` 被 P0 在创建 manifest 和启动 CARLA 前拒绝：配置仍指向旧 evaluator 哈希和旧运行代码锚点。该 launcher 日志与 PID 文件保留为门禁证据，不属于驾驶 attempt。配置随后只把运行代码锚定到 `1d91686` 并更新 evaluator SHA-256；模型、agent、路线、场景和 checkpoint 契约保持不变。

v2 原目录永久保留且禁止 resume、拼接或覆盖。配置提交推送后，使用另一个新 Run ID 对 route6 seed1/2 做生命周期 smoke；只有 2/2 pipeline valid 且进程、2155/2255、GPU 6/7 全部归零，才允许从 14 个 attempt 完整重跑 seeds1/2。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
