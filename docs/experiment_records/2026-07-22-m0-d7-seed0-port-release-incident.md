# M0 D7 seed0 端口释放事故记录

| 字段 | 内容 |
| --- | --- |
| Run ID | `b0-d7-seed0-20260722` |
| 运行代码 | `3745481bcb9f722f5d45e2e93a3e5b62ecc148b3` |
| Runner SHA-256 | `d8cf9260dd63e093eb25ad046b9f7d0ca9e39c6098be329de48e6f7057bfc2fb` |
| 原始结果 | `results/thesis_m0/b0-d7-seed0-20260722/` |
| 结论 | **整批无效，不进入 M0 基线统计** |

## 1. 已观察事实

运行计划包含 seed0 的 7 条 D7 路线，顺序为 `18、6、12、30、36、39、0`。manifest 最终记录 `7` 个 attempt，其中只有 `route_18_seed_0` pipeline valid，其余 `6` 个均为基础设施失败。

`route_18_seed_0` 完成了真实 CARLA 场景执行：Driving Score 为 `1.2423184302`，Route Completion 为 `2.9695176524%`，Infraction Score 为 `0.4183569777`，状态为 `Failed - Agent deviated from the route`。该结果属于有效的模型驾驶失败。

route18 在 `2026-07-22T03:40:51.756755+00:00` 完成。下一条 route6 于约 `40 ms` 后开始端口预检，并报告 `RunnerError: required ports are already in use: [2155]`。随后五条路线也在同一秒内以相同错误结束，没有启动 CARLA 或 evaluator。

## 2. 根因与证据边界

旧 runner 会等待 CARLA 启动脚本进程退出，但未等待 CARLA 的 RPC socket 真正释放。连续 attempt 之间因此存在进程已退出、`2155` 仍短暂监听的生命周期竞态。

六条失败路线没有游戏时长、驾驶分数或模型行为，不能解释为 InterFuser 性能。虽然 route18 自身 pipeline valid，但该 run 未完成冻结的 7 路线集合，所以整个 run 不进入 D7 聚合或 seed 间波动统计。原目录保持只读，禁止覆盖或删除。

## 3. 修复与验证

提交 `f132929` 为 runner 增加 CARLA 退出后的端口释放等待，最长等待 60 秒，并将等待时长写入 attempt manifest。若端口仍未释放，runner 会写入 `cleanup_error`、将 attempt 标为 pipeline invalid，并立即终止整批，避免级联生成伪失败。

修复后的 runner 定向测试为 `8/8` 通过；带完整 CARLA、Leaderboard 和 Scenario Runner `PYTHONPATH` 的测试集为 `141/141` 通过。

下一次 seed0 必须使用新 Run ID 从 7 条路线完整重跑。只有 `7/7` pipeline valid 才允许进入 seed1/2。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
