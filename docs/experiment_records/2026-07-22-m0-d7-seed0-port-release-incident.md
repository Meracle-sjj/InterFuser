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

提交 `f132929` 为 runner 增加 CARLA 退出后的端口释放等待，最长等待 60 秒，并将等待时长写入 attempt manifest。若端口仍未释放，runner 会写入 `cleanup_error`、将 attempt 标为 pipeline invalid，并立即终止整批，避免级联生成伪失败。该提交是第一阶段修复，后续真实运行证明它尚未覆盖显存和进程组回收。

修复后的 runner 定向测试为 `8/8` 通过；带完整 CARLA、Leaderboard 和 Scenario Runner `PYTHONPATH` 的测试集为 `141/141` 通过。

## 4. v2 真实验证与深层根因

`b0-d7-seed0-20260722-v2` 在提交 `e88913c` 上验证第一阶段修复。route18 再次 pipeline valid，记录 `port_release_wait_seconds=0.501`；紧随其后的 route6 未启动 CARLA，原因是 GPU 7 仍占用 `2223 MiB`，超过 `1024 MiB` 门槛。runner 当时未对普通 pipeline-invalid attempt fail-fast，约一秒后又启动 route12，因此该 run 被人工中止并整体判为无效。

中止 route12 后，evaluator 已退出，但 CARLA binary 仍存活，进程状态为 `PPID=1`、`PGID=1808159`，其进程组 leader `CarlaUE4.sh` 已退出。旧 `_stop_process_group` 只等待 leader 的 `Popen` 状态；leader 先退出时，忽略 SIGTERM 的 CARLA 子进程不会再收到 SIGKILL。这同时解释了端口、显存和孤儿进程问题。

## 5. 最终修复与重新准入

提交 `d95b176` 将进程组、端口和 GPU 生命周期抽离到 `runtime_resources.py`。回收逻辑会在 SIGTERM 后检查整个 POSIX 进程组是否消失，宽限期结束仍有成员则向同一进程组发送 SIGKILL；随后分别等待 TCP 端口和 GPU 显存低于门槛。任何 pipeline-invalid attempt 都会在 manifest 落盘后立即终止批次。

回归测试使用真实独立 session 构造“leader 退出、child 忽略 SIGTERM”的故障并证明整个进程组最终消失；完整测试集为 `144/144` 通过。下一次必须先完成 route18→route6 双路线生命周期 smoke，再使用新 Run ID 从 7 条 D7 路线完整重跑。只有 `7/7` pipeline valid 才允许进入 seed1/2。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
