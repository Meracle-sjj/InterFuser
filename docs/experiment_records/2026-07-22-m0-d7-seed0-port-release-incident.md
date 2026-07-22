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

## 6. 生命周期 smoke 准入结果

`b0-d7-lifecycle-smoke-20260722` 在提交 `cffcca0` 上按 route18→route6 顺序执行完成，汇总为 `2/2 pipeline valid`、`0 pipeline invalid`：

| Attempt | Leaderboard 状态 | DS | RC | IS | 端口释放等待 | GPU 释放等待 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `route_18_seed_0` | `Failed - Agent deviated from the route` | 1.242318 | 3.003681% | 0.413599 | 0.0 s | 0.118 s |
| `route_06_seed_0` | `Completed` | 70.000000 | 100.000000% | 0.700000 | 0.0 s | 0.242 s |

route18 结束后其 CARLA 进程组完全消失，route6 使用新的 PID/PGID 和 evaluator 正常启动；route6 结束后 launcher 正常退出，`2155/2255` 无监听，GPU 6/7 回到运行前显存水平。该结果解除资源生命周期门禁，允许正式 D7 seed0 使用新 Run ID 重跑，但不替代 7 路线基线结果。

## 7. v3 CARLA readiness 原生崩溃

`b0-d7-seed0-20260722-v3` 在冻结运行 Git `634639596a9a4e6911f434e60c74224da99c9503` 上完成了前 5 条路线，总 manifest 为 `recorded=5/7`、`pipeline_valid=5`、`pipeline_invalid=0`。route39 启动 CARLA 后，2 秒 readiness RPC 在 CARLA Python 原生层抛出 `TimeoutException` 并调用 `std::terminate`，launcher 因此越过 Python `finally` 直接退出。

route39 的初始 `attempt_manifest.json`、`route.xml`、`carla.log` 和空 `evaluator.log` 保留在原 Run ID 目录；evaluator 从未启动，该事故不进入模型失败或 D7 统计。孤儿 CARLA 进程组 `PGID=1893117` 经命令行、工作目录与 2155 监听三重归属确认后回收，GPU 6/7 显存回到 `81/45 MiB`。

修复边界是将 readiness 和地图加载 RPC 放入短命子进程：原生崩溃只使单次探针失败，父 runner 仍能重试或进入 attempt 清理回路。同时在 CARLA 启动后立即原子落盘 PID 与命令，缩小 launcher 遭遇不可恢复信号时的证据空洞。

定向 runner/资源回归为 `13/13` 通过；带完整 CARLA、Leaderboard 和 Scenario Runner `PYTHONPATH` 的 unittest 为 `149/149` 通过。v3 保持原地不续跑，因为在同一 run manifest 中混用两个 runner hash 会破坏实验 provenance；修复后的 seed0 必须使用新 Run ID 从 7 条路线重新开始。

## 8. v4 GPU compute-owner 门禁事故

readiness 隔离修复提交 `2f99401f3db99e766073501b92f612f9867a1b2a` 后，`b0-d7-seed0-20260722-v4` 在启动前检测到 GPU 7 存在外部计算进程 `PID=1903747`，但旧资源守卫只依赖整卡显存读数与 `1024 MiB` 阈值，当时读数低于阈值而错误放行。这证明“低显存”不等于“无同 GPU 作业”。

v4 在 route18 开始阶段被人工终止，总 manifest 仍为空 attempts/空 summary；原目录、初始 attempt manifest、CARLA/evaluator 日志与中止时的 Leaderboard JSON 全部保留，不进入模型失败或 D7 统计。只回收了 v4 的 launcher、CARLA 与 evaluator，未向外部 PID 发送信号；回收后 `2155/2255` 无监听，GPU 6/7 回到 `81/45 MiB`。

资源门禁必须同时满足两个条件：选定 GPU 显存不超阈值，且通过 GPU UUID 查询不存在任何 active compute process。任一条件失败都必须在启动 CARLA 前拒绝整批。

新门禁的定向 runner/资源回归为 `14/14` 通过；带完整 `PYTHONPATH` 的 unittest 为 `150/150` 通过。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
