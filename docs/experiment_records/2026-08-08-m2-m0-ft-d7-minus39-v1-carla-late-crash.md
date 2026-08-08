# M2 M0-FT D7-minus-route39 v1 CARLA 晚发崩溃记录

| 字段 | 内容 |
| --- | --- |
| Run ID | `m2-interfuser-m0-ft-d7-minus39-seeds0-2-zero-ft-20260805-v1` |
| 运行提交 | `f05ea7e772c3662341482b81f9614e29c7f28906` |
| Runner SHA-256 | `7939310f396ef19d6b521220d166c7e7416965a73f3e8ea6d43c6f24c151b347` |
| 配置 SHA-256 | `1719d76cfe2955bb4e5e4201978e4fcdd3ef94373c15fafbfa1f2b9d5b2d9a90` |
| Manifest SHA-256 | `17ddb66712073556162dee9b25ae217b059461128115aea3e337ac7234fbd7d5` |
| 原始结果 | `results/thesis_m2/m2-interfuser-m0-ft-d7-minus39-seeds0-2-zero-ft-20260805-v1/` |
| 结论 | **v1 批次无效并已 fail-fast；不得进入 D7 聚合** |

## 1. 停止边界

v1 计划按路线 `[18,6,12,30,36,0]` 和每路线 seeds `[0,1,2]` 运行 18 个 attempt。runner 在第 11 个 `route_30_seed_1` 后停止：manifest 为 `recorded=11`、`planned=18`、`pipeline_valid=10`、`pipeline_invalid=1`。

| 路线 | 已记录 seed | Pipeline 结果 | 证据边界 |
| --- | --- | --- | --- |
| 18 | 0、1、2 | 3/3 valid | 未完整批次的有效前缀，不进入聚合 |
| 6 | 0、1、2 | 3/3 valid | 未完整批次的有效前缀，不进入聚合 |
| 12 | 0、1、2 | 3/3 valid | 未完整批次的有效前缀，不进入聚合 |
| 30 | 0 | valid | 未完整批次的有效前缀，不进入聚合 |
| 30 | 1 | **invalid** | CARLA `139`，evaluator `-6` |
| 30 | 2 | 未运行 | fail-fast 阻止 |
| 36、0 | 0、1、2 | 未运行 | fail-fast 阻止 |

10 个 pipeline-valid attempt 证明 M0-FT 能完成对应的单次驾驶评测，但不能从未完整批次中抽取成绩与其他 Run ID 拼接。`route_30_seed_1` 已写出 `Failed - Agent deviated from the route`，但该分数不具备 pipeline 准入资格。

## 2. 根因与模型边界

`route_30_seed_1` 在仿真时间 25.7 s 停止路线并完成统计注册；两秒后 `carla.log` 记录 `Signal 11 caught`、`CommonUnixCrashHandler: Signal=11` 与 `Segmentation fault`。评测器随后在销毁阶段等待 localhost:2155 上已死亡的 simulator 120 s，抛出 `carla::client::TimeoutException` 并以 SIGABRT 退出。

因此异常链为 CARLA 清理窗口的晚发崩溃，与 2026-07-22 记录的生命周期失败同构，不是模型前向、checkpoint 加载或 GPU OOM。该 attempt 的路线偏离仍是待比较的闭环现象，但只有在新的 pipeline-valid 重复中才能作为模型证据。

## 3. 重新准入

v1 原目录永久保留且禁止 resume、覆盖或拼接。恢复路径固定为：

1. 用 `m2-interfuser-m0-ft-route30-seed1-lifecycle-smoke-20260808-v1` 单独复现同一 `route30 / seed1`；
2. smoke 必须同时满足 `pipeline_valid=true`、evaluator exit `0`、`carla_exited_before_cleanup=false`、2155/2255 无监听、GPU 6/7 无本试验残留；
3. 通过后，用 `m2-interfuser-m0-ft-d7-minus39-seeds0-2-zero-ft-20260808-v2` 从 18 个 attempt 完整重跑；
4. 只有 v2 18/18 pipeline-valid 才解锁 M0-V，任一新失败仍 fail-fast 并另行记录。

2026-08-08 复核时，2155/2255 已释放，但 GPU 6/7 正被外部用户的多卡训练占用约 21 GiB/卡。这不是 v1 崩溃原因；v1 每个 attempt 启动前 GPU 6/7 仅为 81/45 MiB。后续必须等待原资源空闲，不改占 GPU，也不干预他人进程。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
