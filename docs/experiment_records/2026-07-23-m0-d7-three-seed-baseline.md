# M0 D7 三种子冻结基线记录

| 字段 | 内容 |
| --- | --- |
| Seed 0 Run ID | `b0-d7-seed0-20260722-v5` |
| Seeds 1/2 Run ID | `b0-d7-seeds1-2-20260722-v3` |
| 冻结驾驶代码 | checkpoint、agent、controller、模型、路线、场景及环境哈希一致 |
| 汇总工具提交 | 本记录同提交引入 `tools/evaluation/summarize_thesis_baseline.py` |
| 汇总报告 | `results/thesis_m0/d7_b0_three_seed_summary_20260722T1653Z.json` |
| 汇总报告 SHA-256 | `41ce89b89915f1a747b4eb3b37b4a6d5e65468cb0fc580c308437f9c75d70459` |
| 结论 | **M0 D7 已冻结：21/21 pipeline valid，允许以 A36 作为论文主比较集并推进 M1** |

## 1. 完整性与来源

seed0-v5 manifest SHA-256 为 `cee24367b632a83dedbb6bd85511e8c948b1a6f8014edb497fe7a5c68c83f978`，记录 `7/7 pipeline valid`；seeds1/2-v3 manifest SHA-256 为 `0cf2e54af70633da39f049ea0f5929600981228fb9bca80c518b5c3e4767f1ea`，记录 `14/14 pipeline valid`。两批合计覆盖路线 `0、6、12、18、30、36、39` 与随机种子 `0、1、2` 的完整笛卡尔积，无缺失、重复或 pipeline-invalid attempt。

seeds1/2-v3 的 launcher PID `2116540` 已退出，结束后不存在属于该 run 的 runner、evaluator 或 CARLA 进程，`2155/2255` 无监听；GPU 6/7 分别为 `81/45 MiB` 且无 compute owner。21 个 attempt 的 `cleanup_error` 均为空，端口释放最长等待 `0.0 s`，GPU 释放最长等待 `0.503 s`，GPU 6/7 峰值分别为 `1308/5982 MiB`。

## 2. 受控 provenance 差异

seed0-v5 的运行 Git/runner 为 `ebc3ee81d4e289822b07b3e7219352762d9014f2` / `003173a345c3f899e8b3034e48098dde87b9d801551f94a93724838bf0fc45d3`；seeds1/2-v3 为 `1e474bcdd8672e919f1f9835979ed7c9006f607d` / `7939310f396ef19d6b521220d166c7e7416965a73f3e8ea6d43c6f24c151b347`，运行代码锚点为 `1d9168650711b46d6f2e754005b864c165a10f50`。

两批唯一的冻结输入哈希差异是 `leaderboard_evaluator`：`a823d80d...` 到 `490c9bce...`。代码审计确认变更发生在路线成绩落盘后的同步模式退出和 actor 回收阶段，不改变 checkpoint、agent、controller、模型定义、路线、场景、背景交通、随机种子或路线执行逻辑。汇总器默认拒绝该差异；本次命令显式使用 `--allow-input-drift leaderboard_evaluator`，并把两个哈希完整写入报告。该许可只适用于本次已审计的生命周期修复，不构成未来输入漂移的通用豁免。

## 3. 三种子路线结果

`DS/RC/IS` 分别表示 Driving Score、Route Completion 和 Infraction Score。驾驶失败状态属于有效模型结果，不是基础设施失败。

| Route | Seed | DS | RC | IS | Leaderboard status |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0 | 1.404419 | 29.124797 | 0.048221 | `Failed - Agent got blocked` |
| 0 | 1 | 7.863262 | 25.318974 | 0.310568 | `Failed - Agent got blocked` |
| 0 | 2 | 7.707946 | 25.318974 | 0.304434 | `Failed - Agent got blocked` |
| 6 | 0 | 38.343744 | 67.526591 | 0.567832 | `Failed - Agent timed out` |
| 6 | 1 | 70.000000 | 100.000000 | 0.700000 | `Completed` |
| 6 | 2 | 60.792554 | 65.724413 | 0.924962 | `Failed - Agent timed out` |
| 12 | 0 | 28.603825 | 53.492990 | 0.534721 | `Failed - Agent deviated from the route` |
| 12 | 1 | 18.746923 | 42.559712 | 0.440485 | `Failed - Agent timed out` |
| 12 | 2 | 30.979773 | 53.172373 | 0.582629 | `Failed - Agent deviated from the route` |
| 18 | 0 | 0.817611 | 3.003681 | 0.272203 | `Failed - Agent deviated from the route` |
| 18 | 1 | 0.807507 | 2.969518 | 0.271932 | `Failed - Agent deviated from the route` |
| 18 | 2 | 0.817611 | 3.003681 | 0.272203 | `Failed - Agent deviated from the route` |
| 30 | 0 | 12.330754 | 12.330754 | 1.000000 | `Failed - Agent deviated from the route` |
| 30 | 1 | 7.967951 | 11.619780 | 0.685723 | `Failed - Agent deviated from the route` |
| 30 | 2 | 12.330754 | 12.330754 | 1.000000 | `Failed - Agent deviated from the route` |
| 36 | 0 | 19.093002 | 42.169597 | 0.452767 | `Failed - Agent got blocked` |
| 36 | 1 | 18.734840 | 41.859164 | 0.447568 | `Failed - Agent got blocked` |
| 36 | 2 | 27.544760 | 42.480030 | 0.648417 | `Failed - Agent got blocked` |
| 39 | 0 | 37.336627 | 82.058521 | 0.455000 | `Failed - Agent timed out` |
| 39 | 1 | 36.987015 | 81.290144 | 0.455000 | `Failed - Agent timed out` |
| 39 | 2 | 36.987015 | 81.290144 | 0.455000 | `Failed - Agent timed out` |

## 4. 冻结汇总

严格按协议先对同一路线三个种子求均值，再对七条路线作宏平均。跨种子标准差由每个种子的七路线宏平均计算，使用总体标准差。

| 指标 | 三种子路线宏平均 | 跨种子标准差 |
| --- | ---: | ---: |
| Driving Score | 22.676090 | 2.300508 |
| Route Completion | 41.840219 | 1.339341 |
| Infraction Score | 0.515698 | 0.058373 |

状态计数为：`Completed=1`、`deviated=8`、`blocked=6`、`timed out=6`。汇总器对相同输入复算后与正式 JSON 字节一致，证明统计不依赖输入顺序、当前时间或手工表格操作。

## 5. 证据边界与下一阶段

D7 是开发回归集，不是最终论文主分数。Town06 资产仍缺失，因此 A36 冻结为 Town01-Town05 的 36-route available-map subset；B0、V、L、V+L 必须使用相同 A36 路线、场景和种子，Town06 不计零。

M0 的 P0、21 条 D7 结果、确定性汇总、资源证据与 A36 决策均已闭合。下一阶段回到 M1：基于已通过的 dataset_index pilot 冻结 sequence 级无泄漏 split manifest，并完成每个核心类别的 RGB/mask 人工对齐复核；在这两项完成前不启动 M2 预训练。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
