# docs/experiment_records/
> L2 | 父级: ../CLAUDE.md

## 成员清单

CLAUDE.md: 实验事实记录模块地图，约束摘要只能引用已完成 run 的原始 manifest、结果哈希和 Git 提交。
2026-07-22-m0-route18-smoke.md: M0 D7 runner 首次真实场景 smoke，记录一次无效 setup run 与修复后的有效驾驶 run。
2026-07-22-m0-d7-seed0-port-release-incident.md: M0 D7 seed0 运行暴露的端口、显存、进程组回收与 CARLA readiness 原生崩溃问题及证据边界。
2026-07-22-m0-d7-seed0-v5.md: M0 D7 seed0 最终有效重跑，固化 7/7 pipeline-valid 路线指标、运行哈希与资源释放证据。
2026-07-22-m0-d7-seeds1-2-v1-carla-late-crash.md: M0 D7 seeds1/2 首次批次的 CARLA 晚发段错误，固化 fail-fast 边界、有效前缀与重跑准入条件。
2026-07-22-m0-d7-seeds1-2-v2-cleanup-crash.md: M0 D7 seeds1/2 第二次批次复现 route6 清理竞态，固化同步模式退出、单次 actor 回收与定向 smoke 准入边界。
2026-07-22-m1-semantic-index-pilot.md: M1 dataset_index 分层抽样 pilot，固化抽样 provenance、类别准入结果与两个候选不足分层。

记录只陈述事实和结论边界；实验协议归 `../baseline_evaluation_protocol_v1.md`，大体积原始结果归远端 `results/`。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
