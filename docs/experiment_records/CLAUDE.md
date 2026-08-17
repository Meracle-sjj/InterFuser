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
2026-07-23-m0-d7-three-seed-baseline.md: M0 D7 三种子冻结记录，连接 seed0-v5 与 seeds1/2-v3 的 21 条有效结果、确定性汇总和受控生命周期哈希差异。
2026-08-14-m2-official-visual-swap-route39-zero-ft.md: M2 H1 作者发布底座上的交通语义 RGB 骨干替换记录，固化 strict-load 不变量、route39 三 seed 零微调结果与 D7 准入边界。
2026-08-14-m2-official-modality-ablation.md: M2 H1 官方 B0/B0-V 的 validation 与 route39 模态因果审计，固化 v1 空 LiDAR 失效、v2 坐标密度准入、RGB 实质依赖及视觉替换任务表征退化边界。
2026-08-15-m2-official-stage2-data-contract-preflight.md: M2 H1 官方 B0/V Stage 2 数据契约预检，固化 compass 轨迹旋转、LiDAR `+1`、缺失导航帧剔除、真实样本自证与旧离线指标失效边界。
2026-08-15-m2-official-stage2-pilot-v1-single-gpu-validation-incident.md: M2 H1 三轮 pilot 首次单卡运行事故，固化 B0 epoch0 完成后 validation 未赋值归约张量、无 checkpoint 可续跑及新 Run ID 全量重跑边界。
2026-08-16-m2-downstream-scene-split-v2.md: M2 H1 行人 holdout 测量修复记录，固化全库 census、45/8/8 分层、semantic-train 零泄漏、loader/哈希确定性与三相机连续帧可信度。
2026-08-17-m2-stage2-scene-validation-v1.md: M2 H1 scene-split v2 的 validation-only 分层评估，固化整体2/5、行人条件4/5、非行人0/5、路线异质性、连续帧退化与 test 保持冻结结论。
2026-08-17-m2-negative-transfer-repair-probes.md: M2 H1 负迁移修复总账，固化两轮插值失败、B0锚定近门槛、L2-SP无效、BN缓冲根因与冻结BN候选精确超限0.00577个百分点的停止结论。
2026-08-17-m2-calibration-diagnostic-v1.md: M2 H1 纯测量校准诊断记录，Q1排序改善与Q2阈值偏移均成立但v5门禁判定不变，附runner层落盘实现偏差说明。
2026-08-17-m2-distill-probe-v1.md: M2 H1 v7 B0 教师特征蒸馏终局记录，固化语义门禁4/4、scene 保持三项全过、行人核心 2/5 发丝级失败、蒸馏后阈值漂移消隐，及按 v7 协议 §3 关闭视觉修复路线的裁决。
2026-08-17-m2-linear-stitch-diagnostic-v1.md: M2 H1 v8a 线性缝合诊断，固化双臂逐 stage 误差消减率、v7 残差非线性的主臂 R=0.0108<0.5 判定，及 v8c 降级、v8b 功能蒸馏优先的方向取舍。
2026-07-23-m1-semantic-split-and-alignment.md: M1 数据 v1 冻结记录，固化 Town+route 无泄漏划分、三组类别覆盖、内容哈希与九类 RGB/mask 人工对齐结论。
2026-07-23-m2-semantic-smoke-v1.md: M2 首次语义预训练 smoke，记录 deterministic CUDA NLL 失败前序、修复后的有效训练/验证、checkpoint 与同构骨干迁移证据。
2026-07-23-m2-semantic-learning-curve-v1.md: M2 三点数据量 pilot，固化嵌套 25%/50%/100% train、完整 validation、逐类指标、产物哈希与优化/数据边界结论。
2026-07-23-m2-semantic-optimization-probe-v1.md: M2 五轮无权重优化充分性 probe，固化逐轮/逐类指标、best checkpoint、产物哈希与类别权重准入结论。
2026-07-23-m2-semantic-class-weight-probe-v1.md: M2 inverse-sqrt 类别权重 probe，固化配对预算、逐类收益/代价、best 骨干哈希与暂不补采的证据边界。
2026-07-24-m2-interfuser-visual-pair-smoke-v1.md: M2 H1 下游 B0/V 配对 smoke，固化单变量参数、两组有效训练、checkpoint schema 和资源释放证据。
2026-07-31-m2-visual-test-frozen-eval.md: M2 H1 冻结 test 配对评估，固化 B0/V 单帧任务、连续帧稳定性指标与 D7 准入事实。
2026-08-04-m1-pedestrian-hazard-audit-and-augmentation.md: M1.1 行人碰撞威胁审计，固化 584 sequence 的特权真值来源、连续可见性、train-only 扩充与 route-group 专项 holdout。
2026-08-04-m2-visual-d7-continuous-frame-failure-analysis.md: M2 H1 D7 连续帧失败归因，固化 route39 三 seed 起步轨迹/转向偏移与 3.05 s 同位置护栏碰撞。
2026-08-04-m2-semantic-pedestrian-hazard-probe-v1.md: M2 行人碰撞威胁 train-only 扩充 probe，固化原 validation、8+8 个 route-group 专项 holdout、InterFuser 严格迁移与下游 smoke。
2026-08-05-m2-m0-visual-swap-route39-zero-ft.md: M2 固定历史 M0 底座的语义 ResNet50d 零微调替换诊断，固化 route39 三 seed 连续控制帧、闭环配对差值与 D7 准入结论。
2026-08-08-m2-m0-ft-d7-minus39-v1-carla-late-crash.md: M2 M0-FT D7-minus-route39 首批的 CARLA 晚发段错误，固化原始 M0 行为对齐、有效前缀不拼接、route30 seed1 生命周期 smoke 与完整 v2 重跑门禁。
2026-08-09-m2-m0-visual-swap-d7-gpu1-resource-migration.md: M2 M0-FT/M0-V 的 GPU1 同卡恢复记录，固化 route0 进度判停、v4 整卡清理误判，以及 v5 共享容量与本项目资源所有权下的完整 D7 重跑边界。

记录只陈述事实和结论边界；实验协议归 `../baseline_evaluation_protocol_v1.md`，大体积原始结果归远端 `results/`。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
