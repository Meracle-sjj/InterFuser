# M2 H1 V 组 D7 连续帧失败归因

## 结论

冻结 D7 的 V 模型离线指标 5 项中改善 4 项，但闭环 Driving Score 宏平均比 B0 低 7.4180，Route Completion 低 12.3129，因此 H1 v1 结论仍为 `mixed_or_insufficient`，不允许声明闭环非劣。

连续控制帧把主要回归定位到 route39：V 三个 seed 在起步后 step 35–36 越过 1 m 车道偏移，并在游戏时间 3.05 s、坐标约 `(-265.35, -83.4)` 碰撞同一护栏。B0 的对应布局碰撞分别在 34.9/85.15/76.05 s，不是相同的起步回归。

## 冻结总结

- D7 attempt：B0 21/21、V 21/21 pipeline-valid，路线与 seed 完全配对。
- Driving Score pair：V 改善 7、持平 4、变差 10。
- Driving Score 宏平均差（V-B0）：`-7.41802894`。
- Route Completion 宏平均差：`-12.31288489`。
- Infraction Score 宏平均差：`+0.03532452`。
- 只有 route39 在三个 seed 上都出现 10 s 内新增布局碰撞。

## route39 起步 40 控制周期

| 指标 | B0 三 seed 范围 | V 三 seed 范围 |
| --- | ---: | ---: |
| 平均绝对 steer | 0.0693–0.0723 | 0.2663–0.2679 |
| 最大绝对 steer | 0.1319–0.1496 | 0.6723–0.6961 |
| 平均绝对 lane offset | 0.3055–0.3099 m | 0.6412–0.6424 m |
| 最大绝对 lane offset | 0.6095–0.6106 m | 1.6922–1.7492 m |
| 首次 `abs(offset)>=1m` | step 678/1683/1492 | step 36/35/35 |
| 车道中心校正达 0.249 饱和比例 | 0% | 42.5%–45.0% |
| raw junction 阳性比例 | 0%–2.5% | 95.0%–97.5% |
| auxiliary junction 阳性比例 | 85.0%–90.0% | 0% |

raw 与 auxiliary junction 头在 B0/V 中交换了极端分工；控制器的 `consensus=min(raw, aux)` 仍接近 0，因此它不是这次护栏碰撞的直接制动触发因。可见的直接链条是 V 预测轨迹横向分量增大 → steer 增大 → lane-center 校正饱和仍无法抵消 → 越过 1 m 并碰撞护栏。

## 产物与哈希

- B0 D7 manifest：`results/thesis_m2/m2-interfuser-visual-d7-b0-seeds0-2-20260724-v1/run_manifest.json`，SHA-256 `05b6b74614ec7adb616efcc48a97bee3c09055aa9288cdc3564f00a9d9f6841d`。
- V D7 manifest：`results/thesis_m2/m2-interfuser-visual-d7-v-seeds0-2-20260724-v1/run_manifest.json`，SHA-256 `6e4c5891917db429aada80a2e920c7d4f6940bca6e6e1c57771478f11a021775`。
- H1 冻结总结：`results/thesis_m2/m2-interfuser-visual-d7-pair-seeds0-2-20260724-v1/h1_summary.json`，SHA-256 `0a609de271101110eca3fcefd66013eceb52d2014a5cbe11a157baeb0885611c`。
- 连续帧失败分析：`results/thesis_m2/visual_d7_failure_analysis_20260804.json`，SHA-256 `f9f8c62b98fd2abc8afdad8fde3cc5f06f919c13abe94f37b72ca78a843920a9`。

## 准入决策

新的视觉骨干不得仅凭离线 mIoU/轨迹 ADE 进入完整 D7。先在 route39 固定起步片段上检查预测 waypoint 横向幅度、前 40 周期最大车道偏移与早期布局碰撞；只有消除可重复起步回归后才值得执行 42 个 attempt。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
