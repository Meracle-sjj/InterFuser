<!--
[INPUT]: 依赖官方底座视觉替换 initialization manifest、CARLA 0.9.16 route39 三 seed 原始诊断日志与连续可视化帧。
[OUTPUT]: 对外提供 official_b0/official_b0_v 严格张量不变量、600 帧 RC/位移配对结果、视觉复核和 D7 准入结论。
[POS]: docs/experiment_records 的 M2 H1 官方可驾驶底座零微调记录；取代无法归因的自训 M0 底座可行性判断。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
-->

# M2 官方底座视觉骨干替换 route39 零微调记录

| 字段 | 内容 |
| --- | --- |
| 日期 | 2026-08-14 |
| 状态 | **ROUTE39-3SEED-PASSED：不启动短适配，待 D7** |
| Git HEAD | `965148ca13455ff815c12618668e6fc4d37b1673` |
| 初始化 Run ID | `m2-interfuser-official-visual-swap-init-v1-seed20260723-20260814-v1` |
| Initialization manifest SHA-256 | `563782c4b33d6b6b74186961b03ba9eff96b12e02594cda1674320320a4d1755` |

## 1. 初始化不变量

- 作者发布 checkpoint SHA-256：`39548dcfa56cdc06b8ccc52d7f6187a7d70ef36b05b7027db4d64a57ac3e30b2`；原始元数据 `arch=mvt_baseline`、epoch 34。
- `official_b0` 导出 SHA-256：`15d70aa489f33b84b8cad0988aa0b8f1787ee626069d8c95c3e9e518e781a0ce`；其 1,132 个 state tensor 与官方源状态逐值一致。
- `official_b0_v` 导出 SHA-256：`1ca2c2204fd0738f85ed5d098fb4b16036597a4efda05248182da968761d4954`。
- 交通语义 ResNet50d 骨干 SHA-256：`a0673b36325b54695f82aef52829c34aaaa4ed7652ff81ee3bee36bf180ea56d`。
- 只有 330 个唯一 RGB tensor（全模型 alias 共 660 key）变化；非 RGB state SHA-256 两组同为 `bdc334e6ceb3c1c6cd6a4dc08003a859eeaefb77beef9dbb1b50622f68e72162`。
- 两个 checkpoint 均能 strict load；同一随机输入的 7 类前向输出 shape 一致且全部为有限值。

## 2. 闭环固定条件

两组共用作者干净 agent/controller/model 定义、CARLA 0.9.16 兼容层、GNSS 第 0 轴取反、`vehicle.lincoln.mkz_2017` 蓝图别名、planner `min_distance=4.0`、最高速度 5 m/s、原始 steering controller、route39、200 辆背景车、seed 0/1/2 和 600 帧上限。唯一模型变量是 RGB 骨干 state。

## 3. 三 seed 配对结果

| seed | official_b0 RC | official_b0_v RC | V−B0 | official_b0 位移 | official_b0_v 位移 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 61.886267% | 59.555452% | -2.330815 | 132.43107 m | 127.40742 m |
| 1 | 61.886267% | 61.886267% | 0.000000 | 132.49654 m | 131.81635 m |
| 2 | 61.886267% | 60.953946% | -0.932321 | 132.52507 m | 130.64487 m |
| 均值 | 61.886267% | 60.798555% | -1.087712 | 132.48423 m | 129.95621 m |

`official_b0_v` 的平均 RC 相对下降 1.7576%，平均位移相对下降 1.9082%。六次运行均 evaluator exit 0、实际生成 600 张 `meta` 连续帧且无 `DIAG_EVENT`。B0-V seed0 首次运行的 `exit_status.log` 误按不存在的 `rgb_front` 目录计数为 0；`DIAG_COMPLETE frames=600`与实际 600 张 `meta/*.jpg` 一致，后续运行器已更正计数 glob，不改动该原始日志。

## 4. 连续帧复核

B0-V seed0 复核 frame 0/100/300/500/599，seed1/2 复核 frame 300/599；三个 seed 均持续位于车道内，弯道航点连续，没有早期向右护栏收敛。这与自训 epoch26 底座约 frame50 撞护栏的失效模式明显不同。

## 5. 结论与边界

route39 三 seed 证明预训练 RGB 骨干可直接接入作者发布 InterFuser，不会引发灾难性特征分布失配。按冻结协议不启动短适配，下一门禁为官方 B0/B0-V 完整 D7 配对。

当前 B0-V 均值小幅低于 B0，因此不能宣称闭环改善；route39 只是准入路线，600 帧 RC 是运行时快照而非正式 Leaderboard Driving Score。改善、中性或下降必须由同一评测语义下的 D7 多路线结果决定。

## 6. 原始证据

- 初始化：`results/thesis_m2/m2-interfuser-official-visual-swap-init-v1-seed20260723-20260814-v1/`。
- 官方 B0 闭环：`/data/shijj/interfuser_clean_repro_results/official-b0-gnssflip-lincoln-official-control-route39-seed{0,1,2}-600f-20260814-v1/`。
- B0-V 闭环：`/data/shijj/interfuser_clean_repro_results/official-b0-v-traffic-semantic-gnssflip-lincoln-official-control-route39-seed{0,1,2}-600f-20260814-v1/`。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
