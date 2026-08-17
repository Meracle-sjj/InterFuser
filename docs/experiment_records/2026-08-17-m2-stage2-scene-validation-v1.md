<!--
[INPUT]: 依赖 Stage 2 scene-split v2 完成 manifest、validation-only 分层评估 manifest、固定指标方向和8个行人 Town+route 配对结果。
[OUTPUT]: 对外提供 B0/V 整体、行人条件、非行人、连续帧与路线异质性的事实结论及 test/D7 停止边界。
[POS]: docs/experiment_records 的 M2 H1 scene validation 证据锚点；区分类别无关 traffic occupancy 与真正行人实例识别，不把聚合多数票外推为普遍视觉收益。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
-->

# M2 Stage 2 Scene Validation v1 记录

## 1. Provenance

| 字段 | 值 |
| --- | --- |
| 评估代码提交 | `3673d29` |
| Run ID | `m2-interfuser-stage2-scene-validation-v1-seed20260814-20260817-v1` |
| Manifest | `results/thesis_m2/m2-interfuser-stage2-scene-validation-v1-seed20260814-20260817-v1/scene_validation_manifest.json` |
| Manifest SHA-256 | `f9477b9c17425032108854e48efc0c6f0d1b793ee42d723265c9ae63a0ca7bd7` |
| Config SHA-256 | `eb21807427f8bb0770ce4ec420ffc9a02f18a598d985eaa964136dab020f631b` |
| Pair run SHA-256 | `8783fd2851024c24def72949a5204bc8045d36115eca4b2f60b2ef748ce84cc4` |
| Split manifest SHA-256 | `bdb1b02745c791ed2f1edc897505f1ec3f432271ad3edb9a18987610a1aef124` |
| Validation index SHA-256 | `b4b5a62ac57a310c0d53aceadba53059eefce4b149fd47c30ae754b0694403d2` |

运行完成且 `pipeline_valid=true`、errors为空、`test_accessed=false`。B0/V各评估7,437帧，耗时17.252/16.941秒；GPU1监测峰值2,555 MiB。

## 2. 冻结五项指标

方向为 `V−B0`；AP/AUC/IoU越高越好，ADE/FDE越低越好。

| Cohort | AP | AUC | IoU | ADE | FDE-10 | V改善项 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | -8.50% | -0.97% | -4.24% | +1.64% | +6.39% | 2/5 |
| Pedestrian-conditioned | +7.51% | -1.12% | +17.09% | +24.71% | +23.58% | 4/5 |
| Non-pedestrian | -7.87% | -0.89% | -5.51% | -6.64% | -4.69% | 0/5 |

表中ADE/FDE的正百分比表示误差下降。行人条件聚合达到4/5多数票，但整体只有2/5，非行人场景0/5；因此方向冲突，不能判定为H1支持。

## 3. Route-group 异质性

8个行人 route group 中，V 的胜出数为：AP `3/8`、AUC `3/8`、IoU `4/8`、ADE `6/8`、FDE `7/8`。

- Town01 route004/013：五项全部改善；
- Town03 route318/341/346/394：ADE/FDE均改善，但三项 traffic 指标均退化；
- Town04 route094：仅IoU和FDE改善；
- Town05 route025：三项traffic指标改善，但ADE/FDE明显退化。

因此聚合行人AP/IoU提升由少数路线结构贡献，不能写成“在大多数行人路线提高识别率”。轨迹收益跨路线更稳定，但并不与traffic occupancy收益同向。

## 4. 辅助头与连续帧

行人条件下，junction macro-F1 从0.93568降至0.83115（-11.17%），red-light macro-F1从0.91708降至0.66468（-27.52%）。traffic连续帧残差MAE升高55.22%，waypoint连续帧残差ADE升高3.59%，均为退化。

这说明V的部分静态聚合指标改善没有形成更稳定的时间表征，并伴随路口/红灯语义代价。该结果与训练日志中V整体validation loss较高相互印证。

## 5. 结论边界

InterFuser traffic head 是类别无关的20×20 occupancy/geometry/speed网格；本记录的“pedestrian-conditioned traffic AP”表示含行人真值帧上的全部traffic occupancy表现，不等价于actor/grid级行人检测AP。

按 Stage 2 v2 冻结规则，本结果属于“行人条件改善、整体/稳定性退化”的混合证据。test继续冻结，不运行route39或D7，也不扩大到25轮。后续若继续诊断，只允许检查不同route group的真值构成、可见性和特征适配原因，不能通过挑选有利路线改写H1。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
