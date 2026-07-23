# M2 交通语义优化充分性 probe v1 记录

| 字段 | 内容 |
| --- | --- |
| Run ID | `m2-semantic-optimization-probe-v1-full-unweighted-seed20260723` |
| 运行 Git | `ac73a4701ace027e8d86b41e1eabfb00c45b5f21` |
| 配置 | `configs/thesis/semantic_pretraining_optimization_probe_v1.json` |
| 配置 SHA-256 | `624bcad605fea7261e02a4f94b72461e07cafbbbf9e34fafcf58e0abb3b63dae` |
| Manifest | `results/thesis_m2/m2-semantic-optimization-probe-v1-full-unweighted-seed20260723/run_manifest.json` |
| Manifest SHA-256 | `78e21cfe3e5c19268a90344238534270e8100cd4116995a267b2065a65f04fcb` |
| 结论 | **多 epoch 能学会 road_line，但四个稀缺交通类仍持续塌缩，允许进入类别权重 probe** |

## 1. 可比性与运行边界

本次只将 pilot 的全量 `7,590` 张 train 预算从 1 epoch 扩展到 5 epochs。完整 validation 保持 `1,725` 张，split、ImageNet 初始化、ResNet50d-FPN、`160x120`、batch size 16、AdamW、learning rate `1e-4`、无类别权重和 seed `20260723` 不变。

运行前 GPU 6 为 `81 MiB`、无 compute owner，Git 工作树干净且 HEAD 与 origin 一致。运行状态为 `completed`、`pipeline_valid=true`，资源峰值 allocated/reserved 为 `995.666/1164.0 MiB`，退出后 GPU 6 回到 `81 MiB`且无 compute owner。

## 2. 五轮指标

| Epoch | Train loss | Train mIoU | Val loss | Val mIoU | Val macro-F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.610946 | 0.263279 | 0.372628 | 0.326204 | 0.382466 |
| 2 | 0.270635 | 0.373885 | 0.259524 | 0.354565 | 0.407319 |
| 3 | 0.188496 | 0.400683 | 0.223341 | 0.365522 | 0.421742 |
| 4 | 0.152493 | 0.422465 | 0.198845 | 0.378555 | 0.435680 |
| 5 | 0.133189 | 0.437040 | 0.189994 | 0.395259 | 0.458918 |

validation mIoU 五轮单调提升，best epoch 为 5。这证明单 epoch 确实优化不足，但不能解释所有稀缺类失败。

## 3. 验证集逐类 IoU

| 类别 | 支持像素 | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| background | 17,959,688 | 0.930523 | 0.940685 | 0.943931 | 0.950436 | 0.950528 |
| road | 10,622,957 | 0.889366 | 0.901172 | 0.906661 | 0.904093 | 0.910809 |
| sidewalk | 2,110,789 | 0.638879 | 0.698115 | 0.708590 | 0.722050 | 0.730449 |
| road_line | 493,249 | 0 | 0.000099 | 0.052530 | 0.086702 | 0.230667 |
| vehicle | 751,311 | 0.469147 | 0.567643 | 0.591263 | 0.617511 | 0.609667 |
| pedestrian | 2,768 | 0 | 0 | 0 | 0 | 0 |
| rider | 13,022 | 0 | 0 | 0 | 0 | 0 |
| traffic_light | 39,820 | 0 | 0 | 0 | 0 | 0 |
| traffic_sign | 2,758 | 0 | 0 | 0 | 0 | 0 |
| barrier | 1,123,638 | 0.334123 | 0.437934 | 0.452241 | 0.504759 | 0.520469 |

road_line 在更多更新后明确脱离零 IoU，因此不应与稀缺小目标一起盲目补采。pedestrian、rider、traffic_light、traffic_sign 在五轮完整验证上始终没有预测像素；训练集对应支持像素分别为 `20,100/57,845/251,712/13,891`，而 background 为 `81,328,888`。下一个最小诊断是固定 inverse-sqrt frequency 权重，仍不扩充数据。

## 4. 产物与证据边界

| 产物 | SHA-256 |
| --- | --- |
| `checkpoint_last.pth` | `d051fcf8ebf01b52177ab1baded0b40e22b649c341c3e60b135e2e008c77eae2` |
| `checkpoint_best.pth` | `cee38bf33dd3409ea2f4cd4d590eebba76589f3f5ca21945f830b93860e0e5da` |
| `backbone_resnet50d.pth` | `6e4fb148aa39c642bdfa9fc56829097918fba76f157b4473778f3cc0b0b42d00` |

骨干导出包含 330 个参数张量，从 best epoch 5 产生并通过 InterFuser strict load。本 probe 仅证明未加权损失下的优化与类别塌缩边界，不是最终 H1 对照，也不足以单独证明必须补采。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
