# M2 交通语义类别权重 probe v1 记录

| 字段 | 内容 |
| --- | --- |
| Run ID | `m2-semantic-class-weight-probe-v1-full-invsqrt-seed20260723` |
| 运行 Git | `1492ea698ece95d7b151c33ebafc025324b17d5a` |
| 配置 | `configs/thesis/semantic_pretraining_class_weight_probe_v1.json` |
| 配置 SHA-256 | `36e03d89bbe55c2fda05a5c26ee12c7d2692bea4b7bd2018cf4af40700e01249` |
| Manifest | `results/thesis_m2/m2-semantic-class-weight-probe-v1-full-invsqrt-seed20260723/run_manifest.json` |
| Manifest SHA-256 | `8413dab95acce8c5dbfa2c752453918744d16fe1f51f61c049a3c9645f1d011a` |
| 结论 | **inverse-sqrt 权重使四个稀缺类全部脱离零 IoU，当前数据允许冻结 M2 v1 骨干候选，不立即补采** |

## 1. 配对契约

本 run 与无权重 optimization probe 使用相同的 `7,590` 张 train、`1,725` 张 validation、M1 split、ImageNet 初始化、ResNet50d-FPN、`160x120`、batch size 16、AdamW、learning rate `1e-4`、5 epochs 和 seed `20260723`。唯一变量是按无权重 train 支持像素派生、mean-one 归一的 inverse-sqrt 类别权重。

runner manifest 回显了全部 10 个权重，Git 工作树为空。运行前 GPU 6 为 `81 MiB`、无 compute owner；运行峰值 allocated/reserved 为 `995.666/1164.0 MiB`，退出后回到 `81 MiB`且无 compute owner。manifest 为 `status=completed`、`pipeline_valid=true`。

## 2. 五轮指标

| Epoch | Train loss | Train mIoU | Val loss | Val mIoU | Val macro-F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.807486 | 0.348577 | 0.598440 | 0.375555 | 0.471361 |
| 2 | 0.403607 | 0.471027 | 0.461842 | 0.423260 | 0.532634 |
| 3 | 0.299821 | 0.500704 | 0.422705 | 0.453262 | 0.567896 |
| 4 | 0.249492 | 0.518603 | 0.385230 | 0.463243 | 0.574830 |
| 5 | 0.219750 | 0.531447 | 0.381424 | 0.460626 | 0.571435 |

best validation mIoU 在 epoch 4；epoch 5 的 train 继续改善但 validation mIoU 轻微回落 `0.002617`，因此骨干候选必须来自 best checkpoint，不能用 last checkpoint 替换。

## 3. 与无权重 best 的逐类对比

| 类别 | 无权重 epoch 5 IoU | 加权 epoch 4 IoU | 差值 |
| --- | ---: | ---: | ---: |
| background | 0.950528 | 0.939638 | -0.010890 |
| road | 0.910809 | 0.900074 | -0.010736 |
| sidewalk | 0.730449 | 0.731940 | +0.001491 |
| road_line | 0.230667 | 0.329192 | +0.098526 |
| vehicle | 0.609667 | 0.590352 | -0.019315 |
| pedestrian | 0 | 0.056668 | +0.056668 |
| rider | 0 | 0.258753 | +0.258753 |
| traffic_light | 0 | 0.156095 | +0.156095 |
| traffic_sign | 0 | 0.176434 | +0.176434 |
| barrier | 0.520469 | 0.493283 | -0.027186 |
| **mIoU** | **0.395259** | **0.463243** | **+0.067984** |

加权并非无代价：background、road、vehicle、barrier 有小幅回落，但四个稀缺交通类全部形成有效预测，且总体 mIoU 提升。pedestrian 仅 `2,768` 个 validation 支持像素，其 `0.056668` IoU 仍然是最大的方差风险；这要求后续报告置信边界，但不构成立即补采的门禁。

## 4. 产物与后续准入

| 产物 | SHA-256 |
| --- | --- |
| `checkpoint_last.pth` | `440c2ee99dc2b815c16422abcf9cbbc15e5c474efd251c87c182525750bc5995` |
| `checkpoint_best.pth` | `fb7be7eb26bc7efac7089a2a5d5b7cc9c4b81ca96385ca1481f650dc52bdc034` |
| `backbone_resnet50d.pth` | `17aac98fcb3b513a672de6edd68ca4e697157fdfb285c4cc9cbfb1151bb8298e` |

best 骨干包含 330 个参数张量，从磁盘重载后再次通过 InterFuser strict load。它现在是 M2 v1 的视觉骨干候选，下一项工作是以不改变下游结构、数据和预算的方式接入 InterFuser V 组，与通用 ImageNet 初始化作配对对照。本结果仍是单 seed 与低分辨率 pilot，不能单独当作最终 H1 结论。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
