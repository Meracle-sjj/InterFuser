# M2 行人碰撞威胁数据扩充 probe v1

## 结论

在 ResNet50d-FPN、ImageNet 初始化、inverse-sqrt 类别权重、学习率、batch size、5 epoch、seed 和原 validation 完全不变的条件下，仅增加经碰撞威胁审计的 4,317 个 RGB 样本：

- 原 validation mIoU 从 `0.463243` 增至 `0.486751`，绝对提升 `+0.023508`。
- 原 validation macro-F1 从 `0.574830` 增至 `0.605853`，绝对提升 `+0.031023`。
- 原 validation 行人 IoU 从 `0.056668` 增至 `0.120473`，绝对提升 `+0.063805`，相对为 2.126 倍。
- 两个 route-group 隔离的行人危险 holdout 上，mIoU 均提升约 0.105，行人 IoU 分别提升 0.1310 和 0.1103，且 10/10 类 IoU 均改善。

这是当前最强的 H1 数据侧证据：CARLA 特权真值可以只用于发现危险训练片段，而输入仍为 RGB 的视觉骨干能在未见危险路线上取得可量化改善。

## 配对契约

| 字段 | baseline | hazard augmented |
| --- | --- | --- |
| train 样本 | 7,590 | 11,907 |
| validation 样本 | 1,725 | 1,725（key 不变） |
| 新增 train sequence | 0 | 38 |
| epoch | 5 | 5 |
| seed | 20260723 | 20260723 |
| 类别权重 | 同一 inverse-sqrt 向量 | 同一 inverse-sqrt 向量 |
| checkpoint 选择 | 原 validation mIoU | 原 validation mIoU |
| 危险 holdout 参与选模 | 否 | 否 |

此 probe 回答的是“危险数据策略是否有效”，因此 train 样本数的改变是预注册的唯一自变量，不应把它表述为新网络结构的收益。

## 原 validation

| 指标 | baseline | augmented | 差值 |
| --- | ---: | ---: | ---: |
| mIoU | 0.463243 | 0.486751 | +0.023508 |
| macro-F1 | 0.574830 | 0.605853 | +0.031023 |
| 行人 IoU | 0.056668 | 0.120473 | +0.063805 |
| 行人 F1 | 0.107258 | 0.215040 | +0.107781 |
| 车辆 IoU | 0.590352 | 0.633473 | +0.043122 |
| traffic light IoU | 0.156095 | 0.186916 | +0.030821 |
| traffic sign IoU | 0.176434 | 0.249968 | +0.073535 |

10 类中 8 类 IoU 提升；rider `-0.005022`、barrier `-0.012420`。因此原 validation 上的改善很明确，但不是每类严格 Pareto 改善。

## 未见行人危险 route-group holdout

| split | 样本 | 指标 | baseline | augmented | 差值 |
| --- | ---: | --- | ---: | ---: | ---: |
| validation | 1,221 | mIoU | 0.526224 | 0.631036 | +0.104811 |
| validation | 1,221 | macro-F1 | 0.645715 | 0.753965 | +0.108250 |
| validation | 1,221 | 行人 IoU | 0.250361 | 0.381365 | +0.131005 |
| validation | 1,221 | 行人 F1 | 0.400462 | 0.552157 | +0.151695 |
| test | 1,155 | mIoU | 0.485536 | 0.591046 | +0.105510 |
| test | 1,155 | macro-F1 | 0.604928 | 0.717513 | +0.112585 |
| test | 1,155 | 行人 IoU | 0.185520 | 0.295857 | +0.110337 |
| test | 1,155 | 行人 F1 | 0.312976 | 0.456620 | +0.143644 |

validation/test 各包含 8 个 Town 分层 route group，与扩充 train 的 route-group 交集为 0。两个 split 均为 10 类 IoU 提升、0 类持平、0 类变差。

## 原始产物

- baseline run manifest：`results/thesis_m2/m2-semantic-class-weight-probe-v1-full-invsqrt-seed20260723/run_manifest.json`，SHA-256 `8413dab95acce8c5dbfa2c752453918744d16fe1f51f61c049a3c9645f1d011a`。
- baseline best checkpoint：SHA-256 `fb7be7eb26bc7efac7089a2a5d5b7cc9c4b81ca96385ca1481f650dc52bdc034`，best epoch 4。
- augmented run manifest：`results/thesis_m2/m2-semantic-pedestrian-hazard-probe-v1-full-invsqrt-seed20260723/run_manifest.json`，SHA-256 `86da8ebcc25afeaaf94abc04a5cefe0389f0f2bf91e3ba74b4d57d3ba5936a3e`。
- augmented best checkpoint：SHA-256 `345ab7257fc9c20758df3612cac4f1f069570c80d8a90f6c3e454512e94ca757`，best epoch 5。
- augmented 可迁移骨干：SHA-256 `a0673b36325b54695f82aef52829c34aaaa4ed7652ff81ee3bee36bf180ea56d`，330 个参数张量 strict-load 通过。
- baseline 危险 holdout 报告：`results/thesis_m2/semantic_hazard_holdout_baseline_20260804.json`，SHA-256 `991bcfcae047cf676d0488756734b7639b8b8ad45e7f2ee705a3d67a35faa474`。
- augmented 危险 holdout 报告：`results/thesis_m2/semantic_hazard_holdout_augmented_20260804.json`，SHA-256 `62a84f2c3e6a4e51ca50520cd00111e0e423042b2a132e1d165d711f7170fcfb`。
- 配对归约：`results/thesis_m2/semantic_hazard_holdout_paired_summary_20260804.json`，SHA-256 `b9d966b1c2d9e8cfe5e48490b1e08076e021587e61c4b71d6a87d8c3bf2f52b6`。

## InterFuser 迁移与下游 smoke

新骨干已通过 InterFuser 单变量迁移和一轮 B0/V 下游训练 smoke：

- 初始化 manifest pipeline-valid，仅 330 个唯一 RGB 骨干张量（全模型两个 alias 共 660 张量）发生变化，非 RGB state SHA-256 一致，全模型 strict-load 通过。
- B0 初始 checkpoint SHA-256 仍为 `92409ebf2e354595dc400cd73d9e900da582e68ab8d606217281cc04ecab45b0`，与 v1 字节相同；新 V 为 `68b46f522569ba87663dd8cba6d3413ba1cb4acf39754008aceb6e3d74d43ada`。
- smoke 对 B0/V 使用同一 2 个 train sequence、2 个 validation sequence、单 epoch、GPU 6/7 和归一化参数哈希；两者均 pipeline-valid，运行后 GPU 完全释放。
- smoke validation L1 为 B0 `6.437353`、V `6.391468`；此差值只证明链路可运行，样本与 epoch 预算不允许将其当作 H1 收益。

迁移产物：

- 初始化配置 SHA-256 `83296a26ed15fb23deae2e7c2c12a9128a632db0a9d69fe6878dd508d97b9f0d`。
- 初始化 manifest：`results/thesis_m2/m2-interfuser-visual-init-pedestrian-hazard-v1-seed20260723/initialization_manifest.json`，SHA-256 `976be079a4783aaaaa7ebe0137382fd18e750fc389ef38d965168cf06070af9f`。
- smoke 配置 SHA-256 `a9c895f1fe8c607a0863dead517f9da47d5bef83b06ea9bb4bd37ec2b6538212`。
- smoke run manifest：`results/thesis_m2/m2-interfuser-visual-pair-pedestrian-hazard-smoke-v1-seed20260723/run_manifest.json`，SHA-256 `3f262994c7fa5bb4eb24627045e7cabcb7e0d87a9f86af1be531020ec0599a4a`。

## 证据边界与下一门禁

当前只完成一个语义训练 seed，还不是方差估计。这些结果证明离线交通语义与行人危险泛化改善，但不能推导闭环 Driving Score 改善。迁移/smoke 已消除 schema 与运行链路风险；下一步是正式下游训练，然后必须先过 route39 起步 40 周期轨迹/偏移门禁，再决定是否重跑完整 D7。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
