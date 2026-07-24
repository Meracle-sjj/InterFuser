# M2 InterFuser B0/V 配对训练 smoke v1 记录

| 字段 | 内容 |
| --- | --- |
| Run ID | `m2-interfuser-visual-pair-smoke-v1-seed20260723-20260724-v1` |
| 运行 Git | `65f3946d35dc08ad9b378a3b5220ebe1fe081ad6` |
| 配置 | `configs/thesis/interfuser_visual_pair_smoke_v1.json` |
| 配置 SHA-256 | `bffd0847169feb797297333d9c04de7251da0795482e1c6fbb54d3aee7378f30` |
| Manifest | `results/thesis_m2/m2-interfuser-visual-pair-smoke-v1-seed20260723-20260724-v1/run_manifest.json` |
| Manifest SHA-256 | `8f2b44f25f116daf12021695cbf0a77555e256f347c2733c4384e807114c1018` |
| 结论 | **B0/V 下游训练链路均 pipeline valid，允许冻结并执行正式配对训练；smoke 数值不进入 H1 结论** |

## 1. 单变量与数据边界

runner 从冻结 train/validation index 以 seed `20260724` 各抽 2 个完整 sequence，分别得到 32/23 个逻辑帧。B0/V 共享 InterFuser 结构、数据、增强、optimizer、学习率、单 epoch、每 GPU batch 2、GPU 6/7 和训练 seed `20260723`；唯一差异是初始化 checkpoint。归一训练参数 SHA-256 均为 `d25b397700822ee70a420f087aab7fba4e377ceb41edc3664ab12b6938ed189b`。

输入下游 split manifest SHA-256 为 `711a70dcfffd7da9e49d68a71ef83f4ee3f1dc8a49d0b41c6d06f7b3d3bb4f70`，初始化 manifest SHA-256 为 `50ad2018a2c015829e99c56e8f7493ba87e1e7e7781345a8bc47ddf4250a99e5`。B0/V 初始 checkpoint SHA-256 分别为 `92409ebf2e354595dc400cd73d9e900da582e68ab8d606217281cc04ecab45b0` 与 `338e879c3005be0840cfd0813cc8f05bfcf6b8677ca5cf36add8a1fc627a207a`。

## 2. 运行结果

| Variant | Train loss | Validation loss | Validation L1 | 时长 | 峰值显存 GPU 6/7 | Best checkpoint SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| B0 | 0.319727 | 0.341607 | 6.437237 | 21.999 s | 2880/2847 MiB | `658efe1ed413e5c30f7aac88632f4d1dc60b6cea791635cd2b1af4286952d453` |
| V | 0.318663 | 0.339381 | 6.457063 | 25.148 s | 2880/2847 MiB | `9143fd1429f67800e549230a9ac50f84f620dcee52373481c85a71b1cd2309a2` |

两组 process exit code 均为 0，无 external timeout 或 GPU monitor error；best/last checkpoint、summary 和 args 均存在且已记录哈希。两组 checkpoint 均包含 1,132 个张量，schema SHA-256 同为 `8d5c54ed9e951dfbee6a1a07cab7453d860cb29445ca4165e6793d34c5c38c13`。

## 3. 资源释放与证据边界

B0/V 结束后 GPU 6/7 分别回到 `81/45 MiB`，无 compute owner；分布式端口 29655 无监听，launcher 与 worker 进程均退出。B0/V 的单 epoch 数值来自 2+2 sequence，只能证明多视角 RGB、LiDAR、多任务 loss、分布式反向、验证、checkpoint 和资源回收链路，不支持任何模型优劣结论。

smoke 基础设施提交前定向测试 `12/12`、完整 unittest `187/187` 通过；补充 formal test index 强制绑定后定向测试 `5/5`、完整 unittest `188/188` 通过。

正式训练沿用上游 2 GPU、每 GPU batch 16、25 epochs 和 5 warmup epochs 配方，并绑定无泄漏 train/validation/test 三组索引。只有正式 best checkpoint 的冻结 test 评价与两组 D7 配对闭环结果能够回答 H1。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
