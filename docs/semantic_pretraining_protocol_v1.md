# M2 交通语义预训练协议 v1.0

| 字段 | 内容 |
| --- | --- |
| 状态 | **CLASS-WEIGHT-VALID / V-INTEGRATION-PENDING：稀缺类已脱离零 IoU，best 骨干候选已冻结** |
| 生效日期 | 2026-07-23 |
| 服务假设 | H1：交通域 ResNet-50 预训练改善语义质量、时序稳定性与闭环表现 |
| 数据契约 | `docs/traffic_pretraining_dataset_v1.md` |
| Smoke 配置 | `configs/thesis/semantic_pretraining_smoke_v1.json` |
| Pilot 配置 | `configs/thesis/semantic_pretraining_pilot_v1.json` |
| 优化诊断配置 | `configs/thesis/semantic_pretraining_optimization_probe_v1.json` |
| 类别权重配置 | `configs/thesis/semantic_pretraining_class_weight_probe_v1.json` |
| 训练入口 | `tools/training/run_semantic_pretraining.py` |

## 1. v1 模型边界

视觉骨干必须直接复用 InterFuser 仓库内 timm 0.4.13 的 `resnet50d`，特征层固定为 `[1,2,3,4]`。训练时附加轻量 FPN 语义分割 head，head 只服务 10 类 CARLA 交通语义监督，迁移时丢弃。

初始化固定为本机缓存的 timm ImageNet `resnet50d_ra2`，checkpoint SHA-256 必须与配置一致。禁止在同一实验中更换 backbone、初始化来源、类别 schema、split 或数据增强后仍沿用同一 run ID。

## 2. 数据与标签

训练与验证只消费 M1 冻结 split manifest；不得重新随机划分帧。一个样本是一个完整 sequence 中某相机、某 frame ID 的 RGB/semantic 对，原始 CARLA uint8 标签通过 `semantic_classes_v1.json` 唯一映射为 train ID 0-9，`255/Any` 保持 ignore。

Smoke 仅从 train/validation 中以固定 seed 对完整样本键作 SHA-256 排序，分别取 32/16 张并缩放到 `160x120`。该抽样只证明训练链路，不进入 H1 精度结论，也不改变 M1 完整 split。

## 3. 指标与产物

每个 epoch 必须记录训练与验证的交叉熵、pixel accuracy、mIoU、macro-F1、10 类逐类 IoU/F1、样本数和耗时。结果目录必须拒绝覆盖并至少包含：

当前 PyTorch nightly 的 CUDA `nll_loss2d` 不提供 strict deterministic 实现，因此训练器使用数学等价的 `log_softmax + gather + ignore mask` 计算交叉熵；无权重时对有效像素取 mean，加权时按有效 target weight 之和归一。两种实现都必须与 CPU 标准 `CrossEntropyLoss(ignore_index=255)` 通过数值等价测试，禁止降级为 `warn_only` 掩盖不确定算子。

- `run_manifest.json`：Git/config/data/初始化/GPU provenance、逐 epoch 指标和产物哈希；
- `checkpoint_last.pth`：模型与 optimizer 的可恢复状态；
- `backbone_resnet50d.pth`：仅含 `backbone.*` 的 weights-only 迁移产物。

骨干导出必须在落盘前严格加载到 InterFuser 同构 `resnet50d(features_only=True, out_indices=[4])`。任何 missing/unexpected key 都使 pipeline invalid。

## 4. Smoke 完成门槛

Smoke 只有同时满足以下条件才算跑通：

1. 启动前目标 GPU 无外部 compute owner，Git 工作树干净；
2. 32 个 train 与 16 个 validation 样本完成 1 epoch，loss 和指标均为有限值；
3. run manifest 明确为 `pipeline_valid=true`；
4. checkpoint 与骨干导出存在且 SHA-256 已记录；
5. 骨干导出通过 InterFuser 严格加载测试；
6. 相关定向测试和完整 unittest 通过。

Smoke 通过只解除 M2 工程链路门禁，不证明数据量充分、交通域预训练有效或 H1 成立。下一阶段必须先检查学习曲线与稀缺类别方差，再冻结 pilot 训练预算。

## 5. 当前 smoke 结果

有效 Run ID 为 `m2-semantic-smoke-v1-seed20260723-20260723-v2`，运行 Git 为 `dc3beb06a01f4a45daad51f37c0bb34f4f7aaa85`。32 个 train 与 16 个 validation 样本完成 1 epoch，manifest 为 `status=completed`、`pipeline_valid=true`；训练 loss/mIoU/macro-F1 为 `1.747226/0.095270/0.132538`，验证为 `1.420750/0.176070/0.206419`。

验证抽样没有 pedestrian 和 traffic_sign 支持像素，因此上述数值只证明前向、反向、指标与落盘链路有效，不能用于类别精度比较。正式 pilot 必须使用完整 validation split 或显式保证核心类别覆盖，不得沿用 16 张 smoke 指标解释 H1。

有效产物位于 `results/thesis_m2/m2-semantic-smoke-v1-seed20260723-20260723-v2/`。骨干导出包含 330 个参数张量，内存中与落盘后均严格加载成功；GPU 6 峰值 allocated/reserved 为 `503.939/540.0 MiB`，进程退出后回到 `81 MiB` 且无 compute owner。完整事实、失败前序与 SHA-256 记录在 `docs/experiment_records/2026-07-23-m2-semantic-smoke-v1.md`。

## 6. Pilot 学习曲线契约

pilot 使用完整的 1,725 张 validation 图像，并从同一 SHA-256 排序后的 7,590 张 train 图像中取严格嵌套的 `1,898/3,795/7,590` 张，约对应 25%/50%/100%。三次 run 都从相同 ImageNet checkpoint 独立初始化，固定 `160x120`、batch size 16、1 epoch、AdamW、learning rate `1e-4` 与 seed `20260723`；禁止从较小数据 run 的 checkpoint 继续训练较大数据 run。

学习曲线汇总只有在三个 run 全部 pipeline valid、Git/config/split/初始化哈希一致、train 样本严格嵌套、validation 样本键完全一致且 checkpoint/骨干文件哈希仍有效时才生成。该 pilot 用于判断现有数据是否仍处在明显的数据受限区间；单 epoch 结果仍不是最终 H1 对照，后续正式预算必须由曲线形态决定。

## 7. Pilot 学习曲线结果

三个 run 均在 Git `431badc9849489b990d1e661441371dc3d8b87f4` 上完成并通过汇总门禁。验证 loss 随 train 样本 `1,898/3,795/7,590` 从 `0.731764` 降至 `0.535311/0.372628`，mIoU 为 `0.215539/0.243451/0.326204`，macro-F1 为 `0.245208/0.280995/0.382466`。学习曲线报告位于 `results/thesis_m2/semantic_learning_curve_pilot_v1_seed20260723_20260723T030714Z.json`，SHA-256 为 `47503a43ce8b34ce04c9a29c91d2fc788188165e694e9d7370f2ed3e600c9954`。

完整 validation 在三个点均覆盖全部 10 类且样本键完全一致。background、road、sidewalk 随数据稳定改善；vehicle 和 barrier 只在 50% 以后出现有效预测。road_line、pedestrian、rider、traffic_light、traffic_sign 在单 epoch 三个点均为 0。由于固定 epoch 使更大数据同时获得更多 optimizer step，曲线只能证明“更多数据与更多更新共同改善结果”，不能单独估计数据量因果效应。

后续无权重五轮诊断已按第 8-9 节完成。road_line 通过增加优化预算脱离零 IoU，而四个稀缺交通类仍持续塌缩，因此下一步是只改变损失权重的配对 probe。

## 8. 优化充分性 probe 契约

optimization probe 固定使用全部 7,590 张 train 与完整 1,725 张 validation，保持 ImageNet 初始化、`160x120`、batch size 16、AdamW、learning rate `1e-4`、无类别权重和 seed `20260723`，仅把预算扩展为 5 epochs。该 run 不允许 `--train-samples` 改写全量数据边界。

训练器每轮记录完整 train/validation 指标，同时保存可恢复的 `checkpoint_last.pth` 和按 validation mIoU 选择的 weights-only `checkpoint_best.pth`。`backbone_resnet50d.pth` 必须从 best epoch 导出并再次通过 InterFuser strict load。该 probe 只区分优化不足与持续类别塌缩，不是最终预训练预算。

## 9. 优化充分性 probe 结果

Run ID `m2-semantic-optimization-probe-v1-full-unweighted-seed20260723` 在 Git `ac73a4701ace027e8d86b41e1eabfb00c45b5f21` 上完成，manifest 为 `pipeline_valid=true`。validation loss/mIoU/macro-F1 从 epoch 1 的 `0.372628/0.326204/0.382466` 改善到 epoch 5 的 `0.189994/0.395259/0.458918`，best epoch 为 5。

road_line IoU 从 0 增至 `0.230667`，vehicle 和 barrier 分别为 `0.609667/0.520469`。pedestrian、rider、traffic_light、traffic_sign 在五轮 validation 中始终为 0。完整 manifest SHA-256 为 `78e21cfe3e5c19268a90344238534270e8100cd4116995a267b2065a65f04fcb`，事实记录位于 `docs/experiment_records/2026-07-23-m2-semantic-optimization-probe-v1.md`。

## 10. 类别权重 probe 契约

类别权重 probe 保持第 8 节全部数据、模型、初始化、优化器、五轮预算与 seed，唯一变量是交叉熵的显式类别权重。权重从无权重 run 的 epoch 5 train 支持像素以 `1/sqrt(count)` 计算并归一到 mean 1，按 train ID 0-9 固定为：

`[0.04598168, 0.06279949, 0.12245012, 0.29846202, 0.29153371, 2.92488764, 1.72414705, 0.82652360, 3.51836151, 0.18485318]`

inverse-sqrt 在提高稀缺类梯度的同时比直接 inverse-frequency 更克制，避免 5,855 倍像素差异直接转化为极端损失倍率。配置必须显式记录支持像素、派生方法、来源 run 和 manifest 哈希；runner manifest 必须回显实际类别权重。若加权后稀缺类形成有效验证预测，优先冻结合理权重/预算；若仍为 0 或高方差失败，再按 M1 provenance 最小定向补采 pedestrian/rider/traffic-sign，不扩大停止边界方向。

## 11. 类别权重 probe 结果与 M2 v1 候选

Run ID `m2-semantic-class-weight-probe-v1-full-invsqrt-seed20260723` 在 Git `1492ea698ece95d7b151c33ebafc025324b17d5a` 上完成，manifest 为 `pipeline_valid=true`。best epoch 4 的 validation mIoU/macro-F1 为 `0.463243/0.574830`，较无权重 best 的 `0.395259/0.458918` 分别提高 `0.067984/0.115912`。

pedestrian、rider、traffic_light、traffic_sign 的 best-checkpoint validation IoU 为 `0.056668/0.258753/0.156095/0.176434`，四类全部脱离零预测。加权使 background、road、vehicle、barrier 有 `0.0109-0.0272` 的 IoU 回落，但总体 mIoU 与交通关键类覆盖均改善。manifest SHA-256 为 `8413dab95acce8c5dbfa2c752453918744d16fe1f51f61c049a3c9645f1d011a`，best 骨干 SHA-256 为 `17aac98fcb3b513a672de6edd68ca4e697157fdfb285c4cc9cbfb1151bb8298e`，330 个参数张量从磁盘重载后 strict load 成功。

该 best 骨干冻结为 M2 v1 下游候选。当前数据已足以形成可学习的十类交通语义表征，因此不启动补采；下一门禁是将该骨干作为唯一变量接入 InterFuser V 组，并与原始通用预训练在同一下游预算下对照。pedestrian 验证支持仍小，必须作为方差风险保留，不得把单 seed pilot 跨界表述为最终 H1 结论。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
