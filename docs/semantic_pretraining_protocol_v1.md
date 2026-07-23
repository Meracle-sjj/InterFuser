# M2 交通语义预训练协议 v1.0

| 字段 | 内容 |
| --- | --- |
| 状态 | **SMOKE-PENDING：训练与迁移链路待真实运行验证** |
| 生效日期 | 2026-07-23 |
| 服务假设 | H1：交通域 ResNet-50 预训练改善语义质量、时序稳定性与闭环表现 |
| 数据契约 | `docs/traffic_pretraining_dataset_v1.md` |
| Smoke 配置 | `configs/thesis/semantic_pretraining_smoke_v1.json` |
| 训练入口 | `tools/training/run_semantic_pretraining.py` |

## 1. v1 模型边界

视觉骨干必须直接复用 InterFuser 仓库内 timm 0.4.13 的 `resnet50d`，特征层固定为 `[1,2,3,4]`。训练时附加轻量 FPN 语义分割 head，head 只服务 10 类 CARLA 交通语义监督，迁移时丢弃。

初始化固定为本机缓存的 timm ImageNet `resnet50d_ra2`，checkpoint SHA-256 必须与配置一致。禁止在同一实验中更换 backbone、初始化来源、类别 schema、split 或数据增强后仍沿用同一 run ID。

## 2. 数据与标签

训练与验证只消费 M1 冻结 split manifest；不得重新随机划分帧。一个样本是一个完整 sequence 中某相机、某 frame ID 的 RGB/semantic 对，原始 CARLA uint8 标签通过 `semantic_classes_v1.json` 唯一映射为 train ID 0-9，`255/Any` 保持 ignore。

Smoke 仅从 train/validation 中以固定 seed 对完整样本键作 SHA-256 排序，分别取 32/16 张并缩放到 `160x120`。该抽样只证明训练链路，不进入 H1 精度结论，也不改变 M1 完整 split。

## 3. 指标与产物

每个 epoch 必须记录训练与验证的交叉熵、pixel accuracy、mIoU、macro-F1、10 类逐类 IoU/F1、样本数和耗时。结果目录必须拒绝覆盖并至少包含：

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

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
