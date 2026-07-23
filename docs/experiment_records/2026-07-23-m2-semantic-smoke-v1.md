# M2 交通语义预训练 smoke v1 记录

| 字段 | 内容 |
| --- | --- |
| 有效 Run ID | `m2-semantic-smoke-v1-seed20260723-20260723-v2` |
| 运行 Git | `dc3beb06a01f4a45daad51f37c0bb34f4f7aaa85` |
| 配置 | `configs/thesis/semantic_pretraining_smoke_v1.json` |
| 配置 SHA-256 | `47b00c07f39ccf260482fa65ae407a18f1b25d8f87f760899ad7997e75798922` |
| Split SHA-256 | `81d9403a5ecd39fb5c4cf7ac8ffd3c79505a9583d683a960cf5a02b6da1fe8f2` |
| ImageNet 权重 SHA-256 | `464e36baa940a80971e717454e61e8f6fc1a1b284b7a3dbb6f6fd4c3da78533d` |
| Run manifest SHA-256 | `19fec32620e9c501a5bdba26d2888d53bb713d0bd108de58cbcc82fc55170347` |
| 完整 checkpoint SHA-256 | `304d907555b7fc9b99697d353ef4b52e44a74adef5641447baadcc1740a51337` |
| 骨干导出 SHA-256 | `b6d945a17a24ec56ea72c806b19ee48474f00b2525ecf7848ddf7cd4d73f9999` |
| 结论 | **M2 smoke 有效：训练、验证、恢复 checkpoint 与 InterFuser 骨干迁移链路已跑通** |

## 1. 冻结输入与预算

本次直接消费 M1 数据 v1 的冻结 split，不重新划分帧。模型是仓库同构 `resnet50d(features_only=True, out_indices=[1,2,3,4])` 加一次性 FPN 分割 head，初始化来自已缓存并校验哈希的 timm ImageNet `resnet50d_ra2`。输出为 10 类 CARLA 交通语义，raw tag 到 train ID 的映射只来自 `semantic_classes_v1.json`。

Smoke 预算固定为 sample seed `20260723`、32 个 train 样本、16 个 validation 样本、`160x120`、batch size 4、1 epoch、AdamW、learning rate `1e-4`、weight decay `0.01`。抽样对完整 `sequence:camera:frame` 键作 SHA-256 排序，目的是快速覆盖训练全链路，不进入 H1 精度结论。

## 2. v1 失败前序与修复边界

首次 Run ID `m2-semantic-smoke-v1-seed20260723-20260723-v1` 在 Git `44d0d23226075cf7918f65e7b662385162edff11` 上通过 GPU、Git、数据与模型初始化门禁，但在第一个 CUDA loss 前向以非零状态退出。manifest 明确记录 `pipeline_valid=false`，SHA-256 为 `4d3ce890f932095fb3661f08dbe1842f417e03332d89b90422422399946f5fde`；launcher log SHA-256 为 `b242cce3af92641c10aa9c85bfab826d99c7b66efd751ed1c30ddad7b3a85230`。

根因是当前 PyTorch `2.10.0.dev20251104+cu128` 的 CUDA `nll_loss2d` 不提供 strict deterministic 实现，不是数据损坏、模型失败或 GPU 冲突。提交 `dc3beb0` 将损失替换为数学等价的 `log_softmax + gather + ignore mask + mean`；最小 CUDA 探针证明其前后向可在 strict deterministic 下运行，并由单元测试证明与 CPU 标准 `CrossEntropyLoss(ignore_index=255)` 数值一致。没有降级为 `warn_only`。

v1 目录和日志原地保留，不续跑、不覆盖，也不与 v2 结果拼接。

## 3. v2 有效结果

v2 从干净提交 `dc3beb0` 启动，于 `2026-07-23T02:45:03.686668+00:00` 至 `2026-07-23T02:45:06.953127+00:00` 完成。manifest 为 `status=completed`、`pipeline_valid=true`、`errors=[]`。

| Phase | Loss | Pixel accuracy | mIoU | Macro-F1 | 样本 | 计算耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1.747226 | 0.559946 | 0.095270 | 0.132538 | 32 | 1.249 s |
| validation | 1.420750 | 0.786253 | 0.176070 | 0.206419 | 16 | 0.200 s |

train 抽样覆盖全部 10 类支持像素；validation 抽样没有 pedestrian 和 traffic_sign 支持像素，因此这两类指标为 `null`。其余小类多数为 0，符合 ImageNet 初始化模型经过仅 8 个优化 step 的预期。该结果证明 loss、反向、optimizer、验证与混淆矩阵链路工作，不证明交通域预训练有效。

## 4. 产物与迁移自证

有效目录为 `results/thesis_m2/m2-semantic-smoke-v1-seed20260723-20260723-v2/`：

- `checkpoint_last.pth`：约 276 MiB，包含模型和 optimizer 状态，可用于恢复；
- `backbone_resnet50d.pth`：约 91 MiB，只包含带 `backbone.` 前缀的视觉骨干；
- `run_manifest.json`：记录 48 个样本键、依赖版本、逐类指标、资源和全部产物哈希。

骨干导出含 330 个参数张量。训练进程内验证和退出后从磁盘以 `weights_only=True` 重载验证，均能严格加载到 InterFuser 的 `resnet50d(features_only=True, out_indices=[4])`，无 missing 或 unexpected key。

## 5. 资源、测试与证据边界

v2 启动前 GPU 6 为 `81 MiB` 且无 compute owner，GPU 7 的外部任务未被触碰。运行设备为 NVIDIA GeForce RTX 5090，GPU 6 峰值 allocated/reserved 为 `503.939/540.0 MiB`；退出后回到 `81 MiB`、0% utilization 且无 compute owner。

确定性损失修复后，M2 定向测试为 `5/5`，带完整仓库、InterFuser、Leaderboard、Scenario Runner 和 CARLA PythonAPI `PYTHONPATH` 的全量 unittest 为 `168/168`。

本次只解除 M2 工程链路门禁。16 张 validation smoke 样本不具备核心类别完整性，当前指标不得进入论文表格或用于判断 H1。下一步应使用完整 validation split，先跑 25%/50%/100% train 学习曲线或一个冻结预算 pilot，再根据 pedestrian、rider、traffic_sign 的方差决定最小定向扩容。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
