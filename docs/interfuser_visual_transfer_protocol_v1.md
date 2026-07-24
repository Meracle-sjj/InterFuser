# M2 InterFuser 视觉权重迁移与 H1 对照协议 v1.0

| 字段 | 内容 |
| --- | --- |
| 状态 | **SMOKE-PENDING：无泄漏索引与 B0/V strict 初始化对已验证，配对训练 smoke 待运行** |
| 服务假设 | H1：交通域 ResNet-50 初始化优于通用 ImageNet 初始化 |
| 数据配置 | `configs/thesis/interfuser_downstream_split_v1.json` |
| 初始化配置 | `configs/thesis/interfuser_visual_initialization_v1.json` |
| 配对 smoke 配置 | `configs/thesis/interfuser_visual_pair_smoke_v1.json` |
| 索引构建器 | `tools/data/build_interfuser_downstream_indexes.py` |
| 初始化生成器 | `tools/training/interfuser_visual_pair.py` |
| 配对训练 runner | `tools/training/run_interfuser_visual_pair.py` |

## 1. 单变量边界

B0 与 V 必须使用同一 `interfuser_baseline` 结构、非 RGB 参数初始值、下游数据、增强、optimizer、学习率、epoch、batch size、GPU 数和 seed。唯一允许的初始差异是：

- B0 RGB backbone 来自 timm ImageNet `resnet50d_ra2`；
- V RGB backbone 来自 M2 类别权重 probe best epoch 4；
- LiDAR backbone、Transformer、任务头和所有 buffer 必须逐张量相同。

迁移不在 InterFuser 模型定义中增加常驻分支。生成器先创建标准 B0 模型，再 strict-load 330 个唯一 RGB 骨干张量，输出两个全模型 checkpoint；下游复用原生 `--initial-checkpoint` strict load。不传 checkpoint 时上游默认行为不变。

## 2. 下游数据泄漏边界

历史 InterFuser 以 Town01/04/05 训练、Town03 验证，但 M2 语义预训练 train 已包含部分 Town03 sequence，直接复用历史划分会让 V 在预训练阶段看到下游验证样本。因此 v1 以 M1 已冻结的 `Town+route_id` 为原子组，将其归属扩展到全量 9,968 个 sequence：

- 冻结 validation/test route group 在所有天气下仍分别归 validation/test；
- 冻结 train route group 与 M1 未见 route group 归 train；
- 三组 route group 交集必须为空，每条全量索引记录必须恰好分配一次。

validation 只用于训练选择，test 才是 H1 离线结论的权威集。任何泄漏或索引哈希漂移都阻止正式训练。

## 3. 基础设施完成门槛

1. 全量下游索引 manifest 证明 train/validation/test 无 route-group 重叠且记录数守恒；
2. B0 RGB 状态与冻结 ImageNet checkpoint 按 PyTorch strict-load 语义等价；原文件中 55 个 BatchNorm `num_batches_tracked` 从 float32 规范化为目标 int64，其他键/形状/dtype/数值必须严格一致；
3. V 导出 330 个张量 strict-load 到 RGB backbone；
4. 全模型的 660 个 RGB alias state key 变化，其余 state key 内容哈希完全相同；
5. B0/V 初始 checkpoint 都能被 `interfuser_baseline` strict load；
6. 相关定向测试与完整 unittest 通过，产物哈希和 Git provenance 固化后才允许训练 smoke。

## 4. 证据边界

本协议的基础设施只证明“对照可归因”，不证明 V 已优于 B0。必须依次完成配对训练 smoke、正式训练、冻结 test 离线评价与 D7 三种子闭环评测，才能判定 H1。

已验证输入为下游 split manifest `711a70dcfffd7da9e49d68a71ef83f4ee3f1dc8a49d0b41c6d06f7b3d3bb4f70` 与初始化 run `m2-interfuser-visual-init-pair-v1-seed20260723-v3`（manifest `50ad2018a2c015829e99c56e8f7493ba87e1e7e7781345a8bc47ddf4250a99e5`）。前者覆盖 9,968 个 sequence 且 train/validation/test route group 两两无交集；后者证明 330 个唯一 RGB 张量变化、全部 660 个 RGB alias 变化、非 RGB state 哈希一致且两个全模型 checkpoint 均可 strict load。

## 5. 配对训练 smoke 契约

smoke 从已冻结全量 train/validation index 中用 seed `20260724` 分别抽取 2 个完整 sequence，仅用于验证多视角 RGB、LiDAR、多任务 loss、分布式反向、验证、checkpoint 与资源回收链路。B0 和 V 按固定顺序在 GPU 6/7 串行运行，共享单 epoch、每 GPU batch 2、AdamW/cosine 和 seed `20260723`；只有 `initial_checkpoint`、experiment 名和输出目录允许不同。

runner 必须拒绝已有 GPU compute owner、分布式端口占用、Git 脏工作树、索引/初始化哈希漂移、非零进程退出、超时、缺失/非有限 summary 或 checkpoint 结构变化。任一 variant invalid 立即停止准入。smoke 指标不进入 H1 效果结论。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
