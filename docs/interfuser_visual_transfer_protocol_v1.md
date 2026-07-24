# M2 InterFuser 视觉权重迁移与 H1 对照协议 v1.0

| 字段 | 内容 |
| --- | --- |
| 状态 | **FORMAL-RUNNING：B0/V 配对正式训练执行中，冻结 test 指标已预注册但禁止运行** |
| 服务假设 | H1：交通域 ResNet-50 初始化优于通用 ImageNet 初始化 |
| 数据配置 | `configs/thesis/interfuser_downstream_split_v1.json` |
| 初始化配置 | `configs/thesis/interfuser_visual_initialization_v1.json` |
| 配对 smoke 配置 | `configs/thesis/interfuser_visual_pair_smoke_v1.json` |
| 配对正式配置 | `configs/thesis/interfuser_visual_pair_formal_v1.json` |
| 冻结 test 配置 | `configs/thesis/interfuser_visual_pair_test_v1.json` |
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

有效 smoke 为 `m2-interfuser-visual-pair-smoke-v1-seed20260723-20260724-v1`（运行 Git `65f3946d35dc08ad9b378a3b5220ebe1fe081ad6`，manifest `8f2b44f25f116daf12021695cbf0a77555e256f347c2733c4384e807114c1018`）。B0/V 均完成 1 epoch、两进程反向与验证，归一训练参数哈希一致，产出相同的 1,132 张量 schema，退出后 GPU 6/7、端口 29655 和进程组全部释放。

## 6. 正式训练契约

正式预算复用上游 `README.md:157-168` 与 `interfuser/scripts/train.sh` 的 2 GPU 配方：每 GPU batch 16、25 epochs、5 warmup epochs、AdamW、cosine、主学习率 `5e-4`、骨干学习率 `2e-4`、weight decay `0.05`、scale `0.9..1.1`，并显式冻结 `train.py` 默认 color jitter `0.1` 与 log interval `50`。B0/V 共用 seed `20260723` 并按 B0 后 V 串行；唯一变量仍是初始视觉骨干。

formal 配置必须同时绑定 train、validation 与从未参与模型选择的 test index。validation 选择 best checkpoint；test 只在两个正式训练完成后由独立离线 evaluator 一次性评估。后续 D7 必须分别评测本配对训练产出的 B0/V checkpoint，M0 原始 checkpoint 只作为基线校准证据，不替代正式配对 B0。

正式配置 SHA-256 为 `bc77afd5cbc0656d935c70fa15d21509474d533b6668d4c7f0f96a6f1a0c1738`；首次且唯一准入 Run ID 冻结为 `m2-interfuser-visual-pair-formal-v1-seed20260723-20260724-v1`。runner 拒绝覆盖同名目录，后续监控只能续查该产物，不得重复启动。

## 7. 冻结 test 离线评价契约

test 配置在 formal 结果完成前预注册，SHA-256 为 `429fb5722754bb5ab7d7f2172bfb6aaed940706b0b50933e98beb5827d2c15ee`，唯一 Run ID 为 `m2-interfuser-visual-pair-test-v1-seed20260724-20260724-v1`。runner 必须拒绝其他 Run ID，以及未完成、pipeline-invalid、summary 非 25 行、checkpoint/schema/hash 漂移或 B0/V 参数不可比的 formal manifest；只有两组 formal best checkpoint 都通过后，才可在 GPU 6 上按 B0 后 V 串行读取 test index `c77c81f1a11dfdaffb064c2439149a39a834cbc5bc94c12cffea654249c1768d`。

旧配置哈希 `976b7f6826e6668da4a2a8ae2e70c5b617d5f812d8547dc5115afb9ba1578367` 在读取任何 test 样本或结果前被废止：它只覆盖单帧指标，未满足 H1 对连续帧稳定性的显式要求。新版本保持同一 test index、模型、阈值、资源和 Run ID，只增加预注册的严格连续帧归约，不改变训练或模型选择。

指标口径固定为：traffic target channel 0 以 `>=0.01` 定义 occupied，预测以 `>=0.5` 形成固定阈值混淆，同时报告 threshold-independent AP/AUC、occupied IoU、概率/属性/速度 MAE；waypoint 排除绝对值 `>=1000` 的 padding，报告每个 horizon 支持数、距离与坐标 MAE、ADE 和 horizon-10 FDE；junction、red-light、stop-sign 报告混淆矩阵、accuracy、macro precision/recall/F1 和逐类支持/precision/recall/F1。任何二分类缺一类或 waypoint horizon 无支持属于数据评价基础设施不足，不得写成模型失败。

连续帧稳定性固定使用 test index 中 324 个 sequence 的 5,462 个相邻帧对，不跨 sequence 比较。evaluator 必须按冻结索引精确读取指定帧，任何缺失或损坏立即使 pipeline invalid，不得沿用数据集“跳到下一帧”的训练期容错。traffic 报告预测概率变化与 target 概率变化之差的 MAE；waypoint 报告预测轨迹变化与 target 轨迹变化之差的逐 horizon 距离及 ADE；三个二分类头报告预测状态转移是否匹配 target 状态转移的错误率。所有连续帧残差越低越好；它们是目标条件稳定性，不把场景真实变化误判为模型抖动。

上游 `train.py::validate` 的 stop-sign accuracy 当前错误复用 traffic-light 的 `output[3]/target[3]`；该值不进入 summary、best checkpoint 选择或本协议结论。独立 evaluator 固定使用正确的 `output[4]/target[6]`，并由纯指标测试防止回归。

## 8. D7 配对闭环与 H1 判定口径

冻结 test pipeline-valid 后才允许生成 B0/V D7 配置，并绑定 formal manifest 与两个 best checkpoint 的最终哈希。两组必须使用相同的 M0 路线、场景、agent、控制器、CARLA/Leaderboard/Scenario Runner 哈希，相同 GPU 6、graphics adapter 7、端口 2155/2255、背景交通、环境变量与外部超时；唯一模型差异是 checkpoint path/hash/provenance。执行顺序固定为 B0 后 V，每组路线顺序 `18,6,12,30,36,39,0`，每条路线依次 seeds `0,1,2`，共 42 个 attempt。任一 pipeline-invalid 立即停止，不把缺失结果记为零，也不从统计中静默删除。

D7 主指标为 Driving Score，Route Completion 与 Infraction Score 为共同报告的诊断指标。每个 variant 先在同 route 内对三个 seed 求均值，再对七条 route 做宏平均；配对差值始终定义为 `V - B0`，并同时报告 21 个 route×seed 差值、七个 route 均值差、三个 seed 宏平均差、均值、标准差、最小值与最大值。样本量只支持描述性 v1 结论，不用事后选择显著性检验或声称超出冻结 D7 的总体泛化。

离线方向性主指标预先固定为 traffic AP、traffic ROC-AUC、occupied IoU、waypoint ADE、horizon-10 FDE，其中前三项越高越好、后两项越低越好；五项至少三项改善定义为“离线多数支持 V”。连续帧 residual 与二分类 macro-F1 用于机制和失败分析，不参与多数票，避免以大量相关指标重复计票。H1 v1 结论分为：

- **支持**：D7 macro Driving Score 的 `V-B0 > 0`，且离线五项至少三项改善；
- **混合/证据不足**：闭环与离线多数方向不一致，或 D7 差值等于零；
- **不支持**：D7 macro Driving Score 的 `V-B0 < 0`，且离线五项不足三项改善。

没有预注册非劣效界值，因此不得把小幅下降改写为“至少没有损害”。RC/IS、逐路线失败、连续帧指标和语义预训练逐类指标必须完整呈现，用于解释结论而不是改写上述门槛。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
