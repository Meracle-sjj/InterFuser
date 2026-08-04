# M1.1 行人碰撞威胁审计与 train-only 扩充

## 结论

584/584 个行人可见 sequence 都含 AutoPilot 预测的行人—自车轨迹碰撞威胁，且 584/584 都至少有一个威胁与三相机语义 tag 12 同帧可见的帧。因此这批数据可作为“路侧行人冲入自车轨迹”危险样本源，无需依赖单帧 A/B/C 主观分类。

## 特权真值边界

- 威胁字段：`measurements.is_pedestrian_present`。
- 来源：`leaderboard/team_code/auto_pilot.py::_is_walker_hazard`，SHA-256 `ceeb5f474ce270a9fb738310feee6b54e425b5549e34c9490ab05df0a659517f`。
- 含义：AutoPilot 投影自车与 walker 未来轨迹，记录与自车轨迹相交的 walker actor ID。
- `birdview`、actor ID 与测量字段只用于选样/审计；模型输入仍只有 RGB，不存在特权输入泄漏。

## 审计事实

| 指标 | 结果 |
| --- | ---: |
| 行人 sequence | 584 |
| Town+route 组 | 54 |
| 轨迹/可见性签名 | 20 |
| 测量帧 | 22,721 |
| 威胁帧 | 10,979 |
| 威胁且可见帧 | 10,418 |
| 威胁帧可见率 | 94.8902% |
| 有威胁的 sequence | 584/584 |
| 有可见威胁帧的 sequence | 584/584 |

Town 分层威胁帧可见率为 Town01 96.86%、Town03 88.53%、Town04 67.39%、Town05 88.01%。Town04 是当前最弱可见性分层，它是后续数据采集的明确优先级，而不是未经量化的“再多采一些行人”。

## 扩充 manifest

- 以每个 Town+route 组一个 sequence 为选样单位，按可见威胁帧数、合格 mask 数、行人像素数与固定 seed 哈希排序。
- 原 M1 validation/test 逐 sequence 保持不变。
- 向 train 增加 38 个 sequence、1,439 个逻辑帧、4,317 个三相机样本；train 从 7,590 增至 11,907 个样本。
- 对原 split 未见的危险 route group 按 Town 分层，另保留 8 个 validation 组和 8 个 test 组，与扩充 train 的 route-group 交集为 0。

## 原始产物

- 清单：`results/thesis_m1/pedestrian_sequence_inventory.json`，SHA-256 `3597efc69baa124d458ef9cece0b3c68fbb78f807b4b0f916d44cbc2eb95db1e`。
- 威胁审计：`results/thesis_m1/pedestrian_hazard_visibility_20260804.json`，SHA-256 `a939eb3c612d02bb97e512789cfc4d7ed4d97640ab04bddd8ae8e70b5516659e`。
- 扩充 manifest：`results/thesis_m1/semantic_split_pedestrian_hazard_augmented_v1_seed20260804.json`，SHA-256 `ea4e5878bf9b8621d70881507ce44b3f4c25cece274d4580961e554e2e6de29b`。

## 证据边界

本记录证明行人危险数据的来源、可见性、选样及无泄漏划分已可审计；它不证明模型指标已改善。模型收益必须由同预算训练、原 validation 和 route-group 专项 holdout 三层配对结果决定。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
