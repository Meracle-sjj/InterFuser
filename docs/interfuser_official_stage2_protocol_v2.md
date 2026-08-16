<!--
[INPUT]: 依赖 Stage 2 v1 协同适配协议、downstream scene split v2 的冻结结构结果、官方 B0/V 初始化与修正后的 CARLA 0.9.16 坐标契约。
[OUTPUT]: 对外提供 scene-split v2 上三轮 B0/V 重训、validation 判读与冻结 test 解封门禁。
[POS]: docs 的 M2 H1 Stage 2 数据测量修正版；继承 v1 模型预算，只替换经验证的下游索引并禁止复用旧 checkpoint。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
-->

# M2 官方 InterFuser Stage 2 Scene-Split v2 协议

| 字段 | 内容 |
| --- | --- |
| 状态 | **PILOT-FROZEN：split 结构通过后按同预算重训** |
| 上游模型协议 | `docs/interfuser_official_stage2_protocol_v1.md` |
| 数据修复协议 | `docs/interfuser_downstream_scene_split_v2_protocol.md` |
| 配置 | `configs/thesis/interfuser_official_stage2_pilot_scene_split_v2.json` |

## 1. 唯一变化

本协议完整继承 Stage 2 v1 的官方 B0/official V 初始化、全模型端到端三轮预算、优化器、学习率、单卡全局 batch 32、LiDAR `+1`、compass 导航 frame、缺失导航字段剔除和 B0 后 V 串行顺序。

唯一共同变化是 B0/V 同时改用 scene-split v2：train/validation 有效帧从 `171,256/4,389` 变为 `165,264/7,437`，validation 行人覆盖从 1 个 Town+route 组、84 帧变为 8 个组、1,131 帧。该变化服务于 H1 可检验性，不构成新的模型变量。

资源准入沿用 `shared_capacity`，按上一轮16.3 GiB实测峰值把GPU1最小空闲量设为20,000 MiB；该门槛只决定是否启动，不进入训练参数，B0/V共享同一容量和外部进程环境。

## 2. 旧 checkpoint 失效边界

Stage 2 v1 的 B0/V 已在 v2 新增 holdout route group 上训练过，不能在新 validation/test 上报告无泄漏结果。v2 必须从同一官方初始化重新训练；不得加载 v1 best/last checkpoint 续训。

## 3. 运行前门禁

- scene-split v2 manifest `valid=true`；三个 index 哈希、有效帧与 loader 实测一致；
- validation/test 行人路线均为 8 组，覆盖 Town01/03/04/05，semantic-train overlap 为零；
- B0/V 配置除 initial checkpoint 外完全相同；
- Git worktree 干净，GPU1 至少 20,000 MiB 空闲。

## 4. 结果判读

三轮训练首先比较整体 validation loss 与 waypoint L1，但不得据此单独判断 H1。训练结束后必须使用同一 best-checkpoint 选择规则，对完整 validation 输出整体和行人场景分层的 traffic AP/AUC/IoU、waypoint ADE/FDE及连续帧稳定性。

- 整体与行人分层多数同向改善：准入冻结 test；
- 行人改善但整体退化：记为混合，先检查融合/轨迹适配，不直接扩大25轮；
- 行人与整体均无改善：停止该视觉初始化路线，不以更多帧掩盖负结果。

test 在两个 variant 训练完成、checkpoint 选择和阈值冻结前不得读取。test 结果按 Town+route 做配对归约；只有离线证据通过才准入 route39 与 D7。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
