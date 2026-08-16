<!--
[INPUT]: 依赖 scene-split v2 配置/协议、生成器真实运行 manifest、三份 index 哈希、CarlaMVDetDataset 有效帧复核与两个 holdout 的三相机连续帧源数据。
[OUTPUT]: 对外提供 M2 H1 行人 holdout 测量修复的结构事实、视觉可信度、泄漏边界与 Stage 2 重训准入结论。
[POS]: docs/experiment_records 的 split-v2 冻结事实锚点；只证明评价集合可测，不提前陈述 B0/V 模型收益。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
-->

# M2 Downstream Scene-Split v2 冻结记录

## 1. 根因

全量 dataset index 含 9,968 个 sequence、187,295 个声明帧；按 Stage 2 导航字段门禁保留 181,152 帧。有效真值中共有 10,988 个 `is_pedestrian_present` 帧、675 个 sequence、61 个 Town+route 组。

v1 downstream split 将61个行人组分为 train/validation/test=`59/1/1`，对应行人帧 `10,892/84/12`。原因不是全库缺少行人，而是 v1 把 M1 小样本 manifest 未覆盖的 route group 全部投影到 train。

## 2. v2 选择与泄漏边界

v2 保持 M1 semantic manifest 的102/25/26个 train/validation/test route group 原归属，只从51个 semantic-unassigned 行人组中确定性补充两个 holdout。选择种子为 `20260816`，模型输出和 RGB 不参与选择。

最终行人 route group 为 `45/8/8`；validation/test 均覆盖 Town01/03/04/05和全部16种冻结天气。semantic-train 与 downstream holdout 交集为零，三个 downstream route-group 两两交集为零，9,968个源 sequence 恰好分配一次。

## 3. 真实产物

| 产物 | 值 |
| --- | --- |
| Run 目录 | `results/thesis_m2/interfuser_downstream_split_v2_scene_seed20260816_20260816-v1` |
| split manifest SHA-256 | `bdb1b02745c791ed2f1edc897505f1ec3f432271ad3edb9a18987610a1aef124` |
| train index SHA-256 | `877a0c87a707947f50663e07d50cedd9b0e8dfa1cc43982dcfcf7525961e9569` |
| validation index SHA-256 | `b4b5a62ac57a310c0d53aceadba53059eefce4b149fd47c30ae754b0694403d2` |
| test index SHA-256 | `ef90e9845cbddb105de4e3f0de31deb1e729867ca06fcc7e5a6c319976fb9775` |
| scene census SHA-256 | `a8297d0971a2c81efad136a7c1e53f96398bd9ccfb52bf53164d139625024465` |

Loader 使用 `lidar_y_axis_multiplier=+1`、`navigation_frame=carla0916_compass`、`missing_navigation_policy=drop` 实测有效帧为 train/validation/test=`165,264/7,437/8,451`，与 manifest 完全一致。重复生成的三个 index 字节相同、SHA-256相同。

行人覆盖为：train `8,276帧/529 sequence/45组`，validation `1,131/80/8`，test `1,581/66/8`；两个 holdout 行人帧比为 `1.397878`，低于冻结上限1.5。

## 4. 连续帧视觉复核

复核使用模型真实输入的 front/left/right 三相机连续帧，不使用单张最大行人帧：

- validation：`Town01:route013`，ClearNoon sequence `town01/town01_tiny_route13_w0_ClearNoon/route13_route01_12_12_20_06_07`，frame 0000–0015；行人从前/左视野接近并持续进入左相机，`is_pedestrian_present` 从 frame 0004 起与多视角可见过程一致；
- test：`Town01:route191`，ClearSunset sequence `town01/town01_tiny_route191_w1_ClearSunset/route191_route01_12_27_16_59_48`，frame 0000–0015；行人从路侧进入前方冲突区域并穿越，多视角连续运动与从 frame 0002 起的真值一致。

该复核证明两个新增 holdout 至少各有一个真实、多视角可见的连续行人事件。它不证明全部1,131/1,581帧均为前摄可见，也不替代后续 actor/grid 级可见性归约。

## 5. 验证结果与结论边界

- `tests/test_interfuser_downstream_indexes.py`：7/7通过；
- 全仓 `unittest`：266/266通过；
- Stage 2 v2 配置 contract preflight 通过，B0/V 共用相同 train/validation index 与预算，唯一初始化 checkpoint 不同。

因此 scene-split v2 通过“评价集合修复”准入，可以从官方 B0/V 初始化重新执行三轮配对 Stage 2。旧 Stage 2 v1 checkpoint 已见过v2新增 holdout，永久禁止用于新 validation/test 的无泄漏结论。当前记录没有任何新模型结果，不能声称视觉预训练已改善行人识别。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
