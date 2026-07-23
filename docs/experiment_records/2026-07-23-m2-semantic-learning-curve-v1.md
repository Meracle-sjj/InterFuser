# M2 交通语义学习曲线 pilot v1 记录

| 字段 | 内容 |
| --- | --- |
| 运行 Git | `431badc9849489b990d1e661441371dc3d8b87f4` |
| Pilot 配置 | `configs/thesis/semantic_pretraining_pilot_v1.json` |
| 配置 SHA-256 | `494b1b60ecfc6fa17119db250c0bacb1611b87aee36296b3a11babff342d74d8` |
| Split SHA-256 | `81d9403a5ecd39fb5c4cf7ac8ffd3c79505a9583d683a960cf5a02b6da1fe8f2` |
| ImageNet 权重 SHA-256 | `464e36baa940a80971e717454e61e8f6fc1a1b284b7a3dbb6f6fd4c3da78533d` |
| 汇总报告 | `results/thesis_m2/semantic_learning_curve_pilot_v1_seed20260723_20260723T030714Z.json` |
| 汇总 SHA-256 | `47503a43ce8b34ce04c9a29c91d2fc788188165e694e9d7370f2ed3e600c9954` |
| 结论 | **三点曲线有效且仍上升；常见类受益明显，稀缺类仍需区分优化不足与像素失衡** |

## 1. 可比性门禁

三个 run 均从同一 ImageNet `resnet50d_ra2` 独立初始化，固定模型、split、sample seed、输入分辨率、batch size、optimizer、learning rate 和单 epoch 预算。train 样本由同一 SHA-256 排序前缀产生，`1,898 < 3,795 < 7,590` 严格嵌套；validation 固定为完整 1,725 张且三个 manifest 的样本键逐项相同。

确定性汇总器验证三个 manifest 全部 `pipeline_valid=true`，Git/config/class/split/初始化 provenance 一致，预算矩阵无缺失或重复，并重新计算六个 checkpoint/骨干产物哈希。汇总为 `valid=true`、`nested_train_samples=true`、`identical_validation_samples=true`。

## 2. 三点结果

| Train 样本 | Optimizer steps | Train loss | Val loss | Val pixel acc | Val mIoU | Val macro-F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,898 | 119 | 0.981787 | 0.731764 | 0.878938 | 0.215539 | 0.245208 |
| 3,795 | 238 | 0.796778 | 0.535311 | 0.894252 | 0.243451 | 0.280995 |
| 7,590 | 475 | 0.610946 | 0.372628 | 0.917133 | 0.326204 | 0.382466 |

验证指标随数据量和对应 optimizer step 单调改善。该趋势证明当前系统尚未饱和，但固定 epoch 同时改变了样本量与更新次数，因此不能把增益全部归因于数据量。

## 3. 完整 validation 逐类 IoU

| 类别 | 支持像素 | 25% | 50% | 100% |
| --- | ---: | ---: | ---: | ---: |
| background | 17,959,688 | 0.894539 | 0.910519 | 0.930523 |
| road | 10,622,957 | 0.834140 | 0.851772 | 0.889366 |
| sidewalk | 2,110,789 | 0.426710 | 0.553156 | 0.638879 |
| road_line | 493,249 | 0 | 0 | 0 |
| vehicle | 751,311 | 0 | 0.051649 | 0.469147 |
| pedestrian | 2,768 | 0 | 0 | 0 |
| rider | 13,022 | 0 | 0 | 0 |
| traffic_light | 39,820 | 0 | 0 | 0 |
| traffic_sign | 2,758 | 0 | 0 | 0 |
| barrier | 1,123,638 | 0 | 0.067412 | 0.334123 |

background、road、sidewalk 的提升稳定，vehicle 与 barrier 在数据扩大后开始形成有效预测。road_line 虽有近 50 万支持像素仍为 0，说明单纯“样本稀少”不足以解释全部失败；pedestrian/traffic_sign 仅约 2.7k 支持像素，则确实存在严重稀缺。当前首先需要排除单 epoch 优化不足和未加权交叉熵的高频类支配，不能直接启动补采。

## 4. Run 与产物哈希

| Train 样本 | Run ID | Manifest SHA-256 | Checkpoint SHA-256 | Backbone SHA-256 |
| ---: | --- | --- | --- | --- |
| 1,898 | `m2-semantic-pilot-v1-n1898-seed20260723` | `2f28ceb9208b2ad5e79ac85fdce85fc67c4da107a49b8884e2a0122671febf62` | `afbd2558bf10f60abe63d4bc112ab60b3dc596fa79e7164a6ffcfb14dd8ebb20` | `71d42f0882e0a36a1a55ec21b455685618e91690e15ae8b67d9b20420a2dbf22` |
| 3,795 | `m2-semantic-pilot-v1-n3795-seed20260723` | `579d3a8002499e30e76d61f24f37936a7cc17fefa5d5860135d88c022a2da0c7` | `b222067ef5c3610fdb3a39a8193150c831835897506db842878932ce00c2a76a` | `205785906d897cbdc36ccc890aef82a65251b27d0c64396f2fb6548a3e4a62a9` |
| 7,590 | `m2-semantic-pilot-v1-n7590-seed20260723` | `0172b4f4f03d0187719aa2acbd36b3fdbd76539d5c2c5bb4ab380159ee69d742` | `d93341e271a6923469a1a5045138f639ac5d6340c1472935572b0fd80de225ef` | `04425b552ba22e708fb75883e57ef94d0195c0923a97c9e763f48bde9afa2852` |

三个骨干导出均含 330 个参数张量并通过 InterFuser strict load。train/validation 计算耗时分别为 `5.758/1.857`、`9.988/1.827`、`18.619/1.835` 秒；GPU 6 峰值 allocated 为 `995.666 MiB`，reserved 为 `1,156-1,176 MiB`。三次退出后 GPU 6 回到 `81 MiB` 且无 compute owner。

## 5. 测试、边界与下一步

pilot 训练契约与汇总定向测试为 `9/9`，带完整仓库、InterFuser、Leaderboard、Scenario Runner 和 CARLA PythonAPI `PYTHONPATH` 的全量 unittest 为 `172/172`。

该曲线不是 H1 对照：只有单 seed、单 epoch、低分辨率和一次性 FPN head，且不同数据点的 optimizer step 数不同。下一项最小诊断应固定 100% train 与完整 validation，先执行多 epoch 无类别权重训练并保存逐 epoch曲线；若稀缺类仍为 0，再版本化引入类别权重，最后才根据证据决定最小定向补采。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
