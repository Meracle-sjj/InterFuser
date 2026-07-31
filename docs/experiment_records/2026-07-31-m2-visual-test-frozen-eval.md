<!--
[INPUT]: 依赖远端 results/thesis_m2/m2-interfuser-visual-pair-test-v1-...-20260724-v1/test_manifest.json 一手指标、formal pair 训练 manifest 与 interfuser_visual_transfer_protocol_v1.md §7 冻结 test 契约。
[OUTPUT]: 对外提供 H1 冻结 test 离线评测的数值、stop_sign 数据不足边界与执行修复证据，作为 D7 配对闭环的前置门禁凭证。
[POS]: docs/experiment_records 的 M2 H1 冻结 test 完成记录；指向 results/thesis_m2 原始 manifest，不复制大体积产物。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
-->

# M2 视觉迁移冻结 test 离线评测记录

| 字段 | 内容 |
| --- | --- |
| Run ID | `m2-interfuser-visual-pair-test-v1-seed20260724-20260724-v1` |
| 状态 | **FROZEN-TEST-COMPLETED：pipeline_valid，B0/V 双 variant 通过；D7 配对闭环待执行** |
| 评测时间 | 2026-07-31T06:36:47Z |
| Git HEAD | `f2cf77b`（含 weights_only / stop_sign 三个执行修复） |
| Config | `configs/thesis/interfuser_visual_pair_test_v1.json` |
| Config SHA-256 | `429fb5722754bb5ab7d7f2172bfb6aaed940706b0b50933e98beb5827d2c15ee` |
| Test manifest SHA-256 | `51ae8d1d99d976f83465b56681e448437062a3efc20b70dbb9aa2d6fea46bdf3` |
| Formal training | `m2-interfuser-visual-pair-formal-v1-seed20260723-20260724-v1`（pipeline_valid） |

## 1. 评测产物

B0 与 V 在 frozen test index（324 sequence、5,786 logical frame）上各完成 362 batch 前向，GPU 6、batch 16、workers 16、fd 65536，worker exit 0。`test_manifest.json` 为 `status=completed`、`pipeline_valid=true`、`errors=[]`，B0/V 各 `pipeline_valid=true`。

## 2. 离线方向性 5 项（H1 判定主指标）

| 指标 | B0 | V | V−B0 | 方向 |
| --- | ---: | ---: | ---: | --- |
| traffic AP | 0.865835 | 0.869992 | +0.004157 | V 改善 |
| traffic ROC-AUC | 0.996535 | 0.996387 | −0.000148 | V 略差 |
| traffic occupied IoU | 0.501365 | 0.501780 | +0.000415 | V 改善 |
| waypoint ADE | 0.236891 | 0.182474 | −0.054417 | V 改善 |
| waypoint FDE-h10 | 0.466793 | 0.334797 | −0.131996 | V 改善 |

5 项中 4 项改善（AP、occupied IoU、ADE、FDE），ROC-AUC 微降 0.00015。按 protocol §8「≥3 项改善 = 离线多数支持 V」，**离线多数支持 V**。

## 3. 二分类头与连续帧稳定性

| 指标 | B0 | V |
| --- | ---: | ---: |
| junction macro-F1 | 0.993344 | 0.992756 |
| red_light macro-F1 | 0.993877 | 0.996155 |
| stop_sign | insufficient_data（counts `[5786, 0]`，全 absent） | 同 B0 |
| temporal traffic Δ-MAE | 0.006509 | 0.006029 |
| temporal waypoint Δ-ADE | 0.256819 | 0.207638 |

stop_sign 在 frozen test 的 5,786 frame 全为 absent（route136/route206 test 集无 stop_sign 场景），按 protocol §7 标注 `insufficient_data`、不进入多数票。连续帧残差 V 均低于 B0，时序稳定性方向与 H1 一致；用于机制解释，不参与多数票。

## 4. 执行修复（解冻 H1 test 环节过程中暴露，均不改评测语义）

首次端到端执行 frozen test，依次暴露并修复三个独立运行问题：

1. **weights_only（commit `b50528f`）**：`interfuser/timm/models/helpers.py:31` 的 `torch.load` 加 `weights_only=False`。PyTorch 2.6+ 默认 `weights_only=True` 拒绝加载含 `argparse.Namespace` 的 formal checkpoint，导致 worker exit 3；与 `run_interfuser_visual_pair.py:468-476` 训练侧写法一致。
2. **stop_sign 缺一类（commit `f2cf77b`）**：`interfuser_offline_metrics.finalize` 缺一类不再 hard fail，改标注 `insufficient_data`+`class_counts`；`run_interfuser_visual_test._metric_delta.value_at` 改 None-safe。符合 protocol §7「不得写成模型失败」。
3. **fd 限制（运行时 `ulimit -n 65536`）**：DataLoader 16 workers + `persistent_workers` 在默认 fd 软限制 1024 下累积耗尽导致 `received 0 items of ancdata`；提 fd 到 65536 后越过 batch 250 崩点（硬限制 1,048,576）。

四次失败 manifest 已归档至 `results/thesis_m2/_audit/`（GPU 占用 / weights-only / ancdata / stop-sign 各一），失败证据保留。

## 5. 证据边界与下一阶段

- 离线多数支持 V（4/5）且连续帧残差更低，方向与 H1 一致；但单 seed、单分辨率离线评测，不声称统计显著。
- stop_sign 在 frozen test 不可评价（数据不足）；论文须如实写为 test 评价基础设施局限，不写成模型 stop_sign 失败。
- D7 三种子闭环（B 组）尚未执行；按 protocol §8，H1 最终判定需 D7 macro Driving Score 的 V−B0 与离线多数票联合。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
