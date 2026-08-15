# M2 官方 Stage 2 pilot v1 单 GPU validation 归约事故

| 字段 | 内容 |
| --- | --- |
| 日期 | 2026-08-15 |
| 失败 Run ID | `m2-interfuser-official-stage2-pilot-v1-seed20260814-20260815-v1` |
| 运行 Git | `e7991a34c0dc3f188f3626e52a64e301c9d32c7d` |
| 状态 | **PIPELINE-INVALID：保留原始目录，不纳入 B0/V 效果比较** |

## 1. 事实

B0 完成 epoch 0 的 5,351/5,351 个训练 batch，平均总 loss 为 `0.0938`，GPU1 峰值 16,227 MiB，未发生 OOM、NaN、数据读取或坐标门禁失败。进入第一个 validation batch 时，`train.py::validate` 抛出：

`UnboundLocalError: local variable 'reduced_loss_traffic' referenced before assignment`

runner 在 909.950 秒处以 exit code 1 停止，未启动 V。checkpoint 保存位于 validation 之后，因此本次没有可恢复 checkpoint，不能把已训练 epoch 与后续重跑拼接。

## 2. 根因与修复

旧 `validate` 只在 `args.distributed=True` 时赋值 traffic、velocity、waypoint、junction、light、stop-sign loss 及分类 accuracy 的 reduced 张量；单 GPU 分支只赋值总 loss。此前正式训练使用两卡，未触发该缺口；本次 GPU1 单卡 pilot 首次覆盖它。

修复将全部 validation 张量统一交给 `_reduce_validation_tensors`：多卡执行 `reduce_tensor`，单卡执行恒等归约。新增独立测试验证十个张量在单卡路径全部返回；完整测试 263/263 通过。

## 3. 重跑边界

失败目录永久保留。恢复运行必须使用新 Run ID，从同一 official B0 checkpoint 和 epoch 0 重新开始；配置、样本、随机种子、预算和 B0→V 顺序不得改变。v1 的训练 loss 只用于证明数据/反向链路健康，不进入配对效果结论。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
