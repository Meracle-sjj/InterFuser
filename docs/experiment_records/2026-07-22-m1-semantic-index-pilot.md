# M1 dataset_index 分层抽样 pilot 记录

| 字段 | 内容 |
| --- | --- |
| 审计工具提交 | `0fac3e6` 引入；执行时仓库为 `ebc3ee81d4e289822b07b3e7219352762d9014f2` |
| 数据根目录 | `/data1/shijj/interfuser_data` |
| Dataset index | `/data1/shijj/interfuser_data/dataset_index.txt` |
| Dataset index SHA-256 | `56c83f46a1010ee43021bbc2f97cafde9b3c2771522088ae938a2df2ea477ff1` |
| 类别配置 SHA-256 | `796222592efb68407a32bfdf9a03907b4631cf3f30d0d6a81528010e50f0b612` |
| 报告 | `results/thesis_m1/semantic_index_pilot_town_weather_n3_seed20260722_20260722T0847Z.json` |
| 报告 SHA-256 | `16162c5167cbe994c8e22ca5224069f913989c6ca583dbec234ce18285de2614` |
| 结论 | **pilot ready，结构错误为 0** |

## 1. 冻结抽样 provenance

审计使用 sample seed `20260722`，按 Town×weather stratum 最多抽 3 个完整 sequence，固定相机为 `front、left、right`，并启用 `--require-ready`。dataset index 含 9,968 个可选 sequence；60 个 stratum 最终选择 176 个 sequence，覆盖 Town01、Town03、Town04、Town05，共 3,619 个逻辑帧与 10,857 张语义 mask。

176 而非 180 不是抽样丢失：Town04 weather 15 与 weather 20 在 index 中各只有 1 个候选，审计器按 `min(3, available)` 全部纳入。该候选不足事实必须随报告保留；后续若补齐这两个分层，必须生成新报告，不能覆盖本次 JSON。

## 2. 准入结果

报告为 `valid=true`、`ready=true`、`errors=[]`、`readiness.failures=[]`。核心类别均通过 qualified-mask 和 sequence 覆盖门槛：

| 类别 | Qualified masks / 门槛 | 覆盖 sequence / 门槛 |
| --- | ---: | ---: |
| road | 10,856 / 1,000 | 176 / 3 |
| sidewalk | 9,563 / 500 | 148 / 3 |
| road_line | 9,794 / 500 | 176 / 3 |
| vehicle | 8,971 / 500 | 176 / 3 |
| pedestrian | 366 / 200 | 12 / 3 |
| rider | 3,156 / 100 | 128 / 3 |
| traffic_light | 8,087 / 200 | 144 / 3 |
| traffic_sign | 2,461 / 100 | 117 / 3 |
| barrier | 10,058 / 200 | 176 / 3 |

## 3. 证据边界与下一步

该报告证明确定性分层样本满足 M1 pilot readiness，不等于 9,968 个 sequence 的全量结构审计，也不冻结 train/validation/test split。下一步应基于完整 sequence 冻结无泄漏 split manifest，并按核心类别人工复核 RGB/mask 对齐样本；不得因两个夜间天气分层候选较少而盲目全量扫描或扩大停车线采集方向。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
