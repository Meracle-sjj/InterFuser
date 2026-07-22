# M0 基线评测协议 v1.0

| 字段 | 内容 |
| --- | --- |
| 状态 | **CALIBRATING：协议已定义，正式基线尚未冻结** |
| 生效日期 | 2026-07-22 |
| 研究目标 | 为 B0、V、L、V+L 四组实验提供同一套可复现闭环评价 |
| 机器配置 | `configs/thesis/baseline_eval_v1.json` |

## 1. 为什么当前结果不是正式基线

`results/full42_eval/summary_current_model.txt` 记录了一次 2026-07-09 的历史诊断运行：可运行的 36 条路线平均 `Driving Score=11.218`、`Route Completion=23.975`、`Infraction Score=0.498`，但完成路线为 **0/36**。

该运行只能用于定位运行时问题，不能进入论文主表，原因包括：

- 42 条路线中的 Town06 六条路线因地图未安装而跳过；
- 运行使用 `no_scenarios.json`，不等价于 `42scenarios.json` 的 Leaderboard 场景分布；
- route 02 没有有效结果，其余路线全部以 blocked、timeout 或 deviation 等状态结束；
- Town01-Town05 实际加载 `_Opt` 地图，必须验证场景触发和路线坐标仍与原始定义一致；
- 只有单次运行，尚未测量 CARLA 交通随机性带来的方差；
- 运行脚本与结果位于忽略目录，没有形成受版本控制的协议配置。

论文中不得把“Town06 计零后的 conservative42”描述为官方 42-route 结果。

## 2. 固定输入

第一轮校准固定以下事实：

- 代码锚点：`138577f823e5e061645b265f7f44cc3ee9ad95ef`；
- 运行时代码根：`interfuser/`、`leaderboard/`、`scenario_runner/`，相对代码锚点不得有已提交、已暂存或未暂存差异；
- 基线模型：`/home/shijj/interfuser/leaderboard/team_code/model_20260121.tar`；
- 模型 SHA-256：`8feebbe02fab204e25ea19db01a05d7a3b6d109ab993b781da4fc2b1f4d3d8b8`；
- 模型架构：`interfuser_baseline`，checkpoint epoch 26；
- checkpoint 训练 Town：1、4、5，验证 Town：3；
- CARLA：0.9.16；Python：3.10.19；
- 路线定义：`leaderboard/data/42routes/42routes.xml`；
- 场景定义：`leaderboard/data/42routes/42scenarios.json`；
- 最终指标：Driving Score、Route Completion、Infraction Score 及各失败类型。

文件哈希、环境变量、路线集合和随机种子以机器配置为准。后续如果替换基线 checkpoint，必须新建协议版本，不能覆盖 v1。

## 3. 三层评测集合

### P0：静态预检

不运行 CARLA，只验证：

- 42 条路线和场景 JSON 能被当前 Leaderboard 解析；
- 每条路线的 Town 在运行地图清单或明确排除清单中；
- 模型、agent、配置、路线和场景哈希匹配；
- 结果目录为空或带有新的 run ID，禁止静默覆盖。

P0 失败时禁止启动批量评测。

### D7：开发回归集

固定路线 `00、06、12、18、30、36、39`。选择规则是 Town01-Town05 各取最小 route ID，再加入扩展块中 Town03、Town04 的最小 route ID；不是根据模型得分挑选。

D7 用于验证运行链路和快速比较，不作为最终论文分数。第一次校准每条路线运行一次；当 7/7 都生成结构完整的 Leaderboard JSON 后，再使用固定随机种子 `0、1、2` 测量方差。

“生成有效结果”只要求 evaluator 正常记录路线状态和指标。模型因碰撞、超时或偏航失败仍是有效驾驶结果；进程崩溃、无 JSON、地图加载失败和传感器初始化失败不是有效结果。

### A36：论文主比较候选集

固定使用 42-route 文件中 Town01-Town05 的 36 条路线，即 `00-23、30-41`。Town06 不计零，也不伪装为已评测；论文必须明确写成“36-route available-map subset”。

A36 只有在 D7 三个种子均通过运行完整性门槛后才能冻结。B0、V、L、V+L 必须运行相同路线、场景、背景交通配置和种子。

### F42：完整 Leaderboard 扩展集

Town06 资产安装并通过 P0/D7 等价预检后，可以追加完整 42 路线结果。F42 是增强项，不阻塞 A36 完成毕业论文主消融。

## 4. 地图兼容原则

路线 XML 与场景 JSON 保持原始 `TownXX` 语义；运行时允许预加载坐标等价的 `TownXX_Opt`，但必须满足：

- evaluator 的地图一致性检查只归一化 `_Opt` 后缀，不修改路线坐标；
- 场景触发点在实际 world 中成功生成；
- D7 每个 Town 至少人工复核一条路线的起终点、交通参与者和场景触发日志；
- 如果 `_Opt` 导致场景缺失，则该兼容方式作废，不能退回 `no_scenarios.json` 冒充正式评价。

## 5. 汇总规则

每个“模型 × 路线 × 种子”保存独立原始 JSON。先对同一路线三个种子求均值，再对路线作宏平均，避免长路线或重复运行隐式获得更高权重。

必须同时报告：

- 三个 Leaderboard 总指标的均值与标准差；
- 完成、blocked、timeout、deviation 和进程错误数量；
- 碰撞行人、碰撞车辆、碰撞环境、闯红灯、停车标志、偏离路线等违规；
- 每条路线的配对差值，使 V、L、V+L 能与 B0 做同路线比较；
- 失败或缺失运行，不允许只删除后重新计算更好看的均值。

## 6. M0 冻结门槛

M0 只有满足以下条件才能从 `CALIBRATING` 改为 `FROZEN`：

1. P0 可由一个受版本控制的命令重复执行；
2. D7 在种子 0、1、2 下产生 21 个结构完整、来源可追溯的结果；
3. 同一配置重复汇总得到完全一致的统计；
4. 已记录运行时长、GPU、峰值显存和 CARLA 进程退出状态；
5. 已决定 A36 是否足以作为论文主集，并在正文中如实描述 Town06 缺失；
6. 基线结果不再依赖 `results/` 内未版本化脚本解释关键参数。

P0 由 `tools/evaluation/preflight_thesis_baseline.py` 实现，D7 runner 由 `tools/evaluation/run_thesis_baseline.py` 实现。runner 默认只生成计划；只有显式传入 `--execute` 才启动 CARLA。配置分别记录 agent CUDA 设备和 CARLA graphics adapter，命令行覆盖必须写入 run manifest。

当前下一步是先对完整 D7 生成 dry-run 计划，再用 route 18 / seed 0 验证一条真实场景链路；单路线有效后才允许扩展到 D7 单种子，不立即重跑 36 条路线。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
