<!--
[INPUT]: 依赖 docs/interfuser_visual_transfer_protocol_v1.md 冻结的 H1 评测链、远端 results/thesis_m2 一手产物状态与 run_interfuser_visual_test.py / run_interfuser_visual_d7_pair.py 的实际门禁逻辑。
[OUTPUT]: 对外提供 A(冻结 test 离线评测)与 B(B0/V D7 配对闭环)的执行设计，作为 writing-plans 与远端执行的冻结输入。
[POS]: docs/superpowers/specs 的 H1 执行设计基线；不重写 protocol_v1.md 的指标口径，只固化本次解冻与执行决策。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
-->

# H1 冻结 test 与 D7 配对闭环执行设计 v1.0

| 字段 | 内容 |
| --- | --- |
| 状态 | **PROPOSED：待用户复审后转 writing-plans 并同步远端** |
| 生效日期 | 2026-07-31 |
| 解冻范围 | 仅 H1 评测链的 test 与 D7 环节；不解冻训练、数据、M2 预训练预算 |
| 上游协议 | `docs/interfuser_visual_transfer_protocol_v1.md` v1.0 |
| 评测基线协议 | `docs/baseline_evaluation_protocol_v1.md` v1.0 |
| 论文目标基线 | `docs/thesis_goal_v1.md` v1.0 第 3 节 RQ1/H1、第 7 节 B0/V 行 |
| 远端仓库 | `/data/shijj/interfuser_origin`（分支 `codex/stop-boundary-label-export`，HEAD `a22a3aa`） |

## 1. 背景与现状（2026-07-31 一手核实）

H1 评测链在 `interfuser_visual_transfer_protocol_v1.md` 中预注册为强制门禁链：formal 训练 → 冻结 test → D7 配对闭环 → H1 判定。各环节实际完成度如下，均经远端产物或 manifest 核实，非账本转述：

| 环节 | 状态 | 证据 |
| --- | --- | --- |
| 语义预训练 probe + 类权重 | ✅ 完成 | `m2-semantic-optimization-probe-v1-full-unweighted`、`m2-semantic-class-weight-probe-v1-full-invsqrt` |
| 下游 B0/V formal 训练 | ✅ `pipeline_valid` | run_manifest `status=completed`、`only_initial_checkpoint_differs=true`，B0/V 各导出 `model_best.pth.tar` |
| 冻结 test 离线评测（A） | ❌ failed | `m2-interfuser-visual-pair-test-v1-...-20260724-v1/test_manifest.json`，`status=failed`、`variants=[]`、errors 记录 GPU 6 被 PID 4041512 占用 |
| D7 配对闭环（B） | ❌ 未跑 | `results/thesis_m2/m2-interfuser-visual-d7-*` 不存在；config `interfuser_visual_d7_build_v1.json` 为 `preregistered` |

协议状态行仍写 "FORMAL-RUNNING…禁止运行 test"，是 formal 训练期间的过时措辞；formal 已 completed 且 pipeline_valid，满足 test 的前置门禁，重跑 test 符合协议 §4 与 §7。

当前 GPU 2–7 空闲（idle 残留 31–81 MiB），此前占用 GPU 6 的 `4041512` 已退出；GPU 0 上为常开 CARLA（PID 3980783，端口 2000）与 mineru。A 与 B 均具备立即执行条件。

## 2. 目标

把 H1 证据链从“validation-only、single-seed”推进到“test 离线结论 + D7 三种子闭环配对”，使论文不再只能写 1.27% 验证 L1 改善，而是能回答两个审稿核心质疑：改善是否超出 seed 噪声、是否传导到闭环驾驶。结论无论为支持、混合或不支持，均按预注册判定规则如实进入论文。

## 3. 明确不做（YAGNI）

- 不改 M2 语义预训练分辨率、epoch、类别权重或 seed；
- 不重训 B0/V formal 下游（复用已冻结的 `model_best.pth.tar`）；
- 不增加 B0/V 之外的 variant，不引入 L 组点云（属 M3，当前不解冻）；
- 不用 M0 原始 InterFuser checkpoint 替代配对 B0（协议 §6 明确禁止）；
- 不在 D7 之外另选路线、不挑路线、不事后补显著性检验（协议 §8 禁止）；
- 不做语义分割定性图、失败案例视觉分析等需要视觉判断的环节；此类交给视觉 agent。

## 4. A — 冻结 test 离线评测

### 4.1 合规重跑

failed test manifest 与重跑目标同名同路径（`results/thesis_m2/m2-interfuser-visual-pair-test-v1-seed20260724-20260724-v1/`）。test runner 拒绝覆盖 worker 输出，且 config 冻结的 `run_id` 不可改（改则违反协议 §7 冻结的 config SHA `429fb572…`）。因此唯一合规路径：

1. 先把 failed 目录整体移动到审计归档位置（如 `results/thesis_m2/_audit/m2-interfuser-visual-pair-test-v1-...-20260724-v1-failed-20260731/`），保留失败 manifest 与 errors 原状；
2. 以同 Run ID、同 config、同 test index 在干净目录重跑；
3. 重跑 manifest 的 `resume_history` 或备注记录“源自 2026-07-28 GPU 占用失败，归档路径见上”。

执行前先跑 runner 的静态 preflight（不 execute）确认门禁通过、不再报 GPU 占用。

### 4.2 执行参数

- 入口 `tools/evaluation/run_interfuser_visual_test.py`，config `configs/thesis/interfuser_visual_pair_test_v1.json`；
- GPU 6，B0 后 V 串行，读取 test index `c77c81f1…`（324 sequence）；
- formal 前置校验：`run_id`、`status=completed`、`pipeline_valid=true`、clean worktree、config hash 一致、B0/V comparability gate、两 variant checkpoint schema 一致（runner 行 200–246）。

### 4.3 产出指标

单帧：traffic AP/ROC-AUC/occupied IoU、概率/属性/速度 MAE；waypoint 各 horizon MAE/ADE/horizon-10 FDE；junction/red-light/stop-sign 混淆矩阵与 macro-F1。连续帧稳定性（`preregister H1 temporal`）：324 sequence 的 5,462 相邻帧对，traffic 概率变化差 MAE、waypoint 轨迹变化差距离、二分类状态转移错误率。

时长估计：离线前向，数十分钟至约 2 小时量级，不精确。

## 5. B — D7 配对闭环

### 5.1 前置

A 的 test manifest `pipeline_valid=true` 后，由 `tools/evaluation/build_interfuser_visual_d7_configs.py` 一次性生成 B0/V child config 与 pair config（目标已存在则拒绝覆盖）。生成条件含 test 实际 checkpoint 与 formal best 哈希一致。

### 5.2 执行参数

- 入口 `tools/evaluation/run_interfuser_visual_d7_pair.py --execute`，pair config `configs/thesis/interfuser_visual_d7_pair_v1.json`；
- 复用 M0 runner 的 CARLA 生命周期；GPU 6 跑 agent、graphics adapter 7 渲染、端口 2155/2255；**不动常开 CARLA 2000 实例**；
- 评 formal pair 训出的 `b0/v model_best.pth.tar`；
- 42 attempt：B0→V 串行，路线 `[18,6,12,30,36,39,0]` × seed `[0,1,2]`；
- 默认拒绝覆盖；`--resume` 跳过已 pipeline-valid 的 attempt，不完整 attempt 目录须先人工审计，不得自动抹去重跑（runner 行 293–339）。

### 5.3 汇总门禁

每组 variant 必须恰好 21 个 attempt 且全 pipeline-valid（runner 行 274–280）；任一 invalid 立即停止，不从统计中静默删除。`summarize_interfuser_visual_d7.py` 复用 M0 汇总门禁校验两组 21/21，再校验两份 config 除 checkpoint path/hash/epoch 与 variant provenance 外逐字段相同。

主指标 Driving Score，配对差恒为 `V−B0`，报告 21 个 route×seed 差值、7 route 均值差、3 seed 宏平均差、均值/标准差/min/max。RC、IS、逐路线失败状态、连续帧残差、语义逐类指标完整呈现用于解释，不参与多数票。

时长估计：M0 的 21 attempt 占约 4 小时独占窗口；42 attempt 约 8 小时以上量级，路线长短不一，不精确。

## 6. 执行顺序

1. 归档 failed test 目录 → test preflight 验证；
2. 跑 A，直至 `pipeline_valid=true`；
3. 生成 D7 child/pair config；
4. 在用户给定的独占窗口内 `--execute` 串行跑完 42 attempt（一次跑完，不分批）；
5. 汇总并按 §7 判定 H1。

## 7. H1 判定规则（引用协议 §8，不改写）

离线方向性 5 项：traffic AP、traffic ROC-AUC、occupied IoU（越高越好）、waypoint ADE、horizon-10 FDE（越低越好）；≥3 项改善为“离线多数支持 V”。

- **支持**：D7 macro DS 的 `V−B0>0` 且离线多数支持；
- **混合/证据不足**：闭环与离线方向不一致，或 DS 差值=0；
- **不支持**：`V−B0<0` 且离线不足 3 项改善。

无非劣效界值；小幅下降不得改写为“至少没损害”。样本量仅支持描述性 v1 结论。

## 8. 风险与缓解

- **闭环被方差淹没**：B0 基线 DS≈22.68、21 条仅完成 1 条，V 的 1.27% 验证改善可能不传导。缓解：配对差值 + 逐 route×seed 差值 + 离线多数票联合判定，且 negative/neutral 是预注册的合法结论。
- **独占窗口被抢**：test 即因此失败。缓解：B 一次性串行跑完；窗口期间不再启动其他 GPU 6/7 任务；若中途被抢，以 `--resume` 续跑而非覆盖。
- **CARLA 晚发崩溃/清理竞态**：M0 seeds1/2 曾遇。缓解：复用已修复的 M0 runner（同步模式退出、单次 actor 回收），失败 attempt 留证据、人工审计后再续。

## 9. 停止条件与结论边界

A 任一 variant invalid 或 test index 损坏 → 停，修基础设施而非降级容错。B 任一 attempt pipeline-invalid → 停，审计后决定续跑或记录失败。H1 结论只覆盖冻结 D7 路线与 test index，不外推到 A36、F42 或跨 Town 泛化；闭环改善不得声称统计显著。

## 10. 执行前待确认（preflight）

- test runner preflight 不报错且不再有 GPU 占用；
- failed 目录归档后，同 Run ID 重跑被 runner 接受（不触发“refusing to overwrite”）；
- D7 config 生成时 test checkpoint 哈希与 formal best 一致。

## 11. 视觉边界

本设计全程为远端编排与数值指标，不涉及视觉判断。语义分割定性图、闭环失败案例图像分析等视觉环节不进入本执行，交由视觉 agent。

## 12. 交付物

- A：`test_manifest.json`（pipeline_valid，含单帧 + 连续帧指标）；
- B：`m2-interfuser-visual-d7-pair/b0/v-…` 三组 run manifest、H1 汇总 JSON；
- `docs/experiment_records/` 新增两条记录，以 run ID 与 SHA-256 指向 `results/`；
- 更新 `paper/evidence_ledger.md` 的“不能写成的结论”与“投稿前门禁”两节，使 test 与 D7 状态与实测一致。
