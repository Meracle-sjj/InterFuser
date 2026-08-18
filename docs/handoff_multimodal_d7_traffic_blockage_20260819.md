<!--
[INPUT]: 依赖 2026-08-18/19 v9 闭环 D7 双臂 run 的 sensor_data 前摄帧、attempt manifest 数值、
  baseline_eval_v1.json 背景车配置（120-200/城）。
[OUTPUT]: 对外提供多模态判定交接：堵车假设的待判问题、输入帧路径、判定标准与输出口径；
  另含 codex/minimax 接管所需的当前任务全景摘要。
[POS]: docs 的多模态停机线交接文档（HANDOFF §0 协议）；Claude Code 无原生视觉，判定交有视觉能力的 agent。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
-->

# 多模态交接：D7 重背景堵车假设判定（2026-08-19）

## 1. 待判定问题（Owner 假设：背景车横置路面导致无法前进）

- **Q1（主问题）**：route_00 双臂 RC 均 <2%（B0 0.881%、v9 1.492%）但仿真时长 1414s——自车是否长时间被密集/横置的背景车辆物理阻断？
- **Q2**：route_06（v9 RC 19.0%、B0 RC 约 10%量级）中段是否存在同样堵点？
- **Q3（对照，待交通消融跑完后）**：0 背景车配置下同路线是否恢复通行？
  - **2026-08-19 更正**：官方 Leaderboard 1.0 的 BackgroundActivity 默认就是 120-200 辆/城（见 leaderboard/leaderboard/scenarios/background_activity.py 的 town_amount，与 baseline_eval_v1.json 逐城一致）。**重背景矩阵即官方原始条件**；0 背景车跑是"交通消融"诊断，不是"官方风格"。真正的差距疑点：evidence_ledger 门禁#2——live agent LiDAR histogram 疑似近空（未闭合）+ 0.9.16 迁移栈；官方作者在同等车流下 DS 45-65。

## 2. 输入（前摄帧，帧号≈仿真秒×2）

```
B0 臂:  results/thesis_m0/m2-v9-d7-b0-seed0-20260818-v1/attempts/route_00_seed_0/sensor_data/<route_ts>/meta/*.jpg
v9 臂:  results/thesis_m0/m2-v9-d7-v-seed0-20260818-v1/attempts/route_00_seed_0/sensor_data/<route_ts>/meta/*.jpg
route_06 同构。建议抽帧：0100/0300/0500/0800/1100（route_00 共约 1154 帧）。
```

## 3. 判定标准

- **阻断成立**：连续多帧（≥30 帧≈15 仿真秒）前摄近景被车辆占据且无可见通行间隙，或背景车呈现非常规朝向（横置/逆停）；
- **阻断不成立**：前方持续有可通行间隙而自车不动（→ 归因转向控制/规划），或自车在移动但路线极长；
- 输出口径：逐帧判定表（帧号、是否阻断、遮挡物类型）+ 一段结论，写入 `results/thesis_m0/m2-v9-d7-blockage-review-20260819.md`，远端 git 提交。

## 4. 数值证据（已固化，无需复算）

- route_00: B0 [DS 0.374, RC 0.881%, IS 0.424]；v9 [DS 0.523, RC 1.492%, IS 0.350]；DS=RC×IS 已自洽验证；
- route_06: v9 [DS 6.719, RC 19.029%, IS 0.353]；B0 DS 4.389；
- route_12: v9 [1.713, 7.797%, 0.220]；B0 1.925；
- 背景车：Town01:120/Town02:100/Town03:120/Town04:200/Town05:200→120（见 baseline_eval_v1.json）；
- 历史同量级：2026-07-22 M0 seed0 smoke DS 1.24/RC 2.9%（重背景配置一贯如此）；08-14 无重背景诊断 harness 官方 B0 route39 RC 61.89%。

## 5. 当前任务全景（给接手 agent）

**在跑（server-side 自动漫链，全部带 GPU 容量自检，勿手动重启同名 run）**：
1. v 臂 seed0 续跑（--resume，补 route_18/30/36/39）；
2. 随后 b0/v × seed1/2 四个 run（全矩阵，聚合器 `summarize_thesis_baseline.py` 需要 3 seeds × 7 routes 才接受 manifest）；
3. 随后交通消融（0 背景车，configs/thesis/interfuser_v9_d7_{b0,v}_light_v1.json）双臂 seed0——定性为诊断消融而非官方条件；
4. 进度文件 `/tmp/v9_d7_chain_progress.log`；各 run 日志 `/tmp/m2-v9-d7-*.log`。

**决策点（等视觉判定 + 全矩阵）**：
- 若 Q1/Q2 成立（堵车）：主表仍是官方条件（重背景）配对 Δ；交通消融用于量化车流贡献并写进分析节；同时优先关闭 live LiDAR histogram 门禁（近空点云会让双臂的 LiDAR 流形同虚设，放大车流脆弱性）；
- 若不成立（车能动但 RC 低）：归因回到控制/规划，需另立诊断协议。

**关键文件**：协议 `docs/interfuser_architecture_aware_semantic_pretraining_protocol_v9.md`（EXECUTED-PASSED）；论文 `paper/adist_2027_draft_v2.md`（本地，§4.4 待填）；证据账本（本地 `paper/evidence_ledger.md`）；文献 `docs/KEYreferences.md`。远端 HEAD 含轻背景配置提交。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
