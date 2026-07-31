# H1 冻结 test 与 D7 配对闭环 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: 用 superpowers:executing-plans 逐 task 执行（远端运维编排，非编码任务，不适用 subagent 并行实现）。步骤用 `- [ ]` 复选框跟踪。

**Goal:** 执行已预注册的 H1 评测链 A（冻结 test 离线评测）与 B（B0/V D7 配对闭环 42 attempt），产出 H1 判定结论。

**Architecture:** 不写新代码——runner/config/test 均已就绪并有测试覆盖。按 `docs/interfuser_visual_transfer_protocol_v1.md` 的强制门禁链依次运行：归档 failed test → A execute → A pipeline-valid → 生成 D7 config → B execute → 汇总判定。结果落 `results/thesis_m2/`（不进 git），证据摘要落 `docs/experiment_records/`（进 git）。

**Tech Stack:** 远端 `ghbserver02-frpMe`，conda env `interfuser_origin`，PYTHONPATH 含 interfuser/carla/leaderboard/scenario_runner，runner 在 `tools/evaluation/`。

## Global Constraints（逐字来自 spec commit 220e073）

- 解冻范围仅 H1 test 与 D7，不改训练/数据/M2 预训练预算。
- A 的 test manifest `pipeline_valid=true` 是 B 的硬前置；未通过禁止启动 D7。
- B 一次跑完 42 attempt（B0→V 串行，路线 `[18,6,12,30,36,39,0]` × seed `[0,1,2]`），独占窗口，不动常开 CARLA 2000 实例。
- 评 formal pair 训出的 `b0/v model_best.pth.tar`；不用 M0 原始 checkpoint 替代配对 B0。
- H1 判定引用 protocol §8；negative/neutral 是合法结论，无非劣效界值。
- 视觉判断环节（语义定性图、闭环失败案例图）交视觉 agent，不在本计划。

## 远端命令约定

所有命令在远端 `ghbserver02-frpMe` 的 `/data/shijj/interfuser_origin` 下执行。每个 SSH 会话开头先 source：

```bash
MAIN=/data/shijj/interfuser_origin
ORIG=/data/shijj/interfuser_origin
PY=/data1/shijj/conda_envs/interfuser_origin/bin/python
export PYTHONPATH="$MAIN/interfuser:$ORIG/carla/PythonAPI:$ORIG/carla/PythonAPI/examples:$ORIG/carla/PythonAPI/carla:$MAIN/leaderboard:$MAIN/leaderboard/team_code:$MAIN/scenario_runner:$MAIN"
cd "$MAIN"
```

执行入口模板（本机→远端）：`ssh ghbserver02-frpMe 'bash -s' <<'REMOTE' ... REMOTE`，把上述 source 和具体命令一起送入。`results/` 与 `data/` 永远不 `git add`。

---

### Task 1: 执行前置确认

**Files:** 无文件改动；只读校验。

**Interfaces:**
- Consumes: 远端 git HEAD `220e073`、`results/thesis_m2/m2-interfuser-visual-pair-formal-v1-...-20260724-v1/run_manifest.json`（formal 已 pipeline-valid）
- Produces: 一份"green 到可以跑 A"的确认（GPU 空、git 干净、runner 自测通过）

- [ ] **Step 1: 确认 git 工作树干净且在正确分支**

```bash
cd /data/shijj/interfuser_origin && git branch --show-current && git status --porcelain && git log --oneline -1
```
Expected: `codex/stop-boundary-label-export`；`git status --porcelain` 除可能的 `results/`、`data/`（已 gitignore，应不显示）外为空；HEAD=`220e073`。

- [ ] **Step 2: 确认 formal pair 仍 pipeline-valid（A 的输入前提）**

```bash
"$PY" - <<'PYEOF'
import json
p="results/thesis_m2/m2-interfuser-visual-pair-formal-v1-seed20260723-20260724-v1/run_manifest.json"
d=json.load(open(p))
print("formal:", d["status"], d["pipeline_valid"], "comparability:", d["comparability"]["only_initial_checkpoint_differs"])
PYEOF
```
Expected: `formal: completed True comparability: True`

- [ ] **Step 3: 确认 GPU 6/7 空闲（无外部 compute owner）**

```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory,process_name --format=csv,noheader
```
Expected: GPU 6、7 的 used memory 在 idle 量级（<100 MiB），compute-apps 列表里 GPU 6/7 无他人进程。若被占，停止，等资源释放——不要抢（test 就是这样失败的）。

- [ ] **Step 4: 跑 runner 相关 unit test 确认工具未被破坏**

```bash
"$PY" -m unittest tests.test_run_interfuser_visual_test tests.test_build_interfuser_visual_d7_configs tests.test_run_interfuser_visual_d7_pair tests.test_summarize_interfuser_visual_d7 2>&1 | tail -5
```
Expected: `OK`（四个测试模块全过）。若模块名不匹配，用 `"$PY" -m unittest discover -s tests -k visual -v 2>&1 | tail` 定位实际测试名。

---

### Task 2: 归档 failed test manifest + test preflight

**Files:**
- Move: `results/thesis_m2/m2-interfuser-visual-pair-test-v1-seed20260724-20260724-v1/` → `results/thesis_m2/_audit/m2-interfuser-visual-pair-test-v1-seed20260724-20260724-v1-failed-20260731/`

**Interfaces:**
- Consumes: Task 1 的 green 确认
- Produces: 干净的原 test 路径 + preflight 通过（可执行 A）

- [ ] **Step 1: 归档 failed manifest（保留失败证据，不抹去）**

```bash
cd /data/shijj/interfuser_origin
mkdir -p results/thesis_m2/_audit
mv results/thesis_m2/m2-interfuser-visual-pair-test-v1-seed20260724-20260724-v1 \
   results/thesis_m2/_audit/m2-interfuser-visual-pair-test-v1-seed20260724-20260724-v1-failed-20260731
ls results/thesis_m2/_audit/
```
Expected: 归档目录出现；原路径 `results/thesis_m2/m2-interfuser-visual-pair-test-v1-seed20260724-20260724-v1` 不再存在。

- [ ] **Step 2: test preflight（静态校验，不 execute）**

```bash
"$PY" tools/evaluation/run_interfuser_visual_test.py \
  --config configs/thesis/interfuser_visual_pair_test_v1.json --preflight-only 2>&1 | tail -20
```
Expected: 退出码 0；输出含 formal contract 校验通过、test index 哈希匹配、无 GPU 占用错误。若报 formal manifest 问题或 test index 漂移，停止排查，不强行 execute。

---

### Task 3: A 执行（test 离线评测）+ 验证 pipeline-valid

**Files:**
- Create: `results/thesis_m2/m2-interfuser-visual-pair-test-v1-seed20260724-20260724-v1/test_manifest.json`

**Interfaces:**
- Consumes: Task 2 preflight 通过、干净 test 路径
- Produces: test manifest（`pipeline_valid=true`，含单帧 + 连续帧指标）—— B 的硬前置

- [ ] **Step 1: 执行 test 评测（B0 后 V 串行，GPU 6）**

```bash
"$PY" tools/evaluation/run_interfuser_visual_test.py \
  --config configs/thesis/interfuser_visual_pair_test_v1.json \
  --run-id m2-interfuser-visual-pair-test-v1-seed20260724-20260724-v1 2>&1 | tail -30
```
Expected: 退出码 0；日志显示 B0、V 两 variant 先后完成、写出 `test_manifest.json`。run_id 必须与 config 冻结值一致（runner 拒绝其他 run_id）。

- [ ] **Step 2: 验证 pipeline-valid 与 variants**

```bash
"$PY" - <<'PYEOF'
import json
p="results/thesis_m2/m2-interfuser-visual-pair-test-v1-seed20260724-20260724-v1/test_manifest.json"
d=json.load(open(p))
print("status:", d["status"], "pipeline_valid:", d["pipeline_valid"], "errors:", d.get("errors"))
print("variants:", [(v.get("variant"), v.get("pipeline_valid")) for v in d.get("variants",[])])
PYEOF
```
Expected: `status: completed`、`pipeline_valid: True`、`errors: []`、variants 含 B0 和 V 两项且各自 `pipeline_valid=True`。若 `pipeline_valid=False`，读 errors 定位；不进入 B。

- [ ] **Step 3: 抓取离线方向性 5 项（H1 判定要用）**

```bash
"$PY" - <<'PYEOF'
import json
p="results/thesis_m2/m2-interfuser-visual-pair-test-v1-seed20260724-20260724-v1/test_manifest.json"
d=json.load(open(p))
for v in d["variants"]:
    print(v["variant"], json.dumps(v.get("metrics",{}), ensure_ascii=False)[:600])
PYEOF
```
Expected: 每个 variant 打出 traffic AP/AUC、occupied IoU、waypoint ADE、horizon-10 FDE 等字段（具体字段名以 manifest 为准）。记录 B0 与 V 的 5 项数值，后续 §判定用。字段名若与预期不符，以 manifest 实际 schema 为准——这是读取，不是改数据。

---

### Task 4: A 证据记录 + commit

**Files:**
- Create: `docs/experiment_records/2026-07-31-m2-visual-test-frozen-eval.md`

**Interfaces:**
- Consumes: Task 3 的 test manifest（SHA-256 + 指标）
- Produces: 进 git 的 A 阶段证据记录；evidence_ledger 状态更新

- [ ] **Step 1: 计算 test manifest SHA-256**

```bash
sha256sum results/thesis_m2/m2-interfuser-visual-pair-test-v1-seed20260724-20260724-v1/test_manifest.json
```
Expected: 64 字符哈希，记入实验记录。

- [ ] **Step 2: 写 experiment_record**

新建 `docs/experiment_records/2026-07-31-m2-visual-test-frozen-eval.md`，按现有记录风格（表格头部 + Run ID/Git HEAD/SHA-256/指标/结论边界/`[PROTOCOL]` 尾）。内容：test run_id、git head、test_manifest SHA-256、B0/V 离线 5 项、连续帧稳定性残差、失败 manifest 归档路径、结论边界（仅 test 离线，D7 待 B）。

- [ ] **Step 3: 更新 evidence_ledger 与 protocol 状态行**

修改 `paper/evidence_ledger.md`：把"冻结 test 被拦截失败"改为已完成 + 实测数值；移除对应"投稿前门禁"项。修改 `docs/interfuser_visual_transfer_protocol_v1.md` 状态行（FORMAL-RUNNING → test completed / D7 pending）。

- [ ] **Step 4: commit（只动 docs，不碰 results）**

```bash
git add docs/experiment_records/2026-07-31-m2-visual-test-frozen-eval.md paper/evidence_ledger.md docs/interfuser_visual_transfer_protocol_v1.md
git commit -m "docs: record H1 frozen test eval and unlock D7" -m "test pipeline_valid; B0/V offline 5-metric + temporal residual recorded" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
Expected: commit 成功，`git status` 显示 `results/` 未被 add。

---

### Task 5: 生成 D7 config + D7 pair prelight

**Files:**
- Create: `configs/thesis/interfuser_visual_d7_b0_v1.json`、`interfuser_visual_d7_v_v1.json`、`interfuser_visual_d7_pair_v1.json`（由 builder 一次性生成）

**Interfaces:**
- Consumes: Task 3 的 pipeline-valid test manifest（builder 校验 test checkpoint 哈希与 formal best 一致）
- Produces: 三份 D7 config（B 的输入）

- [ ] **Step 1: build preflight（不生成，只 load contract 验证前置）**

```bash
"$PY" tools/evaluation/build_interfuser_visual_d7_configs.py \
  --config configs/thesis/interfuser_visual_d7_build_v1.json 2>&1 | tail -15
```
Expected: 退出码 0；校验 test manifest pipeline-valid、checkpoint 哈希一致、三份输出路径未占用。

- [ ] **Step 2: build（生成三份 config）**

```bash
"$PY" tools/evaluation/build_interfuser_visual_d7_configs.py \
  --config configs/thesis/interfuser_visual_d7_build_v1.json --build 2>&1 | tail -15
ls configs/thesis/interfuser_visual_d7_{b0,v,pair}_v1.json
```
Expected: 三份 config 文件生成；builder 拒绝覆盖已存在目标（首次生成应成功）。

- [ ] **Step 3: D7 pair preflight（构建 42 attempt 计划，不启动 CARLA）**

```bash
"$PY" tools/evaluation/run_interfuser_visual_d7_pair.py \
  --config configs/thesis/interfuser_visual_d7_pair_v1.json 2>&1 | tail -20
```
Expected: 退出码 0；输出 B0、V 各 21 attempt 的计划，路线/seed 顺序符合 `[18,6,12,30,36,39,0]`×`[0,1,2]`。

- [ ] **Step 4: commit config（config 是冻结输入，进 git）**

```bash
git add configs/thesis/interfuser_visual_d7_b0_v1.json configs/thesis/interfuser_visual_d7_v_v1.json configs/thesis/interfuser_visual_d7_pair_v1.json
git commit -m "feat: freeze B0/V D7 pair configs from valid test" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: B 执行（D7 配对闭环 42 attempt）

**Files:**
- Create: `results/thesis_m2/m2-interfuser-visual-d7-pair-seeds0-2-20260724-v1/`、`m2-interfuser-visual-d7-b0-...-v1/`、`m2-interfuser-visual-d7-v-...-v1/`

**Interfaces:**
- Consumes: Task 5 的三份 config + 用户给定的独占窗口
- Produces: 42 个 attempt 的原始 Leaderboard JSON + 两组 run manifest

**前置硬门（执行前人工确认）：** 用户已给定独占窗口；窗口期间 GPU 6/7 不启动其他任务；常开 CARLA 2000 不动。

- [ ] **Step 1: 独占窗口就绪确认**

```bash
nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name --format=csv,noheader
ss -ltnp 2>/dev/null | grep -E ':(2155|2255)\b' || echo "2155/2255 空闲"
```
Expected: GPU 6/7 无他人进程；2155/2255 无监听（runner 会自起 CARLA）。窗口未就绪则不启动。

- [ ] **Step 2: D7 pair execute（B0→V 串行 42 attempt，长任务，后台 + Monitor）**

```bash
"$PY" tools/evaluation/run_interfuser_visual_d7_pair.py \
  --config configs/thesis/interfuser_visual_d7_pair_v1.json \
  --pair-run-id m2-interfuser-visual-d7-pair-seeds0-2-20260724-v1 \
  --execute 2>&1 | tee results/thesis_m2/m2-interfuser-visual-d7-pair-seeds0-2-20260724-v1.launcher.log
```
Expected: 端口 2155/2255、GPU 6/7 交替占用；attempt 逐条产出 Leaderboard JSON。预计 8h+ 量级，不精确。本机用 `run_in_background` 启动 SSH，并用 Monitor 跟进 launcher.log 的 attempt 完成 / pipeline-invalid / crash 标志，覆盖成功与失败两类信号。

- [ ] **Step 3: 若中途断（被抢/崩溃），审计后 resume，不覆盖**

```bash
# 先人工审计不完整 attempt 目录的 cleanup_error 与最后写入
ls results/thesis_m2/m2-interfuser-visual-d7-b0-seeds0-2-20260724-v1/
"$PY" tools/evaluation/run_interfuser_visual_d7_pair.py \
  --config configs/thesis/interfuser_visual_d7_pair_v1.json \
  --pair-run-id m2-interfuser-visual-d7-pair-seeds0-2-20260724-v1 \
  --resume
```
Expected: `--resume` 跳过已 pipeline-valid 的 attempt，不完整目录要求先人工审计（runner 拒绝自动抹去）。

- [ ] **Step 4: 确认两组产物目录已生成（21/21 正式校验由 Task 7 汇总器做）**

```bash
ls results/thesis_m2/m2-interfuser-visual-d7-b0-seeds0-2-20260724-v1/
ls results/thesis_m2/m2-interfuser-visual-d7-v-seeds0-2-20260724-v1/
```
Expected: 两组目录各含 21 个 attempt 产物与一份 run_manifest.json。逐 attempt 的 pipeline-valid 与 21/21 完整性由 Task 7 的 `summarize_interfuser_visual_d7` 强制校验（汇总器拒绝非 21/21 或含 invalid attempt 的输入）。

---

### Task 7: 汇总 + H1 判定

**Files:**
- Create: `results/thesis_m2/m2-interfuser-visual-d7-pair-seeds0-2-20260724-v1/h1_summary.json`

**Interfaces:**
- Consumes: Task 6 两组 run manifest
- Produces: H1 汇总 JSON + 预注册判定（支持/混合/不支持）

- [ ] **Step 1: 定位两组 run manifest（避免猜文件名）**

```bash
find results/thesis_m2/m2-interfuser-visual-d7-b0-seeds0-2-20260724-v1 -name 'run_manifest.json'
find results/thesis_m2/m2-interfuser-visual-d7-v-seeds0-2-20260724-v1 -name 'run_manifest.json'
```
Expected: 各返回一条路径；记为 `$B0_M`、`$V_M`。若返回多条或零条，先排查目录结构再继续。

- [ ] **Step 2: 汇总（复用 M0 汇总门禁 + 配对差值）**

```bash
"$PY" tools/evaluation/summarize_interfuser_visual_d7.py \
  --b0-manifest <B0_M> --v-manifest <V_M> \
  --output results/thesis_m2/m2-interfuser-visual-d7-pair-seeds0-2-20260724-v1/h1_summary.json 2>&1 | tail
```
Expected: `{"valid": true, "output": "..."}`。汇总器校验两组 21/21、两 config 仅 checkpoint 差异、产出 21 个 route×seed 差值与宏平均。

- [ ] **Step 3: 按 protocol §8 判定 H1**

```bash
"$PY" - <<'PYEOF'
import json
s=json.load(open("results/thesis_m2/m2-interfuser-visual-d7-pair-seeds0-2-20260724-v1/h1_summary.json"))
# 以 summarize 实际产出字段为准读取 DS macro 差值与离线 5 项改善计数
print(json.dumps({k:s[k] for k in s if "diff" in k.lower() or "macro" in k.lower() or "ds" in k.lower()}, ensure_ascii=False, indent=2)[:800])
PYEOF
```
判定规则（protocol §8）：D7 macro DS 的 `V−B0>0` 且离线 5 项 ≥3 改善 → 支持；方向不一致或 DS 差=0 → 混合/证据不足；`V−B0<0` 且 <3 改善 → 不支持。**读 summarize 实际字段名**，不预设 schema。把 Task 3 抓的离线 5 项代入多数票。

---

### Task 8: B 证据记录 + 最终 commit

**Files:**
- Create: `docs/experiment_records/2026-07-31-m2-visual-d7-pair-h1.md`
- Modify: `paper/evidence_ledger.md`（D7/H1 结论落账）、`paper/adist_2027_draft.md` 的 limitations（single-seed/validation-only 边界按实测更新）

**Interfaces:**
- Consumes: Task 7 的 h1_summary.json + 判定结论
- Produces: 进 git 的 B 阶段证据 + 论文账本更新

- [ ] **Step 1: 计算 h1_summary SHA-256 + 两组 manifest SHA**

```bash
sha256sum results/thesis_m2/m2-interfuser-visual-d7-pair-seeds0-2-20260724-v1/h1_summary.json
```

- [ ] **Step 2: 写 experiment_record**

`docs/experiment_records/2026-07-31-m2-visual-d7-pair-h1.md`：pair/b0/v run_id、git head、各 SHA-256、D7 macro DS/RC/IS（B0 与 V）、21 个 route×seed 差值摘要、离线 5 项、连续帧残差、H1 判定结论、失败路线清单、`[PROTOCOL]` 尾。

- [ ] **Step 3: 更新 evidence_ledger + draft limitations**

`paper/evidence_ledger.md`：把"B0/V D7 尚未完成"改为实测结论；把"不能写成闭环改善"按 H1 判定结果更新（支持→可写闭环改善；混合/不支持→如实写 negative/neutral）。`paper/adist_2027_draft.md` §4.3 Limitations：按 single-seed/闭环已评 的实际状态调整措辞。

- [ ] **Step 4: 最终 commit**

```bash
git add docs/experiment_records/2026-07-31-m2-visual-d7-pair-h1.md paper/evidence_ledger.md paper/adist_2027_draft.md
git commit -m "docs: record H1 D7 pair result and verdict" -m "D7 closed-loop paired eval frozen; H1 verdict per protocol §8" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 5（可选）: push 到 fork（走代理，间歇性）**

```bash
GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_ed25519 -o ProxyCommand='nc -X connect -x 127.0.0.1:7897 %h %p'" git push origin codex/stop-boundary-label-export 2>&1 | tail
```
Expected: push 成功；若 22/代理超时，从另一会话查 `git rev-parse origin/<branch>` 确认，按 HANDOFF §4.8 重试。

---

## Self-Review

**Spec coverage:** spec §4.1（归档+同 run_id 重跑）→ Task 2；§4.2/4.3（test execute + 指标）→ Task 3；§5.1（build 前置）→ Task 5；§5.2（D7 execute）→ Task 6；§5.3（汇总门禁）→ Task 7；§6 顺序→全计划；§7 判定→ Task 7 Step 3；§8 风险（被抢/崩溃）→ Task 6 Step 1/3；§10 preflight→ Task 1/2/5；§12 交付物→ Task 4/8。覆盖完整。

**Placeholder scan:** Task 7 Step 2 的 `<B0_M>`/`<V_M>` 是 Step 1 `find` 定位后填入的变量，非占位；其余命令均为可执行实文，无 TBD/TODO。

**Type/字段一致性:** summarize 产出字段名未预设（Task 7 Step 3 显式声明按实际 schema 读），避免和 manifest 真实字段冲突。run_id 在 Task 2/3/5/6 一致使用冻结值。
