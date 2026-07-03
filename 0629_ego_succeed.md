# 0629 阶段结论:interfuser_origin 干净基线 —— ego 已能正常行驶

> 日期:2026-06-29 ｜ 位置:`/home/shijj/interfuser_origin`(软链到 `/data/shijj/interfuser_origin`,21T 大盘)
> 结论一句话:**自车(ego)已能在 5090 上正常前进、不偏、不摆**;背景交通乱跑是 CARLA 版本错配,单独问题。

---

## 1. 目标与起点
- 用户目标:在 RTX 5090 机器上装一份原版 InterFuser,让自车正常前进(原 `interfuser` 目录的车左右摆动/走不动)。
- 关键认知纠正:**当年"装不上原版"的真因是 PyTorch 1.x 不支持 5090(sm_120 需 CUDA12.4+/torch2.5+),不是 CARLA 版本**。所以"原版"只能是「opendilab 原版代码 + 现代栈」。

## 2. 搭了什么(interfuser_origin)
- **代码**:全新 `git clone opendilab/InterFuser`,仅 overlay 现仓库已验证能开的行驶栈(`leaderboard/team_code` 的 .py、`interfuser/` 自带 timm 分叉、路由文件);去掉数据采集等冗余。
- **环境**:克隆 `interfuser50` → `interfuser_origin`(python3.10 + torch2.10.dev+cu128 + carla0.9.16 + timm1.0.22),在 5090 上 CUDA 可用。
- **CARLA**:软链接复用现有 0.9.16(`/home/shijj/interfuser/carla`),不复制 19GB。
- **checkpoint**:`model_20260121.tar`(Phase3 最佳,DS68.4)经 `INTERFUSER_MODEL_PATH` 绝对路径引用,不复制;默认不再用会摆动的旧 `interfuser.pth.tar`。
- **冒烟测试**:模型构建 OK(52.9M 参数),`load_state_dict` missing=0/unexpected=0(架构与 checkpoint 完全匹配),全链路 import 通过。

## 3. ✅ 已解决

### 3.1 ego 正常行驶
- 在**专用 CARLA 实例**上跑,speed 巡航 4–5 m/s,steer **0 次正负翻转(无摆动)**,正常停走。
- 关键发现:之前"ego 不走"是**共享 CARLA(port 2000,跑了5天+8卡被占满)过载卡死**,不是模型/安装问题。换专用实例(如 port 2155 / GPU5 / Low 画质)即可正常行驶。

### 3.2 右偏(驶入对向车道)已修复 —— 数据坐实
- `lane_offset` 沿车道右向量投影,正值=偏右。基线均值 **+0.47m(持续右偏)**。
- 修法(不改代码):`INTERFUSER_LANE_CENTER_STEER_GAIN=0.5`(默认 0.0=关闭,虽检测了 lane_offset 却没用上)。
- 效果(499帧对比):

  | 指标 | 基线(居中关) | 居中开(gain=0.5) |
  |---|---|---|
  | lane_offset 均值 | +0.47(右偏) | **−0.21(近中心)** |
  | offset 范围 | 峰值 +1.57 | [−0.43, **0.00**](不再往右) |
  | 转向翻转(摆动) | — | **0 次** |
  | 居中率(\|offset\|<0.3) | 低 | 85% |

  gain=0.5 略过矫正偏左,0.35 更贴中心。右偏根因:模型在 0.9.16+tesla.model3 上 waypoints 本身有右偏(域偏移)+ 横向开环。

## 4. ❌ 未解决:背景交通乱跑(CARLA 0.9.10→0.9.16 错配)
- 现象:背景车 ~60–80% 在路面外,**几乎所有都不动(vel≈0)**,与自车模型无关。
- 数据排除:
  - 不是过载 —— 20 辆比 120 辆**更差**(80% 路外、0/20 动)。
  - 不是 hybrid 物理 —— `set_hybrid_physics_mode` 在 0.9.16 上 `set_hybridphysicsmode_radius` API 缺失,半残,**会把 NPC 冻死**(已默认关闭)。
  - NPC 分散各处(非原点聚集 → spawn 没失败),但 road_dist 4–7m、vel≈0。
- 根因:**Traffic Manager 没在驱动背景车**。本项目内置的 leaderboard/scenario_runner 是 CARLA **0.9.10.1 时代**代码(`setup_carla.sh` 下载的就是 0.9.10.1),其 TM 生成/控制逻辑在 0.9.16 上不工作。这与"5090 装不上原版"是**同一根源**:整套代码面向 0.9.10.1。
- 死结提示:回到 CARLA 0.9.10.1 能修交通,但那样 PyTorch 1.x 又用不了 5090。

## 5. ⚠️ 稳定性(segfault / 加载卡住)
- eval 偶发 `Loading the world` 后段错误 + CARLA 实例挂掉,属 0.9.16 兼容性 + 机器过载的随机抽风。
- 处理:**重启几次直到跑通**(用户经验)。已写自动重试脚本 `results/run_until_ok.sh`(杀 CARLA→重起→跑→检测行驶,最多5轮)。

## 6. 怎么跑
```bash
# 推荐:专用 CARLA 实例 + 车道居中(右偏已修)
# 1) 起专用 CARLA(避开被占的 port 2000/2001/2002)
cd /home/shijj/interfuser_origin/carla
./CarlaUE4.sh --world-port=2155 -graphicsadapter=5 -quality-level=Low -RenderOffScreen &

# 2) 跑 eval
cd /home/shijj/interfuser_origin
INTERFUSER_LANE_CENTER_STEER_GAIN=0.5 \
PORT=2155 TM_PORT=2255 CUDA_DEV=0 \
bash leaderboard/scripts/run_origin_eval.sh
# segfault 就重跑(或用 results/run_until_ok.sh 自动重试)
```
**关键环境变量**(都在 `leaderboard/scripts/run_origin_eval.sh`,可覆盖):
- `INTERFUSER_LANE_CENTER_STEER_GAIN=0.5` —— 车道居中(修右偏,**务必开**)
- `INTERFUSER_MODEL_PATH` —— checkpoint(默认 model_20260121.tar)
- `CUDA_DEV` —— 推理 GPU(选 0% util 的)
- `PORT` / `TM_PORT` —— CARLA / TrafficManager 端口(避开 2000-2002)
- `INTERFUSER_BG_VEHICLES` —— 背景车数(默认 120;**降数量不能修交通**)
- `INTERFUSER_TM_HYBRID_PHYSICS` —— hybrid 物理(**保持 0/关**,开了冻死 NPC)

## 7. 产物路径
- 可视化视频(看行驶):
  - `results/driving_vis_LANECENTER.mp4` —— 车道居中后(右偏已修,**推荐看**)
  - `results/driving_vis.mp4` —— 早期(右偏)
  - `results/driving_vis_TRAFFIC.mp4` —— 交通乱跑视角
- 逐帧截图:`data/eval_origin/routes_town01_tiny3_*/meta/*.jpg`(1200×800,三视角+预测+BEV)
- 结果 JSON:`results/origin_result.json`

## 8. 已落地的代码改动(相对 opendilab 原版)
- `leaderboard/team_code/interfuser_config.py` 等 —— overlay 自现仓库(已验证能开的行驶逻辑 + env-var 坐标变换)。
- `leaderboard/leaderboard/leaderboard_evaluator.py` —— TM 调参块(`INTERFUSER_TM_*`,默认 hybrid 关)+ 5090 必需的 PYTHONPATH/LD_LIBRARY_PATH 由 eval 脚本设置。
- `leaderboard/leaderboard/scenarios/route_scenario.py` —— `[TRAFFIC]` 生成日志 + `INTERFUSER_BG_VEHICLES` 车数覆盖。
- `leaderboard/scripts/run_origin_eval.sh` —— 干净的 5090 eval 入口(自包含激活环境、设库路径)。

## 9. 待讨论(后续)
1. **背景交通**:要不要深入改 scenario_runner 的 TM 逻辑适配 0.9.16(让 NPC 动起来)?工程量中等,且受 0.9.10↔0.9.16 死结限制。
2. **车道居中**调到 gain=0.35 看是否更贴中心;或把它设为默认。
3. **红灯前过早停车**(leaderboard 慢速过路口机制)——你提过先放后面。
4. 稳定性:是否要等机器空闲时跑完整 3 路由拿 RC/DS 得分。
