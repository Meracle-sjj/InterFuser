# M2 固定 M0 视觉替换 route39 三 seed 零微调诊断

| 字段 | 内容 |
| --- | --- |
| 状态 | **COMPLETED：M0-FT/M0-V 6/6 pipeline-valid，无启动期灾难性失配，准入完整 D7** |
| 固定底座提交 | `14c388befebf952ffc83a75d074cb2b247b82f15` |
| 诊断配置提交 | `34d986c91b0af91e7a8c98e75e53b68f4835b2d2` |
| 三 seed 续跑提交 | `67795263e7fe921ff6fea1da56d948d161771ca1` |
| 路线/seed | `route39 / [0,1,2]` |
| 微调 | 无；M0-FT 保留历史 RGB，M0-V 只 strict 替换交通语义 ResNet50d |

## 1. 原始运行

| variant | seeds | Run ID | run manifest SHA-256 |
| --- | --- | --- | --- |
| M0-FT | 0 | `m2-interfuser-m0-ft-route39-seed0-zero-ft-20260805-v1` | `e27d5dafa930cd175111030e51e0c1bb570e2f9627dc99099de6eebbe70a5e60` |
| M0-FT | 1–2 | `m2-interfuser-m0-ft-route39-seeds1-2-zero-ft-20260805-v1` | `64e7040425847b0c1c37c43561c9cb54c745ce8c911f51242c41709518e0d91d` |
| M0-V | 0 | `m2-interfuser-m0-v-route39-seed0-zero-ft-20260805-v1` | `669f8461fcb0c55e18b80e2bb4223db6f8521cd85fbda7c1f8f7b40ac9b3e7e7` |
| M0-V | 1–2 | `m2-interfuser-m0-v-route39-seeds1-2-zero-ft-20260805-v1` | `999ec0ff8e0b247da5a050c1f66c2224cdc962634e6962d4fc48a3c9e9b533c3` |

四个 run 共同绑定配置内的 M0-FT/M0-V checkpoint 哈希、相同 agent/控制器/路线/场景、背景交通、GPU 6、graphics adapter 7、端口 2155/2255 与外部超时。每个 attempt 均 `pipeline_valid=true`，无 `vehicle_blocked`，最终状态均为 `Failed - Agent timed out`。

## 2. 闭环指标

| seed | M0-FT DS | M0-V DS | V-FT | M0-FT RC | M0-V RC | V-FT | M0-FT IS | M0-V IS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 34.8616 | 31.4216 | -3.4400 | 83.3647 | 80.5683 | -2.7965 | 0.4182 | 0.3900 |
| 1 | 37.1512 | 52.6040 | +15.4528 | 81.6511 | 80.9292 | -0.7219 | 0.4550 | 0.6500 |
| 2 | 37.5347 | 31.5624 | -5.9723 | 82.4939 | 80.9292 | -1.5647 | 0.4550 | 0.3900 |
| **均值** | **36.5159** | **38.5293** | **+2.0135** | **82.5032** | **80.8089** | **-1.6943** | **0.4427** | **0.4767** |

M0-V 的 RC 在三个 seed 都略低，但 DS 方向不一致，seed1 的 IS 提升主导了宏平均 DS 正差。因此本路线只支持“未发生灾难性闭环退化”，不支持单路线视觉收益结论。

## 3. 连续控制帧

| 指标（三 seed 均值） | M0-FT | M0-V | V-FT |
| --- | ---: | ---: | ---: |
| 前 40 个运动帧平均绝对转向 | 0.04318 | 0.00935 | -0.03383 |
| 前 40 个运动帧最大绝对车道偏移 | 0.42097 m | 0.17313 m | -0.24784 m |
| 首次 `abs(lane_offset) >= 1 m` step | 713.67 | 683.00 | -30.67 |
| layout collision 次数 | 2 | 3 | +1 |
| vehicle blocked 次数 | 0 | 0 | 0 |

M0-V 三个 seed 的前 40 帧转向响应都显著更小，但首次 1 m 偏移仍发生在 step 679–686，远晚于旧从头重训 V 的 step 35–36；三个 seed 均未命中前 10 秒碰撞、blocked 或前 50 个运动帧 1 m 偏移门禁。固定底座视觉替换没有复现旧 V 的启动轨迹回归。

## 4. 结论边界与下一步

本结果证明交通语义 ResNet50d 可以在不微调的情况下接入已训练 M0，并完成 route39 闭环；它没有证明总体收益。由于 DS/RC/IS 在单路线上的方向混合且 M0-V 无灾难性失配，本轮不启动短微调，按协议准入 D7 剩余六条路线三 seed。完整 D7 必须合并本记录的 route39 三 seed，再决定零微调视觉替换是否值得保留或是否进入配对短微调。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
