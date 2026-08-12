# tools/evaluation/
> L2 | 父级: ../CLAUDE.md

## 成员清单

CLAUDE.md: 闭环评测工具模块地图，约束预检、runner 和汇总器按同一机器配置工作。
preflight_thesis_baseline.py: 在启动 CARLA 前校验 M0 配置、文件哈希、路线/场景覆盖、地图排除和 Git 代码锚点。
runtime_resources.py: 守卫 CARLA 进程组、TCP 端口、GPU compute owner 与显存生命周期，默认独占并只为显式配置提供总显存阈值内的受控共享。
run_thesis_baseline.py: 消费通过 P0 的配置生成 D7/A36 运行计划，传递默认独占/显式共享资源策略与可配置显存释放等待，以短命子进程隔离 CARLA 原生启动 RPC，并将异常退出分类为显式基础设施失败。
summarize_thesis_baseline.py: 验证多个 run manifest 构成无缺失、无重复且输入可比的路线×种子矩阵，并将路线进度 blocked 判据哈希纳入输入契约后按冻结口径生成确定性统计。
summarize_interfuser_visual_d7.py: 分别复用 M0 门禁验证 B0/V 的 21 条 D7 矩阵，只放行 checkpoint/provenance 差异，并结合冻结 test 生成配对差值与 H1 预注册结论。
analyze_interfuser_visual_d7_failures.py: 配对归约 D7 原始违规事件与 control.csv 连续控制帧，定位首碰撞、早期转向/车道偏移及路口头分歧，为 H1 负结果提供路线级归因。
run_interfuser_visual_d7_pair.py: 在冻结 test 有效后构建固定路线/seed 的 B0→V 子计划，以单父级 manifest 串行复用 M0 runner，并在失败时阻止后续 variant。
build_interfuser_visual_d7_configs.py: 消费预注册 build 契约，只在 formal/test 均有效后从实际 best checkpoint 一次性生成 hash-bound B0/V child 与 pair 配置。
interfuser_offline_metrics.py: 纯归约 InterFuser traffic grid、逐时域 waypoint、junction、red-light 与 stop-sign 输出，并以目标条件相邻帧残差度量预测稳定性。
run_interfuser_visual_test.py: 在 formal B0/V 完整有效后，以严格索引的隔离单 GPU worker 串行 strict-load 两个 best checkpoint，守卫 test 帧/相邻帧计数、哈希和资源并生成配对指标差值 manifest。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
