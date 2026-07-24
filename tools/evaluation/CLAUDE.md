# tools/evaluation/
> L2 | 父级: ../CLAUDE.md

## 成员清单

CLAUDE.md: 闭环评测工具模块地图，约束预检、runner 和汇总器按同一机器配置工作。
preflight_thesis_baseline.py: 在启动 CARLA 前校验 M0 配置、文件哈希、路线/场景覆盖、地图排除和 Git 代码锚点。
runtime_resources.py: 守卫 CARLA 进程组、TCP 端口、GPU 计算 owner 与显存生命周期，确保启动前独占且连续 attempt 之间完全归零。
run_thesis_baseline.py: 消费通过 P0 的配置生成 D7/A36 运行计划，以短命子进程隔离 CARLA 原生启动 RPC，并将清理前 CARLA 退出与 evaluator 非零退出分类为显式基础设施失败。
summarize_thesis_baseline.py: 验证多个 run manifest 构成无缺失、无重复且输入可比的路线×种子矩阵，按先路线内种子均值、再路线宏平均的冻结口径生成确定性统计。
interfuser_offline_metrics.py: 纯归约 InterFuser traffic grid、逐时域 waypoint、junction、red-light 与 stop-sign 输出，使用全 test 精确分母生成 AP/AUC/IoU、误差和逐类混淆指标。
run_interfuser_visual_test.py: 在 formal B0/V 完整有效后，以隔离单 GPU worker 串行 strict-load 两个 best checkpoint，守卫 test index/哈希/资源并生成配对指标差值 manifest。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
