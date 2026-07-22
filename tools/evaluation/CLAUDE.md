# tools/evaluation/
> L2 | 父级: ../CLAUDE.md

## 成员清单

CLAUDE.md: 闭环评测工具模块地图，约束预检、runner 和汇总器按同一机器配置工作。
preflight_thesis_baseline.py: 在启动 CARLA 前校验 M0 配置、文件哈希、路线/场景覆盖、地图排除和 Git 代码锚点。
runtime_resources.py: 守卫 CARLA 进程组、TCP 端口与 GPU 生命周期，确保连续 attempt 之间外部资源完全归零。
run_thesis_baseline.py: 消费通过 P0 的配置生成 D7/A36 运行计划，编排每个 route/seed 的隔离执行、原始结果与 attempt manifest。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
