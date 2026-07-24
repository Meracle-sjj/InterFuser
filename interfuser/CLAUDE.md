# interfuser/
> L2 | 父级: ../CLAUDE.md

## 成员清单

CLAUDE.md: InterFuser 下游训练与内置 timm 运行时的局部地图。
train.py: 下游多任务训练入口，协调模型、CARLA 数据、优化、验证和 checkpoint 生命周期。
render.py: 将交通预测与 waypoint 目标渲染为训练诊断图像。
distributed_train.sh: 用 PyTorch distributed launcher 启动多 GPU `train.py`。
monitor_training.sh: 轮询历史训练输出的运行状态辅助脚本。
scripts/: 上游训练命令模板；论文运行以 `configs/thesis/` 和 `tools/training/` 为权威编排层。
timm/: 仓库内置 timm 0.4.13 分支，包含 InterFuser 模型、CARLA dataset 和训练基础设施。
tests/: 上游 timm 层、模型与优化器回归。
docs/: 内置 timm 上游文档，不改变毕业论文实验契约。
setup.py: 内置 timm Python 包安装入口。
setup.cfg: 内置 timm 打包与工具配置。
requirements.txt: 内置 timm 运行依赖基线。
requirements-docs.txt: 内置 timm 文档构建依赖。
requirements-modelindex.txt: 模型索引生成依赖。
README.md: 内置 timm 上游说明。
hubconf.py: PyTorch Hub 模型导出。
model-index.yml: 内置 timm 模型索引。
mkdocs.yml: 内置 timm 文档站配置。
MANIFEST.in: Python 包非代码文件清单。
LICENSE: 上游 timm 许可证。
.gitattributes: 子模块属性配置。
.gitignore: 子模块生成物忽略规则。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
