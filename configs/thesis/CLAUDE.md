# configs/thesis/
> L2 | 父级: ../CLAUDE.md

## 成员清单

CLAUDE.md: 论文配置子模块地图，规定配置文件与研究里程碑的对应关系。
baseline_eval_v1.json: M0 评测校准配置，固定模型和输入哈希、开发/主评测路线、随机种子及运行环境变量。
semantic_classes_v1.json: M1 语义类别配置，唯一映射 CARLA 0.9.16 source tag，并给出 pilot 数据准入阈值。
semantic_split_v1.json: M1 sequence 划分配置，固定 Town+route 原子分组、确定性种子、目标比例与每个 split 的类别/Town 准入门槛。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
