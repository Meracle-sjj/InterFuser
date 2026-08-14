# 官方 InterFuser RGB/LiDAR 模态依赖审计协议 v1.0

| 字段 | 内容 |
| --- | --- |
| 状态 | **FROZEN：结果读取前冻结干预、阈值与 validation 全量样本** |
| 问题 | 作者发布 InterFuser 是否实质依赖 RGB，还是主要由 LiDAR/测量/目标点驱动 |
| 配置 | `configs/thesis/interfuser_official_modality_ablation_v1.json` |
| 运行器 | `tools/evaluation/run_interfuser_modality_ablation.py` |

## 1. 因果问题

视觉骨干拥有更好的语义分割指标，不等于融合后的驾驶模型会使用这些特征。本审计固定 checkpoint、validation 样本、LiDAR、测量、目标点和所有模型参数，只改变一个输入模态；对真值任务指标和同一样本正常输出的配对变化同时归约。

主判断对象为作者发布权重的保真导出 `official_b0`。`official_b0_v` 只用于判断交通语义骨干替换是否改变 RGB 依赖程度，不能反过来定义官方模型的原始行为。

## 2. 冻结输入

- 数据：无泄漏 validation index 的全部 4,613 帧，不读取 frozen test；
- 城镇：Town01、Town03、Town04、Town05；
- checkpoint：`official_b0` 与仅替换 RGB backbone 的 `official_b0_v`；
- 推理：eval + inference mode，batch 16，固定 seed，不做数据增强；
- 所有 condition 在同一个 batch 上依次执行，正常输出只计算一次并作为配对参照。

## 3. 干预条件

1. `normal`：输入不变；
2. `rgb_mean_fill`：四路归一化 RGB 全部置零，即各通道固定在 ImageNet 均值；
3. `rgb_blur`：四路 RGB 使用 `11×11` 平均池化，保留低频布局并破坏细节；
4. `rgb_shuffle`：四路相机保持彼此配对，但在 batch 内整体循环错配一个样本，保留自然图像分布并破坏场景对应关系；
5. `lidar_zero`：只清空处理后的 LiDAR histogram，作为模态效应量参照。

`rgb_mean_fill` 与 `rgb_shuffle` 是依赖判定的强破坏条件；`rgb_blur` 只用于区分模型依赖低频布局还是细粒度视觉，不进入自动阈值。

## 4. 指标

每个 condition 均计算 traffic occupancy、waypoint ADE/FDE、junction、red-light 与 stop-sign 离线指标。相对 `normal` 额外计算：

- traffic probability/full-output MAE；
- waypoint 输出 shift ADE/FDE；
- traffic decoder feature 的全局相对 L2 与余弦距离；
- 三个二分类 head 的正类概率变化和 argmax 翻转率。

这两类指标必须一起解释：真值指标不变但配对输出显著变化，说明替代信息或数据标签不敏感；配对输出也不变，才支持“模型忽略该模态”。

## 5. 预注册判定

对 `rgb_mean_fill` 与 `rgb_shuffle` 取最大效应，并除以 `lidar_zero` 的相同输出效应：

- `weak_rgb_dependency`：RGB/LiDAR 最大输出比小于 `0.25`，且 waypoint ADE 相对恶化小于 `5%`；
- `material_rgb_dependency`：任一输出比不小于 `0.50`，或 waypoint ADE 相对恶化不小于 `10%`；
- 其余为 `mixed_rgb_dependency`。

自动判定是内部审计门槛，不等价于统计显著性。论文若使用结果，仍须报告原始效应量、validation 边界和 LiDAR 置零参照。

## 6. 后续决策

- 若官方 B0 为 weak：停止把“更强 ResNet”作为主创新，转向融合利用率或样本效率问题；
- 若为 material：继续等预算 B0-FT/B0-V-FT，验证交通语义预训练能否转化为下游增益；
- 若为 mixed：先按 RGB view、场景类别和天气分层，不直接扩大到完整 D7。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
