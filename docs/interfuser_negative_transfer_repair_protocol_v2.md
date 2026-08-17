<!--
[INPUT]: 依赖全模型权重插值 v1 的失败 manifest、Stage 2 B0/V best checkpoint、RGB alias 边界与同一 validation retention 门禁。
[OUTPUT]: 对外提供只插值 RGB 浮点张量、完整保留 B0 Transformer/LiDAR/任务头的负迁移定位 probe 及后续训练准入规则。
[POS]: docs 的 M2 H1 负迁移修复第二阶段协议；区分 RGB 表征不兼容与共同适配后的下游头漂移，不扩大 alpha 或读取 test。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
-->

# InterFuser 视觉负迁移修复协议 v2：RGB 隔离插值

| 字段 | 内容 |
| --- | --- |
| 状态 | **PROBE-FROZEN：v1失败后冻结，运行前不读取v2结果** |
| 上游失败证据 | `m2-interfuser-weight-interpolation-probe-v1-20260817-v1` |
| 上游 manifest SHA-256 | `ca5448f1998559f6df755f46fa04b6399b4021113b7258945e909ec46c949f34` |
| 配置 | `configs/thesis/interfuser_weight_interpolation_rgb_probe_v2.json` |

## 1. v1 结论

全模型浮点权重插值的三个alpha均失败。即使最接近B0的alpha=0.25，非行人最坏退化仍为159.29%，整体traffic最坏退化75.04%；alpha=0.75保留行人4/5收益，但非行人最坏退化31.79%、行人时序最坏退化24.33%。

因此两个端到端适配后的完整模型不处于可直接线性连接的低损失盆地。继续增加alpha密度没有依据；v1不准入完整重训。

## 2. v2 假设与唯一变化

v1 同时插值 RGB、Transformer、LiDAR 与任务头，无法区分“RGB表征不兼容”和“下游头在两次训练中共同漂移”。v2 完整复制最终B0状态，只对 `rgb_backbone.*` 与 `rgb_patch_embed.backbone.*` 的浮点张量执行相同 `α={0.25,0.5,0.75}` 插值；全部非RGB浮点张量与所有离散缓冲保留B0。

validation、指标、四项门禁、随机种子与test冻结边界均与v1相同。该probe只定位根因，不把事后RGB隔离方案直接写成论文方法。

## 3. 判定

- 若存在合格alpha：说明B0下游头可读取部分语义方向，准入“B0骨干源点 + 语义损失 + L2-SP/教师特征保持”的训练probe；
- 若所有alpha仍失败：说明简单参数线性混合不能对齐两套RGB表征，应停止插值，转向以官方B0为初始化的受约束语义适配；
- 任一候选必须保持非行人/整体/时序门禁，不能只凭行人聚合收益通过。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
