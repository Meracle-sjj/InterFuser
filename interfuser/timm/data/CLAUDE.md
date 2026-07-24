# interfuser/timm/data/
> L2 | 父级: ../CLAUDE.md

## 成员清单

CLAUDE.md: timm.data 局部地图，约束 CARLA 扩展与通用图像数据工具的依赖方向。
__init__.py: 统一导出数据集、loader、增强与常量 API。
base_io_dataset.py: 使用受控文件生命周期提供文本、JSON、NumPy 和图像基础读取与历史帧回退。
carla_dataset.py: 从显式 dataset index 解析 CARLA 多视角 RGB/LiDAR/测量标签为 InterFuser 多任务样本。
carla_loader.py: 为 CarlaMVDetDataset 绑定多视角变换与分布式 DataLoader。
dataset_factory.py: 按名称创建通用或 CARLA dataset。
dataset.py: 通用 ImageFolder/Tar 数据集封装。
loader.py: 通用图像 DataLoader 工厂。
config.py: 从模型默认值解析输入尺寸、归一化与插值契约。
constants.py: ImageNet 均值、方差与裁剪常量。
transforms_carla_factory.py: 构造 CARLA RGB/语义特定的几何与颜色变换。
transforms_factory.py: 通用训练/验证图像变换工厂。
transforms.py: 通用裁剪、缩放与归一化组件。
augmenter.py: CARLA 样本级颜色与噪声增强。
auto_augment.py: Rand/AutoAugment 策略。
mixup.py: Mixup/CutMix 批次增强。
random_erasing.py: Random Erasing 增强。
tf_preprocessing.py: TensorFlow 风格预处理兼容层。
distributed_sampler.py: 分布式训练/验证采样器。
det_utils.py: InterFuser 交通参与者网格目标生成。
heatmap_utils.py: waypoint 热力图与未来轨迹目标生成。
real_labels.py: ImageNet real-label 评价支持。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
