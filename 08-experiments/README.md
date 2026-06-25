# 07 - 综合实验

> **难度**: ⭐⭐⭐(整机项目)
> **前置**: [01] ~ [06] 全部
> **预计耗时**: 半天 - 2 天(每个实验 30 分钟 - 数小时)
> **硬件**: GPU 必备;显存 ≥ 6GB 推荐(ResNet18 微调 CIFAR-10 跑得动)
>
> 前面每章都是零件,这一章是整机。把所有概念串成一个能"开箱即跑"的工程模板。
>
> 📖 详细讲解看 [`doc/08-experiments.md`](doc/08-experiments.md)

## 目标

- 完整的**项目骨架**:`models/` + `data/` + `train.py` + `evaluate.py` + 配置文件
- 完整的**训练流程**:数据增强 + LR scheduler + 早停 + 最佳权重保存
- 完整的**结果记录**:训练曲线图 + 混淆矩阵 + 分类报告
- 一个能**复现**的实验:ResNet18 微调 CIFAR-10

## 实验列表

| 编号 | 名称 | 难度 | 时间 |
|------|------|:----:|:----:|
| E1 | MLP / CNN 在 MNIST 上的对比 | ✅ | 短 |
| E2 | 数据增强对 CIFAR-10 准确率的影响 |  | 中 |
| E3 | ResNet18 微调 CIFAR-10(迁移学习) |  | 中 |
| E4 | 训练曲线可视化 + 混淆矩阵 |  | 中 |

## 推荐工作流

1. 看 `examples/01_project_skeleton.md` 了解工程模板
2. 跑 `examples/02_train_mnist.py`(几分钟出结果,做 baseline)
3. 跑 `examples/03_train_cifar.py`(几十分钟)
4. 试着改 `examples/04_resnet_finetune.py` 里的超参,观察效果
