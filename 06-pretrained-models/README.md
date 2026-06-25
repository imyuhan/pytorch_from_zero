# 06 - 预训练模型使用

> **难度**: ⭐⭐
> **前置**: [05-dataset-dataloader]()
> **预计耗时**: 45 - 60 分钟(下载权重另算 5 - 10 分钟)
> **硬件**: GPU 推荐(首次下载 ~100MB 预训练权重)
>
> 不必从头训练一个图像模型 —— torchvision 自带一堆 ImageNet 预训练权重,直接拿来推理或微调。
>
> 📖 详细讲解看 [`doc/06-pretrained-models.md`](doc/06-pretrained-models.md)

## 核心概念

| 术语 | 一句话解释 |
|------|-----------|
| **pretrained** | torchvision 提供 ImageNet 上预训练好的权重,`weights="DEFAULT"` 加载 |
| **推理 (inference)** | 关闭梯度,跑前向得到预测 |
| **特征提取 (features)** | 拿掉最后的分类头,把 backbone 当特征提取器 |
| **迁移学习 (transfer learning)** | 冻住 backbone 训练新分类头 / 或微调全部参数 |
| **fine-tune** | 用较小的学习率把所有/部分参数继续训练 |
| **timm** | PyTorch Image Models,模型更多更全(可选) |

## 课程大纲

| 序号 | 主题 | 重点 | 难点 |
|------|------|:----:|:----:|
| 1 | torchvision 模型列表与 `weights="DEFAULT"` | ✅ |  |
| 2 | 推理:前向 + softmax + top-5 | ✅ |  |
| 3 | 看模型结构:`print(model)` / `summary` | ✅ |  |
| 4 | 替换分类头(`model.fc = nn.Linear(...)`) |  | ✅ |
| 5 | 冻 vs 不冻(`param.requires_grad`) |  | ✅ |
| 6 | timm 简介(可选模型库) |  |  |

## 学习顺序

1. 跑通 5 个 examples
2. 做完 4 道 exercises(替换头 + 迁移学习雏形)
3. 进入 06 手写 CNN,理解 backbone 怎么来的
