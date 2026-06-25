# 06 - 卷积神经网络 (CNN)

> **难度**: ⭐⭐⭐
> **前置**: [02-autograd](), [05-dataset-dataloader]()
> **预计耗时**: 90 - 120 分钟
> **硬件**: GPU 必备(MNIST/CIFAR-10 训练)
>
> 上一章用了现成模型,这一章从零写一个,理解 backbone 怎么搭出来的。
>
> 📖 详细讲解看 [`doc/07-cnn.md`](doc/07-cnn.md)

## 核心概念

| 术语 | 一句话解释 |
|------|-----------|
| **`nn.Module`** | 所有模型的基类,要写 `__init__` 定义层,`forward` 定义数据流 |
| **Conv2d** | 2D 卷积,滑动窗口提取特征;关键参数 `in_channels / out_channels / kernel_size / stride / padding` |
| **MaxPool2d** | 池化下采样,降维 + 提升平移不变性 |
| **ReLU** | 激活函数,加非线性 |
| **BatchNorm2d** | 批归一化,稳定训练 |
| **Dropout** | 随机丢一些激活,防过拟合 |
| **Flatten / Linear** | 展平 → 全连接 → 输出 logits |
| **CrossEntropyLoss** | 多分类标配 loss(自带 softmax) |
| **optimizer** | `SGD` / `Adam` / `AdamW`,把梯度更新到参数上 |
| **train / eval 模式** | `model.train()` / `model.eval()`,影响 BN / Dropout |

## 课程大纲

| 序号 | 主题 | 重点 | 难点 |
|------|------|:----:|:----:|
| 1 | `nn.Module` 三件套:`__init__` / `forward` / `parameters()` | ✅ |  |
| 2 | 单层卷积 + ReLU + Pool 的输出 shape 推导 | ✅ |  |
| 3 | LeNet-5 for MNIST 完整实现 |  | ✅ |
| 4 | 训练循环:forward → loss → backward → step → zero_grad | ✅ |  |
| 5 | 验证循环 + 准确率 | ✅ |  |
| 6 | 模型保存 / 加载(`state_dict`) |  | ✅ |
| 7 | GPU 训练(`.to(device)`) | ✅ |  |

## 最小示例

```python
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(16*4*4, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

## 学习顺序

1. 跑通 7 个 examples
2. 做完 5 道 exercises
3. 进入 07 完整实验(CIFAR-10 微调)
