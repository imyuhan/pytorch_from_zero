"""练习 02:实现一个 3 层 CNN

要求:
  - 3 个卷积块(每块:Conv → ReLU → MaxPool),通道数 [3 → 32 → 64 → 128]
  - 2 个全连接层(128*4*4 → 256 → 10)
  - 输入 [B, 3, 32, 32](CIFAR)
  - 用 torchinfo.summary 打印结构
"""
import torch
import torch.nn as nn
from torchinfo import summary

# 你的代码 ↓
class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO
        pass

    def forward(self, x):
        # TODO
        pass
