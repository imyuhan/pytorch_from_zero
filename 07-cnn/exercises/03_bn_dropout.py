"""练习 03:加 BatchNorm 和 Dropout

要求:
  - 在 05-3 的 LeNet 上加 BatchNorm2d 和 Dropout(0.3)
  - 每个 Conv 后加 BN
  - 第一个 ReLU 后加 Dropout
  - 打印 summary 验证
"""
import torch
import torch.nn as nn
from torchinfo import summary

# 你的代码 ↓
class LeNetBNDrop(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO
        pass

    def forward(self, x):
        # TODO
        pass
