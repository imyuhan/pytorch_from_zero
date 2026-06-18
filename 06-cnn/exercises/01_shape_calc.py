"""练习 01:卷积输出 shape 计算

不写代码,先手算每个 shape(写在注释里),再写代码验证。
输入: [1, 3, 32, 32]

  Conv2d(3, 16, 3, padding=1) →
  ReLU →
  MaxPool2d(2, 2) →
  Conv2d(16, 32, 3, padding=1) →
  ReLU →
  MaxPool2d(2, 2) →
  Flatten →
  Linear(? , 10)

补全问号,再用代码验证。
"""
import torch
import torch.nn as nn

# 你的代码 ↓
