"""练习 03:NumPy 互操作

要求:
  1. 创建一个 numpy 数组 a = [1, 2, 3, 4, 5]
  2. 用 from_numpy 转换,改 a[0] 验证共享
  3. 用 torch.tensor 转换,改 a[0] 验证不共享
  4. 把 GPU tensor 正确转成 numpy
"""
import numpy as np
import torch

# 你的代码 ↓
