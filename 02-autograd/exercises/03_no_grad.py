"""练习 08:no_grad

要求:
  1. 在 no_grad 上下文外,计算 y1 = (w**2).sum() → 打印 requires_grad
  2. 在 no_grad 上下文内,计算 y2 = (w**2).sum() → 打印 requires_grad
  3. 用 .detach() 替代 no_grad,验证效果一样
"""
import torch

# 你的代码 ↓
