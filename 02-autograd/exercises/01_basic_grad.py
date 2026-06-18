"""练习 06:requires_grad 与 backward

要求:
  1. 创建 w = torch.tensor(3.0, requires_grad=True)
  2. 计算 y = w**3 + 2*w + 1
  3. y.backward()
  4. 手动算 dy/dw 在 w=3 的值,跟 w.grad 对照(应该是 3*9 + 2 = 29)
"""
import torch

# 你的代码 ↓
