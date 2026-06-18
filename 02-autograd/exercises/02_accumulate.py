"""练习 07:梯度累积

要求:
  1. 创建一个 w, requires_grad=True
  2. 不调用 zero_grad,连续 backward 两次 y1 = w**2, y2 = w**3
  3. 观察 w.grad 是累加的
  4. 然后 zero_grad() 再 backward,对比 w.grad 是不是"干净"的
"""
import torch

# 你的代码 ↓
