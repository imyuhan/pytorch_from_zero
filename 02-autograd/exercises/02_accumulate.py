"""练习 02:梯度累积

要求:
  1. 创建一个 w, requires_grad=True
  2. 不调用 grad.zero_(),连续 backward 两次 y1 = w**2, y2 = w**3
  3. 观察 w.grad 是累加的
  4. 然后 grad.zero_() 再 backward,对比 w.grad 是不是"干净"的
"""
import torch

# 你的代码 ↓
w = torch.tensor(1.0,requires_grad=True)

y1 = w**2
y1.backward()
print(f"第一次w.grad = {w.grad}")

y2 = w**3
y2.backward()
print(f"第二次w.grad = {w.grad}")

w.grad.zero_()
y3 = w**3
y3.backward()
print(f"清零后w.grad = {w.grad}")
