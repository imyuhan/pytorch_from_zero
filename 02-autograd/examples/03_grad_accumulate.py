"""02-3 梯度累积与 zero_grad

对应 autograd 章节第 3 条(重点)。
"""
import torch

w = torch.tensor(2.0, requires_grad=True)

# 第一次: y = w^2
y1 = w ** 2
y1.backward()
print(f"第 1 次 backward 后: w.grad = {w.grad}")

# 第二次:不 zero_grad,梯度会累加!
y2 = w ** 3
y2.backward()
print(f"第 2 次 backward 后: w.grad = {w.grad}  # 累加了 2*2 + 3*4 = 4+12 = 16")

# 正确的训练循环写法
w.grad.zero_()  # 显式清零
print(f"\nzero_grad 后: w.grad = {w.grad}")
y3 = w ** 2
y3.backward()
print(f"清零后再 backward: w.grad = {w.grad}  # 干净的值 4")
