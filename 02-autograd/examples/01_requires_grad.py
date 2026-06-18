"""02-1 requires_grad 与 grad_fn 追踪

对应 autograd 章节第 1 条(重点)。
"""
import torch

# 叶子节点:直接创建的 tensor
x = torch.tensor(3.0, requires_grad=True)
print(f"x:        value={x.item()}, grad={x.grad}, grad_fn={x.grad_fn}")
# grad_fn=None 因为是叶子

# 非叶子节点:由 op 产生
y = x ** 2
print(f"y:        value={y.item()}, grad_fn={y.grad_fn}")
# grad_fn=<PowBackward0> 表示是幂运算产生的

z = y * 2 + 1
print(f"z:        value={z.item()}, grad_fn={z.grad_fn}")

# 追溯整条链路
print(f"\n张量链路: x → y(PowBackward0) → z(AddBackward0)")

# 反向传播
z.backward()
print(f"\n反向传播后:")
print(f"  x.grad = {x.grad}  # 应该是 2*3*2 = 12")
# 数学验证: z = 2*x^2 + 1, dz/dx = 4x = 12 ✓
