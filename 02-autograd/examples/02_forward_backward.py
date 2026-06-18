"""02-2 Forward → Loss → Backward 链路底层

对应 autograd 章节第 2 条(难点)。
"""
import torch

# 模拟一个最简单的"线性回归一步"
torch.manual_seed(0)

# 参数(模型)
w = torch.randn(1, requires_grad=True)   # 权重
b = torch.zeros(1, requires_grad=True)  # 偏置

# 数据
x = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([2.0, 4.0, 6.0])   # 真实 y = 2x

# --- forward ---
y_pred = w * x + b
loss = ((y_pred - y_true) ** 2).mean()
print(f"forward 后:")
print(f"  w.grad_fn = {w.grad_fn}")     # None(叶子)
print(f"  y_pred.grad_fn = {y_pred.grad_fn}")  # AddBackward0
print(f"  loss.grad_fn = {loss.grad_fn}")      # MeanBackward0
print(f"  loss = {loss.item():.4f}")

# --- backward ---
loss.backward()
print(f"\nbackward 后:")
print(f"  w.grad = {w.grad}  # 应该是 ∂loss/∂w")
print(f"  b.grad = {b.grad}  # 应该是 ∂loss/∂b")
