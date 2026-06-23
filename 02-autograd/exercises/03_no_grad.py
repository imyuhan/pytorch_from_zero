"""练习 03:no_grad

要求:
  1. 在 no_grad 上下文外,计算 y1 = (w**2).sum() → 打印 requires_grad
  2. 在 no_grad 上下文内,计算 y2 = (w**2).sum() → 打印 requires_grad
  3. 用 .detach() 替代 no_grad,验证效果一样
"""
import torch

# 你的代码 ↓
w = torch.tensor(3.0,requires_grad=True)

y1 = (w**2).sum()
y1.backward()
print(f"在no_grad上下文外：{w.grad},require_grad:{y1.requires_grad}")

w.grad.zero_()
with torch.no_grad():
    y2 = (w**2).sum()
    # 在grad.zero()里的op不计入计算图，没有grad_fn
    print(f"在no_grad上下文内的require_grad：{w.grad}")

w.grad.zero_()
y3 = (w**2).sum().detach()
print(f"用.detach()后的require_grad:{w.grad}")
