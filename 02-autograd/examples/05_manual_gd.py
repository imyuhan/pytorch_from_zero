"""02-5 手写梯度下降(带 LR 控制)

对应 autograd 章节第 5 条(综合难点),也对应学习目标 9 的一部分。
"""
import torch

# 目标:用梯度下降最小化 f(w) = (w - 5)^2
# 理论最优: w* = 5
# 梯度: df/dw = 2(w - 5)

w = torch.tensor(0.0, requires_grad=True)   # 初始值
lr = 0.1                                    # 学习率

print(f"{'step':>4} {'w':>8} {'loss':>10} {'w.grad':>10}")
for step in range(8):
    # forward
    loss = (w - 5) ** 2

    # backward
    loss.backward()

    # 打印
    print(f"{step:>4} {w.item():>8.4f} {loss.item():>10.4f} {w.grad.item():>10.4f}")

    # 手动更新(优化器内部就是干这个)
    with torch.no_grad():           # 更新时不需要梯度
        w -= lr * w.grad

    # 清零梯度
    w.grad.zero_()

print(f"\n最终 w = {w.item():.4f}(理论值 5)")
