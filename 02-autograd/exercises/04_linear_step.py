"""练习 04:线性回归一步

要求:
  1. 准备数据 x = [[1], [2], [3]], y_true = [[3], [5], [7]](真实斜率=2, 截距=1),用torch.tensor创建
  2. 初始化 w, b 为 requires_grad=True 的随机数
  3. forward: y_pred = w * x + b
  4. loss = MSE = ((y_pred - y_true)**2).mean()
  5. loss.backward()
  6. 打印 w.grad, b.grad
  7. 手动更新 w, b(lr=0.01),zero_grad()
  8. 多跑 5 个 step,看 w 是不是慢慢逼近 2、b 是不是慢慢逼近 1
"""
import torch

# 你的代码 ↓
# 初始化数据
x = torch.tensor([[1.0], [2.0], [3.0]])
y_true = torch.tensor([[3.0], [5.0], [7.0]])
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)
lr = 0.01

for step in range(5):
    # forward
    y_pred = w * x + b

    loss = ((y_pred - y_true)**2).mean()
    loss.backward()

    print(f"w: {w.item():.4f},    w.grad: {w.grad.item():.4f},   b: {b.item():.4f},    b.grad: {b.grad.item():.4f}")

    # 手动更新 w,b
    # 更新不需要进计算图
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    # 清 0 梯度，为下次计算做准备
    w.grad.zero_()
    b.grad.zero_()

print(f"\n最终 w = {w.item():.4f}(理论值 2.0)")
print(f"\n最终 b = {b.item():.4f}(理论值 1.0)")

