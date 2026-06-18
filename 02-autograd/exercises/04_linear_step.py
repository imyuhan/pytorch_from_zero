"""练习 09:线性回归一步

要求:
  1. 准备数据 x = [[1], [2], [3]], y_true = [[3], [5], [7]](真实斜率=2, 截距=1)
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
