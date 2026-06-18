"""练习 10:迷你训练循环(综合)

要求:
  1. 数据: y = 3x + 2 + 噪声,生成 100 个点
  2. 参数: w, b, requires_grad=True,初始为 0
  3. 训练 200 步,每步:
     - forward: y_pred = w*x + b
     - loss = MSE
     - backward + 手动 SGD 更新(lr=0.01)
     - zero_grad
  4. 每 20 步打印一次 w, b, loss
  5. 训练完打印最终学到的 w(应该接近 3)、b(应该接近 2)
"""
import torch

torch.manual_seed(0)
x = torch.linspace(0, 10, 100)
y_true = 3 * x + 2 + torch.randn(100) * 0.5

# 你的代码 ↓
