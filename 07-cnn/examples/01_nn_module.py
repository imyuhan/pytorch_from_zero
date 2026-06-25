"""05-1 nn.Module 三件套

所有 PyTorch 模型都继承 nn.Module,要实现 __init__ 和 forward。
"""
import torch
import torch.nn as nn


class TwoLayerMLP(nn.Module):
    """最简单的两层感知机"""

    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# 实例化
model = TwoLayerMLP(in_dim=784, hidden=128, out_dim=10)
print("模型结构:")
print(model)

# 自动收集所有可学习参数
total = sum(p.numel() for p in model.parameters())
print(f"\n总参数量: {total:,}")

# 验证:一个 (4, 784) 的输入能跑
x = torch.randn(4, 784)
y = model(x)
print(f"输入: {x.shape} → 输出: {y.shape}")
