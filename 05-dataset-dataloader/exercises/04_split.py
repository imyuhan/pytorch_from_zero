"""练习 04:train / val split

要求:
  1. 用 MNIST(从 03-6 抄)
  2. 不直接用 torchvision 自带的 train=True/False 切
  3. 改用 random_split 把 train_set 切成 90% 训练 / 10% 验证
  4. 验证:len(train_ds) + len(val_ds) == 60000
  5. 用 DataLoader 验证 90% 数据可迭代
"""
import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
full = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)

# 你的代码 ↓
