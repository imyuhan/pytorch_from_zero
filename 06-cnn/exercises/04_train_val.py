"""练习 04:写一个完整 train + val 循环

要求:
  1. 加载 MNIST(用 05-4 / 05-5 的方法)
  2. 写一个简化 MLP(Flatten → Linear(784, 128) → ReLU → Linear(128, 10))
  3. 训练 2 个 epoch,每 epoch 跑完后在测试集上评估
  4. 打印每个 epoch 的 train_loss / val_loss / val_acc
  5. 观察 val_acc 是不是逐 epoch 上升
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 数据(从 05-4 / 05-5 抄)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_set = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
test_set = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)

# 你的代码 ↓
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 10),
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# TODO: 训练 + 验证循环
