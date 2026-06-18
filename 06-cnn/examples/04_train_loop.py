"""05-4 完整训练循环(1 epoch MNIST)

把 01-05 全部串起来:DataLoader → 模型 → loss → optimizer → 训练
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 路径配置
data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
os.makedirs(data_root, exist_ok=True)

# 数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_set = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)

# 模型 + 损失 + 优化器
device = "cuda" if torch.cuda.is_available() else "cpu"
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 10),
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练 1 个 epoch
model.train()
total_loss = 0
n_batches = 0
for xb, yb in train_loader:
    xb, yb = xb.to(device), yb.to(device)

    logits = model(xb)
    loss = criterion(logits, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    n_batches += 1
    if n_batches % 200 == 0:
        print(f"  batch {n_batches}, loss = {loss.item():.4f}")

print(f"\n平均 loss: {total_loss / n_batches:.4f}")
