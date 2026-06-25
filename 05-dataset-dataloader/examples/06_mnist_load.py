"""03-6 MNIST 实战:加载 + 可视化 + DataLoader

第一次接触真实数据集,跑通就能上手 04/05 了。

注意:MNIST 会自动下载到 data/ 目录,约 50 MB。
"""
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 数据存到项目下 data/(避免污染根目录)
data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
os.makedirs(data_root, exist_ok=True)

# 经典三件套
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 的全局均值/标准差
])

train_set = datasets.MNIST(
    root=data_root, train=True, download=True, transform=transform
)
test_set = datasets.MNIST(
    root=data_root, train=False, download=True, transform=transform
)
print(f"训练集: {len(train_set)} 条")
print(f"测试集: {len(test_set)} 条")

# 拿一个样本
img, label = train_set[0]
print(f"单样本: img shape = {img.shape}, label = {label}")
# img shape = torch.Size([1, 28, 28])  # 单通道 28x28

# DataLoader
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)

xb, yb = next(iter(train_loader))
print(f"\nbatch x: {xb.shape}  # [B, C, H, W]")
print(f"batch y: {yb.shape}  # [B]")
print(f"label 分布: {torch.bincount(yb)}")

# --- 可视化(需要 matplotlib)---
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 8, figsize=(12, 3))
for i, ax in enumerate(axes.flat):
    img, label = train_set[i]
    ax.imshow(img.squeeze(), cmap="gray")
    ax.set_title(f"{label}")
    ax.axis("off")
plt.suptitle("MNIST 前 16 个样本")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist_preview.png")
plt.savefig(out, dpi=100, bbox_inches="tight")
print(f"\n已保存预览图: {out}")
