"""练习 05:CIFAR-10 + transforms 综合

要求:
  1. 加载 CIFAR-10 train(自动下载到 data/)
  2. 写一个 train_transform:Resize → RandomCrop(32, padding=4) → RandomHorizontalFlip → ToTensor → Normalize
  3. 写一个 val_transform:Resize → ToTensor → Normalize
  4. 用 random_split 切 90% train / 10% val
  5. 用 DataLoader 拿一个 batch 打印 shape(B, 3, 32, 32)
  6. 用 matplotlib 把这个 batch 前 8 张画出来(要反归一化才能看)
"""
import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")

# 你的代码 ↓
