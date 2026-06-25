"""练习 02:加 LR Scheduler

要求:
  1. 复制 06-2 的训练循环
  2. 加 StepLR(每 1 epoch lr * 0.5)或 CosineAnnealingLR
  3. 每个 epoch 打印当前 lr
  4. 观察 loss 收敛是否更平滑
"""
import os
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 你的代码 ↓
