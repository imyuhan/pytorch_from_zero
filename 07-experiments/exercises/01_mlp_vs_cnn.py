"""练习 01:MNIST MLP vs CNN 对比

要求:
  1. 写两个模型:MLP(Flatten → 784→256→10)和 CNN(参考 05-3)
  2. 都训 2 epoch,打印 val_acc
  3. 对比哪个更好,思考为什么
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 你的代码 ↓
