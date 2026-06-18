"""练习 04:画自己的训练曲线

要求:
  1. 在 06-2 的基础上,把每 epoch 的 train_loss / val_loss / val_acc 累加到 history dict
  2. 训练完调用 06-5 的绘图代码,保存到 outputs/training_curves.png
  3. 打开图,看你训练的曲线长啥样
"""
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs")
os.makedirs(out_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 你的代码 ↓
