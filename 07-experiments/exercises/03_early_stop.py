"""练习 03:早停 (Early Stopping)

要求:
  1. 写一个 EarlyStopping 类,记录 val_loss
  2. 如果连续 3 个 epoch val_loss 不降,触发停止
  3. 把早停接到 06-2 的循环里
  4. 验证:在 MNIST 上训练时触发(选 lr 大一点让它快速过拟合)
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 你的代码 ↓
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0):
        # TODO
        pass

    def __call__(self, val_loss):
        # TODO: 返回 True 表示应该停止
        pass
