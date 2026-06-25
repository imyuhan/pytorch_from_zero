"""练习 05:模型保存 + 加载 + 推理

要求:
  1. 在 05-7 基础上(或者重新写),训练 1 epoch CIFAR-10 SimpleCNN
  2. 把 state_dict 保存到 checkpoints/ex05_cnn.pt
  3. 重新 new 一个 SimpleCNN 实例(不训练)
  4. load_state_dict 加载权重
  5. 在测试集前 256 条上跑一次,打印准确率
  6. 验证准确率跟训练后那个模型一致(允许 ±0.5%)
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_path = os.path.join(ckpt_dir, "ex05_cnn.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 你的代码 ↓
