"""练习 01:手写 Dataset

把 03-1 抄一遍,然后改成自己的版本:
- 数据: torch.randn(500, 4) 作为 x,torch.randint(0, 3, (500,)) 作为 y(3 分类)
- 实现 __len__ 和 __getitem__
- 用 DataLoader batch_size=16 跑一个 epoch,打印每个 batch 的 shape
"""
import torch
from torch.utils.data import Dataset, DataLoader

# 你的代码 ↓
