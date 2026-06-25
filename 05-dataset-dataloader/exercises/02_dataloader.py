"""练习 02:DataLoader 三件套

要求:
  1. 创建一个 TensorDataset,数据 shape=(200, 5),标签 0/1 二分类
  2. 分别用 batch_size=16 / 64 / 200 跑一个 epoch
  3. 观察 batch 总数,以及 drop_last=True / False 的区别
  4. 打开 shuffle=True / False,观察同一个 epoch 里数据顺序是否变化
"""
import torch
from torch.utils.data import DataLoader, TensorDataset

# 你的代码 ↓
