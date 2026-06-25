"""03-1 Dataset 抽象类与三件套

理解 Dataset 必须实现 __len__ 和 __getitem__。
"""
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """最简单的 Dataset:从两个 tensor 取样本"""

    def __init__(self, x, y):
        # 一次性把所有数据装进内存
        assert len(x) == len(y), "x 和 y 长度必须一致"
        self.x = x
        self.y = y

    def __len__(self):
        # 告诉 DataLoader 总共有多少样本
        return len(self.x)

    def __getitem__(self, i):
        # 给定索引,返回一个 (样本, 标签) 元组
        return self.x[i], self.y[i]


# 模拟 100 条 3 维数据,2 分类
torch.manual_seed(0)
x = torch.randn(100, 3)
y = torch.randint(0, 2, (100,))

ds = MyDataset(x, y)
print(f"数据集大小: {len(ds)}")            # 100
print(f"第 0 个样本 x: {ds[0][0].shape}")  # torch.Size([3])
print(f"第 0 个样本 y: {ds[0][1].item()}") # 0 或 1
