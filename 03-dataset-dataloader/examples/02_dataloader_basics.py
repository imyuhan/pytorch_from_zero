"""03-2 TensorDataset 与 DataLoader

TensorDataset 是最省事的写法,只要数据全是 tensor 就用它。
"""
import torch
from torch.utils.data import DataLoader, TensorDataset

x = torch.randn(100, 3)
y = torch.randint(0, 2, (100,))

# 一行建数据集
ds = TensorDataset(x, y)

# DataLoader 关键参数
dl = DataLoader(
    ds,
    batch_size=8,         # 每个 batch 多少条
    shuffle=True,         # 每个 epoch 是否打乱
    num_workers=0,        # 数据加载用几个子进程(Windows 一般 0 或 2)
    drop_last=False,      # 最后一个 batch 不够 batch_size 时是否丢弃
)

# 迭代
for epoch in range(1):
    for i, (xb, yb) in enumerate(dl):
        if i == 0:
            print(f"epoch={epoch}, batch={i}, xb.shape={xb.shape}, yb.shape={yb.shape}")
        # 在这里写训练逻辑

# 算一下有多少个 batch
n_batches = len(dl)
print(f"\nbatch 总数: {n_batches} = ceil({len(ds)} / 8)")
