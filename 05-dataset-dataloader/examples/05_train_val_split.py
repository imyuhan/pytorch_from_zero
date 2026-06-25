"""03-5 random_split 切训练 / 验证集

不需要真的把文件复制两份,PyTorch 用索引切。
"""
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

# 1000 条数据
x = torch.randn(1000, 3)
y = torch.randint(0, 2, (1000,))
full = TensorDataset(x, y)

# 8:2 切
n_total = len(full)
n_train = int(0.8 * n_total)
n_val = n_total - n_train
train_ds, val_ds = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(0))

print(f"总数: {n_total}, 训练: {len(train_ds)}, 验证: {len(val_ds)}")

# 验证集通常不 shuffle
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# 实际训练时:
# for epoch in range(num_epochs):
#     model.train()
#     for xb, yb in train_loader:
#         ...
#     model.eval()
#     with torch.no_grad():
#         for xb, yb in val_loader:
#             ...
