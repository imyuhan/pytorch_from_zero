"""01-6 索引 / 切片 / 内存共享 vs 复制

对应目标 12, 14。
"""
import torch

x = torch.arange(12).reshape(3, 4)
print("x:")
print(x)
print()

# --- 索引 ---
print("x[0] (第 0 行):", x[0].shape, "→", x[0])
print("x[:, 0] (第 0 列):", x[:, 0].shape, "→", x[:, 0])
print("x[1, 2] (标量):", x[1, 2].item(), type(x[1, 2].item()))
print("x[0:2, 1:3] (子块):", x[0:2, 1:3].shape)
print()

# --- 内存关系 ---
print("=== 内存关系 ===")
a = torch.arange(6)
b = a.view(2, 3)
c = a.reshape(2, 3)
d = a.clone()

b[0, 0] = 999
print(f"改 b 后 a[0] = {a[0]}  (view 共享内存)")

d[0] = 888
print(f"改 d 后 a[0] = {a[0]}    (clone 独立)")

# contiguous 复制示例
e = torch.arange(6).reshape(2, 3).t()   # 转置后不连续
print(f"\n转置后连续? {e.is_contiguous()}")
e_c = e.contiguous()
print(f"contiguous() 后连续? {e_c.is_contiguous()}")
