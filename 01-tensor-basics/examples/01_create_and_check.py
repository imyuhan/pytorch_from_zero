"""01-1 张量创建与基本检查

对应"学完后应能做到"第 1, 2, 3 条。
"""
import torch

# --- 1. 四种创建方式 ---
a = torch.tensor([[1, 2], [3, 4]])     # 从 Python list
b = torch.ones(2, 3)                    # 全 1
c = torch.zeros(2, 3)                   # 全 0
d = torch.randn(2, 3)                   # N(0, 1) 随机，正态分布

print("a:", a)
print("b:", b)
print("c:", c)
print("d:", d)

# --- 2. 检查属性 ---
print("\n--- 属性检查 ---")
print("shape:        ", a.shape)         # torch.Size([2, 2])
print("dtype:        ", a.dtype)         # torch.int64
print("device:       ", a.device)        # cpu
print("requires_grad:", a.requires_grad) # False
print("numel:        ", a.numel())       # 4
print("element_size: ", a.element_size())  # 8 字节(int64)
print(f"估算内存: {a.numel() * a.element_size()} 字节")
