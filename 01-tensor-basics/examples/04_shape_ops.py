"""01-4 形状操作:view / reshape / transpose / squeeze / unsqueeze

对应张量章节教学大纲前 3 条。
"""
import torch

# --- 1. squeeze / unsqueeze ---
print("=== squeeze / unsqueeze ===")
x = torch.randn(3, 4)              # (3, 4)
x_unsq = x.unsqueeze(0)            # 在第 0 维插入 1→ (1, 3, 4)
print(f"unsqueeze(0): {x_unsq.shape}")

x_sq = x_unsq.squeeze(0)           # 去掉第 0 维的 1 → (3, 4)
print(f"squeeze(0):   {x_sq.shape}")

# --- 2. view / reshape ---
print("\n=== view / reshape ===")
a = torch.arange(12)               # (12,)
b = a.view(3, 4)                   # 要求原 tensor 内存连续
c = a.reshape(3, 4)                # 不能满足连续时自动复制
print(f"view:    {b.shape}")
print(f"reshape: {c.shape}")

# --- 3. transpose 后 reshape 会复制 ---
print("\n=== transpose + contiguous ===")
mat = torch.arange(6).reshape(2, 3)  # (2, 3)
trans = mat.t()                      # (3, 2),不连续
print(f"mat 连续?    {mat.is_contiguous()}")
print(f"trans 连续? {trans.is_contiguous()}")
# trans.view(...) 会报错!必须先 contiguous()
trans_c = trans.contiguous()
print(f"contiguous() 后: {trans_c.is_contiguous()}, shape: {trans_c.shape}")

# --- 4. cat vs stack ---
print("\n=== cat vs stack ===")
t1 = torch.tensor([[1, 2], [3, 4]])
t2 = torch.tensor([[5, 6], [7, 8]])

cat_res = torch.cat([t1, t2], dim=0)  # 沿已有维度拼接
print(f"cat dim=0 shape: {cat_res.shape}  # (4, 2)")
print(cat_res)
stack_res = torch.stack([t1, t2], dim=0)  # 新增一个维度
print(f"stack dim=0 shape: {stack_res.shape}  # (2, 2, 2)")
print(stack_res)