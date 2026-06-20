"""练习 04:形状操作

要求:
  1. 创建一个 shape=(12,) 的 arange tensor
  2. 用 view 变成 (3, 4)
  3. 用 unsqueeze(0) 变成 (1, 3, 4)
  4. transpose 第 0 和第 1 维 → (3, 1, 4),再 squeeze(1) 回到 (3, 4)
  5. cat 两个 (2, 3) 沿 dim=0;stack 两个 (2, 3) 沿 dim=0,打印两者 shape
"""
import torch

# 你的代码 ↓

# view 变换
a = torch.arange(12)
print(a.shape)
b = a.view(3,4)
print("view：",b.shape)

# unsqueeze 增加维度
c = b.unsqueeze(0)
print("\nunsqueeze:",c.shape)

# transpose 交换维度，squeeze 删减维度
d = c.transpose(0, 1)
print("\ntranspose:",d.shape)
e = d.squeeze(0)
print("squeeze:",e.shape)

# cat 与 stack
t1 = torch.tensor([[1,2,3],[4,5,6]])
t2 = torch.tensor([[7,8,9],[10,11,12]])

cat_res = torch.cat((t1,t2),0)
print("\ncat:",cat_res.shape)

stack_res = torch.stack((t1,t2),0)
print("stack:",stack_res.shape)
