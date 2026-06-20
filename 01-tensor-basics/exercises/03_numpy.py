"""练习 03:NumPy 互操作

要求:
  1. 创建一个 numpy 数组 a = [1, 2, 3, 4, 5]
  2. 用 from_numpy 转换,改 a[0] 验证共享
  3. 用 torch.tensor 转换,改 a[0] 验证不共享
  4. 把 GPU tensor 正确转成 numpy
"""
import numpy as np
import torch

# 你的代码 ↓

# 验证共享
a = np.array([1, 2, 3, 4, 5])
t_share = torch.from_numpy(a)
a[0] = 0
print(f"a = {a},t_share = {t_share}(共享)")

# 验证不共享
t_copy = torch.tensor(a)
t_copy[0] = 999
print(f"a = {a},t_copy = {t_copy}(不共享)")

# 把 gpu tensor 转 numpy
gpu_t = torch.tensor([1,2,3,4,5],device="cuda")
numpy_t = gpu_t.cpu().numpy()
print("\ngpu 转 numpy:", numpy_t)
