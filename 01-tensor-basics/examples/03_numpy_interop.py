"""01-3 NumPy ↔ Tensor

对应目标 7, 8。
"""
import numpy as np
import torch

# --- numpy → tensor ---
np_arr = np.array([1.0, 2.0, 3.0])

# 方式 1: torch.from_numpy 共享内存
t_shared = torch.from_numpy(np_arr)
print("from_numpy 共享内存:", t_shared)

# 方式 2: torch.tensor 复制
t_copy = torch.tensor(np_arr)
print("torch.tensor 复制:  ", t_copy)

# 验证共享
np_arr[0] = 999
print(f"\n改了 np_arr[0]=999 后:")
print(f"  t_shared[0] = {t_shared[0]}  (会变,共享)")
print(f"  t_copy[0]   = {t_copy[0]}    (不变,独立)")

# --- tensor → numpy(必须先到 CPU!) ---
gpu_t = torch.tensor([1.0, 2.0, 3.0], device="cuda")
# np.array(gpu_t)            # 直接报错!
np_back = gpu_t.cpu().numpy()  # 正确:先 .cpu() 再 .numpy()
print(f"\nGPU tensor → NumPy: {np_back}")
