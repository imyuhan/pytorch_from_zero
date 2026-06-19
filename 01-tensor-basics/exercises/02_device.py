"""练习 02:设备迁移

要求:
  1. 创建两个 1000x1000 的 randn tensor
  2. 把它们搬到 GPU,记录耗时
  3. 在 GPU 上做矩阵乘法,记录耗时
  4. 对比纯 CPU 的耗时
  5. 打印:加速比 = CPU时间 / GPU时间
"""
import torch
import time

# 你的代码 ↓
a = torch.randn(1000,1000)
b = torch.randn(1000,1000)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("当前使用设备：",device)

# 记录迁移耗时
start = time.perf_counter()
a_gpu = a.to(device)
b_gpu = b.to(device)
end = time.perf_counter()
print(f"转移到gpu耗时：{end - start:.4f}s")

# 记录gpu矩阵乘法耗时
torch.cuda.synchronize()
start = time.perf_counter()
c_gpu = a_gpu @ b_gpu
torch.cuda.synchronize()
t_gpu = time.perf_counter() - start
print(f"gpu矩阵乘法耗时：{t_gpu:.4f}s")

# 记录cpu矩阵乘法耗时
start = time.perf_counter()
c_cpu = a @ b
t_cpu = time.perf_counter() - start
print(f"cpu矩阵乘法耗时：{t_cpu:.4f}s")

# 加速比
print(f"加速比(cpu/gpu):{t_cpu / t_gpu:.2f}x")