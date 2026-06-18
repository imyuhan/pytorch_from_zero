"""01-2 设备迁移:CPU ↔ GPU

对应目标 5, 6。
"""
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# CPU → GPU
x = torch.randn(3, 3)
print("x 在哪:       ", x.device)

x_gpu = x.to(device)   # 返回新 tensor
print("x_gpu 在哪:   ", x_gpu.device)
print("原 x 不变:    ", x.device)  # 仍然是 cpu

# 验证 .to() 返回新对象
print("是同一对象?", x is x_gpu)  # False

# GPU → CPU
x_back = x_gpu.to("cpu")
print("搬回后:       ", x_back.device)

# 显存查询
if torch.cuda.is_available():
    print(f"\n当前显存占用: {torch.cuda.memory_allocated() :.2f} B") #如果算MB就除1024**2
    print(f"峰值显存占用: {torch.cuda.max_memory_allocated() :.2f} B")
