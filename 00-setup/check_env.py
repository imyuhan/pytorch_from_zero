"""环境验证:确认 PyTorch + CUDA 可用"""
import torch

print("=" * 50)
print("PyTorch 环境自检")
print("=" * 50)
print(f"torch 版本:        {torch.__version__}")
print(f"CUDA 是否可用:     {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本(运行时): {torch.version.cuda}")
    print(f"GPU 数量:          {torch.cuda.device_count()}")
    print(f"当前 GPU 名称:     {torch.cuda.get_device_name(0)}")
    print(f"GPU 显存:          {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 在 GPU 上跑一个小 tensor 验证
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.rand(5, 3, device=device)
print(f"\n5x3 随机 tensor(device={device}):")
print(x)

# 简单 GPU 算力测试
if torch.cuda.is_available():
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    c = a @ b  # 矩阵乘法
    print(f"\n1000x1000 矩阵乘法 done, 结果 shape: {c.shape}")
    print(f"结果 sum: {c.sum().item():.2f}")
