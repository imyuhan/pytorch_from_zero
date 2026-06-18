"""练习 01:张量创建与属性

要求:
  1. 创建一个 3x4 的全 1 float32 tensor,叫 x1
  2. 创建一个 shape=(2, 3, 4) 的 N(0, 1) 随机 tensor,叫 x2
  3. 打印它们的 shape / dtype / device
  4. 估算 x1 占用多少字节,跟 numel() * element_size() 验证
"""
import torch
# 你的代码 ↓
import torch

a = torch.ones(3,3)
print("cpu张量：")
print(a)
print(f"当前device:{a.device}")

if torch.cuda.is_available():
    a = a.cuda()
    print("\nCUDA张量：")
    print(a)
    print(f"当前device:{a.device}")
else:
    print("当前环境没有可用的GPU")
