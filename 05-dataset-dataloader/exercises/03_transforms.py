"""练习 03:transforms 流水线

要求:
  1. 构造一个 transforms.Compose:
     - Resize 到 64x64
     - RandomHorizontalFlip
     - ToTensor
     - Normalize(mean=[0.5], std=[0.5])
  2. 用 PIL.Image.new('RGB', (256, 256), color='red') 造一张纯红图
  3. 跑 3 次 transform,看 output 是否一致(因为有 RandomHorizontalFlip)
  4. 算 output 的 mean 和 std,验证 ≈ 0 / 1
"""
from torchvision import transforms
from PIL import Image
import torch

# 你的代码 ↓
