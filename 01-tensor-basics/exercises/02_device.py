"""练习 02:设备迁移

要求:
  1. 创建一个 1000x1000 的 randn tensor
  2. 把它搬到 GPU,记录耗时
  3. 在 GPU 上做矩阵乘法,记录耗时
  4. 对比纯 CPU 的耗时
  5. 打印:加速比 = CPU时间 / GPU时间
"""
import torch
import time

# 你的代码 ↓
