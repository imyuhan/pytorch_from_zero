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
