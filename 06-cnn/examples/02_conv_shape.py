"""05-2 卷积输出 shape 推导

公式: H_out = (H_in + 2*padding - kernel_size) / stride + 1
"""
import torch
import torch.nn as nn

# 假设输入: [B, C, H, W] = [1, 1, 28, 28](单张 28x28 灰度图)
x = torch.randn(1, 1, 28, 28)

# 第一个卷积块: 1 → 16 channels, 5x5 kernel, padding=2(保持尺寸)
conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

x = conv1(x)
print(f"Conv1: {x.shape}")  # [1, 16, 28, 28]  ← 尺寸不变(padding 补回来了)
x = pool1(x)
print(f"Pool1: {x.shape}")  # [1, 16, 14, 14]  ← 减半

# 第二个卷积块: 16 → 32
conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
pool2 = nn.MaxPool2d(2, 2)
x = conv2(x)
print(f"Conv2: {x.shape}")  # [1, 32, 14, 14]
x = pool2(x)
print(f"Pool2: {x.shape}")  # [1, 32, 7, 7]

# 展平 + 全连接
x = x.flatten(1)            # 保留 batch 维,展平后面所有
print(f"Flatten: {x.shape}")  # [1, 32*7*7] = [1, 1568]

# 关键点:进 Linear 之前必须自己算好 1568 这个数
# (可以靠 torchinfo.summary 自动算)
fc = nn.Linear(32 * 7 * 7, 10)
x = fc(x)
print(f"FC:     {x.shape}")  # [1, 10]
