"""05-3 LeNet-5 for MNIST 完整实现

经典网络:2 卷积 + 2 池化 + 3 全连接,输入 28x28 灰度图,输出 10 类。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 特征提取
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),    # 28 → 24
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                # 24 → 12
            nn.Conv2d(6, 16, kernel_size=5),   # 12 → 8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                # 8 → 4
        )
        # 分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# 验证
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LeNet().to(device)
print(summary(model, input_size=(1, 1, 28, 28), verbose=0))

x = torch.randn(2, 1, 28, 28).to(device)
y = model(x)
print(f"\n输入: {x.shape} → 输出: {y.shape}")
