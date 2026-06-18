"""04-4 替换分类头:迁移学习核心

把 ImageNet 的 1000 类换成自己的 10 类。
"""
import torch
import torch.nn as nn
import torchvision.models as models

# 加载预训练模型
model = models.resnet18(weights="DEFAULT")

# 看原分类头
print("原分类头:", model.fc)

# 替换成自己的 10 分类头
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

print("\n新分类头:", model.fc)
# 注意:新 Linear 的权重是随机初始化的(没有预训练)

# 验证前向能跑
x = torch.randn(2, 3, 224, 224)
out = model(x)
print(f"\n输入 shape: {x.shape}")
print(f"输出 shape: {out.shape}  # [B, num_classes]")
