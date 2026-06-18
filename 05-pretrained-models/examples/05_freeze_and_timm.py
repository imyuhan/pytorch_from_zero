"""04-5 冻 vs 不冻 + timm 简介

迁移学习的两种策略:
  1. 冻住 backbone,只训练新头(快,数据少时用)
  2. 整个模型继续微调(慢,数据多时用)

timm 是 torchvision 之外的另一个模型库,模型更多。
"""
import torch
import torch.nn as nn
import torchvision.models as models

# --- 策略 1:只训练新头 ---
model = models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, 10)  # 新分类头

# 冻住 backbone 所有参数
for param in model.parameters():
    param.requires_grad = False
# 解冻新分类头
for param in model.fc.parameters():
    param.requires_grad = True

# 优化器只收 requires_grad=True 的参数
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

# 数一下:可训练参数 vs 总参数
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数: {total:,}")
print(f"可训练: {trainable:,}(只占 {100*trainable/total:.2f}%)")

# --- 策略 2:全模型微调 ---
# 把上面所有 requires_grad 操作注释掉就行
# 再用小一点的 lr(比如 1e-4)避免破坏预训练权重
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# --- timm 简介(可选)---
# pip install timm
# import timm
# model = timm.create_model("resnet18", pretrained=True, num_classes=10)
# 模型数量比 torchvision 多很多(EfficientNetV2 / ConvNeXt / ViT / Swin ...)
