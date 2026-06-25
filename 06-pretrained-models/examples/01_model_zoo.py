"""04-1 torchvision 模型列表与加载预训练权重

torchvision 提供几十个经典模型,ImageNet 上预训练好,直接 weights="DEFAULT"。
"""
import torchvision.models as models

# 不同模型用法一致
resnet18 = models.resnet18(weights="DEFAULT")
resnet50 = models.resnet50(weights="DEFAULT")
mobilenet = models.mobilenet_v3_small(weights="DEFAULT")
efficientnet = models.efficientnet_b0(weights="DEFAULT")

# 列出一些常见模型(只列名字,真正要哪个再 .xxx(weights="DEFAULT"))
common = [
    "resnet18", "resnet34", "resnet50",
    "vgg16", "vgg19",
    "densenet121",
    "mobilenet_v3_small", "mobilenet_v3_large",
    "efficientnet_b0", "efficientnet_b3",
    "convnext_tiny", "swin_tiny",
]
print("常见 torchvision 模型:")
for name in common:
    print(f"  models.{name}(weights='DEFAULT')")
