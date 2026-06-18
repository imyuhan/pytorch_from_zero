"""04-3 看模型结构

print(model) 不够清晰,用 torchinfo.summary 拿到每层的输入输出 shape。
"""
import torch
import torchvision.models as models

# 没装 torchinfo 的话先装一下
try:
    from torchinfo import summary
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "torchinfo", "-q"])
    from torchinfo import summary

model = models.resnet18(weights="DEFAULT")
model.eval()

# summary 给每层的 input / output shape + 参数量
print(summary(model, input_size=(1, 3, 224, 224), verbose=0))

# 关键观察点:
# - Total params: 11.7M(ResNet18 不大)
# - model.conv1 / bn1 / layer1..layer4 是 backbone
# - model.fc 是分类头(Linear: 512 → 1000)
