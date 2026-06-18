"""04-2 推理 ResNet18

第一次用预训练模型,看它在随机图上输出什么。
"""
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np

# 加载模型 + 推理模式
device = "cuda" if torch.cuda.is_available() else "cpu"
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.eval()
model.to(device)

# 标准的 ImageNet 预处理(从 weights.meta 直接拿,避免硬编码)
preprocess = weights.transforms()
print(f"预处理: {preprocess}")

# 类别名也直接从 weights.meta 拿(不需要联网下文件)
classes = weights.meta["categories"]
print(f"ImageNet 类别数: {len(classes)}")

# 造一张 256x256 的随机图(只是为了看输出 shape 和 top-5)
# 真用时换成: img = Image.open("your.jpg").convert("RGB")
img = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype(np.uint8))
input_tensor = preprocess(img).unsqueeze(0).to(device)  # [1, 3, 224, 224]

# 推理(必须 no_grad)
with torch.no_grad():
    logits = model(input_tensor)               # [1, 1000]
    probs = torch.softmax(logits, dim=1)[0]   # [1000]
    top5_prob, top5_idx = probs.topk(5)

print("\nTop-5 预测(随机图所以大概率是无意义类别):")
for i in range(5):
    print(f"  {classes[top5_idx[i]]:30s} {top5_prob[i].item():.3f}")
