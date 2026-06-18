"""练习 01:用 ResNet50 推理一张图

要求:
  1. 加载 resnet50(weights="DEFAULT")
  2. 用 04-2 的 preprocess 流水线
  3. 拿 PIL.Image.new('RGB', (256, 256), 'blue') 造一张蓝图
  4. no_grad 跑前向,打印 logits 的 shape 和 top-3 类别
"""
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# 你的代码 ↓
