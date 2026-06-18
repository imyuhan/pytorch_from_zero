"""练习 02:看 MobileNetV3 结构

要求:
  1. 加载 mobilenet_v3_small
  2. 用 torchinfo.summary 打印
  3. 找出分类头在哪里(classifier 字段),看它原来是几类
  4. 替换成 5 分类头
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torchinfo import summary

# 你的代码 ↓
