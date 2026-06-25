"""练习 04:多模型对比

要求:
  1. 分别加载 resnet18 / resnet50 / efficientnet_b0
  2. 每个打印:总参数量(用 torchinfo.summary)
  3. 替换为 10 分类头
  4. 同一个随机输入(1, 3, 224, 224),打印前向耗时(用 time)
  5. 思考:在 RTX 4060 上哪个性价比最高?(参数少但精度不差)
"""
import torch
import time
import torchvision.models as models
from torchinfo import summary

# 你的代码 ↓
