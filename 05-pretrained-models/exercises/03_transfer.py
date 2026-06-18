"""练习 03:迁移学习(冻 backbone,训练新头)

要求:
  1. 加载 resnet18,改 fc 为 3 分类
  2. 冻 backbone,只训练 fc
  3. 造 32 条假数据(3 分类,每类约 10 条),shape=(B, 3, 224, 224)
  4. 写训练循环 3 个 step:forward → cross_entropy loss → backward → step → zero_grad
  5. 打印 loss,验证它在下降
"""
import torch
import torch.nn as nn
import torchvision.models as models

torch.manual_seed(0)
x = torch.randn(32, 3, 224, 224)
y = torch.randint(0, 3, (32,))

# 你的代码 ↓
