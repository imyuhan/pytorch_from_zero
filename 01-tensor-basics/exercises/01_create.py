"""练习 01:张量创建与属性

要求:
  1. 创建一个 3x4 的全 1 float32 tensor,变量名 x1
  2. 创建一个 shape=(2, 3, 4) 的 N(0, 1) 随机 tensor,变量名 x2
  3. 打印它们的 shape / dtype / device
  4. 估算 x1 占用多少字节,跟 x1.untyped_storage().nbytes() 验证
"""
import torch
# 你的代码 ↓
import torch

x1 = torch.ones(3,4)
x2 = torch.randn(2,3,4)
print("x1:",x1.shape,x1.dtype,x1.device)
print("x2:",x2.shape,x2.dtype,x2.device)
print("估算x1字节：",x1.numel()*x1.element_size())
print("实际x1字节：",x1.untyped_storage().nbytes())

