"""练习 05:广播 & 索引

要求:
  1. 创建 x = torch.arange(20).reshape(4, 5)
  2. 取出第 2 行(下标 2)
  3. 取出第 3 列(下标 3)
  4. 取出 [0:2, 1:4] 子块
  5. 用广播让 x 减去行均值(沿 dim=1 求 mean,shape=(5,)),验证中心化
"""
import torch

# 你的代码 ↓
x = torch.arange(20).reshape(4,5)
print(x)
print("\nx 的第二行：",x[1,:])
print("x 的第三列：",x[:,2])
print("x 的[0:2,1:4]:",x[0:2,1:4])

# 沿 dim=1 求 mean,先转换 x 的 dtype
# dim 后的值等于几就沿着那个维度走，例：0 就沿着行方向即向下走
x = x.float()
row_mean = x.mean(dim=1, keepdim=True)
print("\nrow_mean:",row_mean.flatten().tolist())

# 广播中心化(减去行均值,让每行均值归零)
x_centered = x - row_mean
print("中心化后：\n",x_centered)

#验证中心化（中心化后每行均值应为0）
print("中心化后每行均值：",x_centered.mean(dim=1).tolist())

