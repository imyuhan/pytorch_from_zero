# 01 - 张量基础 (Tensor Basics)

> **难度**: ⭐⭐
> **前置**: [00-setup]()
> **预计耗时**: 60 - 90 分钟(看 + 跑 + 练习)
> **硬件**: GPU 推荐,CPU 可降级
>
> 对应学习目标前 14 条
>
> 📖 详细讲解看 [`doc/01-tensor-basics.md`](doc/01-tensor-basics.md)

## 核心概念(必须先搞懂)

| 术语 | 一句话解释 |
|------|-----------|
| **Tensor** | 多维数组,可以是标量、向量、矩阵或更高维 |
| **shape** | 每一维的大小,例如 `(3, 4)` 表示 3 行 4 列 |
| **dtype** | 元素类型:`float32` / `int64` / `bool` 等 |
| **device** | 存在哪里:`cpu` 还是 `cuda:0` |
| **requires_grad** | 是否要对这个 tensor 算梯度(打开后 PyTorch 会建计算图) |
| **CPU / GPU** | 计算硬件;GPU 适合大矩阵并行 |
| **CUDA** | NVIDIA 给 GPU 编程的接口;PyTorch 通过它用 GPU |
| **显存** | GPU 上的内存;`torch.cuda.memory_allocated()` 查 |
| **梯度 (grad)** | loss 对参数的导数,反向传播时累加到 `.grad` 属性 |

## 课程大纲(对应学习目标 14 条)

### 1. 张量创建与基本检查 ✅ 重点

```python
import torch

# 4 种最常用的创建方式
a = torch.tensor([[1, 2], [3, 4]])      # 从 Python 列表
b = torch.ones(2, 3)                     # 全 1
c = torch.zeros(2, 3)                    # 全 0
d = torch.randn(2, 3)                    # 正态分布随机

# 基本检查
print(a.shape, a.dtype, a.device, a.requires_grad)
```

### 2. 设备迁移

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(3, 3)               # 默认在 CPU
x_gpu = x.to(device)                # 搬到 GPU(返回新 tensor)
x_back = x_gpu.to("cpu")            # 搬回 CPU
```

> 重点:`.to()` **返回新 tensor**,原 tensor 不会动。

### 3. NumPy ↔ Tensor

```python
import numpy as np
np_arr = np.array([1.0, 2.0, 3.0])

# numpy → tensor(共享内存!改一边另一边会变)
t1 = torch.from_numpy(np_arr)       # 共享内存
t2 = torch.tensor(np_arr)           # 复制(独立)

# tensor → numpy(必须先到 CPU)
gpu_t = torch.randn(3, device="cuda")
np_back = gpu_t.cpu().numpy()       # .cpu() 必要
```

### 4. 显存查询

```python
torch.cuda.memory_allocated()       # 当前已分配(字节)
torch.cuda.max_memory_allocated()   # 进程启动后的峰值
torch.cuda.empty_cache()            # 释放缓存(不释放张量本身)
```

### 5. 形状(对应张量章节教学大纲)

| 序号 | 主题 | 重点 | 难点 |
|------|------|:----:|:----:|
| 1 | `unsqueeze()` / `squeeze()` 升降维 | ✅ |  |
| 2 | `view()` / `reshape()` 连续性差异,`transpose()` 维度互换 | ✅ |  |
| 3 | `expand()` 性能优化,`cat` vs `stack` 新增维度区别 |  | ✅ |
| 4 | Transformer QKV 矩阵乘法 `matmul()` 维度流转 |  | ✅ |
| 5 | 综合示例:手动构建 `(B, S, H)` 三维文本张量 | ✅ |  |

### 6. 索引与切片(对应目标 12)

```python
x = torch.arange(12).reshape(3, 4)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

x[0]           # 第 0 行
x[:, 0]        # 第 0 列
x[1, 2]        # 标量(单个值)
x[0:2, 1:3]    # 子矩阵
```

### 7. 广播 (Broadcasting)(对应目标 13)

规则:从右往左对齐,维度要么相等,要么其中一个是 1,要么不存在。

```python
# (4, 4) 除以 (4,) → (4, 4) / (1, 4) → 自动广播
x = torch.ones(4, 4)
y = torch.tensor([1., 2., 3., 4.])
(x / y).shape   # torch.Size([4, 4])
```

### 8. 内存共享 vs 复制(对应目标 14)

```python
a = torch.arange(6)
b = a.view(2, 3)       # 共享内存,改 b a 也变
c = a.reshape(2, 3)    # 尽量共享,不能共享时自动复制
d = a.clone()          # 强制复制(完全独立)
e = a.t().contiguous() # transpose 不连续,contiguous() 复制成连续
```

## 学习顺序建议

1. 跑通 `examples/01_create_and_check.py` —— 看到 tensor 在 GPU 上
2. 自己改 14 条目标里的每一条,口头解释给"未来的自己"听
3. 做完 `exercises/01_*.py`(5 道小练习)
4. 卡住的地方回头翻 examples,或搜 PyTorch 官方文档

## 练习

`exercises/` 下 5 道题,每题对应一组能力点。做完可对照 `examples/` 里的写法自检。
