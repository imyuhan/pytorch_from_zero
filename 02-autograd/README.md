# 02 - 自动求导 (Autograd)

> **难度**: ⭐⭐⭐
> **前置**: [01-tensor-basics]()
> **预计耗时**: 60 - 90 分钟
> **硬件**: GPU 推荐(梯度计算演示更直观),CPU 可降级
>
> 对应学习目标第 9, 10 条,以及 autograd 章节的 5 条大纲
>
> 📖 详细讲解看 [`doc/02-autograd.md`](doc/02-autograd.md)

## 核心概念

| 术语 | 一句话解释 |
|------|-----------|
| **requires_grad** | 打开后,PyTorch 会记录所有作用于该 tensor 的运算,形成计算图 |
| **grad_fn** | 记录这个 tensor 是怎么来的(由哪个 op 产生);叶子节点是 None |
| **backward()** | 从 loss 反向走计算图,把梯度累加到每个叶子的 `.grad` |
| **zero_grad()** | 每次 backward 之前清空 `.grad`,否则梯度会累加 |
| **no_grad()** | 上下文管理器,关闭梯度计算(推理 / 评估时用) |
| **detach()** | 把 tensor 从计算图里"切"出来,变成普通 tensor |

## 课程大纲(对应 autograd 章节)

| 序号 | 主题 | 重点 | 难点 |
|------|------|:----:|:----:|
| 1 | `requires_grad` 与 `grad` / `grad_fn` 动态追踪 | ✅ |  |
| 2 | Forward → Loss → Backward 链路底层 |  | ✅ |
| 3 | `.backward()` 累积梯度 + `zero_grad()` 清空 | ✅ |  |
| 4 | 推理时用 `with torch.no_grad()` 断梯度树 | ✅ |  |
| 5 | 综合:手写一个带 LR 的梯度下降 |  | ✅ |

## 最小示例(对应目标 9)

```python
import torch

# 1. 创建带梯度的参数
w = torch.tensor(2.0, requires_grad=True)

# 2. forward
y = w ** 2              # y = w^2

# 3. backward
y.backward()            # dy/dw = 2w = 4

print(w.grad)           # tensor(4.)
```

## 详细讲解请看 `examples/` 下 5 个脚本

## 练习

`exercises/` 下 5 道题。
