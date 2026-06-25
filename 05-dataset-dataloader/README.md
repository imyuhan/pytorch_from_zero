# 03 - 数据集与 DataLoader

> **难度**: ⭐⭐
> **前置**: [01-tensor-basics]()
> **预计耗时**: 60 分钟
> **硬件**: GPU 必备(MNIST/CIFAR-10 演示)
>
> 模型吃的是 batch 形式的 tensor,原始数据五花八门(图片/CSV/文本)。这一章就是学会**喂数据**。
>
> 📖 详细讲解看 [`doc/05-dataset-dataloader.md`](doc/05-dataset-dataloader.md)

## 核心概念

| 术语 | 一句话解释 |
|------|-----------|
| **Dataset** | 抽象类,提供 `__len__` 和 `__getitem__`(给索引返回一个样本) |
| **DataLoader** | 把 Dataset 包装成可迭代的 batch,负责 shuffle / 多 worker / 拼 batch |
| **transform** | 预处理流水线,图片常用 `Resize` → `ToTensor` → `Normalize` |
| **collate_fn** | 默认会把样本堆成 batch,自定义场景(变长序列)需要自己写 |
| **batch / shuffle / num_workers** | DataLoader 三个最常用参数 |
| **train / val split** | 训练集 + 验证集;`torch.utils.data.random_split` 切 |

## 课程大纲

| 序号 | 主题 | 重点 | 难点 |
|------|------|:----:|:----:|
| 1 | `Dataset` 抽象类与三件套(`__init__` / `__len__` / `__getitem__`) | ✅ |  |
| 2 | 自定义 `TensorDataset` / 列表数据集 | ✅ |  |
| 3 | `DataLoader` 四件套:batch_size / shuffle / num_workers / drop_last | ✅ |  |
| 4 | `transforms.Compose` 流水线:`Resize` / `ToTensor` / `Normalize` |  | ✅ |
| 5 | `random_split` 切 train / val | ✅ |  |
| 6 | 完整实战:MNIST 加载 + 可视化 + DataLoader 跑通 |  | ✅ |

## 最小示例

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        return self.x[i], self.y[i]

ds = MyDataset(torch.randn(100, 3), torch.randint(0, 2, (100,)))
dl = DataLoader(ds, batch_size=8, shuffle=True)

for xb, yb in dl:
    print(xb.shape, yb.shape)  # torch.Size([8, 3]) torch.Size([8])
    break
```

## 学习顺序

1. 跑通 6 个 examples
2. 做完 5 道 exercises(综合起来就是"自己写个图像分类数据集")
3. 进入 05 看预训练模型怎么吃 DataLoader
