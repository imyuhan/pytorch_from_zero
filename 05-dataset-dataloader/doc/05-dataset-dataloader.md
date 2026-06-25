# 05 - 数据集与 DataLoader(教学文档)

> 神经网络不能直接吃图片/CSV/文本,得先转成 tensor,再按 batch 喂进去。这一节就是学会"喂数据"。

## 5.1 涉及的文件

| 文件 | 主题 |
|------|------|
| `examples/01_dataset_class.py` | `Dataset` 三件套(`__init__` / `__len__` / `__getitem__`) |
| `examples/02_dataloader_basics.py` | `TensorDataset` + `DataLoader` 四件套 |
| `examples/03_list_dataset.py` | 列表型 Dataset(从硬盘读图) |
| `examples/04_transforms.py` | `transforms.Compose` 流水线 |
| `examples/05_train_val_split.py` | `random_split` 切 train/val |
| `examples/06_mnist_load.py` | MNIST 实战:加载 + 可视化 + DataLoader |

## 5.2 基础知识

### 5.2.0 前置:这章会碰到的基本概念(为 Dataset / DataLoader 做铺垫)

> 05 章名词密度大,先把基本术语和"为什么"讲清楚,后面的内容才不会突兀。

#### 1. 一个训练迭代的最小数据流

神经网络训练时,数据是这样流动的:

```
[原始数据](图片 / CSV / 文本)
   ↓ 一个一个样本
[样本](单张图 + 标签)
   ↓ 多个样本堆起来
[batch](比如 32 个样本)
   ↓ 喂给模型
[模型.forward(batch)]
   ↓
[loss] → 反向传播 → 更新参数
```

这一章的三个组件各管一段:

| 组件 | 干什么 | 管的是哪一段 |
|------|--------|------------|
| **Dataset** | 按索引取一条样本 | 原始数据 → 样本 |
| **DataLoader** | 打包 batch + shuffle + 多进程 | 样本 → batch |
| **transform** | 把样本变成 tensor 形式 | 样本 → tensor |

#### 2. 几个绕不开的基本名词

**样本(sample)**:一条数据,比如 1 张图 + 标签。

**batch**:一批样本堆起来同时喂给模型。为什么要 batch?
- 单条样本梯度方向不稳 → 一次看 N 条取平均更稳
- GPU 擅长并行 → 一次算 N 条比 N 次算 1 条快得多
- 常见 batch_size:32、64、128

**epoch**:**完整遍历一遍数据集**。

举例:1000 条样本,batch_size = 100:
- 1 epoch = 1000 / 100 = **10 个 batch**
- 训 10 epoch = 把数据集看 10 遍

**iteration**(或 step):**跑一个 batch** 就叫一次 iteration。

```
1 epoch = ceil(数据集大小 / batch_size) 次 iteration
```

**shuffle**(打乱):每个 epoch 前**随机打乱**样本顺序。
- 为什么?模型按顺序学可能"记住顺序"而不是"学规律"。打乱后梯度更鲁棒。
- 训练 `shuffle=True`,验证/测试 `shuffle=False`(要可复现)。

#### 3. 为什么需要 Dataset 抽象类?——接口约定

如果你自己写加载数据:
```python
for i in range(len(images)):
    img = images[i]
    label = labels[i]
    ...  # 训练
```

问题:shuffle 怎么写?多进程加载怎么写?换数据集要不要重写训练代码?

**Dataset 抽象类**就是 PyTorch 给你的**接口约定**:
```python
class Dataset:
    def __len__(self): ...        # 数据集多大
    def __getitem__(self, i): ... # 给索引取一条样本
```

只要实现这两个方法,PyTorch 的 DataLoader / 训练循环就能用。**不同数据集(图片 / 文本 / CSV)走同一个 API**。

这就是"面向接口编程":不管你数据怎么存,我只要你能按索引取样本。

#### 4. 为什么需要 DataLoader?——它替你干了这些脏活

不用 DataLoader,你得自己写:
- 把 N 个样本 stack 成 batch(`torch.stack`)
- 每个 epoch 前 shuffle 索引
- 多进程并行加载(Windows 上很坑)
- 把 batch 搬到 GPU(配合 `pin_memory`)

用 DataLoader 一行搞定:
```python
loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
```

**DataLoader = batch 打包 + shuffle + 多进程 + GPU 预取,四合一**。

#### 5. 为什么需要 transform?——把世界变成 tensor

神经网络吃 **tensor**(数字),不吃 PIL 图 / numpy 数组 / 文本字符串。

transform 的工作:
1. **格式转换**:PIL Image → tensor(`ToTensor()`),numpy → tensor
2. **数值归一**:像素 [0, 255] → [0, 1](`ToTensor()` 顺手做),再压到"零均一方差一"(`Normalize()`)
3. **数据增强**(训练时):随机水平翻转、随机裁剪——让模型见到更多变体,不容易过拟合

**Compose 就是流水线**:把多个 transform 串起来,上一个的输出当下一个的输入。

```python
transforms.Compose([
    Resize((32, 32)),         # 1. 缩到 32x32
    RandomHorizontalFlip(),   # 2. 随机左右翻转(数据增强)
    ToTensor(),                # 3. 转 tensor + 归一到 [0, 1]
    Normalize(mean, std),      # 4. 标准化
])
```

顺序很关键——**ToTensor 必须先于 Normalize**(Normalize 算的是 tensor,PIL 上算不了)。

#### 6. train / val / test 三件套

| 集合 | 干什么 | 调完超参还能再用吗 |
|------|--------|----------------|
| **train** | 训练模型 | - |
| **val**(验证) | 调超参(选模型、选 lr、选增强) | ✅ 可以 |
| **test**(测试) | 最终评估,**只用一次** | ❌ 用了就"污染" |

**严格流程**:train 训 → val 调 → test 终极评估,test 结果发完论文就再也不能用 test 调任何东西。

教学项目通常只 train + test(把 test 当 val 用),发论文时再补一个 hold-out test。

#### 7. 学完再看 3.2.1

3.2.1 那张数据流图你应该能看懂了:
- 硬盘数据 → Dataset → 样本 → DataLoader → batch → 训练循环
- 每个组件只管一段,合起来就是完整流水线

如果还有名词不懂,继续往下读。**读 example 时回头查这一节**,比硬啃理论更有效。

---

### 5.2.1 整个数据流的层级

```
[硬盘 / 内存上的原始数据]
        ↓ Dataset 负责"按索引取一条样本"
[__getitem__(i) → 一条样本]
        ↓ DataLoader 负责"打包成 batch + shuffle + 多进程"
[DataLoader 迭代器]
        ↓ 训练循环里 for xb, yb in loader:
[模型.forward(xb) ...]
```

**Dataset 决定怎么"取一条"**,**DataLoader 决定怎么"拼 batch"**。两者职责分明。

### 5.2.2 `Dataset` 抽象类

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, ...):       # 一次性准备(读索引、打开文件清单)
        ...
    def __len__(self):             # 返回样本总数
        ...
    def __getitem__(self, i):      # 给索引,返回一条样本
        return x, y                # 一般是 (数据, 标签) 元组
```

只要实现这三个方法,就能塞进 `DataLoader`、随便迭代。

### 5.2.3 `TensorDataset` —— 最快的写法

如果数据**已经全是 tensor**(在内存里),用 `TensorDataset` 一行搞定:

```python
ds = TensorDataset(x, y)             # x 是 [N, ...], y 是 [N,]
for xb, yb in DataLoader(ds, batch_size=32):
    ...
```

### 5.2.4 `DataLoader` 四件套

```python
DataLoader(
    dataset,
    batch_size=32,       # 每个 batch 多大
    shuffle=True,        # 每个 epoch 是否打乱
    num_workers=0,       # 用几个子进程加载数据(Windows 建议 0 或 2)
    drop_last=False,     # 最后一批不够 batch_size 时是否丢弃
)
```

| 参数 | 默认 | 建议 |
|------|------|------|
| `batch_size` | 1 | 视显存调,32/64/128 是常见起点 |
| `shuffle` | False | 训练 True、验证 False |
| `num_workers` | 0 | Linux 可开 4-8;Windows 建议 0(多进程开销大) |
| `drop_last` | False | 数据集刚好整除时无所谓,否则设 True 让每个 batch 大小一致 |
| `pin_memory` | False | GPU 训练时设 True,数据搬运到 GPU 更快 |
| `collate_fn` | default | 自定义场景才改(变长序列等) |

### 5.2.5 `transforms.Compose` —— 预处理流水线

```python
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),     # 数据增强
    transforms.ToTensor(),                  # PIL → tensor, 0-255 → 0-1
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

**为什么训练和验证用不同的 transform?**
- 训练:**随机**增强(RandomFlip、RandomCrop)→ 模型见到更多样
- 验证:**确定**的预处理(只 Resize、不 Random)→ 公平评估

**ImageNet 的 mean/std** 是统计 1.2M 张图算出来的,所有 torchvision 预训练模型都默认用这个。**用预训练模型时,你的数据集也要用这个**才能匹配它的预期输入。

### 5.2.6 `random_split` 切 train/val

```python
train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))
```

- 传入**长度列表**指定怎么切
- 用 `generator` 固定种子,**保证可复现**

## 5.3 逐个 example 讲解

### 5.3.1 `01_dataset_class.py` — Dataset 三件套

```python
class MyDataset(Dataset):
    def __init__(self, x, y):
        assert len(x) == len(y), "x 和 y 长度必须一致"
        self.x = x
        self.y = y
```

`__init__` 做一次性准备:**别读全部数据**!只存**路径列表**或**索引**,真正用的时候再 `__getitem__` 里临时读。

```python
def __len__(self):
    return len(self.x)
```

返回总数。DataLoader 用它算 `len(dataloader) = ceil(n / batch_size)`。

```python
def __getitem__(self, i):
    return self.x[i], self.y[i]
```

返回**一条样本**。DataLoader 默认会用 `torch.stack` 把 N 条堆成 batch,所以这里返回啥形状都行(只要能 stack)。

### 5.3.2 `02_dataloader_basics.py` — TensorDataset + DataLoader

```python
x = torch.randn(100, 3)
y = torch.randint(0, 2, (100,))
ds = TensorDataset(x, y)
dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
```

**`TensorDataset` 内部就是把每个 tensor 当一列,`__getitem__(i)` 返回 `(x[i], y[i], ...)`**。一行顶 5 行自己写。

**`num_workers=0` 在 Windows 上几乎是必须的** —— Windows 多进程要用 `spawn` 方式,启动开销大,小数据集反而更慢。Linux 上 4-8 是标配。

### 5.3.3 `03_list_dataset.py` — 列表型 Dataset

```python
class ImageListDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.paths = image_paths
        self.labels = labels
        self.transform = transform
```

`__init__` **只存路径列表**,不读图,内存友好。

```python
def __getitem__(self, i):
    img = Image.open(self.paths[i]).convert("RGB")
    label = self.labels[i]
    if self.transform is not None:
        img = self.transform(img)
    return img, label
```

每次访问才读盘 + 预处理。**配合 `num_workers > 0`**,多个 worker 并行读图,GPU 不会饿着等数据。

**`Image.open()` 返回的是 lazy 的 PIL 对象** —— 它不立刻读图,你要 `.convert()` 或 `.resize()` 时才真正加载。提前 `.load()` 一下可以更早触发文件读,有时能省事。

### 5.3.4 `04_transforms.py` — transforms 流水线

```python
resize = transforms.Resize((32, 32))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
```

**单步拆解**:
- `Resize((32, 32))`:PIL/numpy → 缩到 32x32(短边缩放 + center crop,或直接拉伸,看参数)
- `ToTensor()`:PIL/numpy `(H, W, C)` 0-255 → tensor `(C, H, W)` 0-1
- `Normalize(mean, std)`:对每个通道 `(x - mean) / std`

```python
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[...], std=[...]),
])
```

**顺序很关键**:
- 必须先 `ToTensor()` 再 `Normalize` —— Normalize 算的是 tensor,PIL 上没法算
- 增强(Flip/Crop)放在 ToTensor 之前,作用于 PIL 图,比"tensor 增强"快 5x

#### 为什么归一化?

```python
# 没归一化:像素值 0-1
# 归一化后:均值约 0,方差约 1
```

深度学习对**输入分布**敏感,数据如果都集中在 0-1,梯度更新方向可能不稳定。归一化后**各通道量级一致**,训练更稳。

均值/方差用 ImageNet 的(0.485/0.456/0.406, 0.229/0.224/0.225)是**因为大多数预训练模型都按这个训的**,你自己数据集想用预训练模型,跟着这个数最稳。

### 5.3.5 `05_train_val_split.py` — random_split

```python
full = TensorDataset(x, y)
n_train = int(0.8 * n_total)
train_ds, val_ds = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(0))
```

**优点**:**不复制数据**!底层还是同一份 tensor,只是切了索引。

`generator` 固定种子,保证每次跑切出来的 train/val 一样,方便对比实验。

### 5.3.6 `06_mnist_load.py` — MNIST 实战

```python
train_set = datasets.MNIST(
    root=data_root, train=True, download=True, transform=transform
)
```

`datasets.MNIST` 是 torchvision 内置的 `Dataset` 实现,自动下载、自动解包。`train=True` 是 6 万训练集,`train=False` 是 1 万测试集。

**首次运行会下载**,约 50MB,放 `data/` 下。

```python
img, label = train_set[0]
print(img.shape)  # torch.Size([1, 28, 28])
```

**MNIST 是单通道 28x28** → tensor 形状 `[C, H, W] = [1, 28, 28]`。

```python
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
xb, yb = next(iter(train_loader))
print(xb.shape)  # [64, 1, 28, 28]
```

DataLoader 自动 stack 成 batch:每个样本 `[1, 28, 28]` → batch `[64, 1, 28, 28]`。

#### 可视化(用 matplotlib 反归一化画图)

```python
fig, axes = plt.subplots(2, 8, figsize=(12, 3))
for i, ax in enumerate(axes.flat):
    img, label = train_set[i]
    ax.imshow(img.squeeze(), cmap="gray")
```

**`img.squeeze()`** 把 `[1, 28, 28]` 压成 `[28, 28]`(matplotlib 不接受 1 通道的 3 维)。`cmap="gray"` 是灰度图(MNIST 是黑白的)。

## 5.4 进阶知识

### 5.4.1 `IterableDataset` —— 流式数据

`Dataset` 是**随机访问**(`__getitem__(i)` 直接拿第 i 个);当数据是**流**(网络流、日志流、视频流)时,用 `IterableDataset`:

```python
from torch.utils.data import IterableDataset

class MyStream(IterableDataset):
    def __iter__(self):
        # 不能被随机访问,只能迭代
        for item in self._stream():
            yield item
```

代价:**不能 `len()`、不能 shuffle、不能随机抽样**。仅在流场景用。

### 5.4.2 自定义 `collate_fn`

默认 `collate_fn` 把 N 条样本 `torch.stack` 成 batch。**变长序列**(NLP 一句话长短不一)时,stack 不上,得用 padding:

```python
def collate_fn(batch):
    # batch: list of (image, label) tuples
    images = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]  # 假设是变长文本
    return images, labels

dl = DataLoader(ds, batch_size=32, collate_fn=collate_fn)
```

NLP 实战里这个坑躲不开,要么 padding、要么用 `DataCollatorWithPadding` 这种现成工具。

### 5.4.3 预取与 `pin_memory`

`pin_memory=True` 让 DataLoader 把 tensor **预先拷贝到 CUDA pinned memory**,GPU 读时带宽更高(PCIe 不需要 page-lock)。

```python
dl = DataLoader(ds, batch_size=64, pin_memory=True, num_workers=2)
for xb, yb in dl:
    xb = xb.to(device, non_blocking=True)  # non_blocking + pin_memory 才有效
    yb = yb.to(device, non_blocking=True)
```

**实测在 GPU 训练时能省 5-15% 的数据加载时间**。

### 5.4.4 `DataLoader` 迭代器协议

```python
dl = DataLoader(ds, batch_size=32)
it = iter(dl)
xb, yb = next(it)         # 拿一个 batch
xb, yb = next(it)         # 拿下一个
# ... 跑完所有 batch,StopIteration 自动抛
```

`for xb, yb in dl:` 内部就是 `it = iter(dl)` + `next(it)` + 捕获 `StopIteration`。**一个 DataLoader 只能完整迭代一次**(epoch),要再跑一个 epoch 就**重新 `for ... in dl`**,内部会重建 iterator。

### 5.4.5 验证集 + 测试集的区别

| 名字 | 作用 | 用来调超参吗? |
|------|------|--------------|
| **训练集 (train)** | 训练模型 | – |
| **验证集 (val)** | 调超参(选模型、选 lr、选增强策略) | ✅ |
| **测试集 (test)** | 最终评估,**只用一次** | ❌ 用了就"污染"了 |

**严格流程**:train 训 → val 调 → test 终极评估,test 结果发完论文就再也不能用 test 调任何东西。

实际教学项目里 `train` / `test` 两个就够(把 `test` 当 `val` 用),发论文时再补一个 hold-out test。

## 5.5 常见坑

| 现象 | 原因 | 修法 |
|------|------|------|
| `RuntimeError: stack expects each tensor to be equal size` | `__getitem__` 返回的样本尺寸不一 | 用 `collate_fn` 自定义 batch 拼装 |
| DataLoader 比 GPU 还慢 | `num_workers=0` 且数据预处理重 | 增 `num_workers` + 用 `pin_memory` |
| 训练时数据全是 label 0 | `shuffle` + `WeightedRandomSampler` 用错 | 改用 `random_split` 切,确认 label 分布 |
| Windows 报 `RuntimeError: DataLoader worker (pid xxx) is killed by signal: Bus error` | 多 worker 内存爆 | `num_workers=0` 或 `persistent_workers=True` |
| `ToTensor()` 后数据范围不是 [0, 1] | 之前做过 `Normalize` 又 `.numpy()` 之类 | 检查 transform 顺序 |

## 5.6 学习自检

- [ ] 能口述 `Dataset` 三件套的作用
- [ ] 能说出 `DataLoader` 至少 4 个关键参数
- [ ] 能解释训练和验证 transform 为什么不一样
- [ ] 能用 `random_split` 切一个数据集
- [ ] 能写出 `train` / `val` 两个 DataLoader

## 5.7 下一步

进入 [`06-pretrained-models/doc/06-pretrained-models.md`](../../06-pretrained-models/doc/06-pretrained-models.md) 学预训练模型 —— torchvision 模型库加载、推理、替换分类头、冻结 backbone。
