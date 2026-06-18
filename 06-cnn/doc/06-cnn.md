# 06 - 卷积神经网络 CNN(教学文档)

> 上一章用了现成的 backbone,这一章**自己写一个**,理解 Conv2d / Pooling / BN 这些层怎么搭出能用的网络。学完你应该能手写 LeNet/CNN,理解训练循环、模型保存、GPU 训练。

## 5.1 涉及的文件

| 文件 | 主题 |
|------|------|
| `examples/01_nn_module.py` | `nn.Module` 三件套 |
| `examples/02_conv_shape.py` | 卷积输出 shape 推导 |
| `examples/03_lenet.py` | LeNet 完整实现 |
| `examples/04_train_loop.py` | 训练循环(1 epoch MNIST) |
| `examples/05_val_loop.py` | 验证循环 + 准确率 |
| `examples/06_save_load.py` | 模型保存 / 加载 |
| `examples/07_cifar10_cnn.py` | 完整 CIFAR-10 训练 |

## 5.2 基础知识

### 5.2.1 为什么用卷积

全连接层(MLP)处理图像:
- 1000x1000 RGB 图 → 300 万维输入
- 第一层 1024 单元 → 参数 30 亿
- **参数量爆炸、过拟合灾难、丢空间信息**

卷积的优点:
- **局部连接**:每个神经元只看一小块
- **权重共享**:同一个卷积核滑遍整张图
- **平移不变**:猫在左还是在右,卷积都能识别

### 5.2.2 `nn.Module` 三件套

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(...)
        self.conv1 = nn.Conv2d(...)

    def forward(self, x):
        x = self.fc1(x)
        return x
```

- 必须继承 `nn.Module`
- `__init__` 里**定义所有层**(实例化)
- `forward` 里**写数据流**(调层)
- **不要在 `__init__` 里写 forward 逻辑**

**为什么这样设计?** Module 自带 `parameters()`、`to(device)`、`state_dict()`、`train()`/`eval()` 等几十个方法,**只要你正确继承,白送**。

### 5.2.3 `nn.Conv2d` 关键参数

```python
nn.Conv2d(
    in_channels=3,        # 输入通道数(RGB=3,灰度=1)
    out_channels=64,      # 输出通道数(本层卷积核数量)
    kernel_size=3,        # 卷积核尺寸(3 表示 3x3)
    stride=1,             # 步长
    padding=1,            # 边缘填充(0 填充宽度)
    bias=True,            # 是否带偏置
)
```

**输出 shape 公式**:
```
H_out = floor((H_in + 2*padding - kernel_size) / stride) + 1
```

例:`H_in=32, kernel=3, padding=1, stride=1` → `H_out = 32`(保持原尺寸)
例:`H_in=32, kernel=3, padding=0, stride=1` → `H_out = 30`
例:`H_in=32, kernel=2, padding=0, stride=2` → `H_out = 16`(常用下采样)

### 5.2.4 池化层

```python
nn.MaxPool2d(kernel_size=2, stride=2)   # 最常用,2x2 → 1x1
nn.AvgPool2d(kernel_size=2, stride=2)   # 全局平均池化常用在 head 前
```

**作用**:降维 + 提升平移不变性。2x2 max pool 把 32x32 → 16x16,参数立刻少 4 倍。

### 5.2.5 激活函数

| 函数 | 公式 | 特点 |
|------|------|------|
| `nn.ReLU` | `max(0, x)` | **最常用**,简单、快、缓解梯度消失 |
| `nn.LeakyReLU` | `x if x>0 else 0.01x` | 解决 ReLU 死亡(神经元一直输出 0) |
| `nn.GELU` | Transformer 标配 | 平滑、效果好但贵 |
| `nn.Sigmoid` | `1/(1+e^-x)` | 输出 0-1,二分类输出层 |
| `nn.Softmax` | – | 多分类输出层,但 `CrossEntropyLoss` 自带 |

**位置**:Conv → BN → ReLU → Pool 是经典顺序。

### 5.2.6 BatchNorm2d —— 批归一化

```python
nn.BatchNorm2d(num_features=64)  # 跟通道数一致
```

**作用**:对一个 batch 内每个通道**做归一化**(均值 0、方差 1),再加可学习的 scale 和 shift。

**为什么有用?**
- 训练更稳:对学习率、初始化不敏感
- 可以用更大的 lr
- 有轻微正则化效果

**位置**:**Conv 后、ReLU 前**(BN-ReLU-Conv 顺序也有人用,但 Conv-BN-ReLU 是默认)。

**注意**:`model.eval()` 时 BN 行为不同:
- 训练:用当前 batch 的 mean/std
- eval:用**训练时累计的移动平均**

### 5.2.7 Dropout

```python
nn.Dropout(p=0.3)   # 训练时随机丢 30% 激活
```

**作用**:防过拟合,迫使网络不依赖特定神经元。

**只在训练时丢,eval 时关闭** —— 这就是 `model.train()` / `model.eval()` 的另一个作用。

### 5.2.8 损失函数

```python
nn.CrossEntropyLoss()             # 多分类标配,自带 softmax
nn.BCEWithLogitsLoss()            # 二分类(多标签)标配
nn.MSELoss()                      # 回归
```

**`CrossEntropyLoss` = `LogSoftmax + NLLLoss`**。所以**最后一层不要加 softmax**,直接出 logits,损失函数会算。

### 5.2.9 优化器

```python
torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
torch.optim.Adam(model.parameters(), lr=1e-3)
torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

| 优化器 | 特点 | 适用 |
|--------|------|------|
| **SGD + momentum** | 经典、稳、慢 | 论文标配,要 fine-tune 时 |
| **Adam** | 自适应 lr、快 | 实验阶段首选 |
| **AdamW** | Adam + 正确实现 L2 正则 | 训练 Transformer 类模型 |
| **Lion** | 2023 年新出,内存省 | 大模型训练 |

学习阶段 **Adam 起步**,稳定后再考虑换 SGD。

## 5.3 逐个 example 讲解

### 5.3.1 `01_nn_module.py` — nn.Module 三件套

```python
class TwoLayerMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
```

**重点**:
- `super().__init__()` **必须调**,Module 内部要注册子模块
- 把层**实例化**在 `__init__`,`forward` 里只调
- `model.parameters()` 会**递归**收集所有 `nn.Parameter` / 子模块的参数

```python
total = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total:,}")
```

784*128 + 128*10 = 100,480 + 1,280 = **101,760**。

### 5.3.2 `02_conv_shape.py` — 卷积 shape 推导

```python
x = torch.randn(1, 1, 28, 28)        # [B, C, H, W] = [1, 1, 28, 28]
conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
x = conv1(x)
print(x.shape)   # [1, 16, 28, 28]  ← 公式 (28+2*2-5)/1+1=28
x = pool1(x)
print(x.shape)   # [1, 16, 14, 14]  ← 28/2=14
```

**手动算** Conv2d shape:

```
H_out = (H_in + 2*padding - kernel_size) / stride + 1
```

**进 Linear 之前必须自己算好展平后的维度** —— 这是新手最大的卡点:

```python
x = x.flatten(1)              # 展平,保留 batch 维
# 上例: [1, 32, 7, 7] → [1, 32*7*7] = [1, 1568]
fc = nn.Linear(32 * 7 * 7, 10)
```

**经验**:用 `torchinfo.summary(model, input_size=(B, C, H, W))` 自动算,**别手算**。

### 5.3.3 `03_lenet.py` — LeNet-5

```python
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),    # 28 → 24
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                # 24 → 12
            nn.Conv2d(6, 16, kernel_size=5),   # 12 → 8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                # 8 → 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

LeNet 是 1998 年 Yann LeCun 写的,**CNN 鼻祖**,结构极简但能训。

**几个细节**:
- `inplace=True`:ReLU 原地改,**省一点内存**;但 autograd 不喜欢 in-place,实战慎用
- `nn.Sequential`:把层**按顺序连成流水线**,不用手写 `x = layer(x)`,清爽
- `num_classes=10` 参数化:换个数据集(比如 100 类)只改参数
- `model = LeNet().to(device)`:放到 GPU 上

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LeNet().to(device)
x = torch.randn(2, 1, 28, 28).to(device)   # 别忘 .to(device)!
y = model(x)                              # 在 GPU 上算
```

**常见错误**:把模型 `.to(device)` 但忘了把**输入**也 `.to(device)`,会报:
```
RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same
```

### 5.3.4 `04_train_loop.py` — 训练循环

```python
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 10),
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

**训练 4 行黄金结构**(循环体):

```python
for xb, yb in train_loader:
    xb, yb = xb.to(device), yb.to(device)

    # 1. forward
    logits = model(xb)
    loss = criterion(logits, yb)

    # 2. backward
    optimizer.zero_grad()    # 清零
    loss.backward()          # 反向

    # 3. step
    optimizer.step()         # 更新
```

**为什么 4 行不是 3 行?** `optimizer.zero_grad()` 必须在 backward **之前**,否则梯度会跟上一批累加。

**Adam 跟手写 GD 的区别**:Adam 内部每个参数维护了**一阶矩(均值)和二阶矩(方差)**,做自适应学习率,比纯 SGD 鲁棒很多。

### 5.3.5 `05_val_loop.py` — 验证循环

```python
model.eval()    # 切到 eval 模式
correct = 0
total = 0
with torch.no_grad():        # 关闭梯度
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        preds = logits.argmax(dim=1)        # 取概率最大的类别
        correct += (preds == yb).sum().item()
        total += xb.size(0)

acc = correct / total
```

**3 个关键**:
1. `model.eval()` —— BN/Dropout 切到 eval 模式
2. `torch.no_grad()` —— 不建图,**省显存、省算力**
3. `argmax(dim=1)` —— logits → 类别索引

**`preds == yb` 是个 bool tensor,`.sum()` 数 True 多少,`.item()` 拿 Python 数**。

### 5.3.6 `06_save_load.py` — 保存 / 加载

```python
# 保存
torch.save(model.state_dict(), ckpt_path)

# 加载
new_model = nn.Sequential(...)
new_model.load_state_dict(torch.load(ckpt_path, weights_only=True))
new_model.eval()
```

**只保存 `state_dict`**(参数字典),不保存整个模型对象:
- 文件小(只存权重数字)
- 跨 PyTorch 版本兼容
- 不会被 pickle 漏洞利用

`weights_only=True` 防 pickle 反序列化攻击(新版本 PyTorch 默认开了,但显式写更稳)。

**`torch.load` 返回什么?** 一个字典,key 是层名,value 是参数 tensor:
```
{'0.weight': tensor(...), '0.bias': tensor(...), '2.weight': tensor(...), ...}
```

### 5.3.7 `07_cifar10_cnn.py` — 完整 CIFAR-10 训练

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 32→16
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2), # 16→8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

**CIFAR-10 是 3 通道 32x32**,所以 `Conv2d(3, 16, ...)`。

**完整流程**(从 03 学的):
- 加载 CIFAR-10(自动下载到 `data/`)
- DataLoader 拼 batch
- 模型到 GPU
- 训练 1 epoch(打印每 100 batch 的 loss)
- 验证(算测试集准确率)

**`num_workers=0` 在 Windows 上必备** —— 多进程 + spawn + 数据小 = 反而慢。

## 5.4 进阶知识

### 5.4.1 权重初始化

**PyTorch 默认初始化**(Conv2d: Kaiming uniform, Linear: Kaiming uniform)对新模型已经够用。但**如果你手动改权重**(比如用预训练模型初始化),要注意:

```python
# 常用初始化
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')  # Conv 默认
nn.init.xavier_uniform_(layer.weight)                       # Linear 默认
nn.init.zeros_(layer.bias)                                  # 偏置通常置 0
```

**烂初始化 = 不收敛 / 训崩**。一般别动,改用预训练模型最稳。

### 5.4.2 Padding 公式与"保持尺寸"的设置

要保持特征图尺寸不变:
- `kernel=3` → `padding=1`
- `kernel=5` → `padding=2`
- `kernel=7` → `padding=3`

口诀:**padding = (kernel - 1) / 2**(对奇数 kernel)。

### 5.4.3 感受野(Receptive Field)

**感受野 = 一个输出神经元"看"到了输入图像多大区域**。

| 层 | 感受野 |
|---|---|
| Conv 3x3 (pad=1) | 3 |
| Conv 3x3 (pad=1) 叠加 | 5 |
| 三层 Conv 3x3 | 7 |
| Pool 2x2 | ×2 |

**深层神经元的感受野大**,所以它能"看到"图像的大块区域,这对识别整体物体很重要。

**设计网络时**:
- 浅层 → 小感受野 → 抓边缘、纹理
- 深层 → 大感受野 → 抓物体部件、整体

### 5.4.4 ResNet 的"残差连接"

ResNet 之前的网络,深了反而**训崩**。ResNet 2015 年提出,核心 trick:

```python
out = conv(x)
out = relu(out)
out = out + x      # 残差连接!关键
```

让网络学习**残差 F(x) = out - x**,而不是直接学 `out`。**网络再深,梯度也能顺利反传**(因为加法给了捷径)。

现代 CNN 基本都带残差连接(ResNet、DenseNet、ConvNeXt、Swin 等)。

### 5.4.5 学习率调度

**学习率不是固定的**!训练过程中通常要**降低**:

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
# 每 10 epoch lr 减半

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
# 余弦退火,前期 lr 高、后期 lr 平滑降到 0

# 训练循环里
for epoch in range(num_epochs):
    train(...)
    scheduler.step()    # 每个 epoch 调一次
```

**经验**:先 Adam 训到 plateau,再开 scheduler 微调,效果更好。

### 5.4.6 混合精度训练 (AMP)

`torch.cuda.amp` 让你**用 fp16 算、用 fp32 存**:
- 显存少一半(大模型时救命)
- 速度提升 1.5-2x(Tensor Core 加速)

```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    loss = criterion(model(x), y)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**代价**:少数情况数值不稳定(rare),可以加 `grad_clip`。

## 5.5 常见坑

| 现象 | 原因 | 修法 |
|------|------|------|
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | 进 Linear 时维度算错 | 用 `torchinfo.summary` 看展平后的维度 |
| loss 一直是 NaN | lr 太大、loss 计算异常、输入有 NaN/inf | 降 lr / 查数据 / 加 `torch.clamp` |
| 训练集 99% 准确,测试集 10% | 严重过拟合 | 加 Dropout / 数据增强 / weight decay |
| 测试集准确率比训练集还高 | 测试集数据泄漏、或者 val 没 shuffle | 严格分 train/val |
| `state_dict` 加载报 `Missing key(s)` | 模型结构和保存时不匹配 | 结构必须严格一致,或用 `strict=False` 容忍 |

## 5.6 学习自检

- [ ] 能口述 `nn.Module` 三件套
- [ ] 能用公式算 Conv2d 输出 shape
- [ ] 能写一个 3 层 CNN(MNIST 或 CIFAR)
- [ ] 能写完整的 train + val 循环
- [ ] 能用 `state_dict` 保存 / 加载模型
- [ ] 能把模型和数据放到 GPU 上训练

## 5.7 下一步

进入 [`07-experiments/doc/07-experiments.md`](../../07-experiments/doc/07-experiments.md) 学综合实验。
