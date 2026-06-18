# 07 - 综合实验(教学文档)

> 前面每章都是零件,这一章是**整机**。一个能跑、能复现、能调参的工程模板。

## 6.1 涉及的文件

| 文件 | 主题 |
|------|------|
| `examples/01_project_skeleton.md` | 推荐的项目骨架(布局) |
| `examples/02_train_mnist.py` | 完整训练(MNIST) |
| `examples/03_train_cifar.py` | 数据增强对比实验(CIFAR-10) |
| `examples/04_resnet_finetune.py` | ResNet18 微调 CIFAR-10 |
| `examples/05_visualize.py` | 训练曲线 + 混淆矩阵(纯 matplotlib) |

## 6.2 基础知识

### 6.2.1 一个完整 PyTorch 项目的标准模块

```
project/
├── models/                  # 模型定义
│   ├── simple_cnn.py
│   └── resnet_finetune.py
├── utils/                   # 工具函数
│   ├── train.py             # 训练 epoch
│   ├── evaluate.py          # 评估
│   └── visualize.py         # 可视化
├── data/                    # 数据缓存
├── checkpoints/             # 模型权重
├── outputs/                 # 训练曲线 / 报告
├── configs/                 # 配置(YAML / argparse)
├── train.py                 # 主入口
└── evaluate.py              # 评估入口
```

学习项目里**不必都建**(有 `examples/` + 顶层 `train.py` 就够),但发论文 / 做项目时这个结构最常见。

### 6.2.2 可复现性的三个 seed

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)            # Python 内置
    np.random.seed(seed)         # NumPy
    torch.manual_seed(seed)      # PyTorch CPU
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU(多卡)
```

**没固定 seed 的实验不可复现**。论文投稿时审稿人可能让你重新跑一遍。

### 6.2.3 argparse —— 传命令行参数

```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()
```

运行:`python train.py --epochs 5 --lr 0.001`。

**比写死好在哪?**
- 一行换超参,不用改代码
- 跑多个实验时方便
- 写脚本批量跑实验(`for lr in 0.01 0.001 0.0001; do python train.py --lr $lr; done`)

### 6.2.4 GPU 训练标准范式

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MyModel().to(device)

for xb, yb in loader:
    xb, yb = xb.to(device), yb.to(device)
    loss = criterion(model(xb), yb)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

**两条铁律**:
1. 模型和数据**都要** `.to(device)`
2. 验证时**必须** `model.eval()` + `torch.no_grad()`

### 6.2.5 数据增强(为什么有效)

**数据增强 = 用现有数据"造"更多数据**。原理:

- 模型见过**变换后**的版本,学到的特征更**鲁棒**
- 训练集变大,过拟合风险降低
- 推理时不需要做增强(只 test-time augmentation:跑多次取平均,有需要时再用)

常用增强:

| 任务 | 增强 |
|------|------|
| 图像分类 | RandomFlip / RandomCrop / ColorJitter / Cutout / MixUp |
| 目标检测 | 同上 + 多尺度 |
| 分割 | 几何变换(同步输入和 mask) |
| NLP | 随机替换 / 回译 |

## 6.3 逐个 example 讲解

### 6.3.1 `01_project_skeleton.md` —— 项目骨架

这是 markdown,不是代码。讲的是**推荐的项目布局**和**关键工程习惯**。

**关键习惯**(都很重要):
- **可复现性**:`set_seed()`
- **配置外置**:超参放 YAML 或 argparse
- **可观测性**:每 epoch 打印指标
- **最佳权重保存**:验证集提升才覆盖 `best.pt`
- **断点续训**:保存 `optimizer.state_dict` + `epoch` + `best_acc`

### 6.3.2 `02_train_mnist.py` —— 完整训练

5 个核心函数 / 块:

```python
# 1. seed
def set_seed(seed=42): ...

# 2. 模型
class SimpleCNN(nn.Module): ...

# 3. 训练一个 epoch
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = criterion(model(xb), yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        # 累加 loss 和正确数
    return avg_loss, accuracy
```

**关键模式**:
- `model.train()`:打开 train 模式(影响 BN/Dropout)
- `optimizer.zero_grad()` 在 backward **之前**
- 每个 epoch 末尾返回 `avg_loss` 和 `acc`

```python
# 4. 评估
@torch.no_grad()                # 装饰器,等价于 with torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    for xb, yb in loader:
        ...
    return avg_loss, accuracy
```

**`@torch.no_grad()` 装饰器** 一次性把整个函数包在 `no_grad` 上下文里,**简洁**。

```python
# 5. 主函数
def main():
    args = parse_args()
    set_seed(args.seed)
    model = SimpleCNN().to(device)
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(...)
        val_loss, val_acc = evaluate(...)
        print(f"epoch {epoch} ...")
```

**`if __name__ == "__main__":`** 保护,让脚本既能直接跑、也能被 import 当模块用。

### 6.3.3 `03_train_cifar.py` —— 数据增强对比

```python
def get_loaders(use_aug):
    if use_aug:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),    # 4 像素 padding 后随机裁 32x32
            transforms.RandomHorizontalFlip(),       # 50% 概率水平翻转
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3),
        ])
    ...
```

**两组对照**:
- A 组:只 ToTensor + Normalize
- B 组:加 RandomCrop + Flip

**预期**:B 组准确率比 A 组高 1-3%(CIFAR-10 上)。增强能有效防过拟合。

**为什么 `RandomCrop(32, padding=4)`?** CIFAR 图就是 32x32,直接 crop 没东西可裁。**先 pad 4 像素**到 36x36,再 random crop 32x32,**等效于"平移增强"**。

**`RandomHorizontalFlip`** = 50% 概率把图水平翻转。CIFAR 这种自然图像左右翻转语义不变,**适合**;但**文字、医学图像不能翻**(语义会变)。

```python
print(f"提升: {(acc_b - acc_a) * 100:+.2f}%")
```

**这个对比的"科研意义"**:体现"同一数据集,不同增强策略,结果差几个百分点"—— 调参时知道**超参的边际收益**大概多少。

### 6.3.4 `04_resnet_finetune.py` —— 迁移学习实战

```python
def build_model(freeze_backbone=True, num_classes=10):
    model = models.resnet18(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
    return model
```

跟 04 章节学的迁移学习一脉相承,只是**整合**到完整训练脚本里。

```python
# 冻 backbone 时 lr 可以大,全模型微调时 lr 要小
lr = 1e-3 if freeze else 1e-4
```

**经验值**:微调时 lr 至少比从头训练**小 10 倍**。原因:预训练权重已经接近好解了,**大 lr 会把它"踢出去"**。

```python
best_acc = 0.0
for epoch in range(1, args.epochs + 1):
    ...
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "resnet18_cifar10_best.pt")
```

**关键模式**:**只保存最佳权重**。训练完用 `best.pt` 测,而不是用最后一 epoch(可能过拟合了)。

```bash
# 实际跑法
python 04_resnet_finetune.py --epochs 3 --mode head_only   # 冻 backbone,快速出结果
python 04_resnet_finetune.py --epochs 5 --mode finetune    # 全模型微调,慢但更准
```

`--mode` 是 argparse 加的命令行参数,**不传默认 head_only**(学习阶段推荐)。

### 6.3.5 `05_visualize.py` —— 训练曲线 + 混淆矩阵

**训练曲线**:

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history["train_loss"], label="train")
axes[0].plot(history["val_loss"], label="val")
axes[0].set_title("Loss")
axes[1].plot(history["val_acc"], marker="o")
axes[1].set_title("Validation Accuracy")
```

**怎么看**:
- train_loss 一路降、val_loss 一路升 → **过拟合**,加正则 / 早停
- 两个 loss 都不降 → **欠拟合**,加模型容量 / 训更久
- 两个 loss 同步降 → **健康**

**混淆矩阵**:

```python
cm = np.zeros((10, 10), dtype=int)
for t, p in zip(y_true, y_pred):
    cm[t, p] += 1
# cm[i, j] 表示真实 i 被预测成 j 的次数
```

**怎么看**:
- 对角线:预测正确的次数(越亮越好)
- 非对角:错分,**找规律**(比如 cat 经常被分成 dog)
- 多类不均衡:某一行特别暗 → 那一类样本少 / 难

**为什么用纯 matplotlib,不用 sklearn?**
- sklearn 在新 Python 3.12 上偶尔有依赖问题
- 写出来逻辑透明,适合教学
- 工业上 sklearn 更快:`from sklearn.metrics import confusion_matrix`

## 6.4 进阶知识

### 6.4.1 早停 (Early Stopping)

```python
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True   # 应该停
        return False
```

**patience=3**:val_loss 连续 3 个 epoch 不降就停。**比训练固定 epoch 更稳**。

### 6.4.2 学习率调度

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 阶梯式:每 step_size 个 epoch lr 乘 gamma
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 余弦退火:lr 沿余弦曲线降到 0
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# 训练循环
for epoch in range(num_epochs):
    train(...)
    val(...)
    scheduler.step()    # 每 epoch 调一次
```

**经验**:
- 先用 Adam + 固定 lr 训到 plateau
- plateau 之后开 scheduler 微调
- 大模型训练时,`CosineAnnealing` + Warmup 几乎是标配

### 6.4.3 混合精度 (AMP)

```python
scaler = torch.cuda.amp.GradScaler()

for xb, yb in loader:
    xb, yb = xb.to(device), yb.to(device)
    optimizer.zero_grad()

    with torch.cuda.amp.autocast():           # 自动选 fp16/fp32
        loss = criterion(model(xb), yb)

    scaler.scale(loss).backward()             # 缩放 loss 防 fp16 梯度消失
    scaler.step(optimizer)                    # 更新参数
    scaler.update()                           # 更新 scaler
```

**收益**:
- 显存少 30-50%(模型大时救命)
- 速度提升 1.5-2x(用 Tensor Core)
- 几乎不影响精度

### 6.4.4 分布式训练 (DDP)

```python
# 多 GPU / 多机训练
torchrun --nproc_per_node=4 train.py
```

- **DDP**(DistributedDataParallel):最常用的多卡方案
- **FSDP**(FullyShardedDataParallel):超大模型(参数装不下单个 GPU)时用
- **DeepSpeed**:微软的库,优化更多

学习阶段**不需要**,单机单卡足够。**真要做大模型实验时再学**。

### 6.4.5 实验管理

| 工具 | 用途 |
|------|------|
| **TensorBoard** | 训练曲线、模型图、嵌入可视化 |
| **Weights & Biases (wandb)** | 团队协作、实验对比、远程监控 |
| **MLflow** | 实验跟踪、模型注册 |
| **Hydra** | 配置组合(替代 argparse) |

教学项目**TensorBoard 起步**:

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/exp1")

for epoch in range(num_epochs):
    writer.add_scalar("loss/train", train_loss, epoch)
    writer.add_scalar("acc/val", val_acc, epoch)
```

`tensorboard --logdir=runs` 启动可视化。

## 6.5 常见坑

| 现象 | 原因 | 修法 |
|------|------|------|
| 训练 1 epoch 准确率 50% 然后卡住 | 学习率不合适 / 模型结构错了 | 降 lr / 简化模型看 baseline |
| val_acc 远低于 train_acc | 过拟合 | 加数据增强 / Dropout / weight decay |
| 训练慢,GPU 利用率低 | DataLoader 是瓶颈 | `num_workers` + `pin_memory=True` |
| 显存 OOM | batch_size 太大 / 用了大模型 | 降 batch_size / 用 AMP / 改小模型 |
| 训练时 `nan` | 数值溢出 / lr 太大 | 降 lr / `torch.clamp` / 加 `grad_clip` |

## 6.6 学习自检

- [ ] 能搭一个完整训练脚本(数据 + 模型 + 训练 + 评估 + 保存)
- [ ] 能用 argparse 跑不同超参实验
- [ ] 能加数据增强并解释为啥有效
- [ ] 能做迁移学习(冻 backbone / 全模型微调)
- [ ] 能画训练曲线 + 解释模型表现

## 6.7 接下来可以做的事

1. **Kaggle 比赛**:CIFAR-10 / MNIST 排行榜,实战检验
2. **论文复现**:挑一篇 SOTA 论文,按它的 recipe 训
3. **kaggle.com/learn**:PyTorch 入门教程(免费,质量高)
4. **d2l.ai** (李沐《动手学深度学习》):系统性教材
5. **PyTorch 官方教程**:`pytorch.org/tutorials`,权威
6. **fast.ai**:偏实战,top-down 学法

**学习资源合集**:做项目时遇到问题,**90% 都能在 PyTorch 官方文档 / GitHub Issues / Stack Overflow 找到答案**。
