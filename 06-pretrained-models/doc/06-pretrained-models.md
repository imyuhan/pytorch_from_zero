# 06 - 预训练模型(教学文档)

> 从零训练一个图像模型要几天、几千张卡。我们用 **别人训好的** —— 这就是迁移学习 / 预训练模型的威力。

## 4.1 涉及的文件

| 文件 | 主题 |
|------|------|
| `examples/01_model_zoo.py` | torchvision 模型库概览 |
| `examples/02_inference.py` | ResNet18 推理 + softmax + top-5 |
| `examples/03_summary.py` | 用 torchinfo 看模型结构和参数量 |
| `examples/04_replace_head.py` | 替换分类头(迁移学习的核心) |
| `examples/05_freeze_and_timm.py` | 冻 backbone + timm 简介 |

## 4.2 基础知识

### 4.2.1 什么是预训练模型

**预训练 (pretrained)**:在大数据集(通常是 ImageNet, 1.2M 张图、1000 类)上训练好的模型。

**ImageNet 预训练为什么这么香?**
- 模型**已经学到了"什么是边缘、什么是纹理、什么是物体部件"** 这些通用特征
- 你自己的小数据集(几百张、几千张)训不出这种特征
- 把预训练模型的"特征提取能力"搬过来,只重训"分类头" → **10x 数据效率、10x 训练速度**

### 4.2.2 关键概念

| 概念 | 解释 |
|------|------|
| **backbone** | 模型前半部分(卷积层),负责提取通用特征 |
| **head** | 模型最后几层(全连接),负责把特征映射到具体类别 |
| **微调 (fine-tune)** | 用小学习率**继续训练**整个模型或部分 |
| **冻 (freeze)** | 让某些参数 `requires_grad=False`,反向时不更新它们 |
| **特征提取 (feature extraction)** | 拿掉 head,只输出 backbone 的中间表示 |
| **迁移学习 (transfer learning)** | 把预训练模型用到新数据集上(统称) |

### 4.2.3 torchvision 模型库概览

```python
import torchvision.models as models

models.resnet18(weights="DEFAULT")         # 11.7M 参数
models.resnet50(weights="DEFAULT")         # 25.6M
models.efficientnet_b0(weights="DEFAULT")  # 5.3M, 性价比高
models.mobilenet_v3_small(weights="DEFAULT")  # 2.5M, 移动端
models.convnext_tiny(weights="DEFAULT")    # 28.6M, 现代架构
models.swin_tiny(weights="DEFAULT")        # 28.3M, Transformer 类
```

`weights="DEFAULT"` 自动选**该模型最佳版本的预训练权重**(通常是 ImageNet-1K,V2 版的比 V1 准)。

**怎么选?**
- 快速实验:`resnet18` / `mobilenet_v3_small`
- 精度优先:`efficientnet_b3` / `convnext_small`
- 视觉 Transformer 路线:`swin_tiny` / `vit_b_16`

### 4.2.4 timm —— torchvision 之外的另一个模型库

`pip install timm` 安装。`timm` 由 Ross Wightman 维护,优势:
- 模型**多得多**(几百种)
- 训练 / 推理 API **更统一**
- 权重**更新更频繁**

```python
import timm
model = timm.create_model("resnet18", pretrained=True, num_classes=10)
```

`pretrained=True` 和 torchvision 的 `weights="DEFAULT"` 等价。**学完 torchvision,timm 几行就能上手**。

## 4.3 逐个 example 讲解

### 4.3.1 `01_model_zoo.py` —— 模型列表

```python
resnet18 = models.resnet18(weights="DEFAULT")
```

第一次跑会**自动下载权重**(约 47MB),存到 `~/.cache/torch/hub/checkpoints/`(Linux)或 `C:\Users\<user>\.cache\torch\hub\checkpoints\`(Windows)。**之后跑直接读缓存**。

```python
common = [
    "resnet18", "resnet34", "resnet50",
    "vgg16", "vgg19",
    "densenet121",
    "mobilenet_v3_small", "mobilenet_v3_large",
    "efficientnet_b0", "efficientnet_b3",
    "convnext_tiny", "swin_tiny",
]
```

这是按"学习优先级"排的常用模型清单:
- ResNet 系列:经典、稳、教程多
- MobileNet:轻量、移动端部署
- EfficientNet / ConvNeXt:现代架构、精度高
- Swin:Transformer 类的代表

### 4.3.2 `02_inference.py` —— 推理 ResNet18

```python
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.eval()
```

`ResNet18_Weights.DEFAULT` 是枚举值,**等价于字符串 `"DEFAULT"`**。`model.eval()` 切换到推理模式:
- BN 用**移动平均**而非 batch 统计
- Dropout **不丢**任何激活

```python
preprocess = weights.transforms()
```

**这一步是新的写法,避免硬编码**。`weights.transforms()` 返回**这个模型期望的预处理**,跟训练时一模一样。打印看看:

```
ImageClassification(
    crop_size=[224], resize_size=[256],
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
    interpolation=InterpolationMode.BILINEAR
)
```

等于 ResNet 训练时的预处理(也是 ImageNet 标配)。

```python
classes = weights.meta["categories"]
print(f"ImageNet 类别数: {len(classes)}")
```

**类别名也直接从 weights 拿**,不需要联网下载 `imagenet_classes.txt`。这是 torchvision 0.13+ 的新 API。

```python
with torch.no_grad():
    logits = model(input_tensor)               # [1, 1000]
    probs = torch.softmax(logits, dim=1)[0]   # [1000]
    top5_prob, top5_idx = probs.topk(5)
```

- `torch.no_grad()` 推理必备(02 章节 02 / 03 / 04 反复强调)
- `softmax` 把 logits(任意实数)转成**概率**(0-1 之间、和为 1)
- `topk(5)` 拿概率最大的 5 个

**Random 图 top-5 经常是水母(jellyfish)、泡泡之类的**,因为 ImageNet 见过很多"模糊色块"的训练样本,模型学到了某种"水母-like"的特征。换成真实图就准了。

### 4.3.3 `03_summary.py` —— 看模型结构

```python
from torchinfo import summary

print(summary(model, input_size=(1, 3, 224, 224), verbose=0))
```

`torchinfo` 是 torchvision 之外的工具,专门打印**带输入输出 shape 的模型结构**:

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
ResNet                                   [1, 1000]                 --
├─Conv2d: 1-1                            [1, 64, 112, 112]         9,408
├─BatchNorm2d: 1-2                       [1, 64, 112, 112]         128
├─AdaptiveAvgPool2d: 1-9                 [1, 512, 1, 1]            --
├─Linear: 1-10                           [1, 1000]                 513,000
==========================================================================================
Total params: 11,689,512
Trainable params: 11,689,512
==========================================================================================
```

**关键信息**:
- 每层**输入输出 shape** 算得清清楚楚
- 每层**参数量**
- 总参数量
- 输入/前向/权重占用的**显存估算**

**没有 torchinfo 怎么算?**
- 整个模型 `print(model)` 也能看,只是没 shape
- 想要 shape 就得自己**手算** 或用 `torch.fx` 之类的工具

**装 torchinfo**:
```bash
uv pip install torchinfo --python .venv/Scripts/python.exe
```

### 4.3.4 `04_replace_head.py` —— 替换分类头

**这是迁移学习最核心的一步**。

```python
model = models.resnet18(weights="DEFAULT")
print("原分类头:", model.fc)
# Linear(in_features=512, out_features=1000, bias=True)
```

ImageNet 上训的 ResNet18,**最后那层 `model.fc` 是 512 → 1000**(1000 类)。**我们要改的是这层**。

```python
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

**关键细节**:
- `model.fc.in_features = 512`:从原 head 里读,**不用硬编码**
- 新 `nn.Linear(512, 10)` 权重是**随机初始化**的
- backbone 仍是预训练好的;只有新 head 需要重训

```python
x = torch.randn(2, 3, 224, 224)
out = model(x)               # torch.Size([2, 10])
```

**前向还能跑**,shape 对得上。

### 4.3.5 `05_freeze_and_timm.py` —— 冻 vs 不冻

```python
model = models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, 10)

# 冻 backbone
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True
```

**关键点**:
- `requires_grad=False` → autograd **不**记录这个参数(节省算力、显存)
- 必须**先把所有参数置 False,再单独打开 head** 的(顺序无所谓,但别忘)
- backbone 既然不参与梯度,**权重保持预训练的状态**,只更新新 head

```python
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)
```

**`filter(lambda p: p.requires_grad, ...)`**:只把可训练参数传给优化器。**不写这一行的话,优化器拿到所有参数,试图更新冻结的会报"无梯度"错**。

输出:
```
总参数: 11,181,642
可训练: 5,130(只占 0.05%)
```

**5,130 个可训练参数**(只算新 head 的权重和偏置),占 0.05% —— 训练 **极快、显存极省**。

#### 两种策略对比

| 策略 | 适用场景 | lr | 训练量 |
|------|---------|----|----|
| **冻 backbone** | 数据少(<1k)、类别少、和 ImageNet 接近 | 1e-3 | 几分钟 |
| **全模型微调** | 数据中等以上、领域差得远 | 1e-4 ~ 1e-5 | 几十分钟到几小时 |

学习阶段先**试冻 backbone**,因为:
- 快(几分钟出结果)
- 不容易破坏预训练权重
- 效果常常就够用

## 4.4 进阶知识

### 4.4.1 预训练权重该用 V1 还是 V2?

torchvision 0.13+ 有 V1 / V2 之分,例如:
- `ResNet18_Weights.IMAGENET1K_V1`:原始 ResNet18
- `ResNet18_Weights.IMAGENET1K_V2`:用更好的 recipe 重训,**精度更高**

`DEFAULT` 默认指向 V2(更新版本的)。**一般用 DEFAULT 就对了**。

### 4.4.2 不同模型对输入的要求

| 模型 | 输入尺寸 | 预处理 |
|------|---------|--------|
| ResNet | 224 | ImageNet mean/std |
| EfficientNet | 224~600(看 b0~b7) | 同上 |
| ViT | 224 或 384 | 同上 |
| Swin | 224 或 384 | 同上 |

**224 是主流**,但 EfficientNet 默认 256,大 EfficientNet 600,得看模型具体设定。**用 `weights.transforms()` 拿官方推荐最稳**。

### 4.4.3 特征图可视化(理解 backbone 学到了什么)

```python
# 拿中间层输出
features = model.layer3[0].conv1.weight  # 或 forward hook
# 可视化
import matplotlib.pyplot as plt
plt.imshow(features[0, 0].detach().cpu(), cmap="gray")
```

进阶用法是 **forward hook**:
```python
activations = {}
def hook(name):
    def fn(module, input, output):
        activations[name] = output.detach()
    return fn

model.layer2.register_forward_hook(hook("layer2"))
_ = model(x)
print(activations["layer2"].shape)  # 中间层输出
```

**这能帮你理解 backbone 各层学到了什么**(浅层:边缘、纹理;深层:物体部件)。

### 4.4.4 迁移学习为什么有效?—— 几个理论视角

1. **低层特征通用**:边缘、纹理、颜色块对几乎所有视觉任务都有用
2. **数据集大小决定深度**:ImageNet 上训的 ResNet 见过 1.2M 张图,远超你手上的几千张
3. **优化起点好**:随机初始化 vs 预训练权重,**后者在 loss landscape 上更接近好解**
4. **正则化效果**:预训练权重本身就是一个"先验",防止过拟合

但**别神化迁移学习**:
- 领域差异极大(医学影像、卫星图)时,ImageNet 预训练**有时反而拖后腿**,不如直接训
- 大规模自监督预训练(DINO、CLIP)正在改变这件事

### 4.4.5 torch.hub —— 直接从 GitHub 加载模型

```python
model = torch.hub.load("pytorch/vision", "resnet18", weights="DEFAULT")
```

等价于 `models.resnet18(weights="DEFAULT")`,**好处是能加载任何 GitHub 仓库的模型**(只要作者按规范写了 `hubconf.py`)。例如:

```python
# 加载某个第三方模型的最新权重
model = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
```

## 4.5 常见坑

| 现象 | 原因 | 修法 |
|------|------|------|
| 推理 accuracy 接近 0 | 预处理不匹配(mean/std 用错 / 尺寸不对) | 用 `weights.transforms()` 不要手写 |
| 替换 head 后 `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | 新 head 的 `in_features` 写错 | 改用 `model.fc.in_features` 别硬编码 |
| 微调效果比从头训还差 | lr 太大,把预训练权重"破坏"了 | lr 降到 1e-4 ~ 1e-5 |
| 推理显存爆 | 忘了 `model.eval()` 或 `torch.no_grad()` | eval loop 严格包两层 |
| 下载权重慢 / 失败 | 跨国带宽问题 | 多试几次,或者挂代理 |

## 4.6 学习自检

- [ ] 能用 ResNet18 跑 inference 拿到 top-5
- [ ] 能用 `torchinfo` 打印模型结构
- [ ] 能把 ResNet18 的 head 替换成自己的类别数
- [ ] 能冻 backbone 只训 head
- [ ] 能解释两种迁移策略的适用场景

## 4.7 下一步

进入 [`07-cnn/doc/07-cnn.md`](../../07-cnn/doc/07-cnn.md) 学卷积神经网络 —— 看看这些 backbone 到底是怎么搭出来的。
