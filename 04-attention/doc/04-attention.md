# 04 - 维度变换与注意力矩阵(教学文档)

> 承接 01 章张量基础,把 `reshape / transpose / matmul` 用到极致——把 Transformer 注意力机制的矩阵流**从头手算**一遍。本节用 3 条手写的小样本,所有 shape 都可以肉眼心算验证。

## 4.1 涉及的文件

| 文件 | 主题 |
|------|------|
| `examples/01_demo.py` | 端到端跑完 9 个小实验(主入口) |
| `examples/02_mask_compare.py` | 改 valid_mask 后看 A002 注意力权重矩阵的变化 |
| `examples/03_batch_size_compare.py` | 用 1/2/3 条样本跑,对照 batch_size 变化 |
| `src/attention_tools.py` | 课堂演示用的工具函数(load_lesson_data、split_heads、手写 SDPA 等) |
| `data/lesson_data.json` | 3 条课堂样本(A001 全有效、A002 后两位 [PAD]、A003 第 4 位 [PAD]) |

## 4.2 为什么单独开一章

01 章 `05_transformer_qkv.py` 只演示了 `Q @ K^T → (B, S, S)` 这一步,**没有 softmax、没有 mask、没有缩放、没有多头**。这些"剩下没讲的部分"恰恰是注意力的灵魂:

| 没讲的部分 | 这一章讲什么 |
|------|------|
| softmax / 缩放 | `softmax(QK^T / √d)` 才是真正的注意力权重 |
| mask | padding 位置必须屏蔽掉,否则模型会去"关注"无意义的位置 |
| 多头 | 真实 Transformer 都用多头,把 hidden 拆成多个 head 独立算 |
| 官方 SDPA | PyTorch 有 `F.scaled_dot_product_attention`,手写公式必须和它对齐验证 |

读完 4.2 这张表,你应该带着三个疑问进 4.3:
1. 注意力机制**到底在做什么**,为什么 `Q @ K^T` 不够、还得加 softmax 和缩放?
2. 真实模型为什么**几乎都是多头**?把一个注意力拆成多个有什么好处?
3. 多头里 `hidden_dim` 怎么拆?`num_heads` 和 `head_dim` 分别是什么?

## 4.3 核心概念:注意力机制与多头注意力

> 本节是整章的**动机铺垫**。读完再去 4.4 看 `reshape / transpose / matmul` 这些操作,就知道"为什么这么改 shape"。如果只看代码不读这节,会觉得多头的拆合头像是凭空冒出来的 shape 杂技。

### 4.3.1 注意力机制要解决什么问题

先看历史背景。Transformer 之前,序列建模的主流架构是 RNN/LSTM 和 CNN,它们各有硬伤:

| 架构 | 痛点 |
|------|------|
| RNN / LSTM | 长距离依赖要等很多步才能传到,容易梯度消失;必须**串行**算,无法 GPU 并行 |
| 1D CNN | 感受野有限,看 100 个 token 之前的关系要叠很多层 |

**注意力机制的核心思想**:让任意两个 token 之间**直接相连**,距离 O(1),且**完全并行**。

具体怎么做到?——给每个 token 三种"身份",让它们互相"查询-应答",这就是下面要讲的 Q / K / V。

### 4.3.2 Q / K / V 三个矩阵的语义

把每个 token 的 embedding 看作一个人,他同时扮演三种角色:

| 角色 | 含义 | 类比 |
|------|------|------|
| **Q (Query)** | 这个 token 拿着"我想找什么"的向量 | 检索时你输入的关键词 |
| **K (Key)** | 这个 token 拿着"我提供什么"的向量 | 索引里每个文档的标题 |
| **V (Value)** | 这个 token 实际携带的内容 | 文档的正文 |

注意力的过程 = **拿每个 Q 去和所有 K 算相关度,再用相关度对所有 V 加权求和**。

```python
q = x @ W_q   # (B, S, H) → (B, S, H_q)
k = x @ W_k   # (B, S, H) → (B, S, H_k)
v = x @ W_v   # (B, S, H) → (B, S, H_v)
```

> **self-attention 的特殊性**:Q、K、V **都从同一个 X 投影而来**(所以叫 self)。通常 `H_q = H_k = H_v = hidden_dim`,三个投影矩阵 `W_q / W_k / W_v` 是模型**唯一要学**的参数。

### 4.3.3 embedding:把符号变成向量

> 本节是注意力机制的"输入端"铺垫。Transformer 处理不了字符串,必须先把 token 变成数字向量才能做矩阵运算——这一步就是 embedding。**embedding 是 tensor 的一种具体用途,不是一种独立的数据结构**。

#### 4.3.3.1 什么是 embedding

**一句话定义**:embedding = 把"不能直接算"的东西(字 / 词 / 图片 / 用户 ID)变成"能算的向量"。

对模型来说,字符串 `"模型"`、像素矩阵、用户 ID `12345` 全都没法做矩阵乘法。Embedding 给每个离散符号配一个**连续向量**,让后续所有运算(注意力、卷积、相似度)有东西可以乘。

对着你 A001 的真实数据看:

```json
{
  "sample_id": "A001",
  "tokens": ["模型", "会", "关注", "重点"],
  "embeddings": [
    [0.90, 0.10, 0.20, 0.30],   ← "模型" 的向量
    [0.20, 0.80, 0.10, 0.40],   ← "会" 的向量
    [0.30, 0.20, 0.90, 0.10],   ← "关注" 的向量
    [0.40, 0.10, 0.30, 0.80]    ← "重点" 的向量
  ]
}
```

`build_embedding_tensor` 把这些数字转成 `(batch=3, seq=4, hidden=4)` 的 Tensor,然后才能喂给 attention。

**这 4 个数有什么含义?** 每一维对应一个**潜在语义特征**——训练完后可视化能看出来大致是"是不是名词""情感倾向""主谓关系"这类隐含属性。**训练前我们不知道每个维度具体代表啥**,重要的是:**语义相近的词,向量也相近**。

举例子(假想的 2 维嵌入):

```text
"猫"   = [0.9, 0.1]
"狗"   = [0.85, 0.15]
"汽车" = [0.1, 0.9]
```

猫和狗都是动物 → 第一维都接近 0.9;汽车不是 → 第一维 0.1。**空间里相近 = 语义相近**。

#### 4.3.3.2 embedding 与 tensor 的关系

**核心关系**:embedding 是 tensor 的一种"用途",tensor 是底层数据结构。

类比:

| 上位概念 | 具体实例 |
|------|------|
| 容器 | 水杯 / 饭碗 / 花盆 |
| 交通工具 | 自行车 / 汽车 / 飞机 |
| **Tensor** | **embedding / 权重矩阵 / 图像像素 / 音频频谱** |

代码证明它们底层完全一样:

```python
import torch
import torch.nn as nn

e = nn.Embedding(num_embeddings=1000, embedding_dim=64).weight
# shape: (1000, 64)  ← 词向量表

w = nn.Linear(64, 64).weight
# shape: (64, 64)    ← 全连接层权重

img = torch.randn(32, 3, 224, 224)
# shape: (32, 3, 224, 224)  ← 图像 batch

print(type(e))    # torch.Tensor
print(type(w))    # torch.Tensor
print(type(img))  # torch.Tensor
```

三个对象**全是 `torch.Tensor`**,底层存储格式、内存布局完全一样。区别只在三件事:

| 维度 | embedding | 权重矩阵 | 图像 |
|------|-----------|----------|------|
| **shape** | `(vocab_size, dim)` | `(in, out)` | `(N, C, H, W)` |
| **怎么来** | 查表 / 反向传播学 | 初始化 + 反向传播学 | 拍照 / 数据增强 |
| **怎么用** | 按 token ID 取行 | 跟输入做矩阵乘 | 喂给 CNN |

**判别口诀**:当且仅当 tensor 的某个维度对应"某个具体符号的 ID"时,这个 tensor 才叫 embedding。

#### 4.3.3.3 embedding 在 attention 里的层级

注意,embedding 这个词在不同地方指代不同的层级:

```python
# 1. embedding 表本身: (vocab_size, hidden_dim) — 2D tensor
emb_table = nn.Embedding(10000, 512).weight   # (10000, 512)

# 2. 一次 lookup 出来的: (seq_len, hidden_dim) — 2D tensor
sentence_vec = emb_table[torch.tensor([12, 45, 89, 102])]   # (4, 512)

# 3. 一个 batch 的句子: (batch, seq_len, hidden_dim) — 3D tensor
batch_vec = emb_table[torch.tensor([
    [12, 45, 89, 102],
    [8, 99, 0, 0],
])]  # (2, 4, 512)
```

**第 1 个**严格叫"embedding 表"(查表用的);
**第 2、3 个**还是 tensor,但角色是"已经查出来的向量序列"——很多人也管这个叫 embedding(广义)。你项目里 `build_embedding_tensor` 返回的 `(3, 4, 4)` 就是这种广义用法。

#### 4.3.3.4 真实项目里 embedding 从哪来

你 `lesson_data.json` 里是**手工写死**的(教学专用,见 `teaching_notes`),真实项目里分三档:

| 方式 | 怎么来 | 适用场景 |
|------|--------|----------|
| **`nn.Embedding` 训练得到** | 随机初始化 → 反向传播更新 | 最常见,从头训练 LLM |
| **预训练模型提取** | 加载 BERT / Word2Vec 的 embedding 表 | 迁移学习、小数据场景 |
| **其他模态编码器** | CNN(图像) / 音频模型 → 输出向量 | 多模态(CLIP) |

最常见写法 `nn.Embedding`:

```python
import torch.nn as nn

# vocab_size = 词表大小,embedding_dim = 向量维度
embedding = nn.Embedding(num_embeddings=10000, embedding_dim=512)

# 输入是 token ID(整数),不是字符串
token_ids = torch.tensor([2, 45, 1024, 7])   # 4 个词的 ID
vectors = embedding(token_ids)                # → (4, 512)

# 带 batch 的版本
batch_ids = torch.tensor([
    [2, 45, 1024, 7],
    [8, 99, 0, 0],
])                                            # (2, 4)
batch_vectors = embedding(batch_ids)          # → (2, 4, 512)
```

**本质就是一张查表表**:`nn.Embedding(vocab_size, dim)` 内部就是一个 `(vocab_size, dim)` 的矩阵,`embedding(id)` 相当于按 ID 取出第 `id` 行。

#### 4.3.3.5 几个常被混淆的概念

| 概念 | 关系 |
|------|------|
| **Tensor** | 多维数字数组,底层数据结构 |
| **Embedding** | tensor 的一种具体用途(每行对应一个符号) |
| Word2Vec / GloVe | **学 embedding 的老算法**(2013-2014),现在基本被 Transformer 取代 |
| one-hot | embedding 的**退化版**——只有一个维度是 1,其它都是 0,信息量极少 |

#### 4.3.3.6 实战坑点

1. **embedding 输入是整数 ID,不是字符串也不是 one-hot**。`nn.Embedding` 内部自己转 one-hot 然后做矩阵乘
2. **embedding_dim 不一定等于 hidden_dim**,但实际代码里经常相等。Transformer 里 Q/K/V 投影矩阵把 `embedding_dim` 映射到 `hidden_dim`(或反过来,看实现)
3. **padding token 的 embedding 也是学出来的**(通常初始化为 0)。它对模型就是"无效信号",靠 mask 屏蔽掉它对其他位置的影响——这就是为啥 attention 一定要 mask

> **与本章的关系**:你练习 01 手写的多头注意力,输入 `(3, 4, 4)` 的 tensor 就是广义 embedding——已经查出来的向量序列。真实项目里在这之前还会有两步:tokenizer(字符串 → token ID)和 `nn.Embedding`(ID → 向量)。你 04 章是从"已经有 embedding"开始练的,前两步留到后面的章节。

### 4.3.4 Scaled Dot-Product Attention 公式

把上面 Q/K/V 的直觉写成公式,就是 Transformer 的核心:

```text
Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
```

拆成 5 步看每一步做了什么:

1. **打分**:`scores = Q @ K^T` → `(B, S, S)`,第 `(i, j)` 个元素是"第 i 个 token 对第 j 个 token 的相关度"
2. **缩放**:除以 `√d_k`。点积方差随 `d_k` 线性增长,数值太大会让 softmax 把概率压到某个 token 上,梯度消失。详细数学动机见 4.4.7
3. **mask**:padding 位置填 `-inf`,softmax 后该位置权重自动为 0
4. **softmax(dim=-1)**:每行变概率分布,和为 1
5. **加权求和**:`context = weights @ V` → `(B, S, H)`,每个位置拿到"融合了所有位置信息"的新表示

手写版只有 5 行:

```python
def manual_attention(q, k, v, valid_mask):
    d_k = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    scores = scores.masked_fill(~valid_mask.unsqueeze(1), float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)
```

> **01 章 `05_transformer_qkv.py` 只演示了第 1 步**(Q @ K^T),所以你之前看到的 `(B, S, S)` 矩阵**还不是注意力权重**,只是未缩放、未 softmax 的原始分数。完整公式必须把缩放、softmax、mask 一起写上。

### 4.3.5 多头注意力:多个"观察者"并行

单头注意力的局限很直觉:**只有一个 Q/K/V 投影组合,只能学一种关注模式**。

类比审稿:
- **单头**:一个人审稿,他只能从一种风格 / 一种经验去判断
- **多头**:8 个审稿人并行,有人看语法、有人看指代关系、有人看长距离依赖——最后合起来判断

Transformer 论文里的观察确实如此:训练好的 BERT/GPT 里,不同 head 学到的关注模式差异非常大。有的 head 几乎只关注下一个 token(像 bigram),有的 head 关注固定某个位置(像位置编码的某种偏移),有的 head 关注远距离的指代。

### 4.3.6 为什么把 hidden_dim 拆成 num_heads × head_dim

**关键观察:拆前后总维度不变,所以参数量几乎一样。**

| 方案 | 总维度 | 参数量 | 学到的模式数 |
|------|--------|--------|-----------|
| 单头 (hidden=512) | 512 | 1× | 1 种 |
| **8 头 × head_dim=64** | **8×64=512** | **≈1×** | **8 种** |
| 8 头都保留 hidden_dim=512 | 8×512=4096 | 8× | 高度冗余 |

所以拆分是**白嫖**:参数量守恒,白得多视角。

那为什么不让每个头都保持 512 维?——那样 8 个头总维度 4096,参数量爆炸 8 倍,而且 8 个独立的大矩阵容易**学重复模式**(梯度信号雷同,优化方向一致),反而浪费容量。

### 4.3.7 num_heads / head_dim 的具体含义

| 变量 | 含义 | 常见取值 | 备注 |
|------|------|----------|------|
| `hidden_dim` | 输入/输出的总维度 | 768 / 4096 / 8192 | **拆头前后不变** |
| `num_heads` | 头数,即"几个观察者" | 8 / 12 / 16 / 32 / 64 | 越多越能学到多样化模式,但单 head 越窄 |
| `head_dim` | 每个头内部的特征维度 | 64 / 128 | 每个观察者视野多宽 |

**唯一硬约束**:`hidden_dim = num_heads × head_dim`(必须能整除)。改 num_heads 时 head_dim 自动跟着变。

**head_dim 不是越大越好**:
- 头内维度太大会让 QK^T 数值过大,softmax 输出过尖,接近 one-hot,梯度消失
- 经验上 64–128 是甜蜜区。`d_k = 64` 就是 Transformer 论文的原始设定

**实战常见配置**:

| 模型 | hidden_dim | num_heads | head_dim |
|------|-----------|-----------|----------|
| BERT-base | 768 | 12 | 64 |
| GPT-2 small | 768 | 12 | 64 |
| LLaMA-7B | 4096 | 32 | 128 |
| LLaMA-70B | 8192 | 64 | 128 |
| ViT-B/16 | 768 | 12 | 64 |

观察规律:**head_dim 普遍在 64–128 之间**,而不是跟随 hidden_dim 一起放大。`hidden_dim` 主要靠堆 `num_heads` 来增加容量,而不是靠扩大单个 head。

### 4.3.8 shape 流转:从单头到多头

把单头和多头的 shape 摆一起看,差异一目了然:

```text
【单头】
输入:           (B, S, H)
Q, K, V:        (B, S, H)
Q @ K^T:        (B, S, S)              ← 每个 token 对所有 token 的分数
softmax @ V:    (B, S, H)              ← 融合后的表示,shape 回到 H


【多头】
输入:           (B, S, H)                                  H = num_heads × head_dim
reshape:        (B, S, num_heads, head_dim)                ← 拆开 hidden 维
transpose:      (B, num_heads, S, head_dim)                ← 把 heads 提到 seq 前面
Q @ K^T:        (B, num_heads, S, S)                       ← 每个 head 独立打分
softmax @ V:    (B, num_heads, S, head_dim)                ← 每个 head 独立加权
transpose:      (B, S, num_heads, head_dim)                ← 合回去
reshape:        (B, S, H)                                  ← 拼回原 hidden
```

**两个关键点**:
1. **transpose 把 heads 维提到 seq 前面**:这样后续的 matmul 会把 heads 当作 batch 维的一部分,**天然按 head 并行**算,不需要写 for 循环
2. **mask 在多头里要从 `(B, S)` 升到 `(B, 1, 1, S)`**:
   - 第 2 个 `1`(heads 维):所有头共享同一份 mask
   - 第 3 个 `1`(query 维):每个 query 位置用同一份 mask
   - 只有最后 `S`(key 维)是真实维度
   
   直接 `(B, S)` 跟 `(B, num_heads, S, S)` 广播对不上。这步升维是练习 01 的必踩坑。

对应到 `src/attention_tools.py` 里的 `multi_head_demo`,可以一行行对着上面 6 步 shape 看。

## 4.4 基础知识

### 4.4.1 文本张量的标准形状

```text
embeddings.shape = (batch_size, seq_len, hidden_dim)
```

| 维度 | 含义 | 本节样本值 |
|------|------|------|
| `batch_size` | 一次送进模型的句子数 | 3 |
| `seq_len` | 每句 token 数 | 4 |
| `hidden_dim` | 每个 token 的向量长度 | 4 |

`batch_size=3` 来自 `lesson_data.json` 的 3 条样本;`seq_len=4` 是因为 `[PAD]` 也算一个 token;`hidden_dim=4` 是**故意做小**,方便手算。

### 4.4.2 升维 / 降维: `unsqueeze` / `squeeze`

模型接口通常要求输入有 batch 维,但你手头可能只有 `(seq, hidden)` 的单句。`unsqueeze(0)` 在最前面补一个长度为 1 的 batch 维:

```python
sentence = embeddings[0]              # (4, 4) 单句
with_batch = sentence.unsqueeze(0)    # (1, 4, 4) 一批只有 1 句
```

`squeeze()` 默认删除**所有**长度为 1 的维度,初学阶段建议显式写 `dim`:

```python
x = torch.ones(1, 1, 4, 4)
x.squeeze(1)    # (1, 4, 4)  只删第 1 维
x.squeeze()     # (4, 4)      删所有长度 1 的维(小心!)
```

### 4.4.3 拆多头: `reshape` + `transpose` + `contiguous`

多头注意力要把 `hidden_dim` 拆成 `num_heads` 和 `head_dim`。这一步是**最容易卡住**的地方。具体为什么这么拆,见 4.3.6。

```python
batch_size, seq_len, hidden_dim = 3, 4, 4
num_heads = 2
head_dim = hidden_dim // num_heads   # 2

# 第 1 步:reshape 把 hidden 维拆成 (num_heads, head_dim)
split = embeddings.reshape(batch_size, seq_len, num_heads, head_dim)
# shape: (3, 4, 2, 2)

# 第 2 步:transpose 把 heads 维提到 seq 维前面
heads_first = split.transpose(1, 2)
# shape: (3, 2, 4, 2)

# 第 3 步:contiguous() 强制连续,后面再 reshape/view 才不会报错
heads_first = heads_first.contiguous()
```

**为什么要 transpose?** 每个 head 要独立算 QK^T,所以 heads 维必须**在 seq 前面**,这样矩阵乘法才能按 head 并行。完整 shape 流转见 4.3.8。

**为什么要 contiguous?** transpose 只是改 stride,不动数据,内存布局变成不规则的。后续如果再 `reshape` / `view` 会报错。

### 4.4.4 `expand` vs `repeat`

| 操作 | 复制数据吗? | 典型用途 |
|------|------|------|
| `expand` | ❌ 只读视图(零拷贝) | 位置偏置、广播场景 |
| `repeat` | ✅ 真实复制 | 需要写、序列化场景 |

```python
position_bias = torch.arange(4).reshape(1, 4, 1).float()
expanded = position_bias.expand(3, 4, 4)   # (3, 4, 4) 共享视图
repeated = position_bias.repeat(3, 1, 4)   # (3, 4, 4) 真实拷贝
```

### 4.4.5 `cat` vs `stack`

| 操作 | 沿哪一维 | 输出 shape |
|------|------|------|
| `cat([a, b], dim=0)` | 已有第 0 维拼接 | `(a[0]+b[0], ...)` |
| `stack([a, b], dim=0)` | 新建一个维度 | `(2, a[0], ...)` |

口诀:**cat 是加长,stack 是加厚**。

### 4.4.6 矩阵乘法 `matmul` 的 shape 规则

PyTorch 的 `matmul` 规则:
1. 如果两个输入都是二维 → 普通矩阵乘
2. 如果有一个输入维度 ≥ 3 → **只对最后两维**做矩阵乘,前面的维度视为 batch

```python
x = torch.randn(3, 4, 4)          # (B=3, S=4, H=4)
y = x.transpose(-2, -1)            # (B=3, H=4, S=4)
scores = torch.matmul(x, y)        # (B=3, S=4, S=4)
# 等价于 for b in 3: scores[b] = x[b] @ y[b],但 GPU 一次算完
```

`scores[b, i, j]` 表示第 `b` 句里第 `i` 个 token 对第 `j` 个 token 的"相关度"。

### 4.4.7 缩放:为什么除以 `√hidden_dim`

点积的方差会随 hidden_dim 线性增长。`hidden_dim=64` 时点积方差 64,`hidden_dim=512` 时方差 512。数值太大时,softmax 会把所有概率压到一个 token 上,梯度消失。

除以 `√d` 后,方差归一化到 ~1,softmax 输出不会过尖也不会过平。

```python
scaled_scores = raw_scores / math.sqrt(hidden_dim)
```

### 4.4.8 注意力 mask

真实场景里,一个 batch 里的句子长度不一样,短的会 `[PAD]` 到同样长度。**模型不应该关注 [PAD] 位置**,否则会学到噪声。

做法:padding 位置填 `-inf`,softmax 之后该位置权重自动变成 0:

```python
valid_mask = torch.tensor([[True, True, True, False]])   # 第 4 位是 [PAD]
key_mask = valid_mask.unsqueeze(1)                       # (1, 1, 4) 广播到 (B, S, S)

scores = scores.masked_fill(~key_mask, float("-inf"))
weights = torch.softmax(scores, dim=-1)
# weights[..., -1] 全是 0  ← 模型不再关注 [PAD]
```

> **多头版本**的 mask 形状是 `(B, 1, 1, S)`,跟这里 `(B, 1, S)` 的单头版本不一样。详见 4.3.8 的 mask 升维说明。

## 4.5 逐个 example 讲解

> 04 章的 example 之间有依赖关系,建议按 `01_demo.py → 02_mask_compare.py → 03_batch_size_compare.py` 的顺序跑。

### 4.5.1 `examples/01_demo.py` —— 端到端 9 节小实验(主入口)

`01_demo.py` 是这章的主入口,按顺序跑完 9 节实验。所有实验都用同一批 3 条样本,这样每步输出能前后对照。代码里的演示函数都定义在 `src/attention_tools.py` 里。

#### 4.5.1.1 第 1 节 构造文本 embedding 张量

把 `data/lesson_data.json` 里的手写 embedding 转成 `(batch=3, seq=4, hidden=4)` 的三维 Tensor:

```python
embeddings, tokens, valid_mask = build_embedding_tensor(data)
```

为什么要这一步?——注意力机制要求输入是数字向量,不能直接拿中文字符串做矩阵乘法。embedding 的概念铺垫见 4.3.3。

#### 4.5.1.2 第 2 节 `unsqueeze` / `squeeze`

单条句子没有 batch 维,模型接口却要求有。怎么补?——`unsqueeze(0)`。详细形状变化见 4.4.2。

#### 4.5.1.3 第 3 节 `reshape` / `transpose` / `contiguous`(拆多头)

把 `hidden_dim` 拆成 `num_heads × head_dim`,让 heads 维换到 seq 前面。这一步是**最容易卡住**的地方。详细讲解见 4.4.3。

#### 4.5.1.4 第 3.5 节 `split_heads` 封装函数

把上面拆头两步抽成独立函数,直接调用:

```python
heads = split_heads(embeddings, num_heads=2)   # (3, 2, 4, 2)
```

后面真正写多头注意力时直接用它。

#### 4.5.1.5 第 4 节 `expand` / `repeat`

`expand` 是零拷贝广播视图,`repeat` 是真实复制。详细对比见 4.4.4。

#### 4.5.1.6 第 5 节 `cat` / `stack`

`cat` 沿已有维度接长,`stack` 新建一个维度。详细对比见 4.4.5。

#### 4.5.1.7 第 6 节 `matmul`

二维矩阵乘 vs 批量矩阵乘的 shape 规则。详细讲解见 4.4.6。

#### 4.5.1.8 第 7 节 手写 scaled dot-product attention(对应大纲第 7 条)

完整公式(动机铺垫见 4.3.4):

```python
def manual_attention(q, k, v, valid_mask):
    hidden_dim = q.shape[-1]

    # 1. QK^T: (B, S, H) @ (B, H, S) → (B, S, S)
    scores = torch.matmul(q, k.transpose(-2, -1))

    # 2. scaled
    scores = scores / math.sqrt(hidden_dim)

    # 3. mask: (B, S) → (B, 1, S) 广播到 (B, S, S)
    key_mask = valid_mask.unsqueeze(1)
    scores = scores.masked_fill(~key_mask, float("-inf"))

    # 4. softmax(dim=-1) 每行权重和为 1
    weights = torch.softmax(scores, dim=-1)

    # 5. 加权求和: (B, S, S) @ (B, S, H) → (B, S, H)
    context = torch.matmul(weights, v)

    return context, weights
```

公式本质:`context = softmax(QK^T / √d + mask) @ V`。前面章节的 `Q @ K^T → (B, S, S)` 这一步是核心,但**只有 softmax、scaled、mask 才是完整的注意力**。

#### 4.5.1.9 第 8 节 与官方 SDPA 对齐

`F.scaled_dot_product_attention` 是 PyTorch 2.0+ 的生产实现。手写版本必须和它对齐才能证明公式没写错:

```python
import torch.nn.functional as F

official = F.scaled_dot_product_attention(q, k, v, attn_mask=float_mask)
manual = my_manual_attention(q, k, v, mask)

max_diff = (official - manual).abs().max()    # 本例里应该 ≤ 1e-6
```

`01_demo.py` 第 8 节输出 `max_abs_diff: 9e-08`,说明公式完全对齐。

#### 4.5.1.10 第 9 节 多头注意力的拆分与合并(对应大纲第 9 条)

完整 shape 流转图见 4.3.8,这里给代码骨架:

```python
batch_size, seq_len, hidden_dim = embeddings.shape
num_heads = 2
head_dim = hidden_dim // num_heads

# 拆头: (B, S, H) → (B, S, Heads, Head_dim) → (B, Heads, S, Head_dim)
q = embeddings.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

# 每个 head 内部独立做 SDPA
scores = torch.matmul(q, q.transpose(-2, -1)) / math.sqrt(head_dim)

# mask 形状升到 (B, 1, 1, S) 才能广播到 (B, Heads, S, S)
key_mask = valid_mask[:, None, None, :]
scores = scores.masked_fill(~key_mask, float("-inf"))

weights = torch.softmax(scores, dim=-1)
context_per_head = torch.matmul(weights, q)      # (B, Heads, S, Head_dim)

# 合头: (B, Heads, S, Head_dim) → (B, S, Heads, Head_dim) → (B, S, H)
merged = context_per_head.transpose(1, 2).contiguous().reshape(
    batch_size, seq_len, hidden_dim,
)
```

**多头为什么有效**(对应 4.3.5):每个 head 学到的"关注模式"不一样(主谓关系、长距离依赖等),最后拼回 hidden_dim 等价于把多种模式融合。

### 4.5.2 `examples/02_mask_compare.py` —— mask 演示

对应第 7 节的拓展:改 `valid_mask`,肉眼对照 3 条样本的 `(seq, seq)` 注意力权重矩阵,把被 mask 屏蔽的 padding 位置打上 `*` 号,直接看权重流向。

**关键输出**:A002 的注意力权重矩阵里,被 mask 屏蔽的列全部为 0 ——证明 mask 起作用了。

### 4.5.3 `examples/03_batch_size_compare.py` —— batch shape 演示

对应第 6、7 节的拓展:分别用 1/2/3 条样本跑一遍,只动 batch_size 这一维,其余维度不变。

```python
for n in (1, 2, 3):
    sub["samples"] = sub["samples"][:n]
    embeddings, tokens, valid_mask = build_embedding_tensor(sub)
    print(f"embeddings.shape = {tuple(embeddings.shape)}")
```

**直观体会**:batch 维只动最外层,中间维度(seq / hidden)不动。这是为什么 matmul 规则是"只对最后两维做矩阵乘,前面视为 batch"。

## 4.6 进阶知识

### 4.6.1 广播规则回顾

```text
右对齐;每个维度要么相等、要么其中一个为 1、要么不存在
```

mask 升维时常见的几种 shape 推导:

```python
valid_mask.shape              # (B, S)
valid_mask.unsqueeze(1)       # (B, 1, S)    ← 单头 attention 用
valid_mask[:, None, None, :]  # (B, 1, 1, S)  ← 多头 attention 用
```

### 4.6.2 `contiguous()` 的代价

transpose 不复制,只改 stride。`contiguous()` **会真的复制一份**让它变连续。如果在训练循环里反复 transpose + contiguous,显存可能被打爆。生产代码优先用 `F.scaled_dot_product_attention`,它内部帮你处理好了。

## 4.7 常见坑

| 现象 | 原因 | 修复 |
|------|------|------|
| `RuntimeError: view size is not compatible with tensor's size and stride` | transpose 后直接 view | 先 `.contiguous()` 再 view,或直接 `.reshape()` |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | Q 和 K 最后两维不对齐 | 检查 K 是否 `transpose(-2, -1)` |
| padding 位置权重不是 0 | mask 没 unsqueeze 到能广播的形状 | `(B, S)` → `(B, 1, S)`(单头)/ `(B, 1, 1, S)`(多头) |
| 拆头后 view 报错 | heads_first 不连续 | `.transpose(1, 2).contiguous()` 再 reshape |
| softmax 输出全是 0.25 | mask 形状错了,把有效位置也屏蔽了 | 检查 `~key_mask` 而不是 `key_mask` |
| 多头 attention 报 shape 对不上 | `head_dim` 不整除 `hidden_dim` | 检查 `hidden_dim % num_heads == 0` |

## 4.8 学习自检

- [ ] 能解释注意力机制要解决 RNN/CNN 的什么痛点(见 4.3.1)
- [ ] 能说清 embedding 是什么,以及 embedding 和 tensor 的关系(见 4.3.3)
- [ ] 能说清 Q / K / V 三个矩阵各自的语义和类比(见 4.3.2)
- [ ] 能解释 Scaled Dot-Product Attention 公式中每一步的作用(见 4.3.4)
- [ ] 能解释多头注意力为什么是"白嫖"——参数量几乎不变却多 8 倍视角(见 4.3.6)
- [ ] 能解释 `hidden_dim = num_heads × head_dim` 三者各自代表什么(见 4.3.7)
- [ ] 能解释 `(B, S, H) @ (H, H) → (B, S, H)` 的广播语义(见 4.4.6)
- [ ] 能口述 `reshape → transpose → contiguous` 三步为什么缺一不可(见 4.4.3)
- [ ] 能手写 `scaled dot-product attention` 并和官方 SDPA 对齐到 1e-6(见 4.5.1.8)
- [ ] 能解释多头注意力为什么用 `head_dim = hidden_dim / num_heads`(见 4.3.6)
- [ ] 能解释 mask 形状从 `(B, S)` 怎么一步步升到 `(B, 1, 1, S)`(见 4.3.8)

## 4.9 下一步

进入 [`05-pretrained-models/doc/05-pretrained-models.md`](../../05-pretrained-models/doc/05-pretrained-models.md) 看预训练模型 —— 真实 backbone 都是卷积 + 注意力的混合架构,这章的矩阵流在 ViT / Swin Transformer 里会直接复用。