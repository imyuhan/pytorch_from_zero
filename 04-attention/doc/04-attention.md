# 04 - 维度变换与注意力矩阵(教学文档)

> 承接 01 章张量基础,把 `reshape / transpose / matmul` 用到极致——把 Transformer 注意力机制的矩阵流**从头手算**一遍。本节用 3 条手写的小样本,所有 shape 都可以肉眼心算验证。

## 4.1 涉及的文件

`examples/` 下按知识点分了 11 个独立小文件,每个都可以直接 `python 04-attention/examples/NN_xxx.py` 跑通,不依赖 `src/`。详细讲解见 4.5 节。

| 文件 | 对应大纲 | 主题 |
|------|---------|------|
| `examples/01_embedding_tensor.py` | 大纲 1 | 构造 (batch, seq, hidden) 文本张量 + valid_mask |
| `examples/02_unsqueeze_squeeze.py` | 大纲 2 | `unsqueeze` / `squeeze`:补 batch 维、删多余维 |
| `examples/03_reshape_transpose.py` | 大纲 3 | `reshape` + `transpose` + `contiguous` 拆多头 shape 流转 |
| `examples/04_expand_repeat.py` | 大纲 4 | `expand` 广播视图 vs `repeat` 真实复制 |
| `examples/05_cat_stack.py` | 大纲 5 | `cat` 沿已有维度接长 vs `stack` 新建维度 |
| `examples/06_matmul.py` | 大纲 6 | `matmul` 二维矩阵乘 vs 批量矩阵乘的 shape 规则 |
| `examples/07_manual_attention.py` | 大纲 7 | **手写** scaled dot-product attention(QK^T / scaled / mask / softmax / 加权求和) |
| `examples/08_sdpa_compare.py` | 大纲 8 | 手写版与官方 `F.scaled_dot_product_attention` 对齐验证 |
| `examples/09_multi_head.py` | 大纲 9 | 多头拆头 + 合头的完整 shape 流转 |
| `examples/10_mask_visualize.py` | 大纲 7 拓展 | 把 3 条样本的 (seq, seq) 注意力权重矩阵画出来,padding 位置打 * |
| `examples/11_batch_size.py` | 大纲 6/7 拓展 | 分别取 1/2/3 条样本,看 batch 维变化时 shape 怎么动 |
| `data/lesson_data.json` | 数据 | 3 条课堂样本(A001 全有效、A002 后两位 [PAD]、A003 第 4 位 [PAD]) |

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

对应到 `examples/09_multi_head.py` 里的完整多头流程,可以一行行对着上面 6 步 shape 看。

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

> 04 章的 `examples/` 目录下有 11 个独立小文件,**每个文件都可以直接 `python 04-attention/examples/NN_xxx.py` 单独跑通**,完全自包含、只依赖 torch。建议按编号顺序跑(01 → 11),前一个文件建立的概念是后一个文件的基础。

每个小节的"对应章节"会指出**该 example 用到了 4.3 / 4.4 哪些小节的概念**,方便交叉阅读。

---

### 4.5.1 `examples/01_embedding_tensor.py` —— 构造 (batch, seq, hidden) 张量

**对应大纲**:第 1 条(重点)
**对应文档**:4.3.3(embedding 概念)、4.4.1(张量标准形状)
**覆盖的代码量**:约 70 行,运行后输出 3 条样本的 tokens / embeddings 形状 / valid_mask。

#### 它做了什么?

把 `data/lesson_data.json` 里的手写 embedding 转成 `(batch=3, seq=4, hidden=4)` 的三维 Tensor,同时把 valid_mask 转成 `(batch=3, seq=4)` 的 bool Tensor。这是后续所有 example 的"基础数据"。

#### 关键代码拆解

```python
def build_embedding_tensor(data: dict):
    tokens = [sample["tokens"] for sample in data["samples"]]
    embeddings = torch.tensor(
        [sample["embeddings"] for sample in data["samples"]],
        dtype=torch.float32,
    )
    valid_mask = torch.tensor(
        [sample["valid_mask"] for sample in data["samples"]],
        dtype=torch.bool,
    )
    return embeddings, tokens, valid_mask
```

逐行解释:

1. **`tokens = [sample["tokens"] ...]`**:取出每条样本的 token 列表,用中文词是为了和 Tensor 的行号能对上,方便教学时投屏对照。
2. **`torch.tensor(..., dtype=torch.float32)`**:把嵌套 list 转成 Tensor。`float32` 是默认精度,后续矩阵乘法不会因为 dtype 不匹配报错。
3. **`valid_mask` 用 `torch.bool`**:后续 `masked_fill(~mask, -inf)` 会直接取反,bool 类型天然支持位运算。

#### 预期输出(关键几行)

```text
=== 三条样本的 tokens ===
  A001: ['模型', '会', '关注', '重点']
  A002: ['数据', '需要', '对齐', '[PAD]']
  A003: ['查询', '命中', '缓存', '[PAD]']

=== embeddings 张量 ===
  shape      : (3, 4, 4)
  dtype      : torch.float32
  numel()    : 48  (3 * 4 * 4 = 48)

=== valid_mask ===
  A001: [True, True, True, True]
  A002: [True, True, False, False]   ← 后两位 [PAD]
  A003: [True, True, True, False]    ← 第 4 位 [PAD]
```

#### 重点关注

- `embeddings.shape = (3, 4, 4)` 三个数分别对应 batch / seq / hidden,记牢这个顺序,后面 04 章所有 shape 都按这个读。
- `valid_mask` 的 False 位置就是模型不该关注的 `[PAD]`。后面 4.5.7 / 4.5.10 会用它做 attention mask。

---

### 4.5.2 `examples/02_unsqueeze_squeeze.py` —— 补 batch 维、删多余维

**对应大纲**:第 2 条(重点)
**对应文档**:4.4.2(unsqueeze / squeeze)
**覆盖的代码量**:约 100 行,涵盖 unsqueeze(0/1)、squeeze(1)、squeeze()(无参)、以及 squeeze 非 1 维的反例报错。

#### 它做了什么?

构造一个 `(1, 4, 4)` 的小张量(只有 1 条样本),演示怎么把单条句子的形状在 batch 维上做加 / 减。这是个高频操作 —— PyTorch 的模型接口几乎都要求输入有 batch 维,但你手头经常只有单条样本。

#### 关键代码拆解

```python
sentence = embeddings[0]                # (4, 4) — 取出单句,掉了 batch 维
with_batch = sentence.unsqueeze(0)      # (1, 4, 4) — 在最前面补 batch 维
with_extra = with_batch.unsqueeze(1)    # (1, 1, 4, 4) — 在中间插一维(预演多头)
squeezed = with_extra.squeeze(1)        # (1, 4, 4) — 精准删第 1 维
squeezed_all = with_extra.squeeze()     # (4, 4) — 无参:删全部长度 1 的维
```

#### 为什么这一步重要?

模型 forward 的输入约定几乎都是 `(batch, seq, hidden)`。你手头如果是单条句子 `(seq, hidden)`,必须 `unsqueeze(0)` 才能喂进去;反过来模型输出的 batch 维是 1 的时候,你想拿单条结果要 `squeeze(0)`。

#### 反例警告(文件里有专门演示)

```python
sentence.squeeze(0)   # 报错!
# RuntimeError: ... dimension 0 expected to be 1, but got 4
```

`squeeze(dim)` **只在指定 dim 长度为 1 时才生效**,否则报错。这是初学者最常踩的坑 —— 想 squeeze 一个长度为 4 的维来"压平",结果直接报错。

#### 实战建议

- **初学阶段**:始终显式写 `squeeze(dim=...)`,不要用无参 `squeeze()`。因为无参会把所有长度为 1 的维度一起删,可能误删。
- **更安全的写法**:`tensor.unsqueeze(0)` / `tensor.squeeze(0)` 比 `[None, ...]` / `[..., 0]` 更直白。

---

### 4.5.3 `examples/03_reshape_transpose.py` —— 拆多头的 shape 流转

**对应大纲**:第 3 条(难点)
**对应文档**:4.3.6 / 4.3.7(为什么拆多头)、4.4.3(reshape + transpose + contiguous)
**覆盖的代码量**:约 80 行,4 步把 (B, S, H) → (B, H/Heads, S, H/HeadDim),然后再合回去验证。

#### 它做了什么?

把 `(3, 4, 4)` 的输入按 `num_heads=2` 拆成 `(3, 2, 4, 2)` 的多头形式,然后再合回 `(3, 4, 4)`,验证 `max(|merged - embeddings|) = 0`,证明拆合可逆。

#### 关键代码拆解(4 步)

```python
batch_size, seq_len, hidden_dim = 3, 4, 4
num_heads = 2
head_dim = hidden_dim // num_heads   # = 2

# 第 1 步:reshape 把 hidden 维拆成 (num_heads, head_dim)
split = embeddings.reshape(batch_size, seq_len, num_heads, head_dim)
# shape: (3, 4, 2, 2)  —— 只切最后一维的内部布局

# 第 2 步:transpose(1, 2) 把 heads 维提到 seq 维前面
heads_first = split.transpose(1, 2)
# shape: (3, 2, 4, 2)  —— 只改 stride,不动数据
# heads_first.is_contiguous() == False

# 第 3 步:contiguous() 让它变连续
heads_first = heads_first.contiguous()
# shape: (3, 2, 4, 2)  —— 真的复制了一份让内存连续

# 第 4 步:合回去验证
merged = heads_first.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)
# shape: (3, 4, 4)
# max_diff = 0  ← 拆合完全可逆
```

#### 为什么这一步是难点?

多头注意力的"shape 杂技"全在这三步里。完整原理见 4.3.6 / 4.3.7。这里强调两点:

1. **为什么要 transpose(1, 2)?**——让 heads 维换到 seq 前面。这样后续 `Q @ K^T` 时,matrix mul 把 heads 维当作 batch 处理,**每个 head 天然独立并行**。如果不 transpose,你得写 `for head in num_heads:` 才能逐头算,GPU 并行的优势全没了。
2. **为什么要 contiguous()?**——`transpose` 只改 stride 不复制数据,内存布局变成不规则的。下游如果再 `reshape` / `view` 会直接报错。`contiguous()` 真的复制一份让内存连续,后续操作就稳了。

#### 预期输出(关键验证)

```text
max(|merged - embeddings|) = 0.00000000
→ True (拆头再合回去,元素完全一样)
```

---

### 4.5.4 `examples/04_expand_repeat.py` —— 广播视图 vs 真实复制

**对应大纲**:第 4 条(重点)
**对应文档**:4.4.4(expand vs repeat)
**覆盖的代码量**:约 90 行,演示两种操作的可写性、内存含义差异。

#### 它做了什么?

构造一个 `(1, 4, 1)` 的"位置偏置"(模拟每个位置一个偏置值),分别用 `expand` 和 `repeat` 扩展到 `(3, 4, 4)`,对比二者的内存含义和可写性。

#### 关键代码拆解

```python
position_bias = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1)
# shape: (1, 4, 1)  两个长度为 1 的维度都能扩展

expanded = position_bias.expand(3, 4, 4)
# shape: (3, 4, 4)  ← 零拷贝,只读视图
# expanded._base is not None  → True(底层共享内存)

repeated = position_bias.repeat(3, 1, 4)
# shape: (3, 4, 4)  ← 真实复制,占 3*4*4=48 个 float
# repeated._base is not None  → False(独立内存)
```

#### 二者对比

| 操作 | 复制数据吗? | 可写? | 典型用途 |
|------|------|------|------|
| `expand` | ❌ 只读视图(零拷贝) | ❌ 不能 in-place 写 | 位置偏置、广播场景 |
| `repeat` | ✅ 真实复制 | ✅ 可写 | 需要修改、序列化场景 |

#### 反例警告(文件里专门演示)

```python
expanded[0, 0, 0] = 999.0   # 报错!
# RuntimeError: ... expanded tensor, but the underlying memory is not writable
```

`expand` 出来的张量是"共享底层内存的视图",**不能直接 in-place 写**。想改就得 `clone()` 或者用 `repeat`。

#### 实战建议

- 注意力里的 mask、位置编码等**只读场景**:优先 `expand` 省内存。
- 训练里需要更新的参数、需要 `save` 的中间结果:用 `repeat` 或 `clone`。

---

### 4.5.5 `examples/05_cat_stack.py` —— 接长已有维 vs 新建维度

**对应大纲**:第 5 条(重点)
**对应文档**:4.4.5(cat vs stack)
**覆盖的代码量**:约 80 行,演示 cat 在 dim=0/1 的对比,以及 stack 在 dim=0/1/2 的对比。

#### 它做了什么?

取出两条 `(seq=4, hidden=4)` 的样本,演示 cat 和 stack 在不同 dim 下的形状变化。核心结论:**cat 是加长,stack 是加厚**。

#### 关键代码拆解

```python
first = embeddings[0]                # (4, 4)
second = embeddings[1]               # (4, 4)

# cat: 沿已有维度接长,总维度数不变
cat_seq = torch.cat([first, second], dim=0)
# shape: (8, 4)  ← 把 token 序列从 4 接长到 8

# stack: 新建一个维度,总维度数 +1
stack_batch = torch.stack([first, second], dim=0)
# shape: (2, 4, 4)  ← 新建 batch 维,把两条当 batch=2

# stack 的 dim 控制新维度插的位置
stack_mid = torch.stack([first, second], dim=1)
# shape: (4, 2, 4)  ← 新维度插在 seq 和 hidden 之间
```

#### 选谁?

| 场景 | 选谁 |
|------|------|
| 把两段文本拼成一段(变长序列) | `cat` |
| 把两张同 shape 特征图叠成一组 | `stack`(常用 dim=0 或最后一个) |
| 把一组样本打包成 batch | `stack` |
| 把 batch 维拆开(批维 → 样本) | `cat` 沿 dim=0 |

#### 易混淆点

`torch.cat([a, b], dim=0)` 要求 a 和 b **除 dim=0 外其他维度必须完全相同**。如果维度不一致,直接报错(不是广播)。

`torch.stack([a, b], dim=k)` 要求 a 和 b **所有维度完全相同**,然后在 dim=k 处插入新维度。

---

### 4.5.6 `examples/06_matmul.py` —— 矩阵乘法的 shape 规则

**对应大纲**:第 6 条(重点)
**对应文档**:4.3.4(QK^T)、4.4.6(matmul 规则)
**覆盖的代码量**:约 100 行,对比二维矩阵乘和批量矩阵乘,并把第一条样本的注意力分数矩阵打出来。

#### 它做了什么?

把 embeddings 自己当 Q、K(还没引入投影矩阵 Wq/Wk),演示 `Q @ K^T` 在二维和三维张量上的行为差异 —— 这是注意力机制的核心。

#### 关键代码拆解

```python
# 单条样本:二维矩阵乘
single_sentence = embeddings[0]                          # (4, 4)
single_key_t = single_sentence.transpose(0, 1)          # (4, 4)
single_scores = torch.matmul(single_sentence, single_key_t)  # (4, 4)

# 批量矩阵乘
key_t = embeddings.transpose(-2, -1)                    # (3, 4, 4)
scores = torch.matmul(embeddings, key_t)                # (3, 4, 4)
# 等价于 for b in range(3): scores[b] = embeddings[b] @ key_t[b]
```

#### PyTorch matmul 维度规则

1. 二维 @ 二维  →  普通矩阵乘
2. 一维 @ 二维  →  视为行向量乘矩阵
3. 二维 @ 一维  →  视为矩阵乘列向量
4. **N 维(N≥3) @ N 维  →  只对最后两维做矩阵乘,前面的维度视为 batch**
5. 不匹配的广播  →  按广播规则扩(类似加法)

#### 验证:批量版和单句版对齐

```text
scores[0] == single_scores? True
max_diff = 5.96e-08  ← 浮点误差,可以忽略
```

#### 第一条样本的注意力分数矩阵(raw scores,未缩放/未 softmax)

```text
scores[0] =
    [0.95, 0.4,  0.5,  0.67]
    [0.4,  0.85, 0.35, 0.51]
    [0.5,  0.35, 0.95, 0.49]
    [0.67, 0.51, 0.49, 0.9 ]
```

注意:这是 raw scores,**还没除以 √d、还没 softmax、还没 mask** —— 它还不是"注意力权重",只是"原始分数"。完整公式见 4.5.7。

#### -2, -1 写法的好处

`key_t = embeddings.transpose(-2, -1)` 用负数索引的好处:**对任意维度张量(2D/3D/4D/5D)都通用**。等你写多头注意力时,矩阵变成 4D `(B, Heads, S, HeadDim)`,`transpose(-2, -1)` 一样能拿到最后两维。

---

### 4.5.7 `examples/07_manual_attention.py` —— 手写 scaled dot-product attention

**对应大纲**:第 7 条(难点 / 整章核心)
**对应文档**:4.3.4(SDPA 公式)、4.3.8(mask 升维)、4.4.7(为什么除以 √d)、4.4.8(mask)
**覆盖的代码量**:约 130 行,5 步走完完整公式: QK^T → scaled → mask → softmax → 加权求和。

#### 它做了什么?

从手写的 embeddings 出发,自己用 5 步实现完整 SDPA 公式。这是这一章**最重要**的文件 —— 之后的所有注意力变体(causal、cross、multi-head)都是在这 5 步上加东西。

#### 完整 5 步公式 + 代码

```python
def manual_attention(embeddings, valid_mask):
    hidden_dim = embeddings.shape[-1]
    w_q, w_k, w_v = projection_weights(hidden_dim)

    # 步骤 1: Q、K、V 从 X 投影而来(self-attention)
    q = embeddings @ w_q    # (B, S, H)
    k = embeddings @ w_k    # (B, S, H)
    v = embeddings @ w_v    # (B, S, H)

    # 步骤 2: Q @ K^T 得到注意力分数
    raw_scores = torch.matmul(q, k.transpose(-2, -1))  # (B, S, S)

    # 步骤 3: 缩放 —— 除以 sqrt(hidden_dim) 防止 softmax 过尖
    scaled_scores = raw_scores / math.sqrt(hidden_dim)

    # 步骤 4: mask —— valid_mask (B, S) -> (B, 1, S) 广播到 (B, S, S)
    key_mask = valid_mask.unsqueeze(1)                          # (B, 1, S)
    masked_scores = scaled_scores.masked_fill(~key_mask, -inf)  # padding → -inf

    # 步骤 5: softmax(dim=-1) → 每行变成概率分布
    attention_weights = torch.softmax(masked_scores, dim=-1)    # (B, S, S)

    # 步骤 6: 加权求和: weights @ V
    context = torch.matmul(attention_weights, v)                # (B, S, H)

    return context, attention_weights
```

#### 每一步在干什么?

| 步骤 | 公式 | 形状变化 | 解决什么 |
|------|------|---------|---------|
| 1. 投影 | `X @ W_q` | (B, S, H) → (B, S, H) | 把同一个 X 分成三种角色 |
| 2. QK^T | `Q @ K^T` | (B, S, H) → (B, S, S) | 算"谁看谁"的相关度分数 |
| 3. scaled | `scores / √d` | (B, S, S) | 防止 hidden 大时点积数值爆炸 |
| 4. mask | `masked_fill(~mask, -inf)` | (B, S, S) | 屏蔽 padding 位置 |
| 5. softmax | `softmax(dim=-1)` | (B, S, S) | 每行变成概率分布,和为 1 |
| 6. 加权 | `weights @ V` | (B, S, S) @ (B, S, H) → (B, S, H) | 把"相关信息"融合到每个 token |

#### 关键验证(文件最后会断言)

```text
=== 关键检查 ===
  weights.sum(dim=-1) 每行和: [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], ...]
  A002 query[0] 看 [PAD] 的权重 = 0.0000 (期望 0)  ← mask 起作用了
  A003 query[0] 看 [PAD] 的权重 = 0.0000 (期望 0)
```

- **每行和 = 1**:softmax(dim=-1) 的定义。
- **padding 位置权重 = 0**:mask 把 -inf 填进去,softmax(exp(-inf)) = 0。

#### 跟 01 章 05_transformer_qkv.py 的区别

01 章只演示了步骤 2 的 `Q @ K^T → (B, S, S)` —— 那一步得到的**还不是注意力权重**,只是未缩放、未 softmax、未 mask 的原始分数。本章把剩下 4 步全部补齐,才是完整的注意力。

---

### 4.5.8 `examples/08_sdpa_compare.py` —— 与官方 SDPA 对齐验证

**对应大纲**:第 8 条(重点)
**对应文档**:4.3.4(SDPA 公式)
**覆盖的代码量**:约 110 行,演示 float mask 构造、手写 vs 官方结果对比。

#### 它做了什么?

把 4.5.7 的手写 SDPA 公式**再写一遍**,但用 PyTorch 官方的 `F.scaled_dot_product_attention` 跑一次,对比二者输出。生产代码优先用官方函数,但你必须能验证手写公式是对的,生产时切换才放心。

#### 关键代码拆解

```python
import torch.nn.functional as F

# 官方 SDPA 接受 float mask(0 / -inf)
float_mask = torch.zeros(B, S, S, dtype=torch.float32)
float_mask = float_mask.masked_fill(~valid_mask.unsqueeze(1), float("-inf"))

official_context = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=float_mask,
    dropout_p=0.0,   # 课堂稳定输出
)

# 手写版 —— 把 mask 直接加到分数上,等价于先 masked_fill 再 softmax
manual_context = torch.softmax(scaled_scores + float_mask, dim=-1) @ v
```

#### 验证对齐

```text
max(|official - manual|) = 8.94e-08
→ 完全对齐(误差 < 1e-6)
```

8.94e-08 是浮点误差,工程上认为完全相等。如果误差超过 1e-6,**说明公式写错了**,需要回头对照 4.5.7 检查。

#### 为什么生产代码优先用官方 SDPA?

- **自动选最快的实现**:Flash Attention / Memory-Efficient / Math Cudnn,根据硬件和输入大小自动选
- **显存占用更低**:官方实现不用显式存 `(B, S, S)` 注意力分数矩阵(尤其是 S 很长时省很多显存)
- **支持更多参数**:`dropout_p`、`is_causal`、`need_weights`

#### 实战选择

| 阶段 | 用什么 |
|------|------|
| 学习、教学、写博客 | 手写公式(理解每一步在干什么) |
| 比赛、实际项目 | `F.scaled_dot_product_attention`(生产实现) |
| 自定义 attention(比如加相对位置编码) | 手写公式,在公式里加额外项 |

---

### 4.5.9 `examples/09_multi_head.py` —— 多头拆头 + 合头

**对应大纲**:第 9 条(难点)
**对应文档**:4.3.5(为什么多头)、4.3.6(参数量守恒)、4.3.8(多头 shape 流转)
**覆盖的代码量**:约 120 行,把 4.5.3 的拆头和 4.5.7 的 SDPA 串起来,演示完整多头注意力。

#### 它做了什么?

把 `num_heads=2` 嵌入到 SDPA 里,演示:

1. 拆头:`(B, S, H) → (B, Heads, S, HeadDim)`,heads 维在 seq 前面
2. 每个 head 独立做 SDPA:scores 形状从 `(B, S, S)` 变成 `(B, Heads, S, S)`
3. mask 升维:`(B, S) → (B, 1, 1, S)` 才能广播到 `(B, Heads, S, S)`
4. 合头:`(B, Heads, S, HeadDim) → (B, S, Heads, HeadDim) → (B, S, H)`

#### 关键代码拆解(精简版)

```python
batch_size, seq_len, hidden_dim = embeddings.shape
head_dim = hidden_dim // NUM_HEADS  # = 2

# 拆头: (B, S, H) → (B, S, Heads, HeadDim) → (B, Heads, S, HeadDim)
heads_first = (
    embeddings
    .reshape(batch_size, seq_len, NUM_HEADS, head_dim)  # 切最后一维
    .transpose(1, 2)                                    # heads 换到 seq 前面
    .contiguous()                                       # 让内存连续
)
# shape: (3, 2, 4, 2)  ← (B, Heads, S, HeadDim)

# 多头 SDPA(用 heads_first 当 Q/K/V 演示)
scores = torch.matmul(heads_first, heads_first.transpose(-2, -1))  # (3, 2, 4, 4)
scores = scores / math.sqrt(head_dim)

# 多头 mask 形状:(B, S) → (B, 1, 1, S) 才能广播到 (B, Heads, S, S)
key_mask = valid_mask[:, None, None, :]                           # (B, 1, 1, S)
scores = scores.masked_fill(~key_mask, float("-inf"))
weights = torch.softmax(scores, dim=-1)

# 每个 head 拿到自己的上下文
context_per_head = torch.matmul(weights, heads_first)             # (3, 2, 4, 2)

# 合头: (B, Heads, S, HeadDim) → (B, S, Heads, HeadDim) → (B, S, H)
merged = (
    context_per_head
    .transpose(1, 2)         # heads 换回 seq 后面
    .contiguous()
    .reshape(batch_size, seq_len, hidden_dim)
)
# shape: (3, 4, 4)
```

#### mask 升维是必踩坑

单头时 mask 形状 `(B, 1, S)`,多头时要升到 `(B, 1, 1, S)`。中间两个 1 分别对应 heads 维和 query 维 —— 所有 head、所有 query 共享同一份 key mask。如果忘了升维,会报 `RuntimeError: The size of tensor a must match the size of tensor b`。

#### 拆合可逆性 vs 内容变化

文件最后会验证:

```text
形状相同? True
内容差异(应该非 0): max_diff = 0.5748
→ 形状保持一致,但内容被 attention 改变了 —— 这正是 attention 的作用。
```

- **形状相同**:多头 attention 不改输入形状。
- **内容变了**:attention 把每个 token 跟其它 token 融合,得到新的表示。如果 max_diff = 0,反而说明 attention 啥都没干。

#### 多头为什么有效?

参数量守恒(详细数学见 4.3.6),但能学到多种"关注模式"。训练好的 Transformer 里不同 head 学到的东西差异很大 —— 有的看下一个 token,有的看固定位置,有的看远距离指代。

---

### 4.5.10 `examples/10_mask_visualize.py` —— mask 注意力权重矩阵可视化

**对应大纲**:第 7 条的拓展(重点)
**对应文档**:4.4.8(mask)
**覆盖的代码量**:约 130 行,把 3 条样本各自的 (seq, seq) 注意力权重矩阵用整齐的表格打印,padding 位置打 * 号。

#### 它做了什么?

把 4.5.7 算出的注意力权重矩阵**画出来**(以 ASCII 表格的形式),让 mask 的作用肉眼可见 —— 这是验证 mask 是否正确工作的最快方法。

#### 关键代码拆解

```python
def show_attention_matrix(weights, tokens, valid_mask, batch_idx, label):
    print(f"--- {label}（A00{batch_idx+1}）tokens: {tokens[batch_idx]} ---")
    # 打印表头(列方向是 key)
    header = "       " + " ".join(f"{t:>6}" for t in tokens[batch_idx])
    print(header)
    # 逐行打印(行方向是 query)
    for qi, qt in enumerate(tokens[batch_idx]):
        cells = []
        for ki, kt in enumerate(tokens[batch_idx]):
            weight = float(weights[batch_idx, qi, ki])
            mark = "*" if not valid_mask[batch_idx, ki] else " "
            cells.append(f"{weight:5.3f}{mark}")
        print(f"  {qt:>4} " + " ".join(cells))
```

#### 关键输出(对照 A002)

```text
--- 重点: A002(后两位 [PAD],会有 * 号)tokens: ['数据', '需要', '对齐', '[PAD]'] ---
           数据     需要     对齐  [PAD]
    数据   0.551   0.449   0.000* 0.000*
    需要   0.464   0.536   0.000* 0.000*
    对齐   0.507   0.493   0.000* 0.000*
   [PAD]   0.500   0.500   0.000* 0.000*
  (* 号 = 该 key 位置被 mask 屏蔽)
```

可以看到:**A002 后两列(对应 [PAD] token)权重全是 0,后面带 * 号**。这说明 mask 起作用了 —— 模型不再关注 [PAD] 位置。

#### 验证 mask 正确性的最快方法

跑一遍这个文件,肉眼对照 token 看权重矩阵:

- 有效位置:权重在 0~1 之间,和为 1
- padding 位置:权重 = 0,后面带 *
- 同一列的所有权重加起来不一定为 1(因为每行单独 softmax)

如果 padding 列出现非 0 权重,**说明 mask 没写对**,回去检查 `masked_fill(~mask, -inf)`。

---

### 4.5.11 `examples/11_batch_size.py` —— batch 维变化演示

**对应大纲**:第 6、7 条的拓展(重点)
**对应文档**:4.4.1(张量标准形状)、4.4.6(matmul 规则)
**覆盖的代码量**:约 90 行,分别取 1/2/3 条样本跑 `build_embedding_tensor`,对照 shape 变化。

#### 它做了什么?

强调一个关键事实:**batch 维只动最外层,中间维度(seq / hidden)不动**。这是为什么 matmul 规则是"只对最后两维做矩阵乘,前面视为 batch"。

#### 关键代码拆解

```python
for num_samples in (1, 2, 3):
    sub_samples = all_samples[:num_samples]
    embeddings, tokens, valid_mask = build_embedding_tensor(sub_samples)
    print(f"embeddings.shape = {tuple(embeddings.shape)}")
    print(f"valid_mask.shape = {tuple(valid_mask.shape)}")
```

#### 预期输出

```text
>>> 取前 1 条样本
  embeddings.shape = (1, 4, 4)  ← 第一个数就是 batch_size
  valid_mask.shape = (1, 4)     ← 跟 batch 走

>>> 取前 2 条样本
  embeddings.shape = (2, 4, 4)
  valid_mask.shape = (2, 4)

>>> 取前 3 条样本
  embeddings.shape = (3, 4, 4)
  valid_mask.shape = (3, 4)
```

#### 关键观察

1. `batch_size` 从 1 变 3 时,`seq_len=4` 和 `hidden_dim=4` **始终不变**。
2. `embeddings` 和 `valid_mask` 的形状**同步跟着 batch 变**。
3. 这就是为什么 matmul 规则是"只对最后两维做矩阵乘" —— **batch 维就是额外的并行批次,不会影响 seq/hidden 的语义**。

#### 跟 matmul 的关系

```python
# 批量矩阵乘法里,batch 维就是"并行批次"
scores = torch.matmul(embeddings, embeddings.transpose(-2, -1))
# 等价于:
#   for b in range(B):
#       scores[b] = embeddings[b] @ embeddings[b].T
# 但 GPU 一次算完,不需要 Python 循环。
```

这条规则在多头注意力里更明显 —— `(B, Heads, S, S)` 的 scores,matrix mul 把 heads 维也当 batch,天然按 head 并行。

---

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