# 04 - 维度变换与注意力矩阵(教学文档)

> 承接 01 章张量基础,把 `reshape / transpose / matmul` 用到极致——把 Transformer 注意力机制的矩阵流**从头手算**一遍。本节用 3 条手写的小样本,所有 shape 都可以肉眼心算验证。

## 4.1 涉及的文件

| 文件 | 主题 |
|------|------|
| `examples/02_demo.py` | 端到端跑完 9 个小实验(主入口) |
| `examples/03_mask_compare.py` | 改 valid_mask 后看 A002 注意力权重矩阵的变化 |
| `examples/04_batch_size_compare.py` | 用 1/2/3 条样本跑,对照 batch_size 变化 |
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

## 4.3 基础知识

### 4.3.1 文本张量的标准形状

```text
embeddings.shape = (batch_size, seq_len, hidden_dim)
```

| 维度 | 含义 | 本节样本值 |
|------|------|------|
| `batch_size` | 一次送进模型的句子数 | 3 |
| `seq_len` | 每句 token 数 | 4 |
| `hidden_dim` | 每个 token 的向量长度 | 4 |

`batch_size=3` 来自 `lesson_data.json` 的 3 条样本;`seq_len=4` 是因为 `[PAD]` 也算一个 token;`hidden_dim=4` 是**故意做小**,方便手算。

### 4.3.2 升维 / 降维: `unsqueeze` / `squeeze`

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

### 4.3.3 拆多头: `reshape` + `transpose` + `contiguous`

多头注意力要把 `hidden_dim` 拆成 `num_heads` 和 `head_dim`。这一步是**最容易卡住**的地方。

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

**为什么要 transpose?** 每个 head 要独立算 QK^T,所以 heads 维必须**在 seq 前面**,这样矩阵乘法才能按 head 并行。

**为什么要 contiguous?** transpose 只是改 stride,不动数据,内存布局变成不规则的。后续如果再 `reshape` / `view` 会报错。

### 4.3.4 `expand` vs `repeat`

| 操作 | 复制数据吗? | 典型用途 |
|------|------|------|
| `expand` | ❌ 只读视图(零拷贝) | 位置偏置、广播场景 |
| `repeat` | ✅ 真实复制 | 需要写、序列化场景 |

```python
position_bias = torch.arange(4).reshape(1, 4, 1).float()
expanded = position_bias.expand(3, 4, 4)   # (3, 4, 4) 共享视图
repeated = position_bias.repeat(3, 1, 4)   # (3, 4, 4) 真实拷贝
```

### 4.3.5 `cat` vs `stack`

| 操作 | 沿哪一维 | 输出 shape |
|------|------|------|
| `cat([a, b], dim=0)` | 已有第 0 维拼接 | `(a[0]+b[0], ...)` |
| `stack([a, b], dim=0)` | 新建第 0 维 | `(2, a[0], ...)` |

口诀:**cat 是加长,stack 是加厚**。

### 4.3.6 矩阵乘法 `matmul` 的 shape 规则

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

### 4.3.7 缩放:为什么除以 `√hidden_dim`

点积的方差会随 hidden_dim 线性增长。`hidden_dim=64` 时点积方差 64,`hidden_dim=512` 时方差 512。数值太大时,softmax 会把所有概率压到一个 token 上,梯度消失。

除以 `√d` 后,方差归一化到 ~1,softmax 输出不会过尖也不会过平。

```python
scaled_scores = raw_scores / math.sqrt(hidden_dim)
```

### 4.3.8 注意力 mask

真实场景里,一个 batch 里的句子长度不一样,短的会 `[PAD]` 到同样长度。**模型不应该关注 [PAD] 位置**,否则会学到噪声。

做法:padding 位置填 `-inf`,softmax 之后该位置权重自动变成 0:

```python
valid_mask = torch.tensor([[True, True, True, False]])   # 第 4 位是 [PAD]
key_mask = valid_mask.unsqueeze(1)                       # (1, 1, 4) 广播到 (B, S, S)

scores = scores.masked_fill(~key_mask, float("-inf"))
weights = torch.softmax(scores, dim=-1)
# weights[..., -1] 全是 0  ← 模型不再关注 [PAD]
```

## 4.4 手写 Scaled Dot-Product Attention(对应大纲第 7 条)

完整公式:

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

### 4.4.1 和官方函数对齐

`F.scaled_dot_product_attention` 是 PyTorch 2.0+ 的生产实现,**手写版本必须和它对齐**才能证明公式没写错:

```python
import torch.nn.functional as F

official = F.scaled_dot_product_attention(q, k, v, attn_mask=float_mask)
manual = my_manual_attention(q, k, v, mask)

max_diff = (official - manual).abs().max()    # 本例里应该 ≤ 1e-6
```

`examples/02_demo.py` 第 8 节输出 `max_abs_diff: 9e-08`,说明公式完全对齐。

## 4.5 多头注意力(对应大纲第 9 条)

### 4.5.1 拆头 + 合头

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

### 4.5.2 多头为什么有效

每个 head 有自己的 `head_dim` 维向量,学到的"关注模式"不一样:
- 有的 head 学"上一个 token 看下一个 token"
- 有的 head 学"主语看谓语"
- 有的 head 学"长距离依赖"

最后把多个 head 的输出**拼回 hidden_dim**,等价于把多种关注模式融合。

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
| padding 位置权重不是 0 | mask 没 unsqueeze 到能广播的形状 | `(B, S)` → `(B, 1, S)` |
| 拆头后 view 报错 | heads_first 不连续 | `.transpose(1, 2).contiguous()` 再 reshape |
| softmax 输出全是 0.25 | mask 形状错了,把有效位置也屏蔽了 | 检查 `~key_mask` 而不是 `key_mask` |

## 4.8 学习自检

- [ ] 能解释 `(B, S, H) @ (H, H) → (B, S, H)` 的广播语义
- [ ] 能口述 `reshape → transpose → contiguous` 三步为什么缺一不可
- [ ] 能手写 `scaled_dot-product attention` 并和官方 SDPA 对齐到 1e-6
- [ ] 能解释多头注意力为什么用 `head_dim = hidden_dim / num_heads`
- [ ] 能解释 mask 形状从 `(B, S)` 怎么一步步升到 `(B, 1, 1, S)`

进入 [`05-pretrained-models/doc/05-pretrained-models.md`](../../05-pretrained-models/doc/05-pretrained-models.md) 看预训练模型 —— 真实 backbone 都是卷积 + 注意力的混合架构,这章的矩阵流在 ViT / Swin Transformer 里会直接复用。