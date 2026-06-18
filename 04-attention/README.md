# 04 - 维度变换与注意力矩阵 (Attention Lab)

> **难度**: ⭐⭐⭐
> **前置**: [01-tensor-basics]()
> **预计耗时**: 90 - 120 分钟(配合 [`doc/Attention_Is_All_You_Need_CN.md`](../papers/Attention_Is_All_You_Need_CN.md) 阅读更佳)
> **硬件**: GPU 推荐,CPU 可降级(只是慢)
>
> 上一章数据已经能批量送进模型了。这一章**换一条线**：从零手算一遍注意力机制的矩阵流，把 01 章那些 `reshape / transpose / matmul` 用到极致。
>
> 学完你应该能**不看公式文档**，把 scaled dot-product attention 和多头拆头、合头自己写出来，并和 `F.scaled_dot_product_attention` 对齐验证。
>
> 📖 详细讲解看 [`doc/04-attention.md`](doc/04-attention.md)
> 📄 配套论文:[`papers/Attention_Is_All_You_Need_CN.md`](../papers/Attention_Is_All_You_Need_CN.md)

## 核心概念

| 术语 | 一句话解释 |
|------|-----------|
| **Q / K / V** | 同一组 embedding 经过三个不同投影矩阵得到的查询、键、值向量 |
| **scaled dot-product attention** | `softmax(QK^T / √d) · V`,Transformer 注意力最基础的公式 |
| **scaled** | 除以 `√hidden_dim`,防止点积数值过大让 softmax 过于极端 |
| **attention mask** | 把 padding 等无效位置填 `-inf`,让 softmax 之后该位置权重为 0 |
| **multi-head attention** | 把 hidden 维拆成 `num_heads` 份,每份独立算注意力,最后合回 |
| **head_dim** | 每个注意力头分到的向量长度,`head_dim = hidden_dim / num_heads` |
| **broadcasting** | `(batch, 1, seq)` 自动扩展到 `(batch, seq, seq)`,mask 必备技巧 |
| **expand vs repeat** | expand 是只读视图(零拷贝),repeat 是真实复制 |
| **cat vs stack** | cat 沿已有维度拼接,stack 新增一个维度堆叠 |

## 课程大纲

| 序号 | 主题 | 重点 | 难点 |
|------|------|:----:|:----:|
| 1 | 构造 `(batch, seq, hidden)` 文本张量 + 手写 embedding | ✅ |  |
| 2 | `unsqueeze` / `squeeze`:补 batch 维、删多余维 | ✅ |  |
| 3 | `reshape` + `transpose` + `contiguous`:**拆多头** |  | ✅ |
| 4 | `expand` vs `repeat`:广播视图 vs 真实复制 | ✅ |  |
| 5 | `cat` vs `stack`:接长已有维 vs 新建维度 | ✅ |  |
| 6 | `matmul`:二维矩阵乘和批量矩阵乘的 shape 规则 | ✅ |  |
| 7 | **手写 scaled dot-product attention**(Q/K/V、缩放、mask、softmax) |  | ✅ |
| 8 | 与官方 `F.scaled_dot_product_attention` 对齐 | ✅ |  |
| 9 | **多头注意力的拆头 + 合头 shape 流转** |  | ✅ |

## 涉及的文件

| 文件 | 主题 |
|------|------|
| `examples/02_demo.py` | 端到端跑完 9 个小实验 |
| `examples/03_mask_compare.py` | 改 `valid_mask` 后,肉眼对照 3 条样本的注意力权重矩阵 |
| `examples/04_batch_size_compare.py` | 用 1 / 2 / 3 条样本跑一遍,看 batch_size 变化时 shape 怎么动 |
| `src/attention_tools.py` | 课堂演示用的工具函数库(load_lesson_data、split_heads、手写 SDPA 等) |
| `data/lesson_data.json` | 3 条课堂样本(A001/A002/A003),第 2、3 条带 `[PAD]` 用于演示 mask |

## 最小示例(对应大纲第 7 条:手写 attention)

```python
import torch
import math

# Q、K、V 形状统一为 (batch, seq, hidden)
q = torch.randn(2, 4, 8)
k = torch.randn(2, 4, 8)
v = torch.randn(2, 4, 8)

# 1. 注意力分数 = Q · K^T  (B, S, S)
scores = torch.matmul(q, k.transpose(-2, -1))

# 2. scaled: 除以 √hidden_dim
scores = scores / math.sqrt(8)

# 3. mask: padding 位置填 -inf
mask = torch.tensor([[True, True, True, False]])
scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

# 4. softmax(dim=-1) → 每行权重和为 1
weights = torch.softmax(scores, dim=-1)

# 5. 权重 × V → 上下文向量 (B, S, H)
context = torch.matmul(weights, v)
print(context.shape)  # torch.Size([2, 4, 8])
```

## 学习顺序

1. **先把 01 章 `05_transformer_qkv.py` 跑一遍**,建立 `(B, S, H) @ (H, H) → (B, S, H)` 的 shape 直觉
2. 跑 `examples/02_demo.py` —— 9 节小实验每节都很短,跟着输出读 shape
3. 跑 `examples/03_mask_compare.py` —— 把注意力权重矩阵直接打出来,肉眼对照 token 看权重流向
4. 跑 `examples/04_batch_size_compare.py` —— 体会 batch 维只动最外层
5. 进入 05 看预训练模型怎么"用别人训好的 backbone"

## 练习

`exercises/` 下留空,做完 `examples/` 后自己写 3 道小练习(参考答案见对应 doc 章节末):
- 手写一个 `multi_head_attention(Q, K, V, num_heads)` 函数
- 给 mask 加上 causal(下三角),实现单向注意力
- 用 `F.scaled_dot_product_attention` 重写第 7 节的手写版本