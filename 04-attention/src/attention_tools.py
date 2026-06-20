"""K012 的可运行课堂工具函数。

本文件故意把每个知识点拆成一个小函数，方便教师投屏时逐段演示。
所有注释使用中文，便于初学者直接阅读源码。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def print_section(title: str) -> None:
    """打印分隔线，让课堂输出更容易定位。"""
    line = "=" * 82
    print("\n" + line)
    print(title)
    print(line)


def load_lesson_data(path: Path) -> dict[str, Any]:
    """读取本节 JSON 数据。"""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def tensor_profile(name: str, tensor: torch.Tensor) -> str:
    """返回 Tensor 最关键的调试信息。"""
    return (
        f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"device={tensor.device}, contiguous={tensor.is_contiguous()}"
    )


def build_embedding_tensor(
    data: dict[str, Any],
) -> tuple[torch.Tensor, list[list[str]], torch.Tensor]:
    """把 JSON 中的手写 embedding 转成三维 Tensor。

    输出 embeddings 的形状是：
        (batch_size, seq_len, hidden_dim)
    在本节中就是：
        (3, 4, 4)
    其中 batch_size 等于 data["samples"] 的样本数量（当前 3 条），由 data.json 决定。
    """
    # tokens 保留中文词，方便教师投屏时把 Tensor 行列和原句对应起来。
    tokens = [sample["tokens"] for sample in data["samples"]]

    # embeddings 是真正参与矩阵运算的数据。
    # 注意力无法直接处理字符串，所以这里把手写 embedding 转成 float32 Tensor。
    embeddings = torch.tensor(
        [sample["embeddings"] for sample in data["samples"]],
        dtype=torch.float32,
    )

    # valid_mask 标记哪些 token 有效。True 表示真实 token，False 表示 [PAD]。
    # 后面做 attention mask 时会用它屏蔽 padding 位置。
    valid_mask = torch.tensor(
        [sample["valid_mask"] for sample in data["samples"]],
        dtype=torch.bool,
    )
    return embeddings, tokens, valid_mask


def unsqueeze_squeeze_demo(embeddings: torch.Tensor) -> dict[str, Any]:
    """演示升维和降维：给单条句子补 batch 维。"""
    # embeddings[0] 取出第一条句子，形状从三维变成二维：(seq, hidden)。
    sentence = embeddings[0]

    # 模型接口通常要求输入有 batch 维。
    # unsqueeze(0) 在最前面补一个长度为 1 的维度，表示“这一批只有 1 条句子”。
    with_batch = sentence.unsqueeze(0)

    # 这里故意再补一个维度，让学生看到 dim=1 会插在 batch 和 seq 中间。
    with_extra = with_batch.unsqueeze(1)

    # squeeze(1) 只删除第 1 维，避免误删其他长度为 1 的维度。
    squeezed_back = with_extra.squeeze(1)
    return {
        "sentence": tensor_profile("sentence", sentence),
        "with_batch_unsqueeze_0": tensor_profile("with_batch", with_batch),
        "with_extra_unsqueeze_1": tensor_profile("with_extra", with_extra),
        "squeezed_dim_1": tensor_profile("squeezed_back", squeezed_back),
        "课堂提醒": "squeeze() 不指定 dim 时会删除所有长度为 1 的维度，初学阶段建议显式写 dim。",
    }


def reshape_transpose_demo(embeddings: torch.Tensor, num_heads: int) -> dict[str, Any]:
    """演示把 hidden_dim 拆成 num_heads 和 head_dim。"""
    # 三维文本张量的标准读法：(batch, seq, hidden)。
    batch_size, seq_len, hidden_dim = embeddings.shape

    # 每个 head 分到多少维由 hidden_dim / num_heads 决定。
    # 这里数据刻意设置成 4 和 2，方便学生手算。
    head_dim = hidden_dim // num_heads

    # reshape 的目的：把 hidden_dim 拆成 num_heads 和 head_dim。
    # 元素总数没有变，只是把最后一个维度拆成两个维度。
    split_heads = embeddings.reshape(batch_size, seq_len, num_heads, head_dim)

    # attention 通常希望 heads 在 seq 前面，方便每个 head 独立算注意力。
    # transpose(1, 2) 交换 seq 和 heads 两个维度。
    heads_first = split_heads.transpose(1, 2)

    # transpose 后内存布局常常不连续。
    # contiguous() 生成连续副本，后面再 reshape/view 更稳妥。
    contiguous_heads = heads_first.contiguous()

    # 合并时先把 heads 放回 seq 后面，再把 heads 和 head_dim 合成 hidden_dim。
    merged = contiguous_heads.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)

    return {
        "original": tensor_profile("embeddings", embeddings),
        "split_heads": tensor_profile("split_heads", split_heads),
        "heads_first": tensor_profile("heads_first", heads_first),
        "heads_first_is_contiguous": heads_first.is_contiguous(),
        "contiguous_heads": tensor_profile("contiguous_heads", contiguous_heads),
        "merged": tensor_profile("merged", merged),
        "max_merge_diff": round(float((merged - embeddings).abs().max()), 8),
    }


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """把 (batch, seq, hidden) 拆成多头形式 (batch, heads, seq, head_dim)。

    参数：
        x: 形状为 (batch_size, seq_len, hidden_dim) 的三维张量。
        num_heads: 多头注意力的头数。

    返回：
        形状为 (batch_size, num_heads, seq_len, head_dim) 的四维张量，
        其中 head_dim = hidden_dim // num_heads。

    抛错：
        ValueError: 当 x 不是三维、hidden_dim 不能被 num_heads 整除、
        或 num_heads 小于 1 时抛出。

    示例：
        >>> x = torch.arange(2 * 3 * 4).reshape(2, 3, 4).float()
        >>> split_heads(x, num_heads=2).shape
        torch.Size([2, 2, 3, 2])
    """
    # 第 1 步：检查输入形状必须是三维。
    # 拆头只对 (batch, seq, hidden) 有意义，其他维度直接报错。
    if x.dim() != 3:
        raise ValueError(
            f"split_heads 期望输入是 3 维 (batch, seq, hidden)，"
            f"实际收到 {x.dim()} 维，shape={tuple(x.shape)}"
        )

    batch_size, seq_len, hidden_dim = x.shape

    # 第 2 步：num_heads 必须是正整数。
    if num_heads < 1:
        raise ValueError(f"num_heads 必须是 >= 1 的整数，当前是 {num_heads}")

    # 第 3 步：hidden_dim 必须能被 num_heads 整除。
    # 否则 head_dim 会出现小数，无法把 hidden 维平均切分。
    if hidden_dim % num_heads != 0:
        raise ValueError(
            f"hidden_dim={hidden_dim} 不能被 num_heads={num_heads} 整除，"
            f"请确保 hidden_dim 是 num_heads 的整数倍"
        )

    head_dim = hidden_dim // num_heads

    # 第 4 步：先 reshape 把 hidden 拆成 (num_heads, head_dim)。
    # 这一步不交换维度顺序，只在最后一维内部切分。
    # 输出形状：(batch, seq, num_heads, head_dim)
    split = x.reshape(batch_size, seq_len, num_heads, head_dim)

    # 第 5 步：transpose(1, 2) 把 num_heads 维提到 seq 维之前。
    # 多头注意力每个 head 要独立算 QK^T，所以 heads 维必须在 seq 前面。
    # 输出形状：(batch, num_heads, seq, head_dim)
    heads_first = split.transpose(1, 2)

    # 第 6 步：contiguous() 让张量在内存里重新连续排列。
    # transpose 后内存步长不再是规则的，下游如果再 reshape/view 会报错。
    # 注意：这一行是浅拷贝，reshape 才会有真正的数据搬运。
    return heads_first.contiguous()


def expand_repeat_demo(embeddings: torch.Tensor) -> dict[str, Any]:
    """比较 expand 和 repeat：一个尽量共享视图，一个真实复制数据。"""
    batch_size, seq_len, hidden_dim = embeddings.shape

    # position_bias 模拟“每个位置一个偏置值”。
    # 初始形状是 (1, seq, 1)，两个长度为 1 的维度都可以被扩展。
    position_bias = torch.arange(seq_len, dtype=torch.float32).reshape(1, seq_len, 1)

    # expand 只是把长度为 1 的维度按目标形状“看成更大”。
    # 它适合只读场景，因为通常不会真实复制所有数据。
    expanded = position_bias.expand(batch_size, seq_len, hidden_dim)

    # repeat 会把数据真实重复出来。
    # 输出形状和 expanded 一样，但内存含义不同。
    repeated = position_bias.repeat(batch_size, 1, hidden_dim)

    return {
        "position_bias": tensor_profile("position_bias", position_bias),
        "expanded": tensor_profile("expanded", expanded),
        "repeated": tensor_profile("repeated", repeated),
        "expanded_has_base": expanded._base is not None,
        "repeated_has_base": repeated._base is not None,
        "课堂提醒": "expand 常用于广播视图，repeat 会真实复制数据；只读场景优先理解 expand。",
    }


def cat_stack_demo(embeddings: torch.Tensor) -> dict[str, Any]:
    """比较 cat 和 stack 的维度变化。"""
    # 取出两条句子，每条都是 (seq, hidden)。
    first = embeddings[0]
    second = embeddings[1]

    # cat 沿已有第 0 维拼接，结果是把 token 序列接长。
    cat_seq = torch.cat([first, second], dim=0)

    # stack 会新建一个第 0 维，结果像重新组成 batch。
    stack_batch = torch.stack([first, second], dim=0)

    # stack 的 dim 不同，新维度插入的位置也不同，shape 会跟着变化。
    stack_choice = torch.stack([first, second], dim=1)
    return {
        "first_sentence": tensor_profile("first", first),
        "cat_dim_0": tensor_profile("cat_seq", cat_seq),
        "stack_dim_0": tensor_profile("stack_batch", stack_batch),
        "stack_dim_1": tensor_profile("stack_choice", stack_choice),
        "课堂提醒": "cat 沿已有维度接长；stack 会新建一个维度。",
    }


def matmul_demo(embeddings: torch.Tensor) -> dict[str, Any]:
    """演示 matmul 在二维和三维张量上的结果。"""
    # 这里先不引入投影矩阵，直接把 embeddings 当作 query 和 key。
    # 目的是单独观察 matmul 的 shape 规则。
    query = embeddings
    key = embeddings

    # 注意力分数需要 QK^T，所以要把 key 的最后两个维度交换。
    # -2 和 -1 表示倒数第二维和倒数第一维，写法适合高维 Tensor。
    key_t = key.transpose(-2, -1)

    # 三维 matmul 会把最后两维当矩阵相乘，前面的 batch 维保留下来。
    scores = torch.matmul(query, key_t)

    # 单句版本用于对比：二维矩阵乘法得到 (seq, seq)。
    single_sentence_scores = torch.matmul(embeddings[0], embeddings[0].transpose(0, 1))

    return {
        "query": tensor_profile("query", query),
        "key_transposed": tensor_profile("key_t", key_t),
        "batched_scores": tensor_profile("scores", scores),
        "single_sentence_scores": tensor_profile("single_scores", single_sentence_scores),
        "scores_first_row": [round(float(value), 4) for value in scores[0, 0]],
    }


def projection_weights(hidden_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造固定投影矩阵 (Wq, Wk, Wv),保证课堂输出稳定。

    这是公开函数(无下划线前缀),因为 examples/02_mask_compare.py
    需要在外部脚本里独立构造 Q/K/V 来打印权重矩阵。

    三个矩阵的设计意图:
        - Wq 使用单位矩阵:方便学生看懂,Q 基本保留原始 embedding
        - Wk / Wv 是固定矩阵:每次运行输出一致,课堂上不用解释随机数导致的差异
    """
    # Wq 使用单位矩阵，方便学生看懂：Q 基本保留原始 embedding。
    scale = torch.eye(hidden_dim, dtype=torch.float32)
    w_q = scale

    # Wk 和 Wv 故意设置成固定矩阵。
    # 这样每次运行输出一致，课堂上不用解释随机数导致的差异。
    w_k = torch.tensor(
        [
            [0.9, 0.1, 0.0, 0.0],
            [0.1, 0.9, 0.0, 0.0],
            [0.0, 0.0, 0.9, 0.1],
            [0.0, 0.0, 0.1, 0.9],
        ],
        dtype=torch.float32,
    )
    w_v = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.1],
            [0.0, 1.0, 0.1, 0.0],
            [0.0, 0.1, 1.0, 0.0],
            [0.1, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    return w_q, w_k, w_v


def manual_attention_demo(
    embeddings: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, Any]:
    """手写 scaled dot-product attention。

    计算顺序：
        1. X 乘投影矩阵得到 Q、K、V
        2. Q 和 K 转置后相乘得到注意力分数
        3. 除以 sqrt(hidden_dim) 做缩放
        4. 用 mask 屏蔽 padding
        5. softmax 得到注意力权重
        6. 权重乘 V 得到上下文向量
    """
    hidden_dim = embeddings.shape[-1]
    w_q, w_k, w_v = projection_weights(hidden_dim)

    # 通过矩阵乘法得到 Q、K、V。
    # 这里不是训练参数，只是固定投影，用来讲清注意力公式。
    q = embeddings @ w_q
    k = embeddings @ w_k
    v = embeddings @ w_v

    # QK^T 计算每个 token 对其他 token 的相关分数。
    # 输出最后两维是 (seq, seq)，表示“谁看谁”。
    raw_scores = torch.matmul(q, k.transpose(-2, -1))

    # 除以 sqrt(hidden_dim) 是 scaled dot-product attention 的缩放步骤。
    # 作用是避免 hidden 维变大时点积分数过大，导致 softmax 太极端。
    scaled_scores = raw_scores / math.sqrt(hidden_dim)

    # valid_mask 原形状是 (batch, seq)。
    # unsqueeze(1) 后变成 (batch, 1, seq)，可以广播到 (batch, seq, seq) 的 key 维。
    key_mask = valid_mask.unsqueeze(1)

    # padding 位置填成 -inf。
    # softmax 之后这些位置的概率会变成 0，模型就不会关注 [PAD]。
    masked_scores = scaled_scores.masked_fill(~key_mask, float("-inf"))

    # softmax(dim=-1) 表示每个 query token 对所有 key token 的权重和为 1。
    attention_weights = torch.softmax(masked_scores, dim=-1)

    # 注意力权重乘 V，得到融合上下文后的 token 表示。
    context = torch.matmul(attention_weights, v)

    return {
        "q_shape": tuple(q.shape),
        "k_shape": tuple(k.shape),
        "v_shape": tuple(v.shape),
        "score_shape": tuple(scaled_scores.shape),
        "attention_weights_shape": tuple(attention_weights.shape),
        "context_shape": tuple(context.shape),
        "first_token_weights": [round(float(value), 4) for value in attention_weights[0, 0]],
        "padding_token_weight": round(float(attention_weights[1, 0, -1]), 4),
    }


def compare_sdpa_with_manual(
    embeddings: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, Any]:
    """对比官方 scaled_dot_product_attention 和手写版本。"""
    hidden_dim = embeddings.shape[-1]
    w_q, w_k, w_v = projection_weights(hidden_dim)

    q = embeddings @ w_q
    k = embeddings @ w_k
    v = embeddings @ w_v

    # 官方 SDPA 接收的 attn_mask 要能广播到注意力分数形状。
    # 这里构造 float mask，有效位置是 0，padding 位置是 -inf。
    key_mask = valid_mask.unsqueeze(1)
    float_mask = torch.zeros(
        (embeddings.shape[0], embeddings.shape[1], embeddings.shape[1]),
        dtype=torch.float32,
    )
    float_mask = float_mask.masked_fill(~key_mask, float("-inf"))

    # dropout_p=0.0 是为了课堂输出稳定。
    # 如果打开 dropout，每次输出可能不同，不利于初学者对照。
    official_context = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=float_mask,
        dropout_p=0.0,
    )

    raw_scores = torch.matmul(q, k.transpose(-2, -1))
    scaled_scores = raw_scores / math.sqrt(hidden_dim)

    # 手写版本加上同一个 float_mask，再做 softmax 和乘 V。
    # 最后和官方函数比较最大差异，用来验证公式是否写对。
    manual_context = torch.softmax(scaled_scores + float_mask, dim=-1) @ v

    return {
        "official_context_shape": tuple(official_context.shape),
        "manual_context_shape": tuple(manual_context.shape),
        "max_abs_diff": round(float((official_context - manual_context).abs().max()), 8),
        "课堂提醒": "这里验证官方函数和手写公式在本例中对齐；生产代码优先使用官方函数。",
    }


def multi_head_demo(
    embeddings: torch.Tensor,
    valid_mask: torch.Tensor,
    num_heads: int,
) -> dict[str, Any]:
    """演示多头注意力中最关键的拆头和合头。"""
    batch_size, seq_len, hidden_dim = embeddings.shape

    # head_dim 表示每个注意力头分到的向量长度。
    # hidden_dim 必须能被 num_heads 整除，否则无法平均拆头。
    head_dim = hidden_dim // num_heads

    # 先 reshape 成 (batch, seq, heads, head_dim)，再 transpose 成
    # (batch, heads, seq, head_dim)，这样每个 head 可以独立计算注意力。
    q = embeddings.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = embeddings.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v = embeddings.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    # 多头版本中，每个 head 都有自己的 (seq, seq) 注意力分数矩阵。
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

    # mask 形状扩展为 (batch, 1, 1, seq)，可以广播到
    # (batch, heads, seq, seq)，屏蔽每个 head 中的 padding key。
    key_mask = valid_mask[:, None, None, :]
    scores = scores.masked_fill(~key_mask, float("-inf"))

    # 每个 head 内部独立做 softmax。
    weights = torch.softmax(scores, dim=-1)

    # 每个 head 得到自己的上下文向量。
    context_per_head = torch.matmul(weights, v)

    # 合头：先把 seq 维放回 heads 前面，再把 heads 和 head_dim 合并成 hidden_dim。
    # contiguous() 的作用是让转置后的 Tensor 拥有连续内存布局，reshape 更稳。
    merged = context_per_head.transpose(1, 2).contiguous().reshape(
        batch_size,
        seq_len,
        hidden_dim,
    )

    return {
        "q_heads_shape": tuple(q.shape),
        "scores_shape": tuple(scores.shape),
        "weights_shape": tuple(weights.shape),
        "context_per_head_shape": tuple(context_per_head.shape),
        "merged_shape": tuple(merged.shape),
        "first_head_first_token_weights": [round(float(value), 4) for value in weights[0, 0, 0]],
    }
