"""04-1 练习:手写 multi_head_attention 函数

目标:
    1. 把 examples/09_multi_head.py 里的多头流程抽成一个独立函数:
       def multi_head_attention(x, num_heads, valid_mask=None) -> torch.Tensor
    2. 输入 x 形状 (batch, seq, hidden),输出形状不变
    3. 选做: 用 F.scaled_dot_product_attention 实现内部 SDPA,而不是手写

提示:
    - 拆头用 reshape + transpose + contiguous
    - 合头用 transpose + contiguous + reshape
    - mask 从 (batch, seq) 升到 (batch, 1, 1, seq) 才能广播

参考答案: 见 doc/04-attention.md 第 4.5.9 节
"""
import torch
import torch.nn.functional as F


def multi_head_attention(x: torch.Tensor, num_heads: int, valid_mask=None) -> torch.Tensor:
    # TODO: 在这里实现多头注意力
    batch_size,seq_len,hidden_dim = x.shape
    head_dim = hidden_dim // num_heads

    # 拆多头
    split_heads = x.reshape(batch_size,seq_len,num_heads,head_dim)

    # 交换 seq 和 heads 两个维度
    heads_first = split_heads.transpose(1,2)

    # contiguous() 生成连续副本
    contiguous_heads = heads_first.contiguous()

    # 合并
    merged = contiguous_heads.transpose(1,2).reshape(batch_size,seq_len,hidden_dim)

    return merged
    raise NotImplementedError("请按 doc/04-attention.md 第 4.5.1.10 节实现多头注意力")


if __name__ == "__main__":
    # 最小测试
    x = torch.randn(2, 4, 8)
    out = multi_head_attention(x, num_heads=2)
    print(out.shape)  # 期望 torch.Size([2, 4, 8])