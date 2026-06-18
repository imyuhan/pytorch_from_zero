"""04-2 练习:实现 causal mask(单向注意力)

目标:
    1. 给 valid_mask 加上 causal mask(下三角):
       - 每个 token 只能关注自己和之前的 token
       - 位置 i 不能看到位置 i+1, i+2, ...
    2. 形状: (seq, seq) 的 bool 矩阵,上三角为 False
    3. 把它和 02_demo.py 第 7 节的手写 attention 组合,
       验证:位置 i 的输出权重,只在 0..i 上非零

提示:
    - torch.tril(torch.ones(seq, seq), diagonal=0) 生成下三角
    - 把 causal_mask 和 valid_mask 做 AND
    - masked_fill(~combined_mask, -inf)

参考答案: 见 doc/04-attention.md 第 4.3.8 节
"""
import torch


def make_causal_mask(seq_len: int, device=None) -> torch.Tensor:
    """生成 (seq, seq) 下三角 mask,True 表示允许关注。"""
    # TODO: 在这里实现 causal mask
    raise NotImplementedError("请用 torch.tril 生成下三角 mask")


if __name__ == "__main__":
    m = make_causal_mask(4)
    print(m)
    # 期望:
    # tensor([[ True, False, False, False],
    #         [ True,  True, False, False],
    #         [ True,  True,  True, False],
    #         [ True,  True,  True,  True]])