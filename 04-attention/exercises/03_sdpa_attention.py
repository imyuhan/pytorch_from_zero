"""04-3 练习:用 F.scaled_dot_product_attention 重写手写 attention

目标:
    1. 抄一遍 examples/07_manual_attention.py 的手写流程,但内部用 F.scaled_dot_product_attention
    2. 输出形状要和手写版本一致
    3. 思考题: 官方版本比手写快多少?为什么?

提示:
    - F.scaled_dot_product_attention 接受 attn_mask=float mask(0 / -inf)
    - 也接受 is_causal=True 参数(等价于下三角 mask)

参考答案: 见 doc/04-attention.md 第 4.5.8 节
"""
import torch
import torch.nn.functional as F


def sdpa_attention(x: torch.Tensor, w_q, w_k, w_v, valid_mask: torch.Tensor) -> torch.Tensor:
    """用官方 SDPA 实现 scaled dot-product attention。

    参数:
        x: (batch, seq, hidden)
        w_q, w_k, w_v: 投影矩阵,形状都是 (hidden, hidden)
        valid_mask: (batch, seq),True 表示有效 token

    返回:
        context: (batch, seq, hidden)
    """
    # TODO: 用 F.scaled_dot_product_attention 实现
    # 步骤一 求 Q,K,V
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v

    # 官方 SDPA 接受 float mask
    # 形状:(batch, seq) -> (batch, 1, seq),加在 Q 的 seq 维上,广播到所有 head
    float_mask = valid_mask.unsqueeze(1)
    float_mask = float_mask.masked_fill(~float_mask, float("-inf"))

    official_context = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=float_mask,
        dropout_p=0.0
    )
    return official_context


if __name__ == "__main__":
    x = torch.randn(2, 4, 8)
    w = torch.eye(8)
    mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
    out = sdpa_attention(x, w, w, w, mask)
    print(out.shape)  # 期望 torch.Size([2, 4, 8])