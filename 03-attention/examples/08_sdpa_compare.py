"""04-8 与官方 F.scaled_dot_product_attention 对齐验证

对应大纲第 8 条(重点)。

知识点:
    - PyTorch 官方有 F.scaled_dot_product_attention,生产代码都用它。
    - 自己手写的公式必须和它对齐到 1e-6 才能证明没写错。
    - 官方函数直接接受 float mask(0 / -inf),比手写 + masked_fill 更省事。

运行方式:
    .venv\\Scripts\\python.exe 03-attention\\examples\\08_sdpa_compare.py
"""
import math

import torch
import torch.nn.functional as F


# ---------- 数据 ----------
embeddings = torch.tensor(
    [
        [[0.90, 0.10, 0.20, 0.30],
         [0.20, 0.80, 0.10, 0.40],
         [0.30, 0.20, 0.90, 0.10],
         [0.40, 0.10, 0.30, 0.80]],
        [[0.85, 0.15, 0.25, 0.20],
         [0.15, 0.75, 0.20, 0.30],
         [0.25, 0.20, 0.80, 0.20],
         [0.00, 0.00, 0.00, 0.00]],
        [[0.70, 0.30, 0.40, 0.10],
         [0.30, 0.60, 0.20, 0.50],
         [0.40, 0.20, 0.70, 0.30],
         [0.00, 0.00, 0.00, 0.00]],
    ],
    dtype=torch.float32,
)  # (3, 4, 4)

valid_mask = torch.tensor(
    [
        [True, True, True, True],
        [True, True, False, False],
        [True, True, True, False],
    ],
    dtype=torch.bool,
)  # (3, 4)


def projection_weights(hidden_dim: int):
    """跟 07 文件保持一致的固定投影矩阵。"""
    w_q = torch.eye(hidden_dim, dtype=torch.float32)
    w_k = torch.tensor(
        [[0.9, 0.1, 0.0, 0.0],
         [0.1, 0.9, 0.0, 0.0],
         [0.0, 0.0, 0.9, 0.1],
         [0.0, 0.0, 0.1, 0.9]],
        dtype=torch.float32,
    )
    w_v = torch.tensor(
        [[1.0, 0.0, 0.0, 0.1],
         [0.0, 1.0, 0.1, 0.0],
         [0.0, 0.1, 1.0, 0.0],
         [0.1, 0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    return w_q, w_k, w_v


def main() -> None:
    hidden_dim = embeddings.shape[-1]
    w_q, w_k, w_v = projection_weights(hidden_dim)
    q = embeddings @ w_q
    k = embeddings @ w_k
    v = embeddings @ w_v

    # ========================================================================
    # 1. 构造 float mask —— 官方 SDPA 要求 mask 是 0 / -inf 的 float
    # ========================================================================
    print("=== 1. 构造 float mask ===")
    # 形状 (B, S, S),有效位置是 0,padding 位置是 -inf。
    # valid_mask (B, S) -> unsqueeze(1) -> (B, 1, S),广播到 (B, S, S) 的 key 维。
    key_mask = valid_mask.unsqueeze(1)  # (B, 1, S)

    float_mask = torch.zeros(
        (embeddings.shape[0], embeddings.shape[1], embeddings.shape[1]),
        dtype=torch.float32,
    )
    float_mask = float_mask.masked_fill(~key_mask, float("-inf"))
    print(f"  float_mask.shape = {tuple(float_mask.shape)}")
    print(f"  A002 的 mask 行:")
    for row in float_mask[1].tolist():
        print(f"    {row}")
    print(f"  解读: padding 位置是 -inf,有效位置是 0。\n")

    # ========================================================================
    # 2. 官方 SDPA
    # ========================================================================
    print("=== 2. 官方 F.scaled_dot_product_attention ===")
    # dropout_p=0.0 是为了课堂输出稳定。生产里训练时一般设 0.1。
    official_context = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=float_mask,
        dropout_p=0.0,
    )
    print(f"  official_context.shape = {tuple(official_context.shape)}\n")

    # ========================================================================
    # 3. 手写版
    # ========================================================================
    print("=== 3. 手写版(把公式再写一遍) ===")
    raw_scores = torch.matmul(q, k.transpose(-2, -1))
    scaled_scores = raw_scores / math.sqrt(hidden_dim)
    # 注意: 这里直接用 +float_mask 一步到位,等价于先 masked_fill 再 softmax。
    manual_context = torch.softmax(scaled_scores + float_mask, dim=-1) @ v
    print(f"  manual_context.shape = {tuple(manual_context.shape)}\n")

    # ========================================================================
    # 4. 对齐验证
    # ========================================================================
    print("=== 4. 两者对齐验证 ===")
    max_diff = float((official_context - manual_context).abs().max())
    print(f"  max(|official - manual|) = {max_diff:.2e}")
    if max_diff < 1e-6:
        print(f"  → 完全对齐(误差 {max_diff:.2e} < 1e-6)")
    else:
        print(f"  → 误差 {max_diff:.2e} 偏大,检查公式!")

    # ========================================================================
    # 5. 验证 padding 位置权重
    # ========================================================================
    print("\n=== 5. 验证 padding 位置在 SDPA 里也被屏蔽 ===")
    # 重新算一次 softmax 后的权重,看 padding 列
    weights = torch.softmax(scaled_scores + float_mask, dim=-1)
    print(f"  A002 query[0] 看 [PAD] 的权重 = {weights[1, 0, -1].item():.4f}  (期望 0)")
    print(f"  A002 query[2] 看 [PAD] 的权重 = {weights[1, 2, -1].item():.4f}  (期望 0)")
    print(f"  A003 query[0] 看 [PAD] 的权重 = {weights[2, 0, -1].item():.4f}  (期望 0)")

    # ========================================================================
    # 6. 思考:为什么生产代码优先用官方 SDPA?
    # ========================================================================
    print("\n=== 6. 官方 SDPA 比手写多了什么? ===")
    print("  - 自动选最快的实现:Flash Attention / Memory-Efficient / Math")
    print("  - 显存占用更低(不用显式存 (B, S, S) 注意力分数)")
    print("  - 支持更多参数:dropout、is_causal、need_weights")
    print("  → 教学阶段先手写理解公式,生产阶段必须用 F.scaled_dot_product_attention。")


if __name__ == "__main__":
    main()