"""04-10 mask 注意力权重矩阵可视化

对应大纲第 7 条的拓展:把 valid_mask 改了之后,直接打印 3 条样本各自的
(seq, seq) 注意力权重矩阵,把 mask 屏蔽的 padding 位置打上 * 号,
肉眼对照 token 看权重流向。

运行方式:
    .venv\\Scripts\\python.exe 04-attention\\examples\\10_mask_visualize.py
"""
import math

import torch


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

tokens = [
    ["模型", "会", "关注", "重点"],
    ["数据", "需要", "对齐", "[PAD]"],
    ["查询", "命中", "缓存", "[PAD]"],
]


def projection_weights(hidden_dim: int):
    """跟 07、08 文件一致。"""
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


def show_attention_matrix(
    weights: torch.Tensor,
    tokens_list: list[list[str]],
    valid_mask_tensor: torch.Tensor,
    batch_idx: int,
    label: str,
) -> None:
    """打印第 batch_idx 条样本的 (seq, seq) 注意力权重矩阵。

    被 mask 屏蔽的 padding 位置打上 * 号,方便肉眼对照 token 看权重流向。
    """
    print(f"\n--- {label}（A00{batch_idx + 1}）tokens: {tokens_list[batch_idx]} ---")
    # 打印表头(列方向是 key)
    header = "       " + " ".join(f"{t:>6}" for t in tokens_list[batch_idx])
    print(header)
    # 逐行打印(行方向是 query)
    for qi, qt in enumerate(tokens_list[batch_idx]):
        cells: list[str] = []
        for ki, kt in enumerate(tokens_list[batch_idx]):
            weight = float(weights[batch_idx, qi, ki])
            mark = "*" if not valid_mask_tensor[batch_idx, ki] else " "
            cells.append(f"{weight:5.3f}{mark}")
        print(f"  {qt:>4} " + " ".join(cells))
    print("  (* 号 = 该 key 位置被 mask 屏蔽)")


def main() -> None:
    print("=== 当前 valid_mask ===")
    for i, row in enumerate(valid_mask.tolist()):
        print(f"  A00{i + 1}: {row}  ← tokens={tokens[i]}")

    # 走一遍手写注意力的公式
    hidden_dim = embeddings.shape[-1]
    w_q, w_k, w_v = projection_weights(hidden_dim)
    q = embeddings @ w_q
    k = embeddings @ w_k
    v = embeddings @ w_v

    # 1) QK^T
    raw_scores = torch.matmul(q, k.transpose(-2, -1))
    # 2) scaled
    scaled_scores = raw_scores / math.sqrt(hidden_dim)
    # 3) mask
    key_mask = valid_mask.unsqueeze(1)  # (B, 1, S)
    masked = scaled_scores.masked_fill(~key_mask, float("-inf"))
    # 4) softmax
    weights = torch.softmax(masked, dim=-1)

    # 把三条样本各自的权重矩阵画出来
    show_attention_matrix(weights, tokens, valid_mask, 0, "对照: A001(全有效,应该没 * 号)")
    show_attention_matrix(weights, tokens, valid_mask, 1, "重点: A002(后两位 [PAD],会有 * 号)")
    show_attention_matrix(weights, tokens, valid_mask, 2, "对照: A003(第 4 位 [PAD],最后一列 * 号)")

    print("\n=== A002 第 3 行(query='对齐')看每个 key 的权重 ===")
    for ki, kt in enumerate(tokens[1]):
        weight = float(weights[1, 2, ki])
        masked_note = "← 被 mask" if not valid_mask[1, ki] else ""
        print(f"  '对齐' -> '{kt}': {weight:.4f}  {masked_note}")
    print("  → 注意 '对齐' 的权重在 [PAD] 位置都是 0,这是 mask 起作用的证据。")


if __name__ == "__main__":
    main()