"""04-7 手写 scaled dot-product attention

对应大纲第 7 条(难点,也是这一章最重要的代码)。

公式:
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k) + mask) @ V

本文件覆盖 5 步:
    1. QK^T          (B, S, H) @ (B, H, S) → (B, S, S)
    2. scaled        除以 sqrt(hidden_dim) 防止数值过大
    3. mask          padding 位置填 -inf
    4. softmax       每行变成概率分布
    5. 加权求和       权重 @ V 得到 context

运行方式:
    .venv\\Scripts\\python.exe 04-attention\\examples\\07_manual_attention.py
"""
import math

import torch


# ---------- 1. 准备数据 ----------
# 3 条样本,每条 4 个 token,hidden_dim=4,A002/A003 带 [PAD]
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
        [True, True, False, False],   # A002 后两位 [PAD]
        [True, True, True, False],    # A003 第 4 位 [PAD]
    ],
    dtype=torch.bool,
)  # (3, 4)


# ---------- 2. 固定投影矩阵 Wq/Wk/Wv ----------
# 教学时用固定矩阵,避免随机性导致每次输出不同。
# Wq 用单位矩阵 → Q ≈ embeddings,学生容易验证。
# Wk/Wv 用接近对角的固定矩阵 → 课堂输出稳定。
def projection_weights(hidden_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造固定的 Wq、Wk、Wv。"""
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


def manual_attention(
    embeddings: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """手写 scaled dot-product attention。

    参数:
        embeddings: (batch, seq, hidden) 的输入张量
        valid_mask: (batch, seq) 的 bool mask,True = 有效 token

    返回:
        context: (batch, seq, hidden) 上下文向量
        weights: (batch, seq, seq)    注意力权重(可观察)
    """
    hidden_dim = embeddings.shape[-1]
    w_q, w_k, w_v = projection_weights(hidden_dim)

    # ---------- 步骤 1: X @ W 得到 Q、K、V ----------
    # Q、K、V 都从同一个 X 投影而来(self-attention),形状都是 (B, S, H)
    q = embeddings @ w_q
    k = embeddings @ w_k
    v = embeddings @ w_v
    print(f"  Q.shape = {tuple(q.shape)}")
    print(f"  K.shape = {tuple(k.shape)}")
    print(f"  V.shape = {tuple(v.shape)}")

    # ---------- 步骤 2: Q @ K^T 得到注意力分数 ----------
    # 分数矩阵 (B, S, S) 表示"谁看谁":scores[b, i, j] 是第 b 句 i 对 j 的相关度。
    # 用 -2, -1 写法是为了对任意维度都通用(以后到多头 4D 也能用)。
    raw_scores = torch.matmul(q, k.transpose(-2, -1))
    print(f"  raw_scores.shape = {tuple(raw_scores.shape)}  ← (B, S, S)")

    # ---------- 步骤 3: 缩放 (scaled) ----------
    # 点积的方差会随 hidden_dim 线性增长。除以 √d 后方差归一化到 ~1,
    # softmax 输出不会过尖也不会过平。
    # 详细数学动机见 doc/04-attention.md 第 4.4.7 节。
    scaled_scores = raw_scores / math.sqrt(hidden_dim)
    print(f"  scaled_scores = raw_scores / sqrt({hidden_dim}) = raw_scores / {math.sqrt(hidden_dim):.3f}")

    # ---------- 步骤 4: mask 屏蔽 padding ----------
    # valid_mask 形状 (B, S),加一维变 (B, 1, S),可以广播到 (B, S, S) 的 key 维。
    # padding 位置填 -inf,softmax 后该位置权重自动为 0。
    key_mask = valid_mask.unsqueeze(1)
    print(f"  key_mask.shape = {tuple(key_mask.shape)}  ← (B, 1, S) 广播到 (B, S, S)")

    masked_scores = scaled_scores.masked_fill(~key_mask, float("-inf"))
    print(f"  masked_scores.shape = {tuple(masked_scores.shape)}  ← padding 位置填 -inf")

    # ---------- 步骤 5: softmax ----------
    # dim=-1: 每行变成概率分布,和为 1。
    attention_weights = torch.softmax(masked_scores, dim=-1)
    print(f"  attention_weights.shape = {tuple(attention_weights.shape)}")

    # ---------- 步骤 6: 加权求和 ----------
    # 注意力权重 @ V:每个位置拿到"融合了所有 key 信息"的新表示。
    context = torch.matmul(attention_weights, v)
    print(f"  context.shape = {tuple(context.shape)}")

    return context, attention_weights


def main() -> None:
    print("=== 手写 scaled dot-product attention ===")
    print(f"  embeddings.shape = {tuple(embeddings.shape)}")
    print(f"  valid_mask.shape = {tuple(valid_mask.shape)}\n")

    print("--- 走 5 步公式 ---")
    context, weights = manual_attention(embeddings, valid_mask)

    print("\n=== 关键检查 ===")
    # 1. 形状检查
    assert context.shape == embeddings.shape, "context 形状应该和输入一致"
    assert weights.shape == (3, 4, 4), "权重形状应该是 (B, S, S)"

    # 2. 权重和 = 1
    weight_sums = weights.sum(dim=-1)
    print(f"  weights.sum(dim=-1) 每行和: {weight_sums.tolist()}")

    # 3. padding 位置的权重应该全是 0
    pad_weight_a002 = weights[1, 0, -1]  # A002 query[0] 看最后一个 key(是 [PAD])
    pad_weight_a003 = weights[2, 0, -1]
    print(f"  A002 query[0] 看 [PAD] 的权重 = {pad_weight_a002.item():.4f} (期望 0)")
    print(f"  A003 query[0] 看 [PAD] 的权重 = {pad_weight_a003.item():.4f} (期望 0)")

    print("\n=== 结论 ===")
    print("  - context.shape == embeddings.shape: attention 不改输入形状")
    print("  - 权重矩阵每行和 = 1: softmax(dim=-1) 的定义")
    print("  - padding 位置权重 = 0: mask 起作用了")


if __name__ == "__main__":
    main()