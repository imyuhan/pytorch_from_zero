"""04-6 matmul:矩阵乘法的 shape 规则

对应大纲第 6 条(重点)。

知识点:
    - 二维 matmul: 普通矩阵乘, (M, K) @ (K, N) → (M, N)。
    - 三维及以上 matmul: 只对最后两维做矩阵乘,前面的维度视为 batch。
    - 注意力的核心就是 Q @ K^T: (B, S, H) @ (B, H, S) → (B, S, S)。

运行方式:
    .venv\\Scripts\\python.exe 03-attention\\examples\\06_matmul.py
"""
import torch

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


def profile(name: str, t: torch.Tensor) -> str:
    return f"{name:30s} shape={tuple(t.shape)}"


def main() -> None:
    # ----------------------------------------------------------------
    # 1. 把 Q、K 当成 embeddings 自己(Q = K = X)
    # ----------------------------------------------------------------
    print("=== 1. 准备 Q 和 K ===")
    print(f"  {profile('Q = embeddings', embeddings)}")
    print(f"  {profile('K = embeddings', embeddings)}")
    print(f"  注意: 还没引入投影矩阵 Wq/Wk,先用 X 自己当 Q、K 演示 matmul。\n")

    # ----------------------------------------------------------------
    # 2. 单条样本的二维矩阵乘
    # ----------------------------------------------------------------
    print("=== 2. 单条样本: 二维矩阵乘 ===")
    single_sentence = embeddings[0]  # (4, 4)
    single_key_t = single_sentence.transpose(0, 1)  # (4, 4)
    single_scores = torch.matmul(single_sentence, single_key_t)  # (4, 4)
    print(f"  {profile('single_sentence', single_sentence)}")
    print(f"  {profile('single_key_t', single_key_t)}")
    print(f"  {profile('single_scores', single_scores)}")
    print(f"  解读: (seq, hidden) @ (hidden, seq) → (seq, seq)")
    print(f"        single_scores[i, j] = 第 i 个 token 对第 j 个 token 的点积分数\n")

    # ----------------------------------------------------------------
    # 3. 批量矩阵乘
    # ----------------------------------------------------------------
    print("=== 3. 批量矩阵乘 ===")
    # K 的最后两维要交换,得到 (B, H, S)
    # 用 -2, -1 写法的好处:不管张量是几维,都能拿到最后两维
    key_t = embeddings.transpose(-2, -1)
    print(f"  {profile('K.transpose(-2,-1)', key_t)}")
    print(f"  解读: -2, -1 表示'倒数第二维'和'倒数第一维',对 3D/4D/5D 都通用\n")

    scores = torch.matmul(embeddings, key_t)
    print(f"  {profile('Q @ K^T (batched)', scores)}")
    print(f"  解读: (B, S, H) @ (B, H, S) → (B, S, S)")
    print(f"        scores[b, i, j] = 第 b 句里第 i 个 token 对第 j 个 token 的分数")
    print(f"        等价于:")
    print(f"          for b in range(B): scores[b] = embeddings[b] @ key_t[b]")
    print(f"        但 GPU 一次算完,不需要 Python 循环。\n")

    # ----------------------------------------------------------------
    # 4. 验证批量版和单句版一致(只看 batch[0])
    # ----------------------------------------------------------------
    print("=== 4. 验证:批量版和单句版对齐 ===")
    print(f"  scores[0] == single_scores? {torch.allclose(scores[0], single_scores)}")
    print(f"  max_diff = {(scores[0] - single_scores).abs().max().item():.2e}\n")

    # ----------------------------------------------------------------
    # 5. 单条样本的注意力分数矩阵(取第一条样本看具体数字)
    # ----------------------------------------------------------------
    print("=== 5. 第 1 条样本的注意力分数矩阵(未缩放、未 softmax) ===")
    print(f"  scores[0] =")
    for row in scores[0].tolist():
        print(f"    {[round(v, 3) for v in row]}")
    print(f"  注意: 这是 raw scores,还没除以 √d、还没 softmax、还没 mask。")
    print(f"        它还不是'注意力权重',只是'原始分数'。\n")

    # ----------------------------------------------------------------
    # 6. 总结:matmul 维度规则
    # ----------------------------------------------------------------
    print("=== 6. PyTorch matmul 维度规则 ===")
    print("  1) 二维 @ 二维  →  普通矩阵乘")
    print("  2) 一维 @ 二维  →  视为行向量乘矩阵")
    print("  3) 二维 @ 一维  →  视为矩阵乘列向量")
    print("  4) N 维(N>=3) @ N 维  →  只对最后两维做矩阵乘,前面的维度视为 batch")
    print("  5) 不匹配的广播 →  按广播规则扩(类似加法)")
    print("  → 注意力分数 scores = Q @ K^T 就是规则 4,只看最后两维。")


if __name__ == "__main__":
    main()