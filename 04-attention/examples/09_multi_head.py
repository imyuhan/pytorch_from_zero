"""04-9 多头注意力:拆头 + 合头的 shape 流转

对应大纲第 9 条(难点)。

知识点:
    - 多头 = 把 hidden_dim 拆成 num_heads × head_dim。
    - 拆头两步: reshape(B,S,Hidden)→(B,S,Heads,HeadDim)→transpose(1,2)→(B,Heads,S,HeadDim)。
    - 合头两步: transpose(1,2)→(B,S,Heads,HeadDim)→reshape→(B,S,Hidden)。
    - 多头里 mask 形状要从 (B, S) 升到 (B, 1, 1, S) 才能广播到 (B, Heads, S, S)。
    - heads 维在 seq 前面,matrix mul 会把 heads 当 batch 处理,天然按 head 并行。

运行方式:
    .venv\\Scripts\\python.exe 04-attention\\examples\\09_multi_head.py
"""
import math

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

valid_mask = torch.tensor(
    [
        [True, True, True, True],
        [True, True, False, False],
        [True, True, True, False],
    ],
    dtype=torch.bool,
)  # (3, 4)

NUM_HEADS = 2


def profile(name: str, t: torch.Tensor) -> str:
    return f"{name:35s} shape={tuple(t.shape)}"


def main() -> None:
    batch_size, seq_len, hidden_dim = embeddings.shape
    head_dim = hidden_dim // NUM_HEADS
    print(f"=== 输入张量 ===")
    print(f"  embeddings.shape = {tuple(embeddings.shape)}")
    print(f"  NUM_HEADS = {NUM_HEADS}, head_dim = {head_dim}")
    print(f"  关系: hidden_dim = num_heads × head_dim = {NUM_HEADS} × {head_dim}\n")

    # ========================================================================
    # 拆头: (B, S, H) → (B, S, Heads, HeadDim) → (B, Heads, S, HeadDim)
    # ========================================================================
    print("=== 拆头 (split heads) ===")
    # 步骤 1: reshape 把 hidden 拆成 (Heads, HeadDim)。这一步不交换维度顺序。
    split = embeddings.reshape(batch_size, seq_len, NUM_HEADS, head_dim)
    print(f"  {profile('1) reshape', split)}")

    # 步骤 2: transpose(1, 2) 把 heads 维提到 seq 维前面
    heads_first = split.transpose(1, 2)
    print(f"  {profile('2) transpose(1,2)', heads_first)}")
    print(f"  contig? {heads_first.is_contiguous()}  (False,内存不连续)")

    # 步骤 3: contiguous() 让它变连续
    heads_first = heads_first.contiguous()
    print(f"  {profile('3) contiguous()', heads_first)}")
    print(f"  contig? {heads_first.is_contiguous()}  (True)\n")

    # ========================================================================
    # 模拟 Q/K/V —— 这里用同一份 X 当 Q/K/V 演示形状流转
    # ========================================================================
    q = k = v = heads_first  # (B, Heads, S, HeadDim)

    # ========================================================================
    # 多头注意力计算
    # ========================================================================
    print("=== 多头 SDPA 计算 ===")

    # 1) Q @ K^T: (B, Heads, S, HeadDim) @ (B, Heads, HeadDim, S) → (B, Heads, S, S)
    scores = torch.matmul(q, k.transpose(-2, -1))
    print(f"  {profile('Q @ K^T', scores)}")

    # 2) scaled —— 这里除以 sqrt(head_dim),不再是 sqrt(hidden_dim)
    scores = scores / math.sqrt(head_dim)
    print(f"  scaled by sqrt(head_dim={head_dim}) = {math.sqrt(head_dim):.3f}\n")

    # 3) mask —— 多头里的 mask 形状
    print("=== mask 升维(单头 (B,1,S) → 多头 (B,1,1,S)) ===")
    print(f"  valid_mask.shape      = {tuple(valid_mask.shape)}  ← (B, S)")
    # 在第 1 维(seq 维之前)和第 2 维(query 维)各插一个 1,得到 (B, 1, 1, S)
    key_mask = valid_mask[:, None, None, :]
    print(f"  key_mask.shape        = {tuple(key_mask.shape)}  ← (B, 1, 1, S)")
    print(f"  解读: 中间两个 1 分别对应 heads 维和 query 维,只有最后一维是 S(key)")
    print(f"        这样可以广播到 scores (B, Heads, S, S) 的最后两维。")
    print(f"        heads 维和 query 维都填 1 = 所有 head、query 共享同一份 key mask。\n")

    scores = scores.masked_fill(~key_mask, float("-inf"))

    # 4) softmax(dim=-1) —— 每个 head 内部独立
    weights = torch.softmax(scores, dim=-1)
    print(f"  {profile('softmax(dim=-1)', weights)}")

    # 5) 加权求和: (B, Heads, S, S) @ (B, Heads, S, HeadDim) → (B, Heads, S, HeadDim)
    context_per_head = torch.matmul(weights, v)
    print(f"  {profile('weights @ V', context_per_head)}\n")

    # ========================================================================
    # 合头: (B, Heads, S, HeadDim) → (B, S, Heads, HeadDim) → (B, S, Hidden)
    # ========================================================================
    print("=== 合头 (merge heads) ===")
    # 步骤 1: transpose(1, 2) 把 heads 维换回 seq 后面
    merge_step1 = context_per_head.transpose(1, 2)
    print(f"  {profile('1) transpose(1,2)', merge_step1)}")

    # 步骤 2: contiguous() 再 reshape —— 理由跟拆头时一样
    merge_step1 = merge_step1.contiguous()

    # 步骤 3: reshape 把 heads 和 head_dim 合并回 hidden
    merged = merge_step1.reshape(batch_size, seq_len, hidden_dim)
    print(f"  {profile('2) reshape → hidden', merged)}")

    print(f"\n=== 拆合可逆性验证 ===")
    print(f"  输入 embeddings.shape  = {tuple(embeddings.shape)}")
    print(f"  输出 merged.shape     = {tuple(merged.shape)}")
    print(f"  形状相同? {merged.shape == embeddings.shape}")
    # 注意: merged 是多头注意力的输出,和输入 embeddings 不同。
    #       attention 把每个 token 跟其它 token 融合,得到新的表示。
    #       如果这里 max_diff = 0,反而说明 attention 啥都没干。
    print(f"  内容差异(应该非 0): max_diff = {(merged - embeddings).abs().max().item():.4f}")
    print(f"  → 形状保持一致,但内容被 attention 改变了 —— 这正是 attention 的作用。")

    print(f"\n=== 多头 vs 单头 形状对比 ===")
    print(f"  单头 SDPA: scores = (B, S, S)")
    print(f"  多头 SDPA: scores = (B, Heads, S, S)  ← 多了一维,每个 head 独立算")
    print(f"  → heads 维相当于一个额外的 batch,矩阵乘法天然按 head 并行。")
    print(f"  → 这就是为什么需要 transpose 把 heads 提到 seq 前面。")


if __name__ == "__main__":
    main()