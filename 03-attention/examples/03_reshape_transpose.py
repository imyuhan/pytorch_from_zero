"""04-3 reshape / transpose / contiguous:把 hidden_dim 拆成多头

对应大纲第 3 条(难点)。

知识点:
    - 多头注意力要把 hidden_dim 拆成 num_heads × head_dim。
    - reshape 只切最后一维的内部布局,维度顺序不变。
    - transpose 交换两个维度的位置,只改 stride 不复制数据。
    - contiguous() 强制让张量在内存里连续,后续 reshape/view 才不报错。

运行方式:
    .venv\\Scripts\\python.exe 03-attention\\examples\\03_reshape_transpose.py
"""
import torch

# 一批样本: 3 条句子,每句 4 个 token,每个 token 用 4 维向量表示
embeddings = torch.tensor(
    [
        # A001
        [[0.90, 0.10, 0.20, 0.30],
         [0.20, 0.80, 0.10, 0.40],
         [0.30, 0.20, 0.90, 0.10],
         [0.40, 0.10, 0.30, 0.80]],
        # A002
        [[0.85, 0.15, 0.25, 0.20],
         [0.15, 0.75, 0.20, 0.30],
         [0.25, 0.20, 0.80, 0.20],
         [0.00, 0.00, 0.00, 0.00]],
        # A003
        [[0.70, 0.30, 0.40, 0.10],
         [0.30, 0.60, 0.20, 0.50],
         [0.40, 0.20, 0.70, 0.30],
         [0.00, 0.00, 0.00, 0.00]],
    ],
    dtype=torch.float32,
)  # shape: (3, 4, 4)  ← (batch, seq, hidden)

NUM_HEADS = 2


def profile(name: str, t: torch.Tensor) -> str:
    """返回调试信息。"""
    return f"{name:30s} shape={tuple(t.shape)}, contig={t.is_contiguous()}"


def main() -> None:
    print("=== 输入张量 ===")
    print(f"  embeddings.shape = {tuple(embeddings.shape)}")
    print(f"  解读: (batch=3, seq=4, hidden=4)\n")

    # ----------------------------------------------------------------
    # 第 1 步:reshape 把最后一维(hidden)拆成 (num_heads, head_dim)
    # ----------------------------------------------------------------
    batch_size, seq_len, hidden_dim = embeddings.shape
    head_dim = hidden_dim // NUM_HEADS
    print(f"=== 第 1 步 reshape ===")
    print(f"  hidden_dim={hidden_dim}, num_heads={NUM_HEADS}, head_dim={head_dim}")
    print(f"  关系: hidden_dim = num_heads * head_dim = {NUM_HEADS} * {head_dim}")

    split_heads = embeddings.reshape(batch_size, seq_len, NUM_HEADS, head_dim)
    print(f"  {profile('split_heads', split_heads)}")
    print(f"  解读: hidden_dim 这一维被内部切成了 (num_heads, head_dim) = (2, 2)")
    print(f"  元素总数不变: {split_heads.numel()} == {embeddings.numel()}\n")

    # ----------------------------------------------------------------
    # 第 2 步:transpose(1, 2) 把 heads 维提到 seq 维前面
    # ----------------------------------------------------------------
    print("=== 第 2 步 transpose(1, 2) ===")
    heads_first = split_heads.transpose(1, 2)
    print(f"  {profile('heads_first', heads_first)}")
    print(f"  heads_first.is_contiguous() = {heads_first.is_contiguous()}")
    print(f"  解读: 把 heads 维(原 dim=2)换到 seq 维(原 dim=1)前面")
    print(f"  为什么? —— 多头注意力每个 head 要独立算 QK^T,matrix mul 会把 heads 当 batch 处理。")
    print(f"  heads 必须在 seq 前面,后续的 matmul 才能按 head 并行。\n")

    # ----------------------------------------------------------------
    # 第 3 步:contiguous() 让内存连续
    # ----------------------------------------------------------------
    print("=== 第 3 步 contiguous() ===")
    contiguous_heads = heads_first.contiguous()
    print(f"  {profile('contiguous_heads', contiguous_heads)}")
    print(f"  contiguous_heads.is_contiguous() = {contiguous_heads.is_contiguous()}")
    print(f"  解读: transpose 只改 stride 不复制数据,内存变得不规则。")
    print(f"        contiguous() 真的复制一份让它变连续,后面再 reshape/view 才不会报错。\n")

    # ----------------------------------------------------------------
    # 第 4 步:合头 —— 拆回去,验证元素不变
    # ----------------------------------------------------------------
    print("=== 第 4 步 合头(merge) ===")
    print("  步骤: 先 transpose 把 heads 换回 seq 后面,再 reshape 合并最后一维")

    # 4.1 transpose(1, 2) 把 heads 维再换回 seq 后面
    merge_step1 = contiguous_heads.transpose(1, 2)
    print(f"  4.1 transpose(1,2): shape = {tuple(merge_step1.shape)}")
    # 这里不 contiguous 也可以,reshape 内部会自动处理;但显式 contiguous 更稳。

    # 4.2 reshape 把 (batch, seq, heads, head_dim) 合并回 (batch, seq, hidden)
    merged = merge_step1.reshape(batch_size, seq_len, hidden_dim)
    print(f"  4.2 reshape:         shape = {tuple(merged.shape)}")

    # 验证拆合可逆
    max_diff = float((merged - embeddings).abs().max())
    print(f"\n=== 拆合可逆性验证 ===")
    print(f"  max(|merged - embeddings|) = {max_diff:.8f}")
    print(f"  → {max_diff < 1e-6} (拆头再合回去,元素完全一样)")


if __name__ == "__main__":
    main()