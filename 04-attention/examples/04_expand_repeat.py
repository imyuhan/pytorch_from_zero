"""04-4 expand vs repeat:广播视图 vs 真实复制

对应大纲第 4 条(重点)。

知识点:
    - expand(*sizes) 对长度为 1 的维度按目标形状"扩展",通常零拷贝(只读视图)。
    - repeat(*repeats) 沿每个维度真实复制数据。
    - 输出形状可能一样,但内存含义完全不同 —— 只读场景用 expand 更省。

运行方式:
    .venv\\Scripts\\python.exe 04-attention\\examples\\04_expand_repeat.py
"""
import torch

# 输入张量: (batch=3, seq=4, hidden=4) —— 跟其他 example 保持一致
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
)


def main() -> None:
    batch_size, seq_len, hidden_dim = embeddings.shape
    print(f"=== 输入张量 ===")
    print(f"  embeddings.shape = {tuple(embeddings.shape)}\n")

    # ----------------------------------------------------------------
    # 1. 构造"位置偏置",初始形状 (1, seq_len, 1)
    # ----------------------------------------------------------------
    # position_bias 模拟 "每个位置一个偏置值",
    # 形状 (1, seq_len, 1) —— 两个长度为 1 的维度都可以被扩展。
    print("=== 1. 构造位置偏置 position_bias ===")
    position_bias = torch.arange(seq_len, dtype=torch.float32).reshape(1, seq_len, 1)
    print(f"  position_bias.shape = {tuple(position_bias.shape)}")
    print(f"  position_bias =\n{position_bias.squeeze(-1)}\n")

    # ----------------------------------------------------------------
    # 2. expand:零拷贝广播视图
    # ----------------------------------------------------------------
    print("=== 2. expand:零拷贝广播视图 ===")
    expanded = position_bias.expand(batch_size, seq_len, hidden_dim)
    print(f"  expanded.shape        = {tuple(expanded.shape)}")
    print(f"  expanded.numel()      = {expanded.numel()} (4 * 4 * 4 = 64)")
    print(f"  expanded.is_contiguous() = {expanded.is_contiguous()}")
    print(f"  expanded._base is not None? = {expanded._base is not None}")
    print(f"  解读: 输出形状和 embeddings 一样,但底层不真的复制。")
    print(f"        只是把 (1, 4, 1) 按目标形状 (3, 4, 4) '看成' 更大。\n")

    # ----------------------------------------------------------------
    # 3. repeat:真实复制
    # ----------------------------------------------------------------
    print("=== 3. repeat:真实复制 ===")
    repeated = position_bias.repeat(batch_size, 1, hidden_dim)
    print(f"  repeated.shape        = {tuple(repeated.shape)}")
    print(f"  repeated.numel()      = {repeated.numel()}")
    print(f"  repeated.is_contiguous() = {repeated.is_contiguous()}")
    print(f"  repeated._base is not None? = {repeated._base is not None}")
    print(f"  解读: 输出形状一样,但 repeat 把数据真的复制了 {batch_size}*{hidden_dim} = {batch_size*hidden_dim} 份。\n")

    # ----------------------------------------------------------------
    # 4. 内存与可写性差异
    # ----------------------------------------------------------------
    print("=== 4. 内存 / 可写性差异 ===")

    # 4.1 expand 出来的张量共享底层,不能 in-place 修改会报错
    print(f"  expanded[0, 0, 0] = {expanded[0, 0, 0].item()}")
    try:
        expanded[0, 0, 0] = 999.0  # 试图 in-place 写入
    except RuntimeError as exc:
        print(f"  expanded[0,0,0] = 999  报错 RuntimeError: {exc}")
        print(f"  → expand 出来的视图不能直接 in-place 写。\n")

    # 4.2 repeat 是独立内存,可以随便改
    repeated_copy = repeated.clone()  # 复制一份避免污染原数据演示
    repeated_copy[0, 0, 0] = 999.0
    print(f"  repeated_copy[0, 0, 0] = 999 后,repeated[0, 0, 0] = {repeated[0, 0, 0].item()}")
    print(f"  → repeat 是真复制,改 repeated_copy 不影响 repeated。\n")

    # ----------------------------------------------------------------
    # 5. 对比表
    # ----------------------------------------------------------------
    print("=== 5. expand vs repeat 对比 ===")
    print("  expand: 零拷贝视图,只读为主,内存省")
    print("  repeat: 真实复制,可写,内存多")
    print("  → 只读场景(广播、位置偏置)优先用 expand。")


if __name__ == "__main__":
    main()