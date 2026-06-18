"""01-5 Transformer QKV 三维张量与矩阵乘法

对应张量章节教学大纲第 4 条(难点)。
"""
import torch

torch.manual_seed(42)
B, S, H = 2, 4, 8  # Batch=2, Seq_len=4, Hidden=8

# 文本嵌入后的形状: (B, S, H)
x = torch.randn(B, S, H)
print(f"输入 x shape: {x.shape}  # (Batch, Seq, Hidden)")

# QKV 权重: (H, H)
W_q = torch.randn(H, H)
W_k = torch.randn(H, H)
W_v = torch.randn(H, H)

# --- 关键:把最后两维拆开做矩阵乘法 ---
# x @ W 之前需要 (B, S, H) @ (H, H) → PyTorch 会自动把 (B, S, H) 看作 (B*S, H)
# 即最后两维做 matmul
Q = x @ W_q   # (B, S, H) @ (H, H) → (B, S, H)
K = x @ W_k
V = x @ W_v
print(f"Q shape: {Q.shape}  # 同输入")

# --- 注意力分数: Q @ K^T,需要 K 转置最后两维 ---
K_t = K.transpose(-2, -1)               # (B, S, H) → (B, H, S)
scores = Q @ K_t                         # (B, S, H) @ (B, H, S) → (B, S, S)
print(f"注意力分数 shape: {scores.shape}  # (Batch, Seq, Seq)")

# 简单注意力输出
attn = scores @ V                        # (B, S, S) @ (B, S, H) → (B, S, H)
print(f"注意力输出 shape: {attn.shape}")

# --- 广播示例 ---
print("\n=== 广播 (Broadcasting) ===")
a = torch.ones(4, 4)
b = torch.tensor([1., 2., 3., 4.])      # (4,)
print(f"a / b shape: {(a / b).shape}  # (4,4) / (4,) → 自动广播")
