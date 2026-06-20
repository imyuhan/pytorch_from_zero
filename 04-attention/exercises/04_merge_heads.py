# @File     : merge_heasds.py
# @Time     : 2026/6/12 21:14
import torch

def merge_heads(x):
    batch, heads, seq, head_dim = x.shape
    # (batch, heads, seq, head_dim) -> (batch, seq, heads, head_dim)
    x = x.transpose(1, 2)
    # 压平成 (batch, seq, heads * head_dim)
    x = x.reshape(batch, seq, heads * head_dim)
    return x

if __name__ == "__main__":
    torch.manual_seed(42)
    B, H, S, D = 2, 8, 6, 16
    x = torch.randn(B, H, S, D)
    print(f"输入 shape:  {x.shape}")  # (2, 8, 6, 16)
    y = merge_heads(x)
    print(f"输出 shape:  {y.shape}")  # (2, 6, 128)