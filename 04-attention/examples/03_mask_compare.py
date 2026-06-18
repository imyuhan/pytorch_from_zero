"""把 valid_mask 改了之后，直观看 A002 的注意力权重矩阵发生了什么。

对应 K012 课堂第 7 节(手写 scaled dot-product attention)的拓展：
直接打印 3 条样本各自的 (seq, seq) 注意力权重矩阵，
把 mask 屏蔽的 padding 位置打上 * 号，肉眼对照 token 看权重流向。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import torch  # noqa: E402

from attention_tools import _projection_weights, load_lesson_data  # noqa: E402

data = load_lesson_data(ROOT.parent / "data" / "lesson_data.json")
embeddings = torch.tensor([s["embeddings"] for s in data["samples"]], dtype=torch.float32)
valid_mask = torch.tensor([s["valid_mask"] for s in data["samples"]], dtype=torch.bool)
tokens = [s["tokens"] for s in data["samples"]]

print("当前 valid_mask（每行一个样本）：")
for i, row in enumerate(valid_mask.tolist()):
    print(f"  A00{i+1}: {row}  ← tokens={tokens[i]}")

# 走一遍手写注意力的公式
hidden_dim = embeddings.shape[-1]
w_q, w_k, w_v = _projection_weights(hidden_dim)
q = embeddings @ w_q
k = embeddings @ w_k
v = embeddings @ w_v
raw_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hidden_dim)
key_mask = valid_mask.unsqueeze(1)  # (batch, 1, seq)
masked = raw_scores.masked_fill(~key_mask, float("-inf"))
weights = torch.softmax(masked, dim=-1)


def show(batch_idx: int, label: str) -> None:
    print(f"\n--- {label}（A00{batch_idx+1}）tokens: {tokens[batch_idx]} ---")
    print("  " + " ".join(f"{t:>6}" for t in tokens[batch_idx]))
    for qi, qt in enumerate(tokens[batch_idx]):
        row = ["  "]
        for ki, kt in enumerate(tokens[batch_idx]):
            w = float(weights[batch_idx, qi, ki])
            mark = "*" if not valid_mask[batch_idx, ki] else " "
            row.append(f"{w:5.3f}{mark}")
        print(f"  {qt:>4} " + " ".join(row[1:]))
    print("  (* 号 = 该 key 位置被 mask 屏蔽)")


# 重点看 A002（batch_idx=1）
show(0, "对照：A001（valid_mask 全 True，应该没变化）")
show(1, "重点：A002（valid_mask 第 3 位也变 False 了）")
show(2, "对照：A003（valid_mask 没动）")

print("\n" )
print("看 A002 第 3 行（'对齐' 作为 query）看别的 key 的权重：")
for ki, kt in enumerate(tokens[1]):
    w = float(weights[1, 2, ki])
    masked = "← 被 mask" if not valid_mask[1, ki] else ""
    print(f"  '对齐' -> '{kt}': {w:.4f}  {masked}")
