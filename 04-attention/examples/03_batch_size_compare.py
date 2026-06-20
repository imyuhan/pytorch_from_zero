r"""分别用 1/2/3 条样本跑一遍，对照 batch_size 变化时张量 shape 怎么动。

对应 K012 课堂第 6 节(matmul)与第 7 节(手写 attention)的拓展：
只动 batch_size 这一维，其余维度不变，看每个样本各自的注意力形状。

运行方式(从项目根目录):
    .venv\Scripts\python.exe 04-attention\examples\03_batch_size_compare.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from attention_tools import build_embedding_tensor, load_lesson_data  # noqa: E402


def main() -> None:
    """依次取前 1/2/3 条样本跑 build_embedding_tensor，对照 shape。"""
    data = load_lesson_data(ROOT.parent / "data" / "lesson_data.json")

    print("lesson_data.json 里共有样本数：", len(data["samples"]))

    for num_samples in (1, 2, 3):
        # 只取前 num_samples 条喂给 build_embedding_tensor。
        # build_embedding_tensor 内部只读不写,不需要 deepcopy。
        sub = {**data, "samples": data["samples"][:num_samples]}
        embeddings, tokens, valid_mask = build_embedding_tensor(sub)
        print(f"\n>>> 取前 {num_samples} 条样本：")
        print(f"  embeddings.shape = {tuple(embeddings.shape)}  ← 第一个数就是 batch_size")
        print(f"  valid_mask.shape = {tuple(valid_mask.shape)}  ← 第一个数跟着 batch 走")
        print(f"  tokens[0]        = {tokens[0]}")


if __name__ == "__main__":
    main()