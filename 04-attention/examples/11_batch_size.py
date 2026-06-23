"""04-11 batch 维变化演示:分别用 1/2/3 条样本跑,看 batch_size 怎么动 shape

对应大纲第 6、7 条的拓展:
    只动 batch_size 这一维,其余维度(seq / hidden)不变,看每个样本的
    embeddings 和 valid_mask 形状变化。

为什么单独一节?
    - 强调 batch 维只动最外层,中间维度不动。
    - 这是为什么 matmul 规则是"只对最后两维做矩阵乘,前面视为 batch"。

运行方式:
    .venv\\Scripts\\python.exe 04-attention\\examples\\11_batch_size.py
"""
import torch


# 三条原始样本,跟 04 章其它 example 保持一致
all_samples = [
    {  # A001
        "sample_id": "A001",
        "tokens": ["模型", "会", "关注", "重点"],
        "valid_mask": [True, True, True, True],
        "embeddings": [
            [0.90, 0.10, 0.20, 0.30],
            [0.20, 0.80, 0.10, 0.40],
            [0.30, 0.20, 0.90, 0.10],
            [0.40, 0.10, 0.30, 0.80],
        ],
    },
    {  # A002
        "sample_id": "A002",
        "tokens": ["数据", "需要", "对齐", "[PAD]"],
        "valid_mask": [True, True, False, False],
        "embeddings": [
            [0.85, 0.15, 0.25, 0.20],
            [0.15, 0.75, 0.20, 0.30],
            [0.25, 0.20, 0.80, 0.20],
            [0.00, 0.00, 0.00, 0.00],
        ],
    },
    {  # A003
        "sample_id": "A003",
        "tokens": ["查询", "命中", "缓存", "[PAD]"],
        "valid_mask": [True, True, True, False],
        "embeddings": [
            [0.70, 0.30, 0.40, 0.10],
            [0.30, 0.60, 0.20, 0.50],
            [0.40, 0.20, 0.70, 0.30],
            [0.00, 0.00, 0.00, 0.00],
        ],
    },
]


def build_embedding_tensor(samples: list[dict]):
    """跟 01 文件的逻辑一致,这里直接展开,方便单个文件独立运行。"""
    embeddings = torch.tensor(
        [sample["embeddings"] for sample in samples],
        dtype=torch.float32,
    )
    tokens = [sample["tokens"] for sample in samples]
    valid_mask = torch.tensor(
        [sample["valid_mask"] for sample in samples],
        dtype=torch.bool,
    )
    return embeddings, tokens, valid_mask


def main() -> None:
    print(f"=== 原始数据共有 {len(all_samples)} 条样本 ===")
    for sample in all_samples:
        print(f"  {sample['sample_id']}: tokens={sample['tokens']}")
    print()

    # 分别取前 1 / 2 / 3 条样本构造张量,看 batch 维怎么动
    for num_samples in (1, 2, 3):
        print(f">>> 取前 {num_samples} 条样本")

        # 只切 samples 列表,不影响其它数据
        sub_samples = all_samples[:num_samples]
        embeddings, tokens, valid_mask = build_embedding_tensor(sub_samples)

        print(f"  embeddings.shape = {tuple(embeddings.shape)}  ← 第一个数就是 batch_size")
        print(f"  valid_mask.shape = {tuple(valid_mask.shape)}  ← 跟 batch 走")
        print(f"  tokens[0]        = {tokens[0]}")
        print()

    print("=== 关键观察 ===")
    print("  1) batch_size 变化时,seq_len 和 hidden_dim 完全不变")
    print("  2) embeddings 和 valid_mask 的形状同步跟着 batch 变")
    print("  3) 这就是为什么 matmul 规则说:")
    print('     "只对最后两维做矩阵乘,前面的维度视为 batch"')
    print("     —— batch 维就是额外的并行批次,不会影响 seq/hidden 的语义。")


if __name__ == "__main__":
    main()