"""04-1 构造 (batch, seq, hidden) 文本张量

对应大纲第 1 条(重点)。

知识点:
    - 文本不能直接喂给注意力,必须先变成数字向量 —— 这就是 embedding。
    - 一批样本的张量形状约定: (batch_size, seq_len, hidden_dim)
    - 同时维护一个 valid_mask,标记哪些 token 是真实的、哪些是 [PAD]。

运行方式:
    .venv\\Scripts\\python.exe 03-attention\\examples\\01_embedding_tensor.py
"""
import json
from pathlib import Path

import torch

# 课堂样本从 data/lesson_data.json 里读 —— 这样跟 04 章其他 example 共用一份数据。
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "lesson_data.json"


def load_lesson_data(path: Path) -> dict:
    """读取课堂 JSON 数据。"""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_embedding_tensor(data: dict):
    """把 JSON 里的手写 embedding 转成三维 Tensor。

    返回:
        embeddings: 形状 (batch_size, seq_len, hidden_dim) 的 float32 Tensor
        tokens:     List[List[str]],每条样本的 token 列表
        valid_mask: 形状 (batch_size, seq_len) 的 bool Tensor
    """
    # 1. 取出 tokens —— 用中文词是为了和 Tensor 的行号能对上,方便教学时投屏对照。
    tokens = [sample["tokens"] for sample in data["samples"]]

    # 2. 取出所有样本的 embedding 列表,一次转成 Tensor。
    #    注意 data["samples"] 里有几条样本,batch_size 就是几(这里固定 3 条)。
    embeddings = torch.tensor(
        [sample["embeddings"] for sample in data["samples"]],
        dtype=torch.float32,
    )

    # 3. valid_mask 是 (batch, seq) 的 bool Tensor,True 表示真实 token,False 表示 [PAD]。
    #    后面做 attention mask 时会把它升维后用来屏蔽 padding 位置。
    valid_mask = torch.tensor(
        [sample["valid_mask"] for sample in data["samples"]],
        dtype=torch.bool,
    )

    return embeddings, tokens, valid_mask


def main() -> None:
    """主入口:加载数据 → 构造张量 → 打印调试信息。"""
    data = load_lesson_data(DATA_FILE)

    print("课程:", data["title"])
    print("课堂项目:", data["project"])
    print(f"num_heads(将要拆的头数): {data['num_heads']}\n")

    embeddings, tokens, valid_mask = build_embedding_tensor(data)

    print("=== 三条样本的 tokens ===")
    for i, tk in enumerate(tokens):
        print(f"  A00{i + 1}: {tk}")

    print("\n=== embeddings 张量 ===")
    print(f"  shape      : {tuple(embeddings.shape)}")
    print(f"  dtype      : {embeddings.dtype}")
    print(f"  device     : {embeddings.device}")
    print(f"  numel()    : {embeddings.numel()}  (3 * 4 * 4 = {3 * 4 * 4})")
    print(f"  第一条样本:")
    print(f"    {embeddings[0]}")

    print("\n=== valid_mask ===")
    print(f"  shape      : {tuple(valid_mask.shape)}")
    print(f"  dtype      : {valid_mask.dtype}")
    for i, row in enumerate(valid_mask.tolist()):
        print(f"  A00{i + 1}: {row}")


if __name__ == "__main__":
    main()