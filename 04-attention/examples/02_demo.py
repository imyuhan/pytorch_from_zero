r"""
04-attention 章节的端到端 demo：按顺序跑完 9 个小实验。

运行方式(从项目根目录):
    .venv\Scripts\python.exe 04-attention\examples\02_demo.py

课堂定位：
    本节承接 01-tensor-basics 的 QKV shape 铺垫，重点练习维度变换
    和注意力矩阵运算。所有示例都使用小张量，方便初学者逐行观察
    shape 如何变化。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from attention_tools import (  # noqa: E402
    build_embedding_tensor,
    cat_stack_demo,
    compare_sdpa_with_manual,
    expand_repeat_demo,
    load_lesson_data,
    manual_attention_demo,
    matmul_demo,
    multi_head_demo,
    print_section,
    reshape_transpose_demo,
    split_heads,
    tensor_profile,
    unsqueeze_squeeze_demo,
)


# DATA_FILE 是本节的课堂样本数据，里面保存了 token、padding mask 和手写 embedding。
DATA_FILE = ROOT.parent / "data" / "lesson_data.json"


def main() -> None:
    """按课堂顺序运行 K012 的所有小实验。"""
    # 第一步先加载数据。后面的所有维度变化都基于同一批小样本，
    # 这样学生能把每一步输出和 data.json 中的 token 对上。
    data = load_lesson_data(DATA_FILE)

    print("课程：", data["title"])
    print("课堂项目：", data["project"])

    print_section("1. 构造文本 embedding 张量")
    # 先把文本样本变成 (batch, seq, hidden) 的三维 Tensor。
    # 注意力计算要求输入已经是数字向量，不能直接拿中文字符串做矩阵乘法。
    embeddings, tokens, valid_mask = build_embedding_tensor(data)
    print("tokens:", tokens)
    print(tensor_profile("embeddings", embeddings))
    print(tensor_profile("valid_mask", valid_mask))

    print_section("2. unsqueeze / squeeze：补齐或去掉长度为 1 的维度")
    # 单条句子通常没有 batch 维，很多模型接口却要求 batch 维。
    # 这里演示如何安全地增加和删除长度为 1 的维度。
    for key, value in unsqueeze_squeeze_demo(embeddings).items():
        print(f"{key}: {value}")

    print_section("3. reshape / transpose / contiguous：拆分注意力头")
    # 多头注意力要把 hidden_dim 拆成 num_heads 和 head_dim。
    # 这一步专门观察拆头、换维度顺序、再合回原形状的过程。
    for key, value in reshape_transpose_demo(embeddings, num_heads=data["num_heads"]).items():
        print(f"{key}: {value}")

    print_section("3.5 split_heads 封装函数")
    # 把上面拆头两步抽成一个独立函数：输入 (batch, seq, hidden)，
    # 输出 (batch, heads, seq, head_dim)。后面真正写多头注意力时直接调用。
    heads = split_heads(embeddings, num_heads=data["num_heads"])
    print(tensor_profile("split_heads(embeddings, 2)", heads))
    print("元素总数不变：", heads.numel() == embeddings.numel())
    print(
        "拆出来的 head_dim =",
        heads.shape[-1],
        "（应该等于 hidden_dim / num_heads =",
        embeddings.shape[-1] // data["num_heads"],
        "）",
    )

    print_section("4. expand / repeat：广播扩展和真实复制")
    # expand 和 repeat 输出形状可能一样，但内存含义不同。
    # 这里用位置偏置演示：只读扩展通常优先理解 expand，真实复制才用 repeat。
    for key, value in expand_repeat_demo(embeddings).items():
        print(f"{key}: {value}")

    print_section("5. cat / stack：沿已有维度拼接，或新增一个维度")
    # 拼接样本时最容易混淆 cat 和 stack。
    # cat 是接长已有维度，stack 是新建一个维度，输出 shape 完全不同。
    for key, value in cat_stack_demo(embeddings).items():
        print(f"{key}: {value}")

    print_section("6. matmul：二维矩阵乘法和批量矩阵乘法")
    # 注意力分数的核心就是 Q 和 K 的矩阵乘法。
    # 这里先用 embeddings 自己和自己相乘，帮助学生看懂 (batch, seq, seq)。
    for key, value in matmul_demo(embeddings).items():
        print(f"{key}: {value}")

    print_section("7. 手写 scaled dot-product attention")
    # 先手写注意力公式，学生能看到 Q、K、V、mask、softmax 每一步的作用。
    # 后面再对比官方函数，避免一开始就把注意力当成黑盒 API。
    attention = manual_attention_demo(embeddings, valid_mask)
    for key, value in attention.items():
        print(f"{key}: {value}")

    print_section("8. 与 torch.nn.functional.scaled_dot_product_attention 对齐")
    # 官方函数是生产代码更常用的写法。
    # 这里对齐手写结果，是为了证明手写公式理解正确，而不是替代官方实现。
    for key, value in compare_sdpa_with_manual(embeddings, valid_mask).items():
        print(f"{key}: {value}")

    print_section("9. 多头注意力的拆分与合并")
    # 最后把前面的 reshape、transpose、matmul、mask 串起来，
    # 演示真实多头注意力里最关键的拆头和合头形状流转。
    for key, value in multi_head_demo(embeddings, valid_mask, num_heads=data["num_heads"]).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
