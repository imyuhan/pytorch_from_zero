"""04-5 cat vs stack:沿已有维度拼接 vs 新建一个维度

对应大纲第 5 条(重点)。

知识点:
    - cat(tensors, dim) 沿已有维度拼接,总维度数不变。
    - stack(tensors, dim) 在指定位置新建一个维度,总维度数 +1。
    - 口诀: cat 是加长,stack 是加厚。

运行方式:
    .venv\\Scripts\\python.exe 03-attention\\examples\\05_cat_stack.py
"""
import torch

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
)  # shape: (3, 4, 4)


def profile(name: str, t: torch.Tensor) -> str:
    return f"{name:30s} shape={tuple(t.shape)}"


def main() -> None:
    print(f"=== 输入张量 ===")
    print(f"  {profile('embeddings', embeddings)}\n")

    # 取出前两条样本 (seq=4, hidden=4) 用来演示拼接
    first = embeddings[0]
    second = embeddings[1]
    print(f"=== 取出两条样本 ===")
    print(f"  {profile('first', first)}")
    print(f"  {profile('second', second)}\n")

    # ----------------------------------------------------------------
    # 1. cat:沿已有维度拼接
    # ----------------------------------------------------------------
    print("=== 1. cat:沿已有第 0 维拼接(seq 方向接长) ===")
    cat_seq = torch.cat([first, second], dim=0)
    print(f"  {profile('cat_seq', cat_seq)}")
    print(f"  原 first 有 {first.shape[0]} 个 token,cat 后变成 {cat_seq.shape[0]} 个 token")
    print(f"  解读: cat 把 token 序列接长,dim 不变(还是 2D)\n")

    # ----------------------------------------------------------------
    # 2. stack:新建一个维度
    # ----------------------------------------------------------------
    print("=== 2. stack(dim=0):在前面新建一个 batch 维 ===")
    stack_batch = torch.stack([first, second], dim=0)
    print(f"  {profile('stack_batch', stack_batch)}")
    print(f"  解读: stack 新建一维,把两条样本当成 batch=2")
    print(f"        跟 cat 不同 —— stack 后形状从 2D 变成 3D\n")

    # ----------------------------------------------------------------
    # 3. stack 不同 dim:新维度插的位置不一样
    # ----------------------------------------------------------------
    print("=== 3. stack(dim=1):在中间新建一维 ===")
    stack_choice = torch.stack([first, second], dim=1)
    print(f"  {profile('stack_choice', stack_choice)}")
    print(f"  解读: 新维度在 batch=0 和 seq=1 之间(把样本维度放中间)\n")

    print("=== 4. stack(dim=2):在最后新建一维 ===")
    stack_tail = torch.stack([first, second], dim=2)
    print(f"  {profile('stack_tail', stack_tail)}")
    print(f"  解读: 新维度在最后(常用于把两张同 shape 的特征图叠成一组)\n")

    # ----------------------------------------------------------------
    # 5. cat 沿不同维度的对比
    # ----------------------------------------------------------------
    print("=== 5. cat 沿不同维度的对比 ===")
    cat_dim1 = torch.cat([first, second], dim=1)  # 沿 hidden 维拼接
    print(f"  {profile('cat_dim1 (hidden 维接长)', cat_dim1)}")
    print(f"  解读: 沿 hidden 方向把 hidden_dim 从 4 拼成 8")
    print(f"        cat 只能沿已有维度,所以 hidden 必须存在\n")

    # ----------------------------------------------------------------
    # 6. 口诀
    # ----------------------------------------------------------------
    print("=== 6. 总结: cat vs stack ===")
    print("  cat: 沿已有维度接长,总维度数不变")
    print("  stack: 新建一个维度,总维度数 +1")
    print("  口诀: cat 是加长,stack 是加厚。")
    print("  选谁?")
    print("    - 拼接变长序列(如合并两段文本) → cat")
    print("    - 把一组同 shape 张量打包成新 batch → stack")


if __name__ == "__main__":
    main()