"""04-2 unsqueeze / squeeze:补 batch 维、删多余维

对应大纲第 2 条(重点)。

知识点:
    - 单条句子是 (seq, hidden) 的二维张量,模型接口却要求有 batch 维。
    - unsqueeze(dim) 在指定位置插入一个长度为 1 的新维度。
    - squeeze(dim) 删除指定位置长度为 1 的维度(不指定 dim 时删所有长度为 1 的)。

运行方式:
    .venv\\Scripts\\python.exe 03-attention\\examples\\02_unsqueeze_squeeze.py
"""
import torch

# 拿 01 文件里构造张量的逻辑直接复用 —— 为了让每个 example 都能独立跑,
# 这里只 import torch,需要数据时自己造一份小样本。
embeddings = torch.tensor(
    [
        # 第一条样本 (seq=4, hidden=4)
        [[0.90, 0.10, 0.20, 0.30],
         [0.20, 0.80, 0.10, 0.40],
         [0.30, 0.20, 0.90, 0.10],
         [0.40, 0.10, 0.30, 0.80]],
    ],
    dtype=torch.float32,
)  # shape: (1, 4, 4) —— 注意这里 batch=1,只是为了方便演示


def tensor_profile(name: str, t: torch.Tensor) -> str:
    """返回 Tensor 的关键调试信息。"""
    return f"{name:30s} shape={tuple(t.shape)}, dim={t.dim()}, dtype={t.dtype}"


def main() -> None:
    print("=== 原 embeddings 张量 ===")
    print("  " + tensor_profile("embeddings", embeddings))
    print("  解读: (batch=1, seq=4, hidden=4)\n")

    # ----------------------------------------------------------------
    # 1. 取出单条句子 —— 形状从 3D 变成 2D
    # ----------------------------------------------------------------
    # embeddings[0] 沿第 0 维切片,得到第一条样本,形状 (seq, hidden)。
    sentence = embeddings[0]
    print("=== 取出单句 ===")
    print("  " + tensor_profile("sentence = embeddings[0]", sentence))
    print("  解读: 没有了 batch 维,变成 (seq, hidden)\n")

    # ----------------------------------------------------------------
    # 2. unsqueeze(0):在最前面补 batch 维
    # ----------------------------------------------------------------
    # 模型接口通常要求 batch 维在最前面。用 unsqueeze(0) 在 dim=0 处插入新维度。
    with_batch = sentence.unsqueeze(0)
    print("=== unsqueeze(0):补 batch 维 ===")
    print("  " + tensor_profile("with_batch = sentence.unsqueeze(0)", with_batch))
    print("  解读: 形状 (1, 4, 4),最前面多了一个 batch=1 的维\n")

    # ----------------------------------------------------------------
    # 3. unsqueeze(1):在中间补一个"head 数"维(为后续拆多头预演)
    # ----------------------------------------------------------------
    # 这一步故意在 batch 和 seq 之间插一个长度为 1 的维,
    # 让你直观看到 unsqueeze 不一定只能插在最前面。
    with_extra = with_batch.unsqueeze(1)
    print("=== unsqueeze(1):在 batch 和 seq 之间插一维 ===")
    print("  " + tensor_profile("with_extra = with_batch.unsqueeze(1)", with_extra))
    print("  解读: 形状 (1, 1, 4, 4),batch 和 seq 之间多了一维\n")

    # ----------------------------------------------------------------
    # 4. squeeze(1):只删第 1 维
    # ----------------------------------------------------------------
    # squeeze(dim) 只删指定 dim 上长度为 1 的维度;dim 上不是 1 会报错。
    squeezed = with_extra.squeeze(1)
    print("=== squeeze(1):精准删中间那一维 ===")
    print("  " + tensor_profile("squeezed = with_extra.squeeze(1)", squeezed))
    print("  解读: 回到 (1, 4, 4),只删了我们指定的那一维\n")

    # ----------------------------------------------------------------
    # 5. 对比 squeeze()(不带参数) —— 默认会删所有长度为 1 的维
    # ----------------------------------------------------------------
    # 不带参数 = 删所有长度为 1 的维度。教学阶段建议显式写 dim,避免误删。
    squeezed_all = with_extra.squeeze()
    print("=== squeeze():不带参数,删全部长度 1 的维 ===")
    print("  " + tensor_profile("squeezed_all = with_extra.squeeze()", squeezed_all))
    print("  解读: 这里 batch=1 也是长度 1,所以全删掉变成 (4, 4)")
    print("  提醒: 初学阶段建议 squeeze(dim) 显式指定,免得删错。\n")

    # ----------------------------------------------------------------
    # 6. 反例:squeeze 一个非 1 的维会报错
    # ----------------------------------------------------------------
    print("=== 反例:squeeze 一个非 1 的维 ===")
    try:
        sentence.squeeze(0)  # sentence 是 2D,dim=0 的长度是 4,不能 squeeze
    except RuntimeError as exc:
        print(f"  报错 RuntimeError: {exc}")
        print("  提醒: 只有长度为 1 的维度才能 squeeze。\n")

    # ----------------------------------------------------------------
    # 7. 验证:补回去的形状和原 embeddings 一致
    # ----------------------------------------------------------------
    print("=== 验证 unsqueeze/squeeze 的可逆性 ===")
    print(f"  sentence.shape              = {tuple(sentence.shape)}")
    print(f"  sentence.unsqueeze(0).shape = {tuple(sentence.unsqueeze(0).shape)}")
    print(f"  embeddings.shape            = {tuple(embeddings.shape)}")
    print(f"  相同? {sentence.unsqueeze(0).shape == embeddings.shape}")


if __name__ == "__main__":
    main()