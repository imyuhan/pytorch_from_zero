"""04 章重排迁移脚本。

目标:
    - 03-dataset-dataloader → 04-dataset-dataloader
    - 04-attention          → 03-attention
    - 05-pretrained-models  → 06-pretrained-models
    - 06-cnn                → 07-cnn
    - 07-experiments        → 08-experiments
    - 08-fastapi            → 05-fastapi

策略: 两阶段重命名,避免新名字和旧名字冲突
    1. 把所有要改的目录先加前缀 "_tmp_" 变成中间名
    2. 再从中间名改成最终名字

执行:  python tools/migrate_chapters.py
"""
import shutil
from pathlib import Path

ROOT = Path(r"C:\Pytorch\pytorch_from_zero")

# 阶段 1:原名 -> 中间名
STAGE1 = {
    "04-attention": "_tmp_a_attention",           # → 03-attention
    "03-dataset-dataloader": "_tmp_b_dataset",    # → 04-dataset-dataloader
    "05-pretrained-models": "_tmp_c_pretrained",  # → 06-pretrained-models
    "06-cnn": "_tmp_d_cnn",                       # → 07-cnn
    "07-experiments": "_tmp_e_experiments",       # → 08-experiments
    "08-fastapi": "_tmp_f_fastapi",               # → 05-fastapi
}

# 阶段 2:中间名 -> 终名
STAGE2 = {
    "_tmp_a_attention": "03-attention",
    "_tmp_b_dataset": "04-dataset-dataloader",
    "_tmp_c_pretrained": "06-pretrained-models",
    "_tmp_d_cnn": "07-cnn",
    "_tmp_e_experiments": "08-experiments",
    "_tmp_f_fastapi": "05-fastapi",
}


def rename_dir(src: Path, dst: Path) -> None:
    """重命名目录。如果 dst 已存在则报错(脚本设计上是 disjoint 的)。"""
    if dst.exists():
        raise RuntimeError(f"目标已存在,停止: {dst}")
    print(f"  {src.name}  ->  {dst.name}")
    shutil.move(str(src), str(dst))


def stage1() -> None:
    print("=== 阶段 1: 原名 -> 中间名 ===")
    for old, mid in STAGE1.items():
        src = ROOT / old
        if not src.exists():
            raise RuntimeError(f"源目录不存在: {src}")
        dst = ROOT / mid
        rename_dir(src, dst)


def stage2() -> None:
    print("\n=== 阶段 2: 中间名 -> 终名 ===")
    for mid, new in STAGE2.items():
        src = ROOT / mid
        if not src.exists():
            raise RuntimeError(f"中间目录不存在: {src}")
        dst = ROOT / new
        rename_dir(src, dst)


def main():
    print("工作根目录:", ROOT)
    print()
    # 安全检查: 确认没有其他以 _tmp_ 开头的目录
    leftovers = [p.name for p in ROOT.iterdir() if p.name.startswith("_tmp_")]
    if leftovers:
        raise RuntimeError(f"仓库根目录已有 _tmp_ 前缀的目录,先清理: {leftovers}")

    # 安全检查: 确认终名还没被占用
    finals = list(STAGE2.values())
    for name in finals:
        if (ROOT / name).exists():
            raise RuntimeError(f"终名已被占用: {name}")

    stage1()
    stage2()
    print("\n=== 完成 ===")
    print("新章节顺序:")
    for p in sorted(ROOT.iterdir()):
        if p.is_dir() and p.name[:2].isdigit():
            print(f"  {p.name}")


if __name__ == "__main__":
    main()
