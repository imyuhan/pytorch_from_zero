"""把每个章节内部的 doc/NN-chapter.md 文件名跟章节号同步。

新结构:
    00-setup/doc/00-setup.md                (未变)
    01-tensor-basics/doc/01-tensor-basics.md (未变)
    02-autograd/doc/02-autograd.md           (未变)
    03-attention/doc/03-attention.md         (原 04-attention.md)
    04-dataset-dataloader/doc/04-dataset-dataloader.md (原 03-dataset-dataloader.md)
    05-fastapi/doc/05-fastapi.md             (原 08-fastapi.md)
    06-pretrained-models/doc/06-pretrained-models.md
    07-cnn/doc/07-cnn.md
    08-experiments/doc/08-experiments.md
"""
from pathlib import Path

ROOT = Path(r"C:\Pytorch\pytorch_from_zero")

# 新章节号 -> (旧 doc 文件名, 新 doc 文件名)
RENAMES = {
    "03-attention":          ("04-attention.md",          "03-attention.md"),
    "04-dataset-dataloader": ("03-dataset-dataloader.md", "04-dataset-dataloader.md"),
    "05-fastapi":            ("08-fastapi.md",            "05-fastapi.md"),
    "06-pretrained-models":  ("05-pretrained-models.md",  "06-pretrained-models.md"),
    "07-cnn":                ("06-cnn.md",                "07-cnn.md"),
    "08-experiments":        ("07-experiments.md",        "08-experiments.md"),
}


def main() -> None:
    print("=== 重命名 doc/NN-xxx.md 跟章节号同步 ===")
    for chapter, (old_name, new_name) in RENAMES.items():
        doc_dir = ROOT / chapter / "doc"
        old_path = doc_dir / old_name
        new_path = doc_dir / new_name
        if not old_path.exists():
            print(f"  [SKIP]  {old_path.relative_to(ROOT)}  不存在")
            continue
        if new_path.exists():
            print(f"  [SKIP]  {new_path.relative_to(ROOT)}  已存在")
            continue
        print(f"  {old_path.relative_to(ROOT)}")
        print(f"    -> {new_path.relative_to(ROOT)}")
        old_path.rename(new_path)
    print("\n完成")


if __name__ == "__main__":
    main()
