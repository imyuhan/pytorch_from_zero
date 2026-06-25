"""交换 dataset-dataloader 和 fastapi 的章节编号。

旧:  04-dataset-dataloader  05-fastapi
新:  04-fastapi              05-dataset-dataloader
"""
from pathlib import Path
import re
import shutil

ROOT = Path(r"C:\Pytorch\pytorch_from_zero")

# 阶段 1:目录交换
SWAP_DIRS = {
    "04-dataset-dataloader": "_swap_a_dataset",
    "05-fastapi":            "_swap_b_fastapi",
}

# 阶段 2:内部 doc 文件名交换
SWAP_DOCS = {
    "05-dataset-dataloader": ("04-dataset-dataloader.md", "05-dataset-dataloader.md"),
    "04-fastapi":            ("05-fastapi.md",            "04-fastapi.md"),
}

# 阶段 3:文本替换表(目录名 + 文档内数字)
DIR_TEXT_RENAMES = {
    "04-dataset-dataloader": "05-dataset-dataloader",
    "05-fastapi":            "04-fastapi",
}
SORTED_OLD = sorted(DIR_TEXT_RENAMES.keys(), key=len, reverse=True)

# 文档内数字编号(注意:这里用阶段 1 之后的新章节号作 key)
# 05-dataset-dataloader 原本是 04,要把"04 章"等自指改成"05 章"
# 04-fastapi 原本是 05,要把"05 章"等自指改成"04 章"
DOC_NUM_RENAMES = {
    "05-dataset-dataloader": {"4": "5", "04": "05"},
    "04-fastapi":            {"5": "4", "05": "04"},
}

EXCLUDE_DIRS = {".venv", ".git", "__pycache__", "data", "tools"}
TEXT_EXTS = {".md", ".py", ".json", ".txt", ".toml", ".yaml", ".yml"}


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDE_DIRS:
        return True
    if path.is_file() and path.suffix.lower() not in TEXT_EXTS:
        return True
    return False


def phase1_swap_dirs() -> None:
    print("=== 阶段 1: 交换目录 ===")
    for old, mid in SWAP_DIRS.items():
        src = ROOT / old
        dst = ROOT / mid
        if not src.exists():
            raise RuntimeError(f"源目录不存在: {src}")
        if dst.exists():
            raise RuntimeError(f"中间名已存在: {dst}")
        print(f"  {old}  ->  {mid}")
        shutil.move(str(src), str(dst))

    for mid, new in [("_swap_a_dataset", "05-dataset-dataloader"),
                      ("_swap_b_fastapi", "04-fastapi")]:
        src = ROOT / mid
        dst = ROOT / new
        if not src.exists():
            raise RuntimeError(f"中间目录不存在: {src}")
        if dst.exists():
            raise RuntimeError(f"终名已存在: {dst}")
        print(f"  {mid}  ->  {new}")
        shutil.move(str(src), str(dst))


def phase2_rename_docs() -> None:
    print("\n=== 阶段 2: 重命名 doc 文件 ===")
    for chapter, (old_name, new_name) in SWAP_DOCS.items():
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


def phase3_replace_text() -> None:
    print("\n=== 阶段 3: 文本替换(目录名) ===")
    a_files = []
    a_total = 0
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if should_skip(path):
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        # 占位符防链式
        placeholders = {}
        for i, old in enumerate(SORTED_OLD):
            ph = f"\x00PHS{i:02d}\x00"
            content = content.replace(old, ph)
            placeholders[ph] = DIR_TEXT_RENAMES[old]
        for ph, new in placeholders.items():
            count = content.count(ph)
            if count:
                content = content.replace(ph, new)
                a_total += count
        if a_total and path not in [p for p, _ in a_files]:
            pass  # 计数已在上面累计

        # 单独计文件数
        if placeholders and any(content.count(ph) for ph in placeholders):
            path.write_text(content, encoding="utf-8")
            a_files.append((path.relative_to(ROOT), 0))

    # 上面逻辑写乱了,重做更清晰的
    a_files.clear()
    a_total = 0
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if should_skip(path):
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        original = content
        # 占位符
        placeholders = {}
        for i, old in enumerate(SORTED_OLD):
            ph = f"\x00PHS{i:02d}\x00"
            content = content.replace(old, ph)
            placeholders[ph] = DIR_TEXT_RENAMES[old]
        # 还原
        file_changes = 0
        for ph, new in placeholders.items():
            count = content.count(ph)
            if count:
                content = content.replace(ph, new)
                file_changes += count
        if file_changes:
            path.write_text(content, encoding="utf-8")
            a_files.append((path.relative_to(ROOT), file_changes))
            a_total += file_changes

    print(f"阶段 3-目录名: 修改 {len(a_files)} 个文件,总替换 {a_total} 次")
    for rel, n in a_files:
        print(f"  {rel}  ({n} 处)")

    # 阶段 4:文档内数字编号
    print("\n=== 阶段 4: 文档内数字章节号 ===")
    b_files = []
    b_total = 0
    for chapter, renames in DOC_NUM_RENAMES.items():
        doc_dir = ROOT / chapter / "doc"
        for md_path in doc_dir.glob("*.md"):
            try:
                content = md_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            new_content, count = phase_b_apply(content, renames)
            if count == 0:
                continue
            md_path.write_text(new_content, encoding="utf-8")
            b_files.append((md_path.relative_to(ROOT), count))
            b_total += count

    print(f"阶段 4-数字: 修改 {len(b_files)} 个文件,总替换 {b_total} 次")
    for rel, n in b_files:
        print(f"  {rel}  ({n} 处)")


def make_phase_b_handlers(renames: dict):
    handlers = []
    def find_digit_group(m):
        for g in m.groups():
            if g and g.isdigit():
                return g
        return None

    def h_title(m):
        d = find_digit_group(m)
        if d in renames:
            return f"{m.group(1)}{renames[d]}{m.group(3)}"
        return m.group(0)
    handlers.append((re.compile(r"(#{2,6}\s*)(\d+)(\.\d+(?:\.\d+)*)"), h_title))

    def h_see(m):
        d = find_digit_group(m)
        if d in renames:
            return f"{m.group(1)}{renames[d]}{m.group(3)}"
        return m.group(0)
    handlers.append((re.compile(r"((?:见|详见|参?见|参?阅)\s*)(\d+)(\.\d+(?:\.\d+)*)"), h_see))

    def h_paren(m):
        d = find_digit_group(m)
        if d in renames:
            return f"{m.group(1)}{renames[d]}{m.group(3)}{m.group(4)}"
        return m.group(0)
    handlers.append((re.compile(r"([(\[])(\d+)(\.\d+(?:\.\d+)*)([)\]])"), h_paren))

    def h_di(m):
        d = find_digit_group(m)
        if d in renames:
            return f"第 {renames[d]} 章"
        return m.group(0)
    handlers.append((re.compile(r"第\s*(\d+)\s*章"), h_di))

    def h_zhang(m):
        d = find_digit_group(m)
        if d in renames:
            return f"{renames[d]} 章"
        return m.group(0)
    handlers.append((re.compile(r"(?<![\d.])(\d+)\s*章(?![\d.])"), h_zhang))

    return handlers


def phase_b_apply(text: str, renames: dict) -> tuple[str, int]:
    handlers = make_phase_b_handlers(renames)
    total = 0
    for pattern, handler in handlers:
        new, n = pattern.subn(handler, text)
        total += n
        text = new
    return text, total


def main() -> None:
    print("工作根目录:", ROOT)
    # 安全检查
    leftovers = [p.name for p in ROOT.iterdir() if p.name.startswith("_swap_")]
    if leftovers:
        raise RuntimeError(f"已有 _swap_ 前缀目录: {leftovers}")
    for name in ["05-dataset-dataloader", "04-fastapi"]:
        if (ROOT / name).exists():
            raise RuntimeError(f"终名已存在: {name}")

    phase1_swap_dirs()
    phase2_rename_docs()
    phase3_replace_text()

    print("\n=== 最终目录顺序 ===")
    for p in sorted(ROOT.iterdir()):
        if p.is_dir() and p.name[:2].isdigit():
            print(f"  {p.name}")


if __name__ == "__main__":
    main()
