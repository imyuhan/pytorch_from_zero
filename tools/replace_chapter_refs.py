"""统一替换所有受影响文件里的章节号、章节路径。

替换策略(分两阶段):

阶段 A —— 章节目录名(强匹配,只动 6 个目标章节):
    04-attention              -> 03-attention
    03-dataset-dataloader     -> 04-dataset-dataloader
    08-fastapi                -> 05-fastapi
    05-pretrained-models      -> 06-pretrained-models
    06-cnn                    -> 07-cnn
    07-experiments            -> 08-experiments

阶段 B —— 章节级纯数字编号(只对 doc/*.md 做,且用正则限定语境):
    ## 4.1 / ### 4.5.7 / (4.5.7) / 见 4.3.1 / 第 4 章 / 4 章 等

排除:
    - .venv/ .git/ __pycache__/ data/ tools/
    - 二进制/大文件
    - 阶段 B 只在 doc/ 目录的 .md 上跑
"""
import re
from pathlib import Path

ROOT = Path(r"C:\Pytorch\pytorch_from_zero")

# ===== 阶段 A =====
DIR_RENAMES = {
    "04-attention": "03-attention",
    "03-dataset-dataloader": "04-dataset-dataloader",
    "08-fastapi": "05-fastapi",
    "05-pretrained-models": "06-pretrained-models",
    "06-cnn": "07-cnn",
    "07-experiments": "08-experiments",
}
SORTED_OLD = sorted(DIR_RENAMES.keys(), key=len, reverse=True)

# ===== 阶段 B =====
# 文档里"N 章"是引用章节,可能指本章节也可能指别的章节
# 规则: 只有"原章节号"(本章节之前的编号)需要替换,别的章节号不动
# 4 -> 3: 03-attention 的"原章节号"是 04
# 原章节号既可能是裸的 "4" 也可能是带 0 前缀的 "04"(出现在 "04 章" 这种文本里)
DOC_NUM_RENAMES = {
    "03-attention":          {"4": "3", "04": "03"},
    "04-dataset-dataloader": {"3": "4", "03": "04"},
    "05-fastapi":            {"8": "5", "08": "05"},
    "06-pretrained-models":  {"5": "6", "05": "06"},
    "07-cnn":                {"6": "7", "06": "07"},
    "08-experiments":        {"7": "8", "07": "08"},
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


def phase_a_replace(text: str) -> tuple[str, int]:
    """阶段 A: 目录名替换。用占位符防链式匹配。"""
    total = 0
    placeholders = {}
    for i, old in enumerate(SORTED_OLD):
        ph = f"\x00PHA{i:02d}\x00"
        text = text.replace(old, ph)
        placeholders[ph] = DIR_RENAMES[old]
    for ph, new in placeholders.items():
        count = text.count(ph)
        if count:
            text = text.replace(ph, new)
            total += count
    return text, total


def make_phase_b_handlers(renames: dict[str, str]):
    """为阶段 B 生成正则 handlers。"""
    def find_digit_group(m):
        for g in m.groups():
            if g and g.isdigit():
                return g
        return None

    handlers = []

    # 1. 标题前缀: ## 4.1
    def h_title(m):
        d = find_digit_group(m)
        if d in renames:
            return f"{m.group(1)}{renames[d]}{m.group(3)}"
        return m.group(0)
    handlers.append((re.compile(r"(#{2,6}\s*)(\d+)(\.\d+(?:\.\d+)*)"), h_title))

    # 2. 见/详见 4.x.y
    def h_see(m):
        d = find_digit_group(m)
        if d in renames:
            return f"{m.group(1)}{renames[d]}{m.group(3)}"
        return m.group(0)
    handlers.append((re.compile(r"((?:见|详见|参?见|参?阅)\s*)(\d+)(\.\d+(?:\.\d+)*)"), h_see))

    # 3. (4.x.y) / [4.x.y]
    def h_paren(m):
        d = find_digit_group(m)
        if d in renames:
            return f"{m.group(1)}{renames[d]}{m.group(3)}{m.group(4)}"
        return m.group(0)
    handlers.append((re.compile(r"([(\[])(\d+)(\.\d+(?:\.\d+)*)([)\]])"), h_paren))

    # 4. 第 N 章
    def h_di(m):
        d = find_digit_group(m)
        if d in renames:
            return f"第 {renames[d]} 章"
        return m.group(0)
    handlers.append((re.compile(r"第\s*(\d+)\s*章"), h_di))

    # 5. N 章(无前后数字)
    def h_zhang(m):
        d = find_digit_group(m)
        if d in renames:
            return f"{renames[d]} 章"
        return m.group(0)
    handlers.append((re.compile(r"(?<![\d.])(\d+)\s*章(?![\d.])"), h_zhang))

    return handlers


def phase_b_apply(text: str, renames: dict[str, str]) -> tuple[str, int]:
    handlers = make_phase_b_handlers(renames)
    total = 0
    for pattern, handler in handlers:
        new, n = pattern.subn(handler, text)
        total += n
        text = new
    return text, total


def main() -> None:
    print("=== 阶段 A: 目录名替换 ===")
    print(f"替换表(共 {len(DIR_RENAMES)} 项):")
    for old, new in DIR_RENAMES.items():
        print(f"  {old:25s}  ->  {new}")
    print()

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
        new_content, count = phase_a_replace(content)
        if count == 0:
            continue
        path.write_text(new_content, encoding="utf-8")
        a_files.append((path.relative_to(ROOT), count))
        a_total += count

    print(f"阶段 A: 修改 {len(a_files)} 个文件,总替换 {a_total} 次")
    print()

    print("=== 阶段 B: 数字章节号替换(只对 doc/*.md) ===")
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

    print(f"阶段 B: 修改 {len(b_files)} 个文件,总替换 {b_total} 次")
    print()

    print("=== 阶段 A 修改的文件清单 ===")
    for rel, count in a_files:
        print(f"  {rel}  ({count} 处)")
    print()
    print("=== 阶段 B 修改的文件清单 ===")
    for rel, count in b_files:
        print(f"  {rel}  ({count} 处)")


if __name__ == "__main__":
    main()
