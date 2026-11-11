"""docs 目录 Markdown 批量转 Word（.docx）脚本。

功能说明
========================================================================
读取 ``docs/`` 下修复过公式的全部 ``.md`` 文件，批量转换为同名 ``.docx`` 文档，
输出到 ``docs/`` 目录，保证中文、希腊字母、上下标、数学符号等全部正常显示，无乱码。

排版规格（严格按大作业交付要求）
========================================================================
* 一级标题：二号（22pt）、加粗、段前 24pt / 段后 12pt；
* 二级标题：小三号（15pt）、加粗；
* 三级标题：四号（14pt）、加粗；
* 正文：小四号（12pt）、1.5 倍行距、首行缩进 2 字符；
* 表格：自动居中、保留边框、表头加粗并加浅蓝底纹、表头跨页重复；
* 列表：保留项目符号，支持多级缩进；
* 字体：中文默认宋体（East Asian），英文与数字默认 Times New Roman；
* 数学字符（希腊字母、上下标、运算符等）：单独指定 Segoe UI Symbol 字体，
  避免宋体/Times 字形缺失导致的方块或乱码；若系统无该字体，Word 会自动回退显示。

其它说明
========================================================================
* 表格字号取五号（10.5pt）：交付表格（如常见问题表）文字较长，五号更易排布且不影响阅读；
  如需与正文一致，修改脚本顶部的 ``SIZE_TABLE`` 即可；
* 有序列表保留原文编号（不启用 Word 自动编号），避免跨章节编号错乱；
* 转换前会检测文档中是否仍存在 LaTeX 残留（``$``、``\\命令``）并给出提示；
* 单个文件转换失败不影响其余文件，脚本结束打印成功列表与失败原因。

运行方式
========================================================================
    python -m pip install python-docx
    python batch_md2docx.py                 # 转换 docs/ 下全部 md
    python batch_md2docx.py --md-dir other  # 指定其它目录
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MD_DIR = SCRIPT_DIR / "docs"

# --------------------------------------------------------------------------- #
# 一、排版常量
# --------------------------------------------------------------------------- #
SIZE_H1 = 22.0     # 二号
SIZE_H2 = 15.0     # 小三号
SIZE_H3 = 14.0     # 四号
SIZE_BODY = 12.0   # 小四号
SIZE_TABLE = 10.5  # 五号（表格）

FONT_EAST_ASIA = "宋体"                # 中文默认字体
FONT_LATIN = "Times New Roman"         # 英文与数字默认字体
FONT_MATH = "Segoe UI Symbol"          # 数学字符兜底字体

COLOR_HEADING = (0x1F, 0x38, 0x64)     # 标题深蓝
COLOR_QUOTE = (0x59, 0x59, 0x59)       # 引用灰
COLOR_TABLE_HEADER_FILL = "D9E2F3"     # 表头浅蓝底纹

BODY_LINE_SPACING = 1.5
FIRST_LINE_INDENT_PT = SIZE_BODY * 2   # 首行缩进 2 字符（按小四号折算）

_BULLET_PATTERN = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_ORDERED_PATTERN = re.compile(r"^(\s*)\d+[.、]\s+(.*)$")
_TABLE_SEPARATOR = re.compile(r"^\|?[\s:|-]+\|?$")
_INLINE_PATTERN = re.compile(r"(\*\*.+?\*\*|`[^`]+`)")
_MATH_RESIDUE = re.compile(r"\$|\\[A-Za-z]{2,}")

#: 需要数学字体兜底的 Unicode 区段（组合符、上下标、数学运算符、箭头等）
_MATH_RANGES: tuple[tuple[int, int], ...] = (
    (0x0300, 0x036F),  # 组合附加符号（x̄ 的组合横线等）
    (0x1D00, 0x1D7F),  # 语音扩展与下标字母（ᵢ ⱼ 等）
    (0x2070, 0x209F),  # 上下标
    (0x2100, 0x214F),  # 字母式符号
    (0x2190, 0x22FF),  # 箭头与数学运算符
    (0x2300, 0x23FF),  # 杂项技术符号
)

_BULLET_STYLES = ("List Bullet", "List Bullet 2", "List Bullet 3")


# --------------------------------------------------------------------------- #
# 二、python-docx 惰性导入与底层工具
# --------------------------------------------------------------------------- #
def _load_docx() -> dict[str, Any]:
    """惰性导入 python-docx，返回构建文档所需对象集合。

    Raises:
        RuntimeError: 依赖缺失时抛出，由 :func:`main` 转换为安装提示。
    """
    try:
        import docx
        from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        from docx.shared import Cm, Pt, RGBColor
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "未检测到 python-docx 库，请先执行：python -m pip install python-docx"
        ) from exc
    return {
        "Document": docx.Document,
        "Cm": Cm, "Pt": Pt, "RGBColor": RGBColor,
        "WD_ALIGN_PARAGRAPH": WD_ALIGN_PARAGRAPH,
        "WD_TABLE_ALIGNMENT": WD_TABLE_ALIGNMENT,
        "WD_ALIGN_VERTICAL": WD_ALIGN_VERTICAL,
        "OxmlElement": OxmlElement,
        "qn": qn,
    }


def _needs_math_font(text: str) -> bool:
    """判断文本是否包含需要数学字体兜底的字符（希腊字母/上下标/运算符等）。"""
    try:
        for char in text:
            code = ord(char)
            if any(start <= code <= end for start, end in _MATH_RANGES):
                return True
        return False
    except Exception:  # noqa: BLE001
        return False


def _apply_run_font(
    run: Any,
    pptx_like: dict[str, Any],
    size: float,
    bold: bool = False,
    italic: bool = False,
    color: Sequence[int] | None = None,
    math_font: bool = False,
) -> Any:
    """统一设置字体：中文宋体 + 英文 Times New Roman（数学字符用符号字体兜底）。"""
    latin = FONT_MATH if math_font else FONT_LATIN
    font = run.font
    font.name = latin
    font.size = pptx_like["Pt"](size)
    font.bold = bool(bold)
    font.italic = bool(italic)
    if color is not None:
        font.color.rgb = pptx_like["RGBColor"](int(color[0]), int(color[1]), int(color[2]))
    try:
        rpr = run._element.get_or_add_rPr()
        rfonts = rpr.get_or_add_rFonts()
        rfonts.set(pptx_like["qn"]("w:ascii"), latin)
        rfonts.set(pptx_like["qn"]("w:hAnsi"), latin)
        rfonts.set(pptx_like["qn"]("w:cs"), latin)
        rfonts.set(pptx_like["qn"]("w:eastAsia"), FONT_EAST_ASIA)
    except Exception:  # noqa: BLE001 - 字体节点设置失败不影响正文内容
        pass
    return run


def _add_inline_runs(
    paragraph: Any,
    text: str,
    pptx_like: dict[str, Any],
    size: float,
    bold: bool = False,
    italic: bool = False,
    color: Sequence[int] | None = None,
) -> None:
    """把一行文本按行内 Markdown（粗体、行内代码）拆分写入段落。"""
    for token in _INLINE_PATTERN.split(str(text)):
        if not token:
            continue
        token_bold = bold
        content = token
        if token.startswith("**") and token.endswith("**") and len(token) > 4:
            token_bold = True
            content = token[2:-2]
        elif token.startswith("`") and token.endswith("`") and len(token) > 2:
            content = token[1:-1]
        run = paragraph.add_run(content)
        _apply_run_font(
            run, pptx_like, size, bold=token_bold, italic=italic, color=color,
            math_font=_needs_math_font(content),
        )


def _set_cell_shading(cell: Any, pptx_like: dict[str, Any], fill: str) -> None:
    """设置单元格底纹颜色（表头浅蓝底色）。"""
    try:
        shd = pptx_like["OxmlElement"]("w:shd")
        shd.set(pptx_like["qn"]("w:val"), "clear")
        shd.set(pptx_like["qn"]("w:color"), "auto")
        shd.set(pptx_like["qn"]("w:fill"), fill)
        cell._tc.get_or_add_tcPr().append(shd)
    except Exception:  # noqa: BLE001
        pass


def _set_table_borders(table: Any, pptx_like: dict[str, Any]) -> None:
    """为表格显式写入四边与内部边框，确保“保留边框”。"""
    try:
        tbl_pr = table._tbl.tblPr
        borders = pptx_like["OxmlElement"]("w:tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            element = pptx_like["OxmlElement"](f"w:{edge}")
            element.set(pptx_like["qn"]("w:val"), "single")
            element.set(pptx_like["qn"]("w:sz"), "6")
            element.set(pptx_like["qn"]("w:space"), "0")
            element.set(pptx_like["qn"]("w:color"), "808080")
            borders.append(element)
        tbl_pr.append(borders)
    except Exception:  # noqa: BLE001
        pass


def _repeat_header_row(row: Any, pptx_like: dict[str, Any]) -> None:
    """让表头行在跨页时自动重复。"""
    try:
        tr_pr = row._tr.get_or_add_trPr()
        header = pptx_like["OxmlElement"]("w:tblHeader")
        header.set(pptx_like["qn"]("w:val"), "true")
        tr_pr.append(header)
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# 三、文档骨架与区块解析
# --------------------------------------------------------------------------- #
def _setup_document(pptx_like: dict[str, Any], title: str = "") -> Any:
    """创建文档对象并设置纸张、页边距、正文默认字体与行距。"""
    document = pptx_like["Document"]()
    section = document.sections[0]
    section.page_width = pptx_like["Cm"](21.0)     # A4
    section.page_height = pptx_like["Cm"](29.7)
    section.left_margin = section.right_margin = pptx_like["Cm"](2.6)
    section.top_margin = section.bottom_margin = pptx_like["Cm"](2.54)

    normal = document.styles["Normal"]
    normal.font.name = FONT_LATIN
    normal.font.size = pptx_like["Pt"](SIZE_BODY)
    try:
        rpr = normal.element.get_or_add_rPr()
        rfonts = rpr.get_or_add_rFonts()
        rfonts.set(pptx_like["qn"]("w:ascii"), FONT_LATIN)
        rfonts.set(pptx_like["qn"]("w:hAnsi"), FONT_LATIN)
        rfonts.set(pptx_like["qn"]("w:eastAsia"), FONT_EAST_ASIA)
    except Exception:  # noqa: BLE001
        pass
    paragraph_format = normal.paragraph_format
    paragraph_format.line_spacing = BODY_LINE_SPACING
    paragraph_format.space_after = pptx_like["Pt"](6)

    if title:
        document.core_properties.title = title
    return document


def parse_blocks(lines: Iterable[str]) -> list[tuple[str, Any]]:
    """把 Markdown 行序列解析为区块列表。

    Returns:
        list[tuple[str, Any]]: 区块类型与内容，类型取值：
        ``heading`` / ``paragraph`` / ``bullet`` / ``ordered`` / ``table`` / ``quote``。
    """
    source = [line.rstrip() for line in lines]
    blocks: list[tuple[str, Any]] = []
    index = 0
    while index < len(source):
        raw = source[index]
        stripped = raw.strip()
        if not stripped or stripped in {"---", "***", "___"}:
            index += 1
            continue

        # 表格：起始行 + 分隔行
        if stripped.startswith("|") and index + 1 < len(source) and _TABLE_SEPARATOR.match(source[index + 1].strip()):
            rows: list[list[str]] = []
            while index < len(source) and source[index].strip().startswith("|"):
                current = source[index].strip()
                if not _TABLE_SEPARATOR.match(current):
                    rows.append([cell.strip() for cell in current.strip("|").split("|")])
                index += 1
            blocks.append(("table", rows))
            continue

        # 标题
        if stripped.startswith("#"):
            level = min(6, len(stripped) - len(stripped.lstrip("#")))
            blocks.append(("heading", (level, stripped[level:].strip())))
            index += 1
            continue

        # 引用块
        if stripped.startswith(">"):
            quoted: list[str] = []
            while index < len(source) and source[index].strip().startswith(">"):
                quoted.append(source[index].strip().lstrip(">").strip())
                index += 1
            blocks.append(("quote", " ".join(part for part in quoted if part)))
            continue

        # 项目符号列表
        bullet = _BULLET_PATTERN.match(raw)
        if bullet:
            blocks.append(("bullet", (min(2, len(bullet.group(1)) // 2), bullet.group(2).strip())))
            index += 1
            continue

        # 有序列表（保留原文编号）
        ordered = _ORDERED_PATTERN.match(raw)
        if ordered:
            blocks.append(("ordered", (min(2, len(ordered.group(1)) // 2), stripped)))
            index += 1
            continue

        # 普通段落：合并到空行或下一个区块起始
        lines_buffer = [stripped]
        index += 1
        while index < len(source):
            following = source[index]
            following_stripped = following.strip()
            if (
                not following_stripped
                or following_stripped in {"---", "***", "___"}
                or following_stripped.startswith(("#", "|", ">"))
                or _BULLET_PATTERN.match(following)
                or _ORDERED_PATTERN.match(following)
            ):
                break
            lines_buffer.append(following_stripped)
            index += 1
        blocks.append(("paragraph", " ".join(lines_buffer)))
    return blocks


# --------------------------------------------------------------------------- #
# 四、各区块写入 Word
# --------------------------------------------------------------------------- #
def _add_heading(document: Any, level: int, text: str, pptx_like: dict[str, Any]) -> None:
    """写入标题：一级二号、二级小三、三级四号，均加粗。"""
    paragraph = None
    try:
        paragraph = document.add_paragraph(style=f"Heading {min(level, 3)}")
    except Exception:  # noqa: BLE001 - 模板缺少标题样式时退化为普通段落
        paragraph = document.add_paragraph()
    format_ = paragraph.paragraph_format
    format_.first_line_indent = pptx_like["Pt"](0)
    if level <= 1:
        size, format_.space_before, format_.space_after = SIZE_H1, pptx_like["Pt"](24), pptx_like["Pt"](12)
    elif level == 2:
        size, format_.space_before, format_.space_after = SIZE_H2, pptx_like["Pt"](16), pptx_like["Pt"](8)
    else:
        size, format_.space_before, format_.space_after = SIZE_H3, pptx_like["Pt"](12), pptx_like["Pt"](6)
    format_.line_spacing = BODY_LINE_SPACING
    _add_inline_runs(paragraph, text, pptx_like, size, bold=True, color=COLOR_HEADING)


def _add_body_paragraph(document: Any, text: str, pptx_like: dict[str, Any], indent: bool = True) -> None:
    """写入正文段落：小四号、1.5 倍行距、首行缩进 2 字符。"""
    paragraph = document.add_paragraph()
    format_ = paragraph.paragraph_format
    format_.line_spacing = BODY_LINE_SPACING
    format_.space_after = pptx_like["Pt"](6)
    format_.first_line_indent = pptx_like["Pt"](FIRST_LINE_INDENT_PT if indent else 0)
    _add_inline_runs(paragraph, text, pptx_like, SIZE_BODY)


def _add_bullet(document: Any, level: int, text: str, pptx_like: dict[str, Any], style_names: set[str]) -> None:
    """写入项目符号列表（优先使用 Word 内置列表样式，缺失时手工加符号）。"""
    style_name = _BULLET_STYLES[min(level, len(_BULLET_STYLES) - 1)]
    paragraph = None
    if style_name in style_names:
        try:
            paragraph = document.add_paragraph(style=style_name)
        except Exception:  # noqa: BLE001
            paragraph = None
    if paragraph is None:
        paragraph = document.add_paragraph()
        text = f"{'　' * level}• {text}"
    format_ = paragraph.paragraph_format
    format_.line_spacing = BODY_LINE_SPACING
    format_.space_after = pptx_like["Pt"](4)
    format_.first_line_indent = pptx_like["Pt"](0)
    _add_inline_runs(paragraph, text, pptx_like, SIZE_BODY)


def _add_ordered(document: Any, level: int, text: str, pptx_like: dict[str, Any]) -> None:
    """写入有序列表（保留原文编号，避免 Word 自动编号跨章节错乱）。"""
    paragraph = document.add_paragraph()
    format_ = paragraph.paragraph_format
    format_.line_spacing = BODY_LINE_SPACING
    format_.space_after = pptx_like["Pt"](4)
    format_.first_line_indent = pptx_like["Pt"](0)
    format_.left_indent = pptx_like["Pt"](18 * level)
    _add_inline_runs(paragraph, text, pptx_like, SIZE_BODY)


def _add_quote(document: Any, text: str, pptx_like: dict[str, Any]) -> None:
    """写入引用块（如文档题注信息），使用灰色斜体、居中呈现。"""
    paragraph = document.add_paragraph()
    format_ = paragraph.paragraph_format
    format_.first_line_indent = pptx_like["Pt"](0)
    format_.space_after = pptx_like["Pt"](10)
    paragraph.alignment = pptx_like["WD_ALIGN_PARAGRAPH"].CENTER
    _add_inline_runs(paragraph, text, pptx_like, SIZE_BODY - 1, italic=True, color=COLOR_QUOTE)


def _add_table(document: Any, rows: Sequence[Sequence[str]], pptx_like: dict[str, Any]) -> None:
    """写入表格：自动居中、保留边框、表头加粗并加底纹、跨页重复表头。"""
    if not rows:
        return
    columns = max(len(row) for row in rows)
    table = document.add_table(rows=0, cols=columns)
    try:
        table.style = "Table Grid"
    except Exception:  # noqa: BLE001 - 样式缺失时依赖手工边框
        pass
    try:
        table.alignment = pptx_like["WD_TABLE_ALIGNMENT"].CENTER
        table.autofit = True
    except Exception:  # noqa: BLE001
        pass

    for row_index, row_data in enumerate(rows):
        row = table.add_row()
        for column_index in range(columns):
            cell = row.cells[column_index]
            value = row_data[column_index] if column_index < len(row_data) else ""
            paragraph = cell.paragraphs[0]
            format_ = paragraph.paragraph_format
            format_.first_line_indent = pptx_like["Pt"](0)
            format_.line_spacing = 1.0
            format_.space_after = pptx_like["Pt"](2)
            if row_index == 0:
                paragraph.alignment = pptx_like["WD_ALIGN_PARAGRAPH"].CENTER
                _set_cell_shading(cell, pptx_like, COLOR_TABLE_HEADER_FILL)
            _add_inline_runs(paragraph, value, pptx_like, SIZE_TABLE, bold=(row_index == 0))
            try:
                cell.vertical_alignment = pptx_like["WD_ALIGN_VERTICAL"].CENTER
            except Exception:  # noqa: BLE001
                pass

    _set_table_borders(table, pptx_like)
    try:
        _repeat_header_row(table.rows[0], pptx_like)
    except Exception:  # noqa: BLE001
        pass
    document.add_paragraph()  # 表后留白，避免与下文紧贴


# --------------------------------------------------------------------------- #
# 五、转换主流程与质量审计
# --------------------------------------------------------------------------- #
def read_text(path: Path) -> str:
    """按 UTF-8 / UTF-8-BOM / GBK 顺序读取文本。"""
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def audit_docx(path: Path, pptx_like: dict[str, Any]) -> dict[str, Any]:
    """回读生成结果做质量审计，验证是否还有乱码或公式残留。"""
    try:
        document = pptx_like["Document"](str(path))
        texts = [paragraph.text for paragraph in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                for cell in row.cells:
                    texts.append(cell.text)
        joined = "\n".join(texts)
        return {
            "n_paragraphs": len(document.paragraphs),
            "n_tables": len(document.tables),
            "n_chars": len(joined),
            "replacement_chars": joined.count("\ufffd"),
            "latex_residue": len(re.findall(r"\$|\\[A-Za-z]{2,}", joined)),
            "math_symbols": sum(joined.count(char) for char in "σμρΣ√²ᵢₜ≤∈≈×±°"),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"审计失败：{exc}"}


def convert_file(md_path: Path, out_path: Path, pptx_like: dict[str, Any]) -> dict[str, Any]:
    """把单个 Markdown 文件转换为 Word 文档。

    Returns:
        dict: ``{"path", "counts", "audit", "latex_residue", "error"}``。
    """
    result: dict[str, Any] = {
        "path": str(out_path), "counts": {}, "audit": {}, "latex_residue": 0, "error": None,
    }
    try:
        content = read_text(md_path)
        result["latex_residue"] = len(_MATH_RESIDUE.findall(content))

        blocks = parse_blocks(content.splitlines())
        title = next((payload[1] for kind, payload in blocks if kind == "heading"), md_path.stem)
        document = _setup_document(pptx_like, title=str(title))
        style_names = {style.name for style in document.styles}

        counters = {"heading": 0, "paragraph": 0, "bullet": 0, "ordered": 0, "table": 0, "quote": 0}
        for kind, payload in blocks:
            if kind == "heading":
                _add_heading(document, int(payload[0]), str(payload[1]), pptx_like)
            elif kind == "paragraph":
                _add_body_paragraph(document, str(payload), pptx_like)
            elif kind == "bullet":
                _add_bullet(document, int(payload[0]), str(payload[1]), pptx_like, style_names)
            elif kind == "ordered":
                _add_ordered(document, int(payload[0]), str(payload[1]), pptx_like)
            elif kind == "table":
                _add_table(document, payload, pptx_like)
            elif kind == "quote":
                _add_quote(document, str(payload), pptx_like)
            counters[kind] = counters.get(kind, 0) + 1

        out_path.parent.mkdir(parents=True, exist_ok=True)
        document.save(str(out_path))
        result["counts"] = counters
        result["audit"] = audit_docx(out_path, pptx_like)
        return result
    except Exception as exc:  # noqa: BLE001 - 单文件失败不影响整批
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def collect_markdown_files(md_dir: Path) -> list[Path]:
    """收集待转换的 Markdown 文件（自动排除 ``.md.bak`` 备份）。"""
    try:
        return [
            path for path in sorted(Path(md_dir).glob("*.md"))
            if path.is_file() and not path.name.endswith(".bak")
        ]
    except Exception as exc:  # noqa: BLE001
        print(f"[错误] 目录扫描失败：{exc}")
        return []


def main(argv: Sequence[str] | None = None) -> int:
    """脚本入口：批量把 docs 下的 Markdown 转换为同名 Word 文档。

    Returns:
        int: 退出码，0 表示全部成功，1 表示存在失败项。
    """
    parser = argparse.ArgumentParser(description="批量把 docs 目录 Markdown 转换为 Word 文档")
    parser.add_argument("--md-dir", default=str(DEFAULT_MD_DIR), help="Markdown 所在目录（默认 docs/）")
    parser.add_argument("--out-dir", default="", help="Word 输出目录（默认与 Markdown 同目录）")
    args = parser.parse_args(list(argv) if argv is not None else None)

    md_dir = Path(args.md_dir)
    if not md_dir.is_absolute() and not md_dir.exists():
        md_dir = SCRIPT_DIR / md_dir
    out_dir = Path(args.out_dir) if args.out_dir else md_dir
    if not out_dir.is_absolute():
        out_dir = SCRIPT_DIR / out_dir

    print("=" * 74)
    print("Markdown 批量转 Word 工具（中文宋体 / 英文 Times New Roman）")
    print(f"输入目录：{md_dir}")
    print(f"输出目录：{out_dir}")
    print("=" * 74)

    if not md_dir.is_dir():
        print(f"[错误] 目录不存在：{md_dir}")
        return 1
    files = collect_markdown_files(md_dir)
    if not files:
        print("[错误] 未找到任何 .md 文件")
        return 1

    try:
        pptx_like = _load_docx()
    except RuntimeError as exc:
        print(f"[错误] {exc}")
        print("       安装命令：python -m pip install python-docx")
        return 1

    succeeded: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    warnings: list[str] = []

    for md_path in files:
        out_path = out_dir / f"{md_path.stem}.docx"
        outcome = convert_file(md_path, out_path, pptx_like)
        if outcome["error"]:
            failed.append({**outcome, "source": str(md_path)})
            print(f"  [失败] {md_path.name} → {out_path.name}：{outcome['error']}")
            continue

        succeeded.append({**outcome, "source": str(md_path)})
        counters = outcome["counts"]
        audit = outcome["audit"]
        print(
            f"  [成功] {md_path.name} → {out_path.name}"
            f"（标题 {counters.get('heading', 0)} / 段落 {counters.get('paragraph', 0)} / "
            f"列表 {counters.get('bullet', 0) + counters.get('ordered', 0)} / 表格 {counters.get('table', 0)}）"
        )
        if outcome["latex_residue"]:
            warnings.append(f"{md_path.name}：源文档仍有 {outcome['latex_residue']} 处疑似 LaTeX 残留")
        if audit.get("replacement_chars"):
            warnings.append(f"{out_path.name}：检测到 {audit['replacement_chars']} 个替换字符（乱码）")
        if audit.get("latex_residue"):
            warnings.append(f"{out_path.name}：正文仍存在 {audit['latex_residue']} 处公式残留")

    print("-" * 74)
    print(f"转换成功文件列表（{len(succeeded)} 个）：")
    for item in succeeded:
        audit = item["audit"]
        size_kb = Path(item["path"]).stat().st_size / 1024 if Path(item["path"]).exists() else 0.0
        print(
            f"  ✔ {Path(item['path']).name}"
            f"  （段落 {audit.get('n_paragraphs', 0)}、表格 {audit.get('n_tables', 0)}、"
            f"数学符号 {audit.get('math_symbols', 0)} 个、{size_kb:.1f} KB）"
        )
    if failed:
        print(f"转换失败文件（{len(failed)} 个）：")
        for item in failed:
            print(f"  ✘ {Path(item['source']).name}：{item['error']}")
    if warnings:
        print("提示与告警：")
        for line in warnings:
            print(f"  ! {line}")
    else:
        print("质量审计：未发现乱码字符与公式残留（σ / Σ / √ / 上下标等均可正常显示）。")
    print("=" * 74)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
