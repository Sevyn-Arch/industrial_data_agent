"""工业数据分析 Agent 系统 —— 项目简介 PPT 自动生成脚本。

功能说明
========================================================================
读取 ``docs/项目简介PPT文案.md``，按 Markdown 标题层级自动生成一份标准项目简介 PPT，
保存为当前目录下的 ``工业数据分析Agent系统_项目简介.pptx``。

Markdown 解析约定
========================================================================
* ``# 一级标题``            ：PPT 主标题（用于标题页）；
* ``## 第 N 页：页面名称``   ：一页幻灯片，页面名称用于目录页；
* ``### 页面标题``           ：该页在幻灯片上显示的标题；
* ``### 正文内容``           ：该页正文要点，``- `` 列表项逐条呈现；
* ``### 排版建议``           ：不在幻灯片上显示，写入演讲者备注（备注栏）；
* 标题含"封面"的页          ：内容并入标题页，不再单独生成正文页；
* 脚本自动追加               ：目录页（依据各页名称生成）与结束页。

版式说明
========================================================================
脚本内置四种版式：标题页、目录页、正文页、结束页；配色为工业风蓝灰色系
（深海军蓝 + 工业蓝 + 浅灰蓝底纹 + 中性灰文字），全部文本采用"手工文本框"方式
精确控制字号、颜色与位置，避免依赖模板文件。

运行方式
========================================================================
    python -m pip install python-pptx
    python generate_ppt.py

可选参数：
    --md   指定 Markdown 文案路径（默认 docs/项目简介PPT文案.md）
    --out  指定输出 pptx 路径（默认 ./工业数据分析Agent系统_项目简介.pptx）
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

# --------------------------------------------------------------------------- #
# 常量配置
# --------------------------------------------------------------------------- #
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MD = SCRIPT_DIR / "docs" / "项目简介PPT文案.md"
DEFAULT_OUT = "工业数据分析Agent系统_项目简介.pptx"

#: 工业风蓝灰色系配色
COLOR_PRIMARY = (0x1F, 0x38, 0x64)   # 深海军蓝：标题页/页眉底色
COLOR_ACCENT = (0x2E, 0x75, 0xB6)    # 工业蓝：强调元素
COLOR_PANEL = (0xF2, 0xF5, 0xF9)     # 浅灰蓝：正文底纹
COLOR_TEXT = (0x26, 0x26, 0x26)      # 正文深灰
COLOR_MUTED = (0x7F, 0x7F, 0x7F)     # 辅助灰
COLOR_WHITE = (0xFF, 0xFF, 0xFF)
COLOR_LIGHT = (0xD6, 0xE1, 0xEF)     # 浅蓝：深色底上的次级文字

#: 字体（拉丁与数字走 Segoe UI，中文走微软雅黑）
FONT_LATIN = "Segoe UI"
FONT_EAST_ASIAN = "微软雅黑"

#: 报告落款信息
FOOTER_TEXT = "工业数据分析 Agent 系统 ｜ 和利时企业命题"


# --------------------------------------------------------------------------- #
# 一、Markdown 解析
# --------------------------------------------------------------------------- #
_PAGE_PREFIX = re.compile(r"^第\s*\d+\s*页\s*[：:]\s*")
_ORDERED_ITEM = re.compile(r"^\d+[.、]\s*")
_INLINE_MARKS = re.compile(r"(\*\*|__|\*|`)")


def clean_inline(text: str) -> str:
    """清理行内 Markdown 标记（粗体、斜体、行内代码）与多余空白。"""
    try:
        cleaned = _INLINE_MARKS.sub("", str(text))
        return cleaned.strip()
    except Exception:  # noqa: BLE001 - 清洗失败时原样返回
        return str(text).strip()


@dataclass
class SlideContent:
    """单页幻灯片的内容容器。"""

    block_title: str = ""
    title: str = ""
    bullets: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def head_title(self) -> str:
        """幻灯片显示标题：优先使用「页面标题」，缺失时回退到页面名称。"""
        return self.title or self.block_title or "未命名页面"

    @property
    def note_text(self) -> str:
        """演讲者备注文本。"""
        return "\n".join(self.notes)


@dataclass
class DeckContent:
    """整份文案的解析结果。"""

    main_title: str = ""
    cover_lines: list[str] = field(default_factory=list)
    slides: list[SlideContent] = field(default_factory=list)
    generated_at: str = ""


def parse_markdown(md_text: str) -> DeckContent:
    """解析 Markdown 文案，返回结构化内容（界面/PPT 生成共用）。

    Args:
        md_text: Markdown 文案全文。

    Returns:
        DeckContent: 主标题、封面信息与逐页内容。
    """
    deck = DeckContent(generated_at=datetime.now().strftime("%Y-%m-%d"))
    cover: SlideContent | None = None
    current: SlideContent | None = None
    section = ""

    try:
        for raw_line in str(md_text or "").splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped or stripped in {"---", "***", "___"} or stripped.startswith("|"):
                continue

            if stripped.startswith("# ") and not stripped.startswith("## "):
                deck.main_title = clean_inline(stripped[2:])
                continue

            if stripped.startswith("## "):
                block = clean_inline(stripped[3:])
                block_title = _PAGE_PREFIX.sub("", block).strip() or block
                current = SlideContent(block_title=block_title)
                if "封面" in block_title:
                    cover = current
                else:
                    deck.slides.append(current)
                section = ""
                continue

            if stripped.startswith("### "):
                section = clean_inline(stripped[4:])
                continue

            if current is None:
                continue

            if stripped.startswith(("- ", "* ", "+ ")):
                text = clean_inline(stripped[2:])
            elif _ORDERED_ITEM.match(stripped):
                text = clean_inline(_ORDERED_ITEM.sub("", stripped))
            elif stripped.startswith(">"):
                continue  # 引用块不进入正文
            else:
                text = clean_inline(stripped)

            if not text:
                continue
            if "页面标题" in section:
                current.title = text
            elif "排版建议" in section or "备注" in section:
                current.notes.append(text)
            else:
                current.bullets.append(text)

        if cover is not None:
            deck.cover_lines = list(cover.bullets) + list(cover.notes)
        if not deck.main_title:
            deck.main_title = "项目简介"
        return deck
    except Exception as exc:  # noqa: BLE001 - 解析异常时返回已解析部分
        print(f"[警告] Markdown 解析出现异常，已保留已解析内容：{exc}", file=sys.stderr)
        return deck


# --------------------------------------------------------------------------- #
# 二、PPT 构建
# --------------------------------------------------------------------------- #
def build_presentation(deck: DeckContent, pptx_module: Any) -> Any:
    """根据解析结果构建 PPT 演示文稿对象。

    Args:
        deck: :func:`parse_markdown` 的解析结果。
        pptx_module: 已导入的 pptx 相关对象集合（便于统一注入与测试）。

    Returns:
        Presentation: python-pptx 演示文稿对象。
    """
    Presentation = pptx_module["Presentation"]
    prs = Presentation()
    prs.slide_width = pptx_module["Inches"](13.333)   # 16:9
    prs.slide_height = pptx_module["Inches"](7.5)

    _add_title_slide(prs, deck, pptx_module)
    _add_toc_slide(prs, deck, pptx_module)
    total = len(deck.slides)
    for index, slide in enumerate(deck.slides, start=1):
        _add_content_slide(prs, slide, index, total, pptx_module)
    _add_closing_slide(prs, deck, pptx_module)
    return prs


def _load_pptx() -> dict[str, Any]:
    """惰性导入 python-pptx，返回构建 PPT 所需的对象集合。

    Raises:
        RuntimeError: 依赖缺失时抛出，由 :func:`main` 转换为友好的安装提示。
    """
    try:
        from pptx import Presentation
        from pptx.dml.color import RGBColor
        from pptx.enum.shapes import MSO_SHAPE
        from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
        from pptx.oxml.ns import qn
        from pptx.util import Emu, Inches, Pt
    except ImportError as exc:  # pragma: no cover - 环境缺少依赖时触发
        raise RuntimeError(
            "未检测到 python-pptx 库，请先执行：python -m pip install python-pptx"
        ) from exc
    return {
        "Presentation": Presentation,
        "RGBColor": RGBColor,
        "MSO_SHAPE": MSO_SHAPE,
        "MSO_ANCHOR": MSO_ANCHOR,
        "PP_ALIGN": PP_ALIGN,
        "qn": qn,
        "Emu": Emu,
        "Inches": Inches,
        "Pt": Pt,
    }


# ------------------------------ 低层绘制工具 ------------------------------ #
def _rgb(pptx_module: dict[str, Any], color: Sequence[int]) -> Any:
    """把 ``(R, G, B)`` 元组转换为 python-pptx 颜色对象。"""
    return pptx_module["RGBColor"](int(color[0]), int(color[1]), int(color[2]))


def _add_rect(
    slide: Any,
    pptx_module: dict[str, Any],
    left: float,
    top: float,
    width: float,
    height: float,
    fill: Sequence[int] | None = None,
) -> Any:
    """绘制无边框矩形色块（用于页眉、底纹、装饰条）。"""
    shape = slide.shapes.add_shape(
        pptx_module["MSO_SHAPE"].RECTANGLE,
        pptx_module["Inches"](left),
        pptx_module["Inches"](top),
        pptx_module["Inches"](width),
        pptx_module["Inches"](height),
    )
    if fill is None:
        shape.fill.background()
    else:
        shape.fill.solid()
        shape.fill.fore_color.rgb = _rgb(pptx_module, fill)
    shape.line.fill.background()
    try:  # 关闭默认阴影，保持工业风扁平观感
        shape.shadow.inherit = False
    except Exception:  # noqa: BLE001 - 个别版本不支持该属性
        pass
    return shape


def _add_textbox(
    slide: Any,
    pptx_module: dict[str, Any],
    left: float,
    top: float,
    width: float,
    height: float,
    anchor: str = "top",
) -> Any:
    """添加一个自动换行的文本框，并设置垂直对齐方式。"""
    box = slide.shapes.add_textbox(
        pptx_module["Inches"](left),
        pptx_module["Inches"](top),
        pptx_module["Inches"](width),
        pptx_module["Inches"](height),
    )
    frame = box.text_frame
    frame.word_wrap = True
    try:
        frame.vertical_anchor = (
            pptx_module["MSO_ANCHOR"].MIDDLE if anchor == "middle" else pptx_module["MSO_ANCHOR"].TOP
        )
    except Exception:  # noqa: BLE001
        pass
    return box


def _set_east_asian_font(run: Any, pptx_module: dict[str, Any], typeface: str) -> None:
    """为文字设置东亚字体（python-pptx 未提供公开 API，需写入 XML 的 ea 节点）。"""
    try:
        qn = pptx_module["qn"]
        rpr = run._r.get_or_add_rPr()
        latin = rpr.find(qn("a:latin"))
        ea = rpr.find(qn("a:ea"))
        if ea is None:
            ea = rpr.makeelement(qn("a:ea"), {})
            if latin is not None:
                latin.addnext(ea)  # 按 OOXML 顺序置于 latin 之后
            else:
                rpr.append(ea)
        ea.set("typeface", typeface)
    except Exception:  # noqa: BLE001 - 中文字体设置失败不影响其余样式
        pass


def _style_run(
    run: Any,
    pptx_module: dict[str, Any],
    text: str,
    size: float,
    bold: bool = False,
    color: Sequence[int] = COLOR_TEXT,
    italic: bool = False,
) -> Any:
    """统一设置文本段落的字体、字号、字重与颜色。"""
    run.text = str(text)
    run.font.size = pptx_module["Pt"](size)
    run.font.bold = bool(bold)
    run.font.italic = bool(italic)
    run.font.name = FONT_LATIN
    run.font.color.rgb = _rgb(pptx_module, color)
    _set_east_asian_font(run, pptx_module, FONT_EAST_ASIAN)
    return run


def _add_text(
    slide: Any,
    pptx_module: dict[str, Any],
    text: str,
    left: float,
    top: float,
    width: float,
    height: float,
    size: float = 18,
    bold: bool = False,
    color: Sequence[int] = COLOR_TEXT,
    align: str = "left",
    anchor: str = "top",
    italic: bool = False,
) -> Any:
    """在指定位置写入一段文本（单段落），返回文本框。"""
    box = _add_textbox(slide, pptx_module, left, top, width, height, anchor=anchor)
    paragraph = box.text_frame.paragraphs[0]
    if align == "center":
        paragraph.alignment = pptx_module["PP_ALIGN"].CENTER
    elif align == "right":
        paragraph.alignment = pptx_module["PP_ALIGN"].RIGHT
    _style_run(paragraph.add_run(), pptx_module, text, size, bold=bold, color=color, italic=italic)
    return box


def _bullet_font_size(bullets: Iterable[str]) -> float:
    """按正文总量自适应字号，避免文字溢出幻灯片。"""
    try:
        total = sum(len(str(item)) for item in bullets)
    except Exception:  # noqa: BLE001
        return 18.0
    if total <= 200:
        return 20.0
    if total <= 320:
        return 18.0
    if total <= 480:
        return 16.0
    return 14.0


def _fill_bullets(
    box: Any,
    pptx_module: dict[str, Any],
    bullets: Iterable[str],
    size: float,
) -> None:
    """在文本框内逐条写入项目符号正文。"""
    frame = box.text_frame
    items = [str(item) for item in bullets] or ["（本页暂无要点）"]
    for index, text in enumerate(items):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.space_after = pptx_module["Pt"](10)
        paragraph.line_spacing = 1.25
        _style_run(paragraph.add_run(), pptx_module, "▪  ", size, bold=True, color=COLOR_ACCENT)
        _style_run(paragraph.add_run(), pptx_module, text, size, color=COLOR_TEXT)


def _write_notes(slide: Any, notes: Sequence[str]) -> None:
    """把排版建议写入演讲者备注（不占用幻灯片版面）。"""
    try:
        text = "\n".join(str(item) for item in notes if str(item).strip())
        if text:
            slide.notes_slide.notes_text_frame.text = "【排版建议】\n" + text
    except Exception:  # noqa: BLE001 - 备注写入失败不影响主流程
        pass


# ------------------------------ 四种页面版式 ------------------------------ #
def _add_title_slide(prs: Any, deck: DeckContent, pptx_module: dict[str, Any]) -> Any:
    """版式一：标题页（深色底 + 左侧强调竖条 + 副标题信息）。"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_rect(slide, pptx_module, 0, 0, 13.333, 7.5, fill=COLOR_PRIMARY)
    _add_rect(slide, pptx_module, 0, 0, 0.3, 7.5, fill=COLOR_ACCENT)

    _add_text(
        slide, pptx_module, "项目简介 · PROJECT OVERVIEW", 1.15, 1.55, 10.5, 0.5,
        size=15, bold=True, color=COLOR_LIGHT,
    )
    _add_text(
        slide, pptx_module, deck.main_title, 1.15, 2.15, 11.2, 1.5,
        size=40, bold=True, color=COLOR_WHITE,
    )
    _add_rect(slide, pptx_module, 1.2, 3.85, 2.6, 0.06, fill=COLOR_ACCENT)

    if deck.cover_lines:
        box = _add_textbox(slide, pptx_module, 1.15, 4.15, 11.0, 2.2)
        frame = box.text_frame
        for index, line in enumerate(deck.cover_lines):
            paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
            paragraph.space_after = pptx_module["Pt"](6)
            _style_run(paragraph.add_run(), pptx_module, line, 16, color=COLOR_LIGHT)

    _add_text(
        slide, pptx_module, f"{FOOTER_TEXT}     {deck.generated_at}", 1.15, 6.72, 11.0, 0.4,
        size=12, color=COLOR_LIGHT,
    )
    return slide


def _add_toc_slide(prs: Any, deck: DeckContent, pptx_module: dict[str, Any]) -> Any:
    """版式二：目录页（依据各页页面名称自动生成，条目多时自动分两列）。"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_rect(slide, pptx_module, 0, 0, 13.333, 1.15, fill=COLOR_PRIMARY)
    _add_rect(slide, pptx_module, 0, 0, 0.18, 1.15, fill=COLOR_ACCENT)
    _add_text(slide, pptx_module, "目  录", 0.65, 0.18, 8.0, 0.8, size=26, bold=True,
              color=COLOR_WHITE, anchor="middle")
    _add_text(slide, pptx_module, "CONTENTS", 11.0, 0.34, 1.8, 0.5, size=13, color=COLOR_LIGHT,
              align="right", anchor="middle")

    entries = [item.block_title for item in deck.slides] or ["（暂无内容页）"]
    columns = 2 if len(entries) > 8 else 1
    per_column = (len(entries) + columns - 1) // columns
    width = 5.6 if columns == 2 else 11.0
    for column in range(columns):
        chunk = entries[column * per_column:(column + 1) * per_column]
        if not chunk:
            continue
        left = 1.15 + column * 6.05
        box = _add_textbox(slide, pptx_module, left, 1.85, width, 4.9)
        frame = box.text_frame
        for index, name in enumerate(chunk):
            order = column * per_column + index + 1
            paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
            paragraph.space_after = pptx_module["Pt"](14)
            _style_run(paragraph.add_run(), pptx_module, f"{order:02d}", 18, bold=True, color=COLOR_ACCENT)
            _style_run(paragraph.add_run(), pptx_module, "    ", 18, color=COLOR_MUTED)
            _style_run(paragraph.add_run(), pptx_module, name, 18, color=COLOR_TEXT)
    _add_text(slide, pptx_module, FOOTER_TEXT, 1.15, 6.95, 8.0, 0.35, size=10, color=COLOR_MUTED)
    return slide


def _add_content_slide(
    prs: Any,
    content: SlideContent,
    index: int,
    total: int,
    pptx_module: dict[str, Any],
) -> Any:
    """版式三：正文页（页眉标题 + 浅色底纹正文 + 页脚页码）。"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_rect(slide, pptx_module, 0, 0, 13.333, 1.15, fill=COLOR_PRIMARY)
    _add_rect(slide, pptx_module, 0, 0, 0.18, 1.15, fill=COLOR_ACCENT)
    _add_text(
        slide, pptx_module, content.head_title, 0.65, 0.16, 10.9, 0.85,
        size=26, bold=True, color=COLOR_WHITE, anchor="middle",
    )
    if content.block_title and content.block_title != content.head_title:
        _add_text(
            slide, pptx_module, content.block_title, 11.5, 0.16, 1.45, 0.85,
            size=12, color=COLOR_LIGHT, align="right", anchor="middle",
        )

    _add_rect(slide, pptx_module, 0.6, 1.5, 12.13, 5.1, fill=COLOR_PANEL)
    _add_rect(slide, pptx_module, 0.6, 1.5, 0.07, 5.1, fill=COLOR_ACCENT)

    bullets = content.bullets or ["（本页暂无要点）"]
    body = _add_textbox(slide, pptx_module, 0.98, 1.78, 11.35, 4.55)
    _fill_bullets(body, pptx_module, bullets, _bullet_font_size(bullets))

    _add_text(slide, pptx_module, FOOTER_TEXT, 0.65, 6.82, 9.0, 0.35, size=10, color=COLOR_MUTED)
    _add_text(
        slide, pptx_module, f"{index:02d} / {total:02d}", 11.9, 6.82, 0.9, 0.35,
        size=10, bold=True, color=COLOR_MUTED, align="right",
    )
    _write_notes(slide, content.notes)
    return slide


def _add_closing_slide(prs: Any, deck: DeckContent, pptx_module: dict[str, Any]) -> Any:
    """版式四：结束页（致谢 + 项目落款）。"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_rect(slide, pptx_module, 0, 0, 13.333, 7.5, fill=COLOR_PRIMARY)
    _add_rect(slide, pptx_module, 0, 0, 0.3, 7.5, fill=COLOR_ACCENT)
    _add_text(slide, pptx_module, "谢 谢 观 看", 0, 2.65, 13.333, 1.2, size=44, bold=True,
              color=COLOR_WHITE, align="center")
    _add_text(slide, pptx_module, "敬请各位老师批评指正", 0, 4.0, 13.333, 0.7, size=20,
              color=COLOR_LIGHT, align="center")
    _add_rect(slide, pptx_module, 5.87, 4.95, 1.6, 0.06, fill=COLOR_ACCENT)
    _add_text(slide, pptx_module, deck.main_title, 0, 5.25, 13.333, 0.6, size=16,
              color=COLOR_WHITE, align="center")
    _add_text(slide, pptx_module, f"{FOOTER_TEXT}     {deck.generated_at}", 0, 5.95, 13.333, 0.5,
              size=12, color=COLOR_LIGHT, align="center")
    return slide


# --------------------------------------------------------------------------- #
# 三、文件与入口
# --------------------------------------------------------------------------- #
def resolve_paths(md_arg: str, out_arg: str) -> tuple[Path, Path]:
    """解析输入文案与输出文件的绝对路径。

    * 输入文案：相对路径优先按当前工作目录解析，找不到时回退到脚本所在目录；
    * 输出文件：相对路径按当前工作目录解析（即"运行后在当前目录生成 pptx"）。
    """
    md_path = Path(md_arg)
    if not md_path.is_absolute() and not md_path.exists():
        md_path = SCRIPT_DIR / md_path
    out_path = Path(out_arg)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    return md_path, out_path


def read_text(path: Path) -> str:
    """按 UTF-8 / UTF-8-BOM / GBK 顺序读取文本，避免编码问题导致脚本中断。"""
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    """脚本入口：解析文案 -> 构建 PPT -> 写出 pptx 文件。

    Returns:
        int: 进程退出码，0 表示成功，1 表示失败（均不抛出未捕获异常）。
    """
    parser = argparse.ArgumentParser(description="根据 Markdown 文案自动生成项目简介 PPT")
    parser.add_argument("--md", default=str(DEFAULT_MD), help="Markdown 文案路径")
    parser.add_argument("--out", default=DEFAULT_OUT, help="输出 pptx 文件路径")
    args = parser.parse_args(list(argv) if argv is not None else None)

    md_path, out_path = resolve_paths(args.md, args.out)
    print("=" * 66)
    print("工业数据分析 Agent 系统 —— 项目简介 PPT 生成工具")
    print(f"文案输入：{md_path}")
    print(f"文件输出：{out_path}")
    print("=" * 66)

    try:
        if not md_path.exists():
            print(f"[错误] 未找到文案文件：{md_path}")
            print("       请确认 docs/项目简介PPT文案.md 存在，或用 --md 指定路径。")
            return 1

        deck = parse_markdown(read_text(md_path))
        if not deck.slides:
            print("[错误] 文案解析结果为空，请检查 Markdown 中是否包含 `## 第 N 页：xxx` 形式的标题。")
            return 1

        try:
            pptx_module = _load_pptx()
        except RuntimeError as exc:
            print(f"[错误] {exc}")
            print("       安装命令：python -m pip install python-pptx")
            return 1

        presentation = build_presentation(deck, pptx_module)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        presentation.save(str(out_path))

        total = len(presentation.slides)
        print(f"[完成] 已生成幻灯片 {total} 页：")
        print("       01  标题页")
        print(f"       02  目录页（含 {len(deck.slides)} 个条目）")
        for index, slide in enumerate(deck.slides, start=1):
            print(f"       {index + 2:02d}  {slide.head_title}")
        print(f"       {total:02d}  结束页")
        print(f"[输出] {out_path}")
        size_kb = out_path.stat().st_size / 1024 if out_path.exists() else 0.0
        print(f"[大小] {size_kb:.1f} KB")
        return 0
    except Exception as exc:  # noqa: BLE001 - 入口兜底，禁止未捕获异常
        print(f"[错误] PPT 生成失败：{exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
