"""docs 目录 Markdown 公式乱码批量修复脚本。

功能说明
========================================================================
遍历 ``docs/`` 下的全部 ``.md`` 文件，把 LaTeX 公式批量转换为「可直接显示」的
Unicode 字符（希腊字母、上下标、数学运算符、重点符号等），使文档在 Word、记事本、
浏览器等任意环境中都能正常显示，不再出现 ``$R^2$``、``\\sigma`` 一类的源码残留。

主要替换规则（节选）
========================================================================
* ``$R^2$`` → ``R²``；``$3\\sigma$`` → ``3σ``；``$\\mu$`` → ``μ``；``$\\rho$`` → ``ρ``；
* ``$x_i$`` → ``xᵢ``；``$\\bar{x}$`` → ``x̄``；``$\\hat y_i$`` → ``ŷᵢ``；
* ``$\\frac{a}{b}$`` → ``a/b``；``$\\sqrt{x}$`` → ``√(x)``；``$\\sum_{i}$`` → ``Σᵢ``；
* ``$\\text{MSE}$`` / ``$\\mathrm{RMSE}$`` → ``MSE`` / ``RMSE``；
* 常见关系符与运算符：``\\times``→``×``、``\\pm``→``±``、``\\leq``→``≤``、``\\in``→``∈`` 等；
* 公式定界符 ``$`` / ``$$`` 与其后的 LaTeX 花括号、反斜杠命令一并清理。

安全策略
========================================================================
* 覆盖原文件前，先将原始内容备份为同名 ``.md.bak``；
* 若 ``.md.bak`` 已存在则**保留最早备份**，避免重复运行把"原始版本"覆盖为"已修复版本"
  （如需强制刷新备份，可加 ``--overwrite-backup``）；
* 内容未发生变化的文件不做写入，也不会产生备份；
* 支持 ``--dry-run`` 预演，只输出将要修改的文件清单。

运行方式
========================================================================
    python fix_md_formulas.py                # 修复 docs/ 下全部 md 文件
    python fix_md_formulas.py --dry-run      # 仅预演，不写文件
    python fix_md_formulas.py --docs-dir other_docs
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DOCS_DIR = SCRIPT_DIR / "docs"

# --------------------------------------------------------------------------- #
# 一、替换字典
# --------------------------------------------------------------------------- #
#: 希腊字母（含大写）
GREEK_MAP: dict[str, str] = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "varepsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "ϑ",
    "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
    "pi": "π", "varpi": "ϖ", "rho": "ρ", "varrho": "ϱ", "sigma": "σ", "varsigma": "ς",
    "tau": "τ", "upsilon": "υ", "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ",
    "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π",
    "Sigma": "Σ", "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}

#: 数学运算符、关系符与其它常用符号
SYMBOL_MAP: dict[str, str] = {
    "times": "×", "div": "÷", "cdot": "·", "ast": "∗", "pm": "±", "mp": "∓",
    "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "neq": "≠", "ne": "≠",
    "approx": "≈", "equiv": "≡", "sim": "∼", "propto": "∝", "ll": "≪", "gg": "≫",
    "in": "∈", "notin": "∉", "subset": "⊂", "subseteq": "⊆", "supset": "⊃",
    "cup": "∪", "cap": "∩", "emptyset": "∅", "varnothing": "∅",
    "forall": "∀", "exists": "∃", "neg": "¬", "land": "∧", "lor": "∨",
    "partial": "∂", "nabla": "∇", "infty": "∞", "prime": "′", "circ": "°",
    "degree": "°", "angle": "∠", "perp": "⊥", "parallel": "∥",
    "sum": "Σ", "prod": "∏", "int": "∫", "sqrt": "√",
    "rightarrow": "→", "to": "→", "leftarrow": "←", "leftrightarrow": "↔",
    "Rightarrow": "⇒", "Leftarrow": "⇐", "leftrightarrows": "⇆",
    "uparrow": "↑", "downarrow": "↓", "mapsto": "↦",
    "dots": "…", "ldots": "…", "cdots": "⋯", "vdots": "⋮", "ddots": "⋱",
    "quad": " ", "qquad": "  ", ",": "", ";": " ", "!": "", " ": " ",
    "log": "log", "ln": "ln", "lg": "lg", "exp": "exp", "sin": "sin", "cos": "cos",
    "tan": "tan", "cot": "cot", "sec": "sec", "csc": "csc", "arcsin": "arcsin",
    "arccos": "arccos", "arctan": "arctan", "min": "min", "max": "max", "arg": "arg",
    "det": "det", "dim": "dim", "ker": "ker", "deg": "deg", "bmod": "mod", "mod": "mod",
    "lvert": "|", "rvert": "|", "vert": "|", "Vert": "‖", "lVert": "‖", "rVert": "‖",
    "langle": "⟨", "rangle": "⟩", "lceil": "⌈", "rceil": "⌉", "lfloor": "⌊", "rfloor": "⌋",
    "slash": "/", "backslash": "\\", "percent": "%", "amp": "&", "hash": "#",
    "sim": "∼", "star": "⋆", "bullet": "∙", "oplus": "⊕", "otimes": "⊗",
}

#: 上下标字符表
SUPERSCRIPT_MAP: dict[str, str] = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶",
    "7": "⁷", "8": "⁸", "9": "⁹", "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾",
    "n": "ⁿ", "i": "ⁱ", "a": "ᵃ", "b": "ᵇ", "c": "ᶜ", "d": "ᵈ", "e": "ᵉ", "f": "ᶠ",
    "g": "ᵍ", "h": "ʰ", "j": "ʲ", "k": "ᵏ", "l": "ˡ", "m": "ᵐ", "o": "ᵒ", "p": "ᵖ",
    "r": "ʳ", "s": "ˢ", "t": "ᵗ", "u": "ᵘ", "v": "ᵛ", "w": "ʷ", "x": "ˣ", "y": "ʸ",
    "z": "ᶻ", "*": "*", "T": "ᵀ",
}

SUBSCRIPT_MAP: dict[str, str] = {
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆",
    "7": "₇", "8": "₈", "9": "₉", "+": "₊", "-": "₋", "=": "₌", "(": "₍", ")": "₎",
    "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ", "k": "ₖ", "l": "ₗ", "m": "ₘ",
    "n": "ₙ", "o": "ₒ", "p": "ₚ", "r": "ᵣ", "s": "ₛ", "t": "ₜ", "u": "ᵤ", "v": "ᵥ",
    "x": "ₓ",
}

#: 组合重音命令 -> Unicode 组合符
ACCENT_MAP: dict[str, str] = {
    "bar": "\u0304", "hat": "\u0302", "tilde": "\u0303", "dot": "\u0307",
    "ddot": "\u0308", "vec": "\u20d7", "check": "\u030c", "breve": "\u0306",
}

#: 需要「取参数并只保留内容」的命令
ARG_KEEP_COMMANDS: tuple[str, ...] = ("text", "mathrm", "operatorname", "mathit", "mathbf", "mathsf", "mathtt")

_COMMAND_PATTERN = re.compile(r"\\([A-Za-z]+)")
#: 公式内的命令名模式（LaTeX 会吞掉命令名后的一个空格，如 ``\Delta t`` 实际渲染为 Δt）
_MATH_COMMAND_PATTERN = re.compile(r"\\([A-Za-z]+) ?")
_BLOCK_MATH_PATTERN = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE_MATH_PATTERN = re.compile(r"\$([^$\n]+?)\$")
_SUP_BRACED = re.compile(r"\^\{([^{}]*)\}")
_SUB_BRACED = re.compile(r"_\{([^{}]*)\}")
_SUP_SINGLE = re.compile(r"\^(\S)")
_SUB_SINGLE = re.compile(r"_(\S)")
_SPACE_PATTERN = re.compile(r"[ \t]{2,}")


# --------------------------------------------------------------------------- #
# 二、核心转换逻辑
# --------------------------------------------------------------------------- #
def _read_text(path: Path) -> str:
    """按 UTF-8 / UTF-8-BOM / GBK 顺序读取文本，避免编码问题中断批处理。"""
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _translate_chars(text: str, mapping: dict[str, str]) -> str:
    """按字符表翻译（无法映射的字符原样保留）。"""
    return "".join(mapping.get(char, char) for char in text)


def _extract_group(text: str, start: int) -> tuple[str, int]:
    """从 ``start`` 位置（指向 ``{``）提取配对花括号内容，返回 ``(内容, 结束位置)``。"""
    if start >= len(text) or text[start] != "{":
        raise ValueError("参数必须以左花括号开始")
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:index], index + 1
    raise ValueError("花括号不配对")


def _extract_atom(text: str, start: int) -> tuple[str, int]:
    """提取一个「原子」参数：``{...}`` 分组、``\\command`` 或单个字符。

    用于兼容 ``\\frac1n``、``\\sqrt x`` 这类省略花括号的 LaTeX 简写。
    """
    cursor = start
    while cursor < len(text) and text[cursor] in " \t":
        cursor += 1
    if cursor >= len(text):
        return "", cursor
    if text[cursor] == "{":
        return _extract_group(text, cursor)
    if text[cursor] == "\\":
        matched = re.match(r"\\[A-Za-z]+", text[cursor:])
        if matched:
            return matched.group(0), cursor + matched.end()
    return text[cursor], cursor + 1


def _replace_command_args(text: str, command: str, handler: Callable[[list[str]], str], arg_count: int) -> str:
    """替换 ``\\command{a}{b}`` 形式的命令（支持嵌套花括号）。"""
    marker = "\\" + command
    while True:
        position = text.find(marker)
        if position < 0:
            return text
        cursor = position + len(marker)
        args: list[str] = []
        for _ in range(arg_count):
            argument, cursor = _extract_atom(text, cursor)
            args.append(argument)
        text = text[:position] + handler(args) + text[cursor:]


def _wrap_if_needed(expression: str) -> str:
    """分子/分母含运算符或空格时补括号，保证 ``a/b`` 形式语义清晰、无歧义。"""
    stripped = expression.strip()
    if not stripped:
        return stripped
    if stripped.startswith("(") and stripped.endswith(")"):
        return stripped
    if re.search(r"[+\-−×·/]|\s", stripped):
        return f"({stripped})"
    return stripped


def _replace_accent(text: str, command: str, mark: str) -> str:
    """把 ``\\bar{x}`` / ``\\bar y`` 等重音命令转换为组合字符（优先使用预组合字符）。"""
    marker = "\\" + command
    while True:
        position = text.find(marker)
        if position < 0:
            return text
        cursor = position + len(marker)
        while cursor < len(text) and text[cursor] in " \t":
            cursor += 1
        if cursor < len(text) and text[cursor] == "{":
            try:
                base, cursor = _extract_group(text, cursor)
            except ValueError:
                base, cursor = "", cursor + 1
        elif cursor < len(text):
            base, cursor = text[cursor], cursor + 1
        else:
            base = ""
        combined = unicodedata.normalize("NFC", f"{base}{mark}") if base else ""
        text = text[:position] + combined + text[cursor:]


def convert_math(expression: str) -> str:
    """把单条 LaTeX 公式转换为可直接显示的 Unicode 文本。"""
    text = str(expression or "").strip()
    if not text:
        return ""
    try:
        text = text.replace("\\left", "").replace("\\right", "")
        text = text.replace("\\displaystyle", "").replace("\\limits", "")
        # 结构型命令
        text = _replace_command_args(text, "frac", lambda args: f"{_wrap_if_needed(args[0])}/{_wrap_if_needed(args[1])}", 2)
        text = _replace_command_args(text, "dfrac", lambda args: f"{_wrap_if_needed(args[0])}/{_wrap_if_needed(args[1])}", 2)
        text = _replace_command_args(text, "tfrac", lambda args: f"{_wrap_if_needed(args[0])}/{_wrap_if_needed(args[1])}", 2)
        text = _replace_command_args(text, "sqrt", lambda args: f"√({args[0]})", 1)
        for command in ARG_KEEP_COMMANDS:
            text = _replace_command_args(text, command, lambda args: args[0], 1)
        # 重音命令
        for command, mark in ACCENT_MAP.items():
            text = _replace_accent(text, command, mark)
        # 希腊字母与符号（用 lambda 替换，避免替换串中的反斜杠被当作转义序列）
        for command in sorted({**GREEK_MAP, **SYMBOL_MAP}, key=len, reverse=True):
            replacement = GREEK_MAP.get(command) or SYMBOL_MAP.get(command) or ""
            if command.isalpha():
                # 字母命令需要后向断言（避免 \in 误匹配 \infty），并吞掉其后的一个空格
                pattern = r"\\" + command + r"(?![A-Za-z]) ?"
            else:
                # 符号命令（\, \; \! \ 等）后面常紧跟字母或数字，不能加字母断言
                pattern = re.escape("\\" + command)
            text = re.sub(pattern, lambda _match, value=replacement: value, text)
        # 上下标
        text = _SUP_BRACED.sub(lambda match: _translate_chars(match.group(1), SUPERSCRIPT_MAP), text)
        text = _SUB_BRACED.sub(lambda match: _translate_chars(match.group(1), SUBSCRIPT_MAP), text)
        text = _SUP_SINGLE.sub(lambda match: _translate_chars(match.group(1), SUPERSCRIPT_MAP), text)
        text = _SUB_SINGLE.sub(lambda match: _translate_chars(match.group(1), SUBSCRIPT_MAP), text)
        # 残留命令与花括号清理（公式内同样吞掉命令名后的空格）
        text = _MATH_COMMAND_PATTERN.sub(lambda match: match.group(1), text)
        text = text.replace("{", "").replace("}", "").replace("\\", "")
        text = _SPACE_PATTERN.sub(" ", text)
        return text.strip()
    except Exception:  # noqa: BLE001 - 单条公式转换失败时退化为去符号文本
        return text.replace("$", "").replace("\\", "").strip()


def convert_text(content: str) -> tuple[str, int]:
    """转换整篇文档中的公式，返回 ``(新内容, 处理公式数量)``。"""
    counter = {"count": 0}

    def _block(match: re.Match[str]) -> str:
        counter["count"] += 1
        return " " + convert_math(match.group(1)) + " "

    def _inline(match: re.Match[str]) -> str:
        counter["count"] += 1
        return convert_math(match.group(1))

    try:
        text = _BLOCK_MATH_PATTERN.sub(_block, str(content or ""))
        text = _INLINE_MATH_PATTERN.sub(_inline, text)
        # 文档中未被 $ 包裹的外露 LaTeX 命令（如说明文字里的 \sigma）
        text = _COMMAND_PATTERN.sub(lambda match: GREEK_MAP.get(match.group(1), SYMBOL_MAP.get(match.group(1), match.group(1))), text)
        return text, counter["count"]
    except Exception:  # noqa: BLE001 - 转换异常时保持原文，避免损坏文档
        return str(content or ""), 0


def fix_file(path: Path, overwrite_backup: bool = False, dry_run: bool = False) -> dict[str, Any]:
    """修复单个 Markdown 文件。

    Returns:
        dict: ``{"path", "changed", "formulas", "backup", "error"}``。
    """
    result: dict[str, Any] = {
        "path": str(path), "changed": False, "formulas": 0, "backup": None, "error": None,
    }
    try:
        original = _read_text(path)
        fixed, count = convert_text(original)
        result["formulas"] = count
        if fixed == original:
            return result

        result["changed"] = True
        if dry_run:
            return result

        backup = path.with_suffix(path.suffix + ".bak")
        if overwrite_backup or not backup.exists():
            backup.write_text(original, encoding="utf-8")
        result["backup"] = str(backup)
        path.write_text(fixed, encoding="utf-8")
        return result
    except Exception as exc:  # noqa: BLE001 - 单文件失败不影响整批处理
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


# --------------------------------------------------------------------------- #
# 三、批处理与命令行入口
# --------------------------------------------------------------------------- #
def collect_markdown_files(docs_dir: Path) -> list[Path]:
    """收集目录下的 Markdown 文件（自动排除 ``.md.bak`` 备份文件）。"""
    try:
        files = [
            path for path in sorted(Path(docs_dir).glob("*.md"))
            if path.is_file() and not path.name.endswith(".bak")
        ]
        return files
    except Exception as exc:  # noqa: BLE001
        print(f"[错误] 目录扫描失败：{exc}")
        return []


def main(argv: Sequence[str] | None = None) -> int:
    """脚本入口：批量修复 ``docs/`` 下的公式乱码。

    Returns:
        int: 进程退出码，0 表示全部成功，1 表示存在失败项。
    """
    parser = argparse.ArgumentParser(description="批量修复 docs 目录 Markdown 文件的公式乱码")
    parser.add_argument("--docs-dir", default=str(DEFAULT_DOCS_DIR), help="待处理文档目录（默认 docs/）")
    parser.add_argument("--dry-run", action="store_true", help="仅预演，不写入文件")
    parser.add_argument("--overwrite-backup", action="store_true", help="强制刷新 .md.bak 备份")
    args = parser.parse_args(list(argv) if argv is not None else None)

    docs_dir = Path(args.docs_dir)
    if not docs_dir.is_absolute() and not docs_dir.exists():
        docs_dir = SCRIPT_DIR / docs_dir

    print("=" * 70)
    print("Markdown 公式乱码批量修复工具")
    print(f"目标目录：{docs_dir}")
    print(f"运行模式：{'预演（不写文件）' if args.dry_run else '实际修复（自动备份 .md.bak）'}")
    print("=" * 70)

    if not docs_dir.is_dir():
        print(f"[错误] 目录不存在：{docs_dir}")
        return 1

    files = collect_markdown_files(docs_dir)
    if not files:
        print("[错误] 未找到任何 .md 文件")
        return 1

    changed: list[dict[str, Any]] = []
    unchanged: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for path in files:
        outcome = fix_file(
            path,
            overwrite_backup=bool(args.overwrite_backup),
            dry_run=bool(args.dry_run),
        )
        if outcome["error"]:
            failed.append(outcome)
            print(f"  [失败] {path.name}：{outcome['error']}")
        elif outcome["changed"]:
            changed.append(outcome)
            backup_name = Path(outcome["backup"]).name if outcome["backup"] else "（预演，未备份）"
            print(f"  [已修复] {path.name}：转换公式 {outcome['formulas']} 处 ｜ 备份 {backup_name}")
        else:
            unchanged.append(outcome)
            print(f"  [无需处理] {path.name}")

    print("-" * 70)
    print(f"修复完成文件列表（{len(changed)} 个）：")
    for item in changed:
        print(f"  ✔ {Path(item['path']).name}  （公式 {item['formulas']} 处）")
    if unchanged:
        print(f"无需修改文件（{len(unchanged)} 个）：")
        for item in unchanged:
            print(f"  – {Path(item['path']).name}")
    if failed:
        print(f"处理失败文件（{len(failed)} 个）：")
        for item in failed:
            print(f"  ✘ {Path(item['path']).name}：{item['error']}")
    print("=" * 70)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
