"""工业数据分析 Agent 系统 —— 大作业提交材料打包脚本。

功能说明
========================================================================
把分散在项目中的交付物整理为标准提交目录结构::

    工业数据分析Agent系统_提交材料/
    ├── 01_项目源代码/industrial_data_agent/   整个项目（剔除缓存与临时目录、清除密钥）
    ├── 02_交付文档/*.docx                     五份交付文档（另附项目简介 PPT）
    ├── 03_演示材料/                            演示数据集（演示视频录制后放入此处）
    └── 04_补充材料/                            Markdown 源文件、单元测试证据、提交清单

安全说明
========================================================================
* 复制项目时自动清空 ``config/.env`` 中疑似密钥的变量值（API_KEY / TOKEN / SECRET /
  PASSWORD 等），并在文件头部写入「提交版」提示；**原始项目的 .env 不会被修改**；
* 打包结束后扫描整个提交目录，确认没有残留形如 ``sk-xxxx`` 的密钥字符串。

排除项（复制 01 源代码时）
========================================================================
* ``venv`` / ``.venv`` / ``env`` 等虚拟环境目录：依赖环境体积大且可重建，不属于源代码；
* ``.idea`` / ``.vscode`` 等 IDE 配置与 ``__pycache__`` / ``.pytest_cache`` / ``*.pyc`` 缓存；
* ``output/.sandbox``：沙箱单次运行的临时工作目录（每次执行的代码副本，属运行时垃圾）；
* 其余内容（含 ``output`` 下的图表、报告、模型与示例数据）全部保留，作为系统可运行证据。

运行方式
========================================================================
    python package_submission.py
    python package_submission.py --test-log D:/xxx/pytest.log   # 附带单元测试证据
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import stat
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent

#: 提交包名称与内部目录
PACKAGE_NAME = "工业数据分析Agent系统_提交材料"
DIR_SOURCE = "01_项目源代码"
DIR_DOCS = "02_交付文档"
DIR_DEMO = "03_演示材料"
DIR_EXTRA = "04_补充材料"

#: 五份交付文档（Word）与项目简介 PPT
DOCX_FILES: tuple[str, ...] = (
    "项目概要介绍.docx",
    "项目详细技术方案.docx",
    "产品使用手册.docx",
    "项目简介PPT文案.docx",
    "团队分工与开发过程文档.docx",
)
PPT_FILE = "工业数据分析Agent系统_项目简介.pptx"
DEMO_CSV = "sample_process_data.csv"

#: 复制源代码时排除的目录名（依赖环境、IDE 配置与各类缓存均不属于源代码）
EXCLUDE_DIR_NAMES = {
    "__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".git",
    ".idea", ".vscode", ".mypy_cache", ".ruff_cache",
    "venv", ".venv", "env", "virtualenv", "node_modules", "dist", "build",
}
EXCLUDE_RELATIVE = {"output/.sandbox"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}

#: 单元测试中的占位密钥（属测试夹具，不作为敏感信息报警）
FIXTURE_SECRET_HINTS = (
    "sk-unit-test", "sk-1234567890abcdef", "test-secret", "unit-test-secret-value",
)

#: 疑似敏感变量（.env 变量名命中即清空其值）
_SECRET_KEY_PATTERN = re.compile(
    r"^(?P<key>[A-Za-z0-9_]*(?:API_KEY|_KEY|TOKEN|SECRET|PASSWORD|PASSWD)[A-Za-z0-9_]*)\s*=(?P<value>.*)$"
)
#: 疑似密钥字面量（打包后全局扫描）
_SECRET_VALUE_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}|\bghp_[A-Za-z0-9]{20,}")

ENV_HEADER = (
    "# ------------------------------------------------------------\n"
    "# 【提交版】敏感信息已清空：填入自己的 DEEPSEEK_API_KEY 即可运行；\n"
    "# 留空时系统自动运行于本地模板模式，全部功能仍可正常演示。\n"
    "# ------------------------------------------------------------\n"
)


# --------------------------------------------------------------------------- #
# 一、通用工具
# --------------------------------------------------------------------------- #
def _human_size(num_bytes: float) -> str:
    """把字节数格式化为易读字符串。"""
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} GB"


def _relative_excluded(relative: Path) -> bool:
    """判断相对路径是否属于排除项。"""
    if any(part in EXCLUDE_DIR_NAMES for part in relative.parts):
        return True
    if any(part.endswith(".egg-info") for part in relative.parts):
        return True
    if relative.suffix.lower() in EXCLUDE_SUFFIXES:
        return True
    normalized = relative.as_posix()
    return any(normalized == item or normalized.startswith(f"{item}/") for item in EXCLUDE_RELATIVE)


def _copy_project(project_root: Path, dest: Path) -> dict[str, Any]:
    """复制整个项目到提交目录，返回统计信息。"""
    stats = {"files": 0, "dirs": 0, "bytes": 0, "excluded": []}
    dest.mkdir(parents=True, exist_ok=True)
    for path in sorted(project_root.rglob("*")):
        relative = path.relative_to(project_root)
        if _relative_excluded(relative):
            if len(stats["excluded"]) < 40:
                stats["excluded"].append(relative.as_posix())
            continue
        target = dest / relative
        try:
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                stats["dirs"] += 1
            elif path.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                stats["files"] += 1
                stats["bytes"] += path.stat().st_size
        except Exception as exc:  # noqa: BLE001 - 单个文件失败不影响整体
            print(f"  [警告] 复制失败 {relative}：{exc}")
    return stats


def _sanitize_env(env_path: Path) -> list[str]:
    """清空 .env 中疑似密钥的变量值，返回被清理的变量名列表。"""
    cleared: list[str] = []
    if not env_path.exists():
        return cleared
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
        output: list[str] = []
        for line in lines:
            matched = _SECRET_KEY_PATTERN.match(line.strip())
            if matched and matched.group("value").strip():
                output.append(f"{matched.group('key')}=")
                cleared.append(matched.group("key"))
            else:
                output.append(line)
        env_path.write_text(ENV_HEADER + "\n".join(output) + "\n", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"  [警告] .env 脱敏失败：{exc}")
    return cleared


def _scan_secrets(root: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """扫描目录中残留的疑似密钥。

    Returns:
        tuple[list, list]: ``(需要处理的命中项, 测试夹具类命中项)``；
        后者来自单元测试中的占位密钥（如 ``sk-unit-test``），不视为敏感信息泄露。
    """
    blocking: list[tuple[str, str]] = []
    informational: list[tuple[str, str]] = []
    try:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() in {".png", ".jpg", ".pptx", ".docx", ".mp4"}:
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:  # noqa: BLE001
                continue
            relative = path.relative_to(root).as_posix()
            for matched in _SECRET_VALUE_PATTERN.finditer(content):
                value = matched.group(0)
                item = (relative, value[:16] + "…")
                if (
                    "/tests/" in f"/{relative}"
                    or path.name in {"package_submission.py"}  # 本脚本内含占位密钥样例，属工具自身内容
                    or any(hint in value for hint in FIXTURE_SECRET_HINTS)
                ):
                    informational.append(item)
                else:
                    blocking.append(item)
    except Exception:  # noqa: BLE001
        pass
    return blocking, informational


def _print_tree(root: Path, max_entries: int = 200) -> None:
    """打印提交目录树（超出条目上限时给出省略提示）。"""
    print(f"{root.name}/")
    entries = sorted(root.rglob("*"))
    for index, path in enumerate(entries):
        if index >= max_entries:
            print(f"  ...（其余 {len(entries) - max_entries} 项已省略）")
            break
        depth = len(path.relative_to(root).parts)
        suffix = "/" if path.is_dir() else f"  ({_human_size(path.stat().st_size)})"
        print("  " * depth + path.name + suffix)


def _force_rmtree(path: Path) -> bool:
    """删除目录（兼容 Windows 只读文件），返回是否删除成功。"""

    def _on_error(func: Any, target: str, exc_info: Any) -> None:
        try:
            os.chmod(target, stat.S_IWRITE)
            func(target)
        except Exception:  # noqa: BLE001 - 尽力删除，失败交由调用方判断
            pass

    try:
        shutil.rmtree(path, onerror=_on_error)
    except Exception:  # noqa: BLE001
        pass
    return not path.exists()


# --------------------------------------------------------------------------- #
# 二、单元测试证据整理
# --------------------------------------------------------------------------- #
def _collect_test_overview(project_root: Path) -> dict[str, int]:
    """统计各测试文件包含的用例数量（``--collect-only``，不产生任何缓存文件）。"""
    try:
        completed = subprocess.run(
            [sys.executable, "-B", "-m", "pytest", "tests", "--collect-only", "-q", "-p", "no:cacheprovider"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        counts: Counter[str] = Counter()
        for line in (completed.stdout or "").splitlines():
            if "::" in line and line.startswith("tests"):
                counts[line.split("::", 1)[0].strip()] += 1
        return dict(sorted(counts.items()))
    except Exception as exc:  # noqa: BLE001 - 统计失败不影响打包
        print(f"  [警告] 测试用例统计失败：{exc}")
        return {}


def _render_log_image(log_path: Path, png_path: Path, max_lines: int = 30) -> bool:
    """把测试日志渲染成深色终端风格图片（便于直接排版进交付材料）。"""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("  [提示] 未安装 Pillow，跳过测试结果图片生成")
        return False
    try:
        raw_lines = [
            line.rstrip() for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip()
        ]
        lines = [line if len(line) <= 150 else line[:147] + "…" for line in raw_lines[-max_lines:]]
        title = "pytest 全量测试运行结果（真实终端输出导出）"
        footer = f"导出时间：{datetime.now():%Y-%m-%d %H:%M} ｜ 执行命令：python -m pytest tests -q"

        # 按平台探测字体路径（不使用写死的盘符，兼容 Windows / Linux）
        font_candidates: list[Path] = []
        windir = os.environ.get("WINDIR")
        if windir:
            font_candidates.extend([Path(windir) / "Fonts" / "msyh.ttc", Path(windir) / "Fonts" / "consola.ttf"])
        font_candidates.extend(
            [
                Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
            ]
        )
        font = title_font = None
        for candidate in font_candidates:
            if Path(candidate).exists():
                try:
                    font = ImageFont.truetype(candidate, 15)
                    title_font = ImageFont.truetype(candidate, 18)
                    break
                except Exception:  # noqa: BLE001 - 换下一个候选字体
                    continue
        if font is None or title_font is None:
            try:
                font = ImageFont.load_default(15)
                title_font = ImageFont.load_default(18)
            except Exception:  # noqa: BLE001
                font = title_font = ImageFont.load_default()

        def _width(text: str, target_font: Any) -> float:
            try:
                return float(target_font.getlength(text))
            except Exception:  # noqa: BLE001
                return len(text) * 9.0

        padding, line_height = 20, 24
        measured = [_width(title, title_font), _width(footer, font)] + [_width(line, font) for line in lines]
        width = int(max(900, min(max(measured) + padding * 2, 1500)))
        height = int(padding * 2 + 46 + len(lines) * line_height + 30)

        image = Image.new("RGB", (width, height), (17, 20, 24))
        draw = ImageDraw.Draw(image)
        draw.text((padding, padding), title, font=title_font, fill=(111, 208, 140))
        draw.line([(padding, padding + 30), (width - padding, padding + 30)], fill=(70, 78, 88), width=1)
        cursor = padding + 44
        for line in lines:
            color = (150, 220, 170) if "passed" in line else (208, 212, 218)
            draw.text((padding, cursor), line, font=font, fill=color)
            cursor += line_height
        draw.text((padding, cursor + 6), footer, font=font, fill=(120, 128, 138))

        png_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(png_path)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [警告] 测试结果图片生成失败：{exc}")
        return False


def _write_test_evidence(test_log: Path | None, dest_dir: Path, project_root: Path) -> dict[str, Any]:
    """整理单元测试证据：运行日志、结果图片与说明文件。"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    outcome: dict[str, Any] = {"log": None, "image": None, "overview": {}}

    if test_log is not None and Path(test_log).is_file():
        target = dest_dir / "单元测试运行日志.txt"
        shutil.copy2(test_log, target)
        outcome["log"] = target
        if _render_log_image(Path(test_log), dest_dir / "单元测试结果.png"):
            outcome["image"] = dest_dir / "单元测试结果.png"

    overview = _collect_test_overview(project_root)
    outcome["overview"] = overview
    total = sum(overview.values())

    lines = [
        "单元测试证据说明",
        "=" * 64,
        "1. 单元测试运行日志.txt —— pytest 全量运行的真实终端输出（原始文件，未作任何修改）。",
        "2. 单元测试结果.png —— 由上述真实日志导出的图片，可直接排版进文档。",
        "   特别说明：该图是脚本从真实日志渲染的，不是屏幕截图；若需现场截图，",
        "   请在项目根目录执行  python -m pytest tests -q  后用 Win+Shift+S 截取终端窗口替换本图。",
        "3. 复现命令：python -m pytest tests -q",
        "",
        f"用例分布（共 {total} 个）：",
    ]
    lines.extend(f"  - {name}：{count} 个" for name, count in overview.items())
    lines += [
        "",
        "测试覆盖范围：",
        "  - 算法库：描述性统计、相关性、时滞、3σ 异常检测、三类机器学习模型的训练/评估/保存/推理；",
        "  - 沙箱安全：危险导入与函数、魔术属性、无界循环、路径越界、文件删除、超时终止、密钥隔离；",
        "  - Agent 闭环：意图解析、代码生成（LLM 与本地模板双通道）、沙箱执行、报告生成、历史记录；",
        "  - 报告模块：事实整理、模板报告渲染、Markdown 转 HTML 与图表 Base64 内嵌；",
        "  - LLM 客户端：本地桩服务验证双通道传输、重试机制与模型自动降级；",
        "  - Web 界面：Streamlit 无头冒烟测试（含一次完整的问答式分析流程）。",
    ]
    (dest_dir / "单元测试证据说明.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outcome


def _write_checklist(dest_dir: Path, stats: dict[str, Any]) -> Path:
    """生成提交材料清单，便于对照检查与补录演示视频。"""
    content = [
        "# 提交材料清单",
        "",
        f"生成时间：{datetime.now():%Y-%m-%d %H:%M}",
        "",
        "## 一、目录结构",
        "",
        "```",
        f"{PACKAGE_NAME}/",
        f"├── {DIR_SOURCE}/industrial_data_agent/   项目源代码（已剔除缓存与沙箱临时目录，密钥已清空）",
        f"├── {DIR_DOCS}/                           五份交付文档（Word）+ 项目简介 PPT",
        f"├── {DIR_DEMO}/                           演示数据集（演示视频录制后放入此处）",
        f"└── {DIR_EXTRA}/                          Markdown 源文件、单元测试证据、本清单",
        "```",
        "",
        "## 二、待办事项",
        "",
        f"- [ ] 录制产品交互演示视频，命名为 `产品交互演示视频.mp4`，放入 `{DIR_DEMO}/`；",
        "- [ ] 如需现场截图版测试证据，用真实截图替换 `04_补充材料/单元测试截图/单元测试结果.png`；",
        "- [ ] 提交前确认 `01_项目源代码/industrial_data_agent/config/.env` 中没有真实密钥。",
        "",
        "## 三、复现步骤（供评审老师参考）",
        "",
        "1. 安装依赖：`python -m pip install -r requirements.txt`；",
        "2. （可选）在 `config/.env` 中填入 `DEEPSEEK_API_KEY`；留空时系统自动运行于本地模板模式；",
        "3. 启动系统：`streamlit run app.py`；",
        "4. 自检：`python -m pytest tests -q`（全部用例应通过）。",
        "",
        "## 四、打包统计",
        "",
        f"- 源代码文件：{stats.get('files', 0)} 个，合计 {_human_size(float(stats.get('bytes', 0)))}；",
        f"- 已排除项：{len(stats.get('excluded', []))} 类（Python 缓存、沙箱临时工作目录等）；",
        f"- 已脱敏环境变量：{('、'.join(stats.get('cleared_keys', [])) or '无需清理（原值为空）')}。",
    ]
    target = dest_dir / "提交材料清单.md"
    target.write_text("\n".join(content) + "\n", encoding="utf-8")
    return target


# --------------------------------------------------------------------------- #
# 三、主流程
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    """脚本入口：按提交规范整理打包。

    Returns:
        int: 退出码，0 表示打包完成且无异常，1 表示存在缺失项或安全问题。
    """
    parser = argparse.ArgumentParser(description="整理工业数据分析 Agent 系统的大作业提交材料")
    parser.add_argument("--project", default=str(SCRIPT_DIR), help="项目根目录（默认脚本所在目录）")
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR.parent / PACKAGE_NAME), help="提交包输出目录")
    parser.add_argument("--test-log", default="", help="单元测试日志文件路径（可选）")
    args = parser.parse_args(list(argv) if argv is not None else None)

    project_root = Path(args.project).resolve()
    package_root = Path(args.out_dir).resolve()
    test_log = Path(args.test_log).resolve() if args.test_log else None

    print("=" * 76)
    print("大作业提交材料打包工具")
    print(f"项目根目录：{project_root}")
    print(f"提交包目录：{package_root}")
    print("=" * 76)

    if not (project_root / "app.py").is_file():
        print(f"[错误] 未在 {project_root} 找到 app.py，请用 --project 指定项目根目录")
        return 1
    if str(package_root).startswith(str(project_root)):
        print("[错误] 提交包目录不能位于项目内部，否则会造成递归复制")
        return 1
    if package_root.exists():
        if package_root.name != PACKAGE_NAME:
            print(f"[错误] 目标目录已存在且名称不符合提交包命名，为避免误删已中止：{package_root}")
            return 1
        # 安全保护：目标目录若包含 Git 仓库，说明它已被用作真实工作仓库，
        # 删除会丢失提交历史，必须中止并提示改用新的输出目录。
        git_dirs = [item for item in package_root.rglob(".git") if item.is_dir()]
        if git_dirs:
            print("[错误] 目标目录内检测到 Git 仓库，为避免删除提交历史已中止：")
            for item in git_dirs[:3]:
                print(f"        {item}")
            print("        请改用 --out-dir 指定新的输出目录，或手工同步更新文件。")
            return 1
        if not _force_rmtree(package_root):
            print(f"[错误] 旧提交包目录无法删除（可能被占用）：{package_root}")
            return 1
        print("已清理旧的提交包目录以便重建")

    # ---------- 目录骨架 ----------
    source_dir = package_root / DIR_SOURCE / "industrial_data_agent"
    docs_dir = package_root / DIR_DOCS
    demo_dir = package_root / DIR_DEMO
    extra_dir = package_root / DIR_EXTRA
    for directory in (source_dir, docs_dir, demo_dir, extra_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # ---------- 1. 项目源代码 ----------
    print("\n[1/5] 复制项目源代码 ...")
    stats = _copy_project(project_root, source_dir)
    cleared = _sanitize_env(source_dir / "config" / ".env")
    stats["cleared_keys"] = cleared
    print(f"      已复制 {stats['files']} 个文件、{stats['dirs']} 个目录，合计 {_human_size(stats['bytes'])}")
    print(f"      排除缓存/临时项 {len(stats['excluded'])} 个（__pycache__、.pytest_cache、output/.sandbox 等）")
    print(f"      密钥脱敏：{'、'.join(cleared) if cleared else '无需清理（原文件密钥字段为空）'}")

    # ---------- 2. 交付文档 ----------
    print("\n[2/5] 收集交付文档 ...")
    copied_docs: list[str] = []
    missing_docs: list[str] = []
    for name in DOCX_FILES:
        source = project_root / "docs" / name
        if source.is_file():
            shutil.copy2(source, docs_dir / name)
            copied_docs.append(name)
            print(f"      ✔ {name}")
        else:
            missing_docs.append(name)
            print(f"      ✘ 缺失：{name}（请先运行 batch_md2docx.py 生成）")

    pptx_source = project_root / PPT_FILE
    if pptx_source.is_file():
        shutil.copy2(pptx_source, docs_dir / PPT_FILE)
        copied_docs.append(PPT_FILE)
        print(f"      ✔ {PPT_FILE}（项目简介演示文稿，由 generate_ppt.py 生成）")
    else:
        missing_docs.append(PPT_FILE)
        print(f"      ✘ 缺失：{PPT_FILE}（可运行 generate_ppt.py 生成）")

    # ---------- 3. 演示材料 ----------
    print("\n[3/5] 收集演示材料 ...")
    csv_source = project_root / "data" / DEMO_CSV
    if csv_source.is_file():
        shutil.copy2(csv_source, demo_dir / DEMO_CSV)
        print(f"      ✔ {DEMO_CSV}（示例工业数据集，可用于复现演示）")
    else:
        print(f"      ✘ 缺失：data/{DEMO_CSV}")
    print(f"      · 演示视频待录制：录制完成后命名为 产品交互演示视频.mp4 放入 {DIR_DEMO}/ 即可")

    # ---------- 4. 补充材料 ----------
    print("\n[4/5] 整理补充材料 ...")
    md_dir = extra_dir / "md源文件"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_count = 0
    for md_path in sorted((project_root / "docs").glob("*.md")):
        if md_path.name.endswith(".bak"):
            continue
        shutil.copy2(md_path, md_dir / md_path.name)
        md_count += 1
    print(f"      ✔ Markdown 源文件 {md_count} 份 → {DIR_EXTRA}/md源文件/")

    evidence = _write_test_evidence(test_log, extra_dir / "单元测试截图", project_root)
    if evidence["log"]:
        print(f"      ✔ 单元测试运行日志 → {DIR_EXTRA}/单元测试截图/单元测试运行日志.txt")
    else:
        print("      ! 未提供测试日志（可加 --test-log 参数附带单元测试证据）")
    if evidence["image"]:
        print(f"      ✔ 单元测试结果图片 → {DIR_EXTRA}/单元测试截图/单元测试结果.png")
    print(f"      ✔ 测试用例统计：{sum(evidence['overview'].values())} 个（详见证据说明）")

    checklist = _write_checklist(extra_dir, stats)
    print(f"      ✔ 提交材料清单 → {DIR_EXTRA}/{checklist.name}")

    # ---------- 5. 安全扫描与汇总 ----------
    print("\n[5/5] 安全扫描与汇总 ...")
    blocking, informational = _scan_secrets(package_root)
    if blocking:
        print("      ✘ 检测到疑似真实密钥残留（请手工检查）：")
        for path, fragment in blocking[:10]:
            print(f"        {path}：{fragment}")
    else:
        print("      ✔ 未发现疑似密钥残留（config/.env 密钥字段已清空）")
    if informational:
        fixture_files = sorted({item[0] for item in informational})
        print(f"      · 另有 {len(informational)} 处占位密钥（如 sk-unit-test）位于单元测试夹具中，"
              f"属正常测试数据：{fixture_files[0]} 等 {len(fixture_files)} 个文件")

    all_paths = [path for path in package_root.rglob("*") if path.is_file()]
    total_files = len(all_paths)
    total_bytes = sum(path.stat().st_size for path in all_paths)
    print(f"\n提交包共 {total_files} 个文件，合计 {_human_size(total_bytes)}")
    print("体积最大的 5 个文件（便于确认是否还需要精简）：")
    for path in sorted(all_paths, key=lambda item: -item.stat().st_size)[:5]:
        print(f"      {_human_size(path.stat().st_size):>10}  {path.relative_to(package_root)}")
    print("\n目录结构：")
    _print_tree(package_root, max_entries=80)
    print("\n待办：录制演示视频 → 03_演示材料/产品交互演示视频.mp4")
    print("=" * 76)
    return 0 if not missing_docs and not blocking else 1


if __name__ == "__main__":
    sys.exit(main())
