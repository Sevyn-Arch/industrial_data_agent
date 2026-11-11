"""代码安全沙箱执行器（subprocess 子进程隔离实现）。

安全设计（对应需求「必须设置执行超时、目录权限限制、危险代码校验」）
========================================================================

1. **危险代码校验（静态 AST 白名单）**

   - 模块导入白名单：仅允许数据分析相关库（pandas / numpy / scipy / sklearn /
     matplotlib / seaborn / joblib / 项目自带 analytics 等）；``os``、``sys``、
     ``subprocess``、``socket``、``ctypes``、``pickle``、``importlib`` 等一律拒绝；
   - 禁用内置危险函数：``exec`` / ``eval`` / ``compile`` / ``__import__`` / ``open`` /
     ``input`` / ``getattr`` 系列 / ``globals`` / ``locals``；
   - 禁止访问双下划线魔术属性（``obj.__class__.__bases__`` 等经典逃逸链）；
   - 禁止 ``while True`` 无界循环、超大内存分配等资源耗尽写法；
   - 正则黑名单做二次冗余筛查，与 AST 校验互为补充。

2. **目录权限限制**

   - 子进程工作目录（cwd）为每次运行独立创建的空目录 ``output/.sandbox/run_<uuid>``；
   - 代码中出现的绝对路径字面量必须落在 ``data/`` 或 ``output/`` 之内，否则拒绝执行；
   - 禁止文件系统枚举（glob / walk / listdir / scandir）与删除（unlink / rmtree / rmdir）；
   - 环境变量白名单化：只透传 PATH / SYSTEMROOT 等系统必需变量，
     ``DEEPSEEK_API_KEY`` 等敏感变量**不会**传入子进程，避免密钥经生成代码外泄。

3. **执行超时**

   - ``subprocess`` 级别超时（默认 60s，可配置）；超时后强制杀掉整个进程树
     （Windows 使用 ``taskkill /F /T``，POSIX 使用 ``os.killpg``）；
   - POSIX 平台额外通过 ``resource`` 限制 CPU 时间、地址空间与单文件大小；
   - 标准输出/错误输出按 ``max_output_chars`` 截断，避免内存膨胀。

4. **执行方式**

   - 生成代码写入独立文件后由子进程 ``python -I -X utf8 <script>`` 执行，
     **绝不使用 exec()**；``-I`` 隔离模式使子进程忽略 PYTHONPATH 与用户 site-packages。

残余风险与生产建议
========================================================================
子进程与主进程同属一个操作系统账户，``subprocess`` 隔离**不是**强隔离边界：若攻击者
能完全控制生成代码，理论上仍可能借助合法依赖库的未知漏洞发起攻击。因此生产环境建议：
以低权限专用账户运行、把本沙箱放入 Docker/gVisor 等容器、关闭容器网络出口、对输出
目录设置独立磁盘配额。本模块实现「进程隔离 + 静态校验 + 超时终止 + 目录限制 + 环境
变量脱敏」五道防线，作为容器化之前的第一层防护。
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from analytics import (
    ALLOWED_PATH_ROOTS,
    CHART_DIR,
    DATA_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    SANDBOX_WORK_DIR,
    ensure_dirs,
    fail,
    get_logger,
    ok,
    truncate_text,
)

__all__ = [
    "SandboxPolicy",
    "ExecutionResult",
    "validate_code",
    "check_code_safety",
    "execute_code",
    "build_script",
    "SANDBOX_PRELUDE",
    "ALLOWED_IMPORTS",
    "FORBIDDEN_BUILTINS",
    "FORBIDDEN_ATTRIBUTES",
    "ERROR_MESSAGES",
    "ERROR_TYPE_SECURITY",
    "ERROR_TYPE_RUNTIME",
    "ERROR_TYPE_TIMEOUT",
]

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# 错误类型与面向用户的提示文案（前端与 Agent 共用同一份）
# --------------------------------------------------------------------------- #
ERROR_TYPE_SECURITY = "security"
ERROR_TYPE_RUNTIME = "runtime"
ERROR_TYPE_TIMEOUT = "timeout"

ERROR_MESSAGES: dict[str, str] = {
    ERROR_TYPE_SECURITY: "代码包含危险操作，已被安全沙箱拦截，禁止执行",
    ERROR_TYPE_RUNTIME: "分析代码执行失败，请检查字段名或调整分析需求",
    ERROR_TYPE_TIMEOUT: "分析任务执行超时，已强制终止，请简化分析需求",
}

# --------------------------------------------------------------------------- #
# 一、静态安全规则
# --------------------------------------------------------------------------- #
#: 允许导入的顶层模块白名单（白名单之外一律拒绝）
ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "analytics",  # 项目自带、已通过单元测试的分析算法库
        "pandas",
        "numpy",
        "scipy",
        "sklearn",
        "matplotlib",
        "seaborn",
        "joblib",
        "math",
        "statistics",
        "json",
        "datetime",
        "re",
        "itertools",
        "functools",
        "collections",
        "warnings",
        "typing",
        "pathlib",
        "decimal",
        "fractions",
        "string",
        "textwrap",
        "time",
        "random",
        "copy",
    }
)

#: 禁止出现的内置函数名（以裸名字形式调用）
FORBIDDEN_BUILTINS: frozenset[str] = frozenset(
    {
        "exec",
        "eval",
        "compile",
        "__import__",
        "open",
        "input",
        "breakpoint",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "memoryview",
        "exit",
        "quit",
        "help",
    }
)

#: 禁止调用的对象属性 / 方法名（文件破坏、进程创建、环境变量读写等）
FORBIDDEN_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "read_text",
        "read_bytes",
        "write_text",
        "write_bytes",
        "open",
        "unlink",
        "rmtree",
        "rmdir",
        "removedirs",
        "chmod",
        "chown",
        "symlink",
        "link",
        "scandir",
        "listdir",
        "walk",
        "glob",
        "rglob",
        "iglob",
        "iterdir",
        "rename",
        "mkdtemp",
        "mkstemp",
        "system",
        "popen",
        "spawn",
        "fork",
        "execv",
        "execve",
        "kill",
        "killpg",
        "startfile",
        "putenv",
        "getenv",
        "environ",
    }
)

#: 正则黑名单（第二道冗余筛查，命中即拒绝）
DANGEROUS_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        r"\bimport\s+(os|sys|subprocess|socket|shutil|ctypes|importlib|pickle|marshal|"
        r"multiprocessing|threading|requests|urllib|http|pty|signal|resource|tempfile|glob|"
        r"runpy|code|codeop|builtins|inspect|platform|webbrowser)\b",
        "禁止导入系统级或网络级模块",
    ),
    (
        r"\bfrom\s+(os|sys|subprocess|socket|shutil|ctypes|importlib|pickle|marshal|"
        r"multiprocessing|requests|urllib|http|pty|signal|resource|tempfile|glob)\s+import\b",
        "禁止从系统级或网络级模块导入",
    ),
    (r"\b(exec|eval|compile|__import__)\s*\(", "禁止动态执行或编译代码"),
    (r"\bsubprocess\b|\bos\.system\b|\bos\.popen\b|popen\s*\(", "禁止创建子进程或调用系统命令"),
    (r"\b(socket|requests|urllib|ftplib|smtplib|telnetlib|http\.client)\b", "禁止网络访问"),
    (r"__\w+__", "禁止访问双下划线（魔术）属性"),
    (r"(10\s*\*\*\s*[3-9]\d*|\*\s*10\s*\*\*\s*[3-9])", "疑似超大内存分配，已拒绝执行"),
    (r"\bcurl\b|\bwget\b|powershell|cmd\.exe|/bin/sh|bash\s+-c", "禁止调用外部命令解释器"),
)

_ABSOLUTE_PATH_PATTERN = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|/)")

#: 结果包中已被占用的键名，展开为附加字段时必须剔除，避免关键字冲突
_RESERVED_RESULT_KEYS: frozenset[str] = frozenset({"status", "data", "message", "error", "charts"})


def _extras(payload: dict[str, Any]) -> dict[str, Any]:
    """从执行结果字典中提取可安全展开为结果包附加字段的键值对。"""
    try:
        return {key: value for key, value in dict(payload).items() if key not in _RESERVED_RESULT_KEYS}
    except Exception:  # noqa: BLE001
        return {}


@dataclass
class SandboxPolicy:
    """沙箱执行策略（可按环境调整，默认值面向单机工业分析场景）。"""

    timeout: float = 60.0
    """单次执行超时（秒）。"""
    max_output_chars: int = 20000
    """stdout / stderr 最大字符数。"""
    max_code_chars: int = 20000
    """生成代码最大字符数。"""
    memory_limit_mb: int = 2048
    """POSIX 平台地址空间上限（MB）。"""
    cpu_limit_seconds: int = 60
    """POSIX 平台 CPU 时间上限（秒）。"""
    max_file_size_mb: int = 64
    """POSIX 平台单文件写入上限（MB）。"""
    allow_while_true: bool = False
    """是否允许 while True 无界循环。"""
    keep_workspace: bool = True
    """是否保留运行目录（便于问题追踪）。"""
    max_workspaces: int = 20
    """保留的历史运行目录数量上限。"""

    @classmethod
    def from_env(cls) -> "SandboxPolicy":
        """从环境变量构造策略（解析失败时使用默认值，绝不抛异常）。"""
        policy = cls()
        try:
            policy.timeout = float(os.environ.get("SANDBOX_TIMEOUT", policy.timeout))
            policy.max_output_chars = int(os.environ.get("SANDBOX_MAX_OUTPUT_CHARS", policy.max_output_chars))
            policy.memory_limit_mb = int(os.environ.get("SANDBOX_MEMORY_MB", policy.memory_limit_mb))
        except Exception:  # noqa: BLE001 - 配置解析失败时保持默认值
            logger.warning("沙箱策略环境变量解析失败，已使用默认配置")
        return policy


@dataclass
class ExecutionResult:
    """沙箱执行结果（与统一结果包的 ``data`` 字段结构一致）。"""

    status: str = "error"
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    result: Any = None
    charts: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    timed_out: bool = False
    error: str | None = None
    error_type: str | None = None
    code_path: str | None = None
    result_path: str | None = None
    workdir: str | None = None
    violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """转为可 JSON 序列化的字典。"""
        return {
            "status": self.status,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "result": self.result,
            "charts": list(self.charts),
            "duration_seconds": round(self.duration_seconds, 4),
            "timed_out": self.timed_out,
            "error": self.error,
            "error_type": self.error_type,
            "code_path": self.code_path,
            "result_path": self.result_path,
            "workdir": self.workdir,
            "violations": list(self.violations),
        }


# --------------------------------------------------------------------------- #
# 二、沙箱预置环境（可信代码，注入到生成代码之前）
# --------------------------------------------------------------------------- #
SANDBOX_PRELUDE = """# -*- coding: utf-8 -*-
'''工业数据分析 Agent —— 沙箱运行环境预置代码。

本段由系统自动注入（可信代码），提供数据目录、图表目录、结果输出与算法库访问能力。
生成代码只需关注分析逻辑本身，且必须调用 emit_result(...) 输出结构化结果。
'''

import json as _json
import os as _os
import pathlib as _pathlib
import sys as _sys
import warnings as _warnings

_warnings.filterwarnings("ignore")

# 允许生成代码复用项目内已通过单元测试的分析算法库（只读、无副作用）
_sys.path.insert(0, _os.environ.get("SANDBOX_PROJECT_ROOT", ""))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure as _Figure

# 统一字体配置：Linux（云端）优先 Noto Sans CJK SC，Windows 优先微软雅黑 / 黑体，
# 并关闭 unicode_minus 以修复负号方块问题。配置与项目算法库共用同一实现。
try:
    from analytics import apply_font_to_figure, configure_chart_fonts
except Exception:  # 项目模块不可用时退化为内置候选列表
    def configure_chart_fonts(force=False):
        plt.rcParams["font.sans-serif"] = [
            "Noto Sans CJK SC", "Noto Sans CJK JP", "Microsoft YaHei", "SimHei",
            "WenQuanYi Zen Hei", "Arial Unicode MS", "DejaVu Sans",
        ]
        plt.rcParams["font.family"] = ["sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
        return plt.rcParams["font.sans-serif"][0]

    def apply_font_to_figure(figure):
        return None

configure_chart_fonts()
plt.rcParams["savefig.bbox"] = "tight"

# 生成代码可能自行覆盖 rcParams（例如设置成 DejaVu Sans），导致中文变成方块。
# 这里在每次保存图片前统一校正字体，并回溯修改图内已创建的标题/坐标轴等文本。
_ORIGINAL_FIGURE_SAVEFIG = _Figure.savefig


def _savefig_with_chinese_fonts(self, *args, **kwargs):
    try:
        configure_chart_fonts(force=True)
        apply_font_to_figure(self)
    except Exception:
        pass
    return _ORIGINAL_FIGURE_SAVEFIG(self, *args, **kwargs)


_Figure.savefig = _savefig_with_chinese_fonts

DATA_DIR = _pathlib.Path(_os.environ["SANDBOX_DATA_DIR"])
CHART_DIR = _pathlib.Path(_os.environ["SANDBOX_CHART_DIR"])
OUTPUT_DIR = _pathlib.Path(_os.environ["SANDBOX_OUTPUT_DIR"])
MODEL_DIR = _pathlib.Path(_os.environ["SANDBOX_MODEL_DIR"])
_RESULT_PATH = _pathlib.Path(_os.environ["SANDBOX_RESULT_PATH"])

try:
    CONTEXT = _json.loads(_os.environ.get("SANDBOX_CONTEXT") or "{}")
except Exception:
    CONTEXT = {}

DATA_FILE = CONTEXT.get("data_file")


def _json_default(obj):
    '''把 numpy / pandas 对象转换为可 JSON 序列化的原生对象。'''
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)
    return str(obj)


def load_data(name=None, **kwargs):
    '''读取 data 目录下的 CSV；name 为空时使用本次分析绑定的数据文件。'''
    target = name or DATA_FILE
    if target is None:
        raise ValueError("未指定数据文件：请在 CONTEXT 中提供 data_file 或显式传入文件名")
    path = _pathlib.Path(str(target))
    if not path.is_absolute():
        path = DATA_DIR / path
    return pd.read_csv(path, **kwargs)


def save_chart(name="chart.png", fig=None, dpi=150):
    '''保存图表到 output/charts 并返回路径字符串。'''
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    target = CHART_DIR / _pathlib.Path(str(name)).name
    (fig if fig is not None else plt.gcf()).savefig(target, dpi=dpi)
    return str(target)


def list_models():
    '''返回 output/models 下已保存的模型文件路径（按修改时间升序）。'''
    if not MODEL_DIR.exists():
        return []
    files = [item for item in MODEL_DIR.iterdir() if item.suffix == ".joblib"]
    return sorted(files, key=lambda item: item.stat().st_mtime)


def emit_result(payload):
    '''输出结构化结果（分析代码必须调用一次，用于自动生成报告）。'''
    with open(_RESULT_PATH, "w", encoding="utf-8") as handle:
        _json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
    return str(_RESULT_PATH)
"""


def build_script(user_code: str, context: dict[str, Any] | None = None) -> str:
    """将沙箱预置代码与生成代码拼接为完整可执行脚本。

    Args:
        user_code: 生成代码主体。
        context: 运行上下文（注入为 ``CONTEXT`` 变量，例如 ``{"data_file": "demo.csv"}``）。

    Returns:
        str: 完整脚本内容。
    """
    header = f"# 运行上下文（自动注入）: {json.dumps(context or {}, ensure_ascii=False)}\n"
    return f"{SANDBOX_PRELUDE}\n{header}\n# ===== 以下为生成的分析代码 =====\n{user_code}\n"


# --------------------------------------------------------------------------- #
# 三、静态校验
# --------------------------------------------------------------------------- #
def _check_import(node: ast.AST, violations: list[str]) -> None:
    """校验 import 语句是否位于白名单内。"""
    if isinstance(node, ast.Import):
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root not in ALLOWED_IMPORTS:
                violations.append(f"第 {node.lineno} 行：禁止导入模块 {alias.name}（不在白名单内）")
    elif isinstance(node, ast.ImportFrom):
        if node.level and node.level > 0:
            violations.append(f"第 {node.lineno} 行：禁止使用相对导入")
            return
        root = (node.module or "").split(".")[0]
        if root not in ALLOWED_IMPORTS:
            violations.append(f"第 {node.lineno} 行：禁止从模块 {node.module} 导入（不在白名单内）")


def _check_name(node: ast.AST, violations: list[str]) -> None:
    """校验标识符：禁用内置危险函数与魔术名称。"""
    if not isinstance(node, ast.Name):
        return
    if node.id in FORBIDDEN_BUILTINS:
        violations.append(f"第 {node.lineno} 行：禁止调用 {node.id}()")
    if re.fullmatch(r"__\w+__", node.id):
        violations.append(f"第 {node.lineno} 行：禁止访问魔术名称 {node.id}")


def _check_attribute(node: ast.Attribute, violations: list[str]) -> None:
    """校验属性访问：禁用魔术属性与危险方法。"""
    attr = node.attr
    if re.fullmatch(r"__\w+__", attr):
        violations.append(f"第 {node.lineno} 行：禁止访问魔术属性 .{attr}")
    if attr in FORBIDDEN_ATTRIBUTES:
        violations.append(f"第 {node.lineno} 行：禁止调用属性/方法 .{attr}")


def _check_path_literals(node: ast.AST, violations: list[str], allowed_roots: Sequence[Path]) -> None:
    """校验字符串字面量中的路径是否落在允许目录范围内。"""
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        return
    text = node.value.strip()
    if not text or len(text) > 512:
        return
    try:
        parts = Path(text).parts
    except Exception:  # noqa: BLE001
        return
    looks_like_path = bool(_ABSOLUTE_PATH_PATTERN.match(text)) or ".." in parts
    if not looks_like_path:
        return
    try:
        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = SANDBOX_WORK_DIR / candidate
        resolved = candidate.resolve()
    except Exception:  # noqa: BLE001 - 无法解析的路径一律拒绝
        violations.append(f"第 {node.lineno} 行：路径 {text} 无法解析，已拒绝")
        return
    if not any(resolved.is_relative_to(root) for root in allowed_roots):
        allowed = "、".join(str(root) for root in allowed_roots)
        violations.append(f"第 {node.lineno} 行：路径 {text} 超出允许目录范围（仅允许 {allowed} 之下）")


def _check_while_loop(node: ast.While, violations: list[str], policy: SandboxPolicy) -> None:
    """禁止无界循环（``while True``），避免资源耗尽。"""
    if policy.allow_while_true:
        return
    test = node.test
    if isinstance(test, ast.Constant) and test.value is True:
        violations.append(f"第 {node.lineno} 行：禁止 while True 无界循环（可能造成资源耗尽）")


def _resolve_allowed_roots(extra_roots: Sequence[Path | str] | None) -> tuple[Path, ...]:
    """合并默认允许目录与本次运行的附加允许目录（仅可扩展，不可收窄）。"""
    roots: list[Path] = [Path(root).resolve() for root in ALLOWED_PATH_ROOTS]
    for item in extra_roots or ():
        try:
            candidate = Path(item).resolve()
        except Exception:  # noqa: BLE001
            continue
        if candidate not in roots:
            roots.append(candidate)
    return tuple(roots)


def validate_code(
    code: str,
    policy: SandboxPolicy | None = None,
    allowed_roots: Sequence[Path | str] | None = None,
) -> tuple[bool, list[str]]:
    """对生成代码执行静态安全校验。

    Args:
        code: 待校验的 Python 源码。
        policy: 沙箱策略（控制 while True 等可放宽规则）。
        allowed_roots: 本次运行允许访问的目录（在默认 ``data/``、``output/`` 基础上扩展）。

    Returns:
        tuple[bool, list[str]]: ``(是否通过, 违规说明列表)``。内部异常一律捕获并判定为不通过。
    """
    policy = policy or SandboxPolicy.from_env()
    roots = _resolve_allowed_roots(allowed_roots)
    violations: list[str] = []
    try:
        if not isinstance(code, str) or not code.strip():
            return False, ["代码为空，无法执行"]
        if len(code) > policy.max_code_chars:
            return False, [f"代码长度 {len(code)} 超过上限 {policy.max_code_chars} 字符"]

        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return False, [f"代码存在语法错误：第 {exc.lineno} 行 - {exc.msg}"]

        for pattern, message in DANGEROUS_PATTERNS:
            match = re.search(pattern, code)
            if match:
                line_no = code[: match.start()].count("\n") + 1
                violations.append(f"第 {line_no} 行：命中危险模式（{message}）")

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                _check_import(node, violations)
            elif isinstance(node, ast.Name):
                _check_name(node, violations)
            elif isinstance(node, ast.Attribute):
                _check_attribute(node, violations)
            elif isinstance(node, ast.Constant):
                _check_path_literals(node, violations, roots)
            elif isinstance(node, ast.While):
                _check_while_loop(node, violations, policy)
            elif isinstance(node, ast.Assert):
                violations.append(f"第 {node.lineno} 行：禁止使用 assert（无法作为结果输出手段）")

        violations = list(dict.fromkeys(violations))  # 去重且保持顺序
        return (len(violations) == 0), violations
    except Exception as exc:  # noqa: BLE001 - 校验器自身异常也必须安全返回
        logger.exception("代码安全校验执行异常")
        return False, [f"安全校验器内部异常：{exc}"]


def check_code_safety(
    code: str,
    policy: SandboxPolicy | None = None,
    allowed_roots: Sequence[Path | str] | None = None,
) -> dict[str, Any]:
    """静态安全校验（对外 API，返回统一结果包，供前端展示校验明细）。"""
    try:
        is_valid, violations = validate_code(code, policy, allowed_roots=allowed_roots)
        data = {
            "is_safe": is_valid,
            "violations": violations,
            "n_lines": len(str(code).splitlines()),
            "allowed_imports": sorted(ALLOWED_IMPORTS),
            "forbidden_builtins": sorted(FORBIDDEN_BUILTINS),
            "forbidden_attributes": sorted(FORBIDDEN_ATTRIBUTES),
        }
        message = "安全校验通过，代码可进入沙箱执行" if is_valid else f"安全校验未通过，发现 {len(violations)} 处违规"
        return ok(data, message)
    except Exception as exc:  # noqa: BLE001
        logger.exception("check_code_safety 执行失败")
        return fail("代码安全校验失败", exc)


# --------------------------------------------------------------------------- #
# 四、子进程执行
# --------------------------------------------------------------------------- #
def _build_env(
    run_dir: Path,
    result_path: Path,
    context: dict[str, Any] | None,
    data_dir: Path | str | None = None,
) -> dict[str, str]:
    """构造子进程环境变量（白名单透传 + 沙箱目录注入 + 敏感变量剥离）。

    Args:
        run_dir: 本次运行的独立工作目录。
        result_path: 结构化结果 JSON 的输出路径。
        context: 运行上下文（注入为 ``CONTEXT``）。
        data_dir: 本次运行的数据目录；``None`` 表示使用项目默认 ``data/`` 目录。
    """
    env: dict[str, str] = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(SANDBOX_WORK_DIR / "mplconfig"),
        "SANDBOX_DATA_DIR": str(Path(data_dir) if data_dir else DATA_DIR),
        "SANDBOX_CHART_DIR": str(CHART_DIR),
        "SANDBOX_OUTPUT_DIR": str(OUTPUT_DIR),
        "SANDBOX_MODEL_DIR": str(MODEL_DIR),
        "SANDBOX_PROJECT_ROOT": str(PROJECT_ROOT),
        "SANDBOX_RESULT_PATH": str(result_path),
        "SANDBOX_CONTEXT": json.dumps(context or {}, ensure_ascii=False),
    }
    # 仅透传系统运行必需变量；DEEPSEEK_API_KEY 等敏感信息一律不传入子进程
    for key in ("PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "NUMBER_OF_PROCESSORS", "LANG"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    # HOME / TEMP 指向本次运行目录，避免污染用户配置与系统临时目录
    env.update({"HOME": str(run_dir), "USERPROFILE": str(run_dir), "TEMP": str(run_dir), "TMP": str(run_dir)})
    return env


def _make_preexec_fn(policy: SandboxPolicy) -> Any:
    """返回 POSIX 资源限制函数（Windows 平台返回 None）。"""
    if os.name == "nt":
        return None

    def _limit() -> None:  # pragma: no cover - 仅在 POSIX 平台执行
        try:
            import resource

            memory = int(policy.memory_limit_mb) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
            resource.setrlimit(resource.RLIMIT_CPU, (int(policy.cpu_limit_seconds),) * 2)
            file_size = int(policy.max_file_size_mb) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_FSIZE, (file_size, file_size))
        except Exception:  # noqa: BLE001 - 限制设置失败不阻断执行
            pass

    return _limit


def _terminate_tree(process: Any) -> None:
    """强制结束子进程及其派生的全部子进程。"""
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True,
                check=False,
            )
        else:  # pragma: no cover - 仅在 POSIX 平台执行
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except Exception:  # noqa: BLE001 - 兜底强杀
        try:
            process.kill()
        except Exception:  # noqa: BLE001
            logger.warning("子进程强制结束失败，pid=%s", getattr(process, "pid", "unknown"))


def _snapshot_charts() -> dict[str, float]:
    """记录图表目录的文件快照（路径 -> 修改时间），用于识别本次新增或被覆盖的图表。"""
    try:
        ensure_dirs()
        snapshot: dict[str, float] = {}
        for path in CHART_DIR.glob("*"):
            if path.is_file():
                snapshot[str(path)] = path.stat().st_mtime
        return snapshot
    except Exception:  # noqa: BLE001
        return {}


def _diff_charts(before: dict[str, float], after: dict[str, float]) -> list[str]:
    """对比前后快照，返回本次新增或被覆盖的图表文件列表。"""
    try:
        return sorted(path for path, mtime in after.items() if before.get(path) != mtime)
    except Exception:  # noqa: BLE001
        return []


def _prune_workspaces(policy: SandboxPolicy) -> None:
    """清理历史运行目录，避免磁盘无限增长。"""
    try:
        if policy.max_workspaces <= 0:
            return
        runs = sorted(
            (path for path in SANDBOX_WORK_DIR.glob("run_*") if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for stale in runs[int(policy.max_workspaces):]:
            shutil.rmtree(stale, ignore_errors=True)
    except Exception:  # noqa: BLE001
        logger.warning("历史运行目录清理失败", exc_info=True)


def _read_result(result_path: Path) -> tuple[Any, str | None]:
    """读取生成代码写出的结构化结果 JSON。"""
    if not result_path.exists():
        return None, "生成代码未调用 emit_result()，未产出结构化结果"
    try:
        content = result_path.read_text(encoding="utf-8")
        if not content.strip():
            return None, "结果文件为空"
        return json.loads(content), None
    except Exception as exc:  # noqa: BLE001
        return None, f"结果文件解析失败：{exc}"


def execute_code(
    code: str,
    policy: SandboxPolicy | None = None,
    context: dict[str, Any] | None = None,
    validate: bool = True,
    data_dir: Path | str | None = None,
) -> dict[str, Any]:
    """在隔离子进程中执行生成代码，并返回统一结果包。

    Args:
        code: 待执行的 Python 源码（由 LLM 生成或模板生成）。
        policy: 沙箱策略；``None`` 时读取环境变量并回落到默认值。
        context: 运行上下文，例如 ``{"data_file": "process.csv", "question": "..."}``。
        validate: 是否执行静态安全校验（生产环境应保持 ``True``）。
        data_dir: 本次运行的数据目录（默认项目 ``data/``）；该目录会同时被加入
            静态校验的「允许路径范围」，保证自定义数据目录场景下路径校验依然严格。

    Returns:
        dict: 统一结果包；``data`` 为 :meth:`ExecutionResult.to_dict` 的结构，
        ``charts`` 为本次新增的图表文件列表。
        校验未通过时 ``status="error"``，``data["violations"]`` 给出违规明细。
    """
    policy = policy or SandboxPolicy.from_env()
    effective_data_dir = Path(data_dir).resolve() if data_dir else DATA_DIR
    result = ExecutionResult()
    started = time.perf_counter()
    run_dir: Path | None = None
    try:
        ensure_dirs()

        # 1) 危险代码校验
        if validate:
            is_valid, violations = validate_code(code, policy, allowed_roots=(effective_data_dir,))
            if not is_valid:
                result.violations = violations
                result.error_type = ERROR_TYPE_SECURITY
                result.error = "代码未通过安全校验：" + "；".join(violations[:5])
                result.duration_seconds = time.perf_counter() - started
                payload = result.to_dict()
                return fail(ERROR_MESSAGES[ERROR_TYPE_SECURITY], result.error, data=payload, **_extras(payload))

        # 2) 独立运行目录（目录权限限制：cwd 限定在空目录内）
        run_dir = SANDBOX_WORK_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        result.workdir = str(run_dir)

        script_path = run_dir / "analysis_code.py"
        result_path = run_dir / "result.json"
        script_path.write_text(build_script(code, context), encoding="utf-8")
        result.code_path = str(script_path)
        result.result_path = str(result_path)

        # 3) 子进程执行（python -I 隔离模式；绝不使用 exec）
        env = _build_env(run_dir, result_path, context, data_dir=effective_data_dir)
        command = [sys.executable, "-I", "-X", "utf8", str(script_path)]
        creationflags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
        charts_before = _snapshot_charts()

        process = subprocess.Popen(
            command,
            cwd=str(run_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
            start_new_session=(os.name != "nt"),
            preexec_fn=_make_preexec_fn(policy),
        )
        try:
            stdout, stderr = process.communicate(timeout=float(policy.timeout))
            result.exit_code = process.returncode
        except subprocess.TimeoutExpired:
            result.timed_out = True
            result.error_type = ERROR_TYPE_TIMEOUT
            _terminate_tree(process)
            try:
                stdout, stderr = process.communicate(timeout=5)
            except Exception:  # noqa: BLE001
                stdout, stderr = "", ""
            result.exit_code = process.returncode
            result.error = f"执行超时（超过 {policy.timeout} 秒），进程树已强制终止"

        result.stdout = truncate_text(stdout, policy.max_output_chars)
        result.stderr = truncate_text(stderr, policy.max_output_chars)

        # 4) 结果解析与图表收集
        payload, result_note = _read_result(result_path)
        result.result = payload
        charts_after = _snapshot_charts()
        result.charts = _diff_charts(charts_before, charts_after)

        if result.timed_out:
            result.status = "error"
        elif result.exit_code == 0:
            result.status = "success"
            if result_note:
                result.error = result_note
        else:
            result.status = "error"
            result.error_type = ERROR_TYPE_RUNTIME
            tail = [line for line in (result.stderr or "").strip().splitlines() if line.strip()]
            result.error = tail[-1] if tail else f"子进程异常退出，退出码 {result.exit_code}"

        result.duration_seconds = time.perf_counter() - started
        data = result.to_dict()

        if result.status == "success":
            message = f"代码执行成功（耗时 {data['duration_seconds']:.2f} 秒）"
            if result.charts:
                message += f"，生成 {len(result.charts)} 张图表"
            if result.error:
                message += f"；提示：{result.error}"
            return ok(data, message, charts=result.charts, **_extras(data))
        user_message = ERROR_MESSAGES.get(result.error_type or "", result.error or "分析代码执行失败")
        return fail(user_message, result.error or "代码执行失败", data=data, charts=result.charts, **_extras(data))
    except Exception as exc:  # noqa: BLE001 - 沙箱自身异常也必须安全返回
        logger.exception("沙箱执行器内部异常")
        result.status = "error"
        result.error_type = ERROR_TYPE_RUNTIME
        result.error = f"沙箱执行器内部异常：{exc}"
        result.duration_seconds = time.perf_counter() - started
        payload = result.to_dict()
        return fail(ERROR_MESSAGES[ERROR_TYPE_RUNTIME], result.error, data=payload, **_extras(payload))
    finally:
        try:
            if run_dir is not None and not policy.keep_workspace:
                shutil.rmtree(run_dir, ignore_errors=True)
            _prune_workspaces(policy)
        except Exception:  # noqa: BLE001
            logger.warning("运行目录清理失败", exc_info=True)


def _self_check() -> dict[str, Any]:
    """执行器自检：分别运行一段合法代码与一段非法代码，验证两条路径都符合预期。"""
    try:
        safe_code = (
            "import pandas as pd\n"
            "values = pd.Series([1, 2, 3, 4, 100])\n"
            "plt.figure(figsize=(4, 3))\n"
            "plt.plot(values)\n"
            "chart = save_chart('self_check.png')\n"
            "emit_result({'mean': float(values.mean()), 'chart': chart})\n"
            "print('self-check ok')\n"
        )
        good = execute_code(safe_code, context={"data_file": None})
        bad = execute_code("import os\nprint(os.getcwd())\n")
        data = {
            "safe_run_status": good["status"],
            "safe_run_stdout": good["data"]["stdout"].strip(),
            "safe_run_charts": len(good["charts"]),
            "unsafe_run_status": bad["status"],
            "unsafe_run_violations": bad["data"].get("violations", []),
        }
        passed = good["status"] == "success" and bad["status"] == "error"
        return ok(data, "沙箱执行器自检通过" if passed else "沙箱执行器自检未达预期")
    except Exception as exc:  # noqa: BLE001
        return fail("沙箱执行器自检失败", exc)


if __name__ == "__main__":
    print(json.dumps(_self_check(), ensure_ascii=False, indent=2))
