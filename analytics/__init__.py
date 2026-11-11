"""数据分析算法库（analytics）包初始化模块。

本模块是全项目的「公共基础层」，职责有三：

1. 定义唯一的路径常量（数据目录、输出目录、图表目录等），其它模块禁止硬编码路径；
2. 定义统一的函数返回结构（结果包），保证所有模块的返回值格式一致、可被上层统一渲染；
3. 提供 matplotlib 中文绘图样式、日志对象、JSON 安全转换等公共能力。

统一返回结构（所有对外函数必须遵守）::

    {
        "status":  "success" | "error",   # 执行状态
        "data":    Any,                   # 业务数据（保证可 JSON 序列化）
        "message": str,                   # 面向用户的简要说明
        "error":   str | None,            # 失败原因（成功时为 None）
        "charts":  list[str],             # 本次分析产出的图表文件绝对路径
        # ... 允许通过 **extra 追加业务字段（例如沙箱执行器的 stdout / duration）
    }

分层依赖约定（上层可依赖下层，下层禁止反向依赖）::

    analytics  <-  sandbox  <-  agent  <-  report  <-  app
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    # 路径常量
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "DATA_DIR",
    "OUTPUT_DIR",
    "CHART_DIR",
    "REPORT_DIR",
    "MODEL_DIR",
    "SANDBOX_WORK_DIR",
    "ENV_FILE",
    "ALLOWED_PATH_ROOTS",
    "CHART_EXT",
    "DEFAULT_ENCODING",
    # 目录工具
    "ensure_dirs",
    "new_chart_path",
    # 统一返回结构
    "make_result",
    "ok",
    "fail",
    # 字段解析
    "resolve_column",
    # 通用工具
    "to_jsonable",
    "safe_float",
    "truncate_text",
    "get_logger",
    "setup_plot_style",
]

# --------------------------------------------------------------------------- #
# 一、路径常量（全项目唯一来源）
# --------------------------------------------------------------------------- #
# analytics/__init__.py -> analytics -> industrial_data_agent
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CONFIG_DIR: Path = PROJECT_ROOT / "config"
DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
CHART_DIR: Path = OUTPUT_DIR / "charts"
REPORT_DIR: Path = OUTPUT_DIR / "reports"
MODEL_DIR: Path = OUTPUT_DIR / "models"
SANDBOX_WORK_DIR: Path = OUTPUT_DIR / ".sandbox"
ENV_FILE: Path = CONFIG_DIR / ".env"

#: 沙箱代码中允许出现的绝对路径必须落在这些根目录之内（目录权限限制）
ALLOWED_PATH_ROOTS: tuple[Path, ...] = (DATA_DIR.resolve(), OUTPUT_DIR.resolve())

DEFAULT_ENCODING: str = "utf-8"
CHART_EXT: str = ".png"


def ensure_dirs() -> None:
    """创建项目运行所需的全部目录（幂等，可安全重复调用）。

    Raises:
        OSError: 目录创建失败时抛出，由调用方（均为带异常处理的函数）兜底。
    """
    for directory in (
        CONFIG_DIR,
        DATA_DIR,
        OUTPUT_DIR,
        CHART_DIR,
        REPORT_DIR,
        MODEL_DIR,
        SANDBOX_WORK_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# 二、统一返回结构
# --------------------------------------------------------------------------- #
def make_result(
    status: str,
    data: Any = None,
    message: str = "",
    error: str | None = None,
    charts: Sequence[str] | None = None,
    json_safe: bool = True,
    **extra: Any,
) -> dict[str, Any]:
    """构造统一格式的结果包。

    Args:
        status: 执行状态，取值为 ``"success"`` 或 ``"error"``。
        data: 业务数据（默认做 JSON 安全转换）。
        message: 面向用户的简要说明。
        error: 失败原因文本，成功时传 ``None``。
        charts: 本次产出的图表绝对路径列表。
        json_safe: 是否对 ``data`` 执行 JSON 安全转换；返回 DataFrame 等重型对象时传 ``False``。
        **extra: 需要追加到结果包中的业务字段。

    Returns:
        dict: 统一格式的结果包。
    """
    result: dict[str, Any] = {
        "status": "success" if status == "success" else "error",
        "data": to_jsonable(data) if (data is not None and json_safe) else data,
        "message": str(message or ""),
        "error": None if error is None else str(error),
        "charts": [str(item) for item in (charts or [])],
    }
    result.update(extra)
    return result


def ok(
    data: Any = None,
    message: str = "",
    charts: Sequence[str] | None = None,
    json_safe: bool = True,
    **extra: Any,
) -> dict[str, Any]:
    """构造「成功」结果包，等价于 ``make_result("success", ...)``。"""
    return make_result("success", data=data, message=message, charts=charts, json_safe=json_safe, **extra)


def fail(
    message: str,
    error: Exception | str | None = None,
    data: Any = None,
    charts: Sequence[str] | None = None,
    json_safe: bool = True,
    **extra: Any,
) -> dict[str, Any]:
    """构造「失败」结果包，避免任何异常向上抛出（规范二.2）。

    Args:
        message: 面向用户的失败说明。
        error: 原始异常对象或错误文本。
        data: 失败时仍希望返回的部分数据。
        charts: 失败前已产出的图表。
        json_safe: 是否对 ``data`` 执行 JSON 安全转换（返回 DataFrame 时传 ``False``）。
        **extra: 追加字段。
    """
    detail = "" if error is None else (str(error).strip() or type(error).__name__)
    text = str(message or "执行失败")
    if detail and detail not in text:
        text = f"{text}：{detail}"
    return make_result(
        "error",
        data=data,
        message=str(message or "执行失败"),
        error=detail or "未知错误",
        charts=charts,
        json_safe=json_safe,
        **extra,
    )


def to_jsonable(obj: Any) -> Any:
    """将任意对象递归转换为可 JSON 序列化的 Python 原生对象。

    支持 numpy 标量/数组、pandas DataFrame/Series/Timestamp、Path、集合等常见类型，
    无法识别的对象统一退化为 ``str(obj)``，确保结果包永远可以写入 JSON。
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Mapping):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(item) for item in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode(DEFAULT_ENCODING, errors="replace")

    # numpy 标量 / 数组
    if hasattr(obj, "item") and hasattr(obj, "dtype") and getattr(obj, "ndim", 0) == 0:
        try:
            return to_jsonable(obj.item())
        except Exception:  # noqa: BLE001 - 转换失败退化为字符串
            return str(obj)
    if hasattr(obj, "tolist") and hasattr(obj, "dtype"):
        try:
            return to_jsonable(obj.tolist())
        except Exception:  # noqa: BLE001
            return str(obj)

    # pandas 对象
    if hasattr(obj, "isoformat"):  # datetime / pandas.Timestamp
        try:
            return obj.isoformat()
        except Exception:  # noqa: BLE001
            return str(obj)
    if hasattr(obj, "to_dict"):  # DataFrame 使用 records 便于前端渲染
        try:
            return to_jsonable(obj.to_dict(orient="records"))
        except Exception:  # noqa: BLE001
            return str(obj)
    if hasattr(obj, "to_list"):
        try:
            return to_jsonable(obj.to_list())
        except Exception:  # noqa: BLE001
            return str(obj)
    return str(obj)


def safe_float(value: Any, default: float = float("nan")) -> float:
    """将任意值安全转换为 float，转换失败返回 ``default``。"""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result


def resolve_column(df: Any, column: Any) -> Any:
    """将用户传入的字段名解析为 DataFrame 中真实存在的列标签。

    工业场景中字段名可能来自前端表单（字符串）而 DataFrame 列标签为其它类型，
    因此这里做一次「同名优先、字符串等价次之」的宽松匹配。

    Args:
        df: pandas.DataFrame。
        column: 用户传入的字段名。

    Returns:
        Any: DataFrame 中真实存在的列标签。

    Raises:
        KeyError: 字段不存在时抛出，由调用方的异常处理转换为失败结果包。
    """
    try:
        labels = list(df.columns)
    except Exception as exc:  # noqa: BLE001
        raise TypeError(f"无法读取 DataFrame 的列名：{exc}") from exc
    if column in labels:
        return column
    text = str(column)
    for label in labels:
        if str(label) == text:
            return label
    raise KeyError(f"字段不存在：{column}；可用字段：{[str(label) for label in labels]}")


def truncate_text(text: Any, limit: int = 20000, tail_note: str = "\n...[输出过长已截断]") -> str:
    """截断过长文本，避免前端/日志被超大输出拖垮。"""
    try:
        content = "" if text is None else str(text)
    except Exception:  # noqa: BLE001
        return ""
    limit = max(0, int(limit))
    if limit and len(content) > limit:
        return content[:limit] + tail_note
    return content


# --------------------------------------------------------------------------- #
# 三、绘图公共配置
# --------------------------------------------------------------------------- #
#: 中文字体候选列表（按优先级排列）
_CJK_FONT_CANDIDATES: tuple[str, ...] = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
    "WenQuanYi Zen Hei",
    "Arial Unicode MS",
)
_PLOT_STYLE_READY: bool = False
_PLOT_FONT_NAME: str = ""


def setup_plot_style() -> str:
    """初始化 matplotlib 绘图样式（Agg 后端 + 中文字体），幂等。

    Returns:
        str: 实际生效的中文字体名；未找到中文字体时返回空字符串
        （此时中文可能显示为方块，但不会影响程序运行）。
    """
    global _PLOT_STYLE_READY, _PLOT_FONT_NAME
    if _PLOT_STYLE_READY:
        return _PLOT_FONT_NAME
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)  # 无界面后端，必须在 pyplot 之前设置
        from matplotlib import font_manager, rcParams

        # 注意：seaborn 主题会覆盖 rcParams，因此必须先应用主题、再设置中文字体
        try:  # seaborn 为可选依赖，缺失时不影响绘图
            import seaborn as sns

            sns.set_theme(style="whitegrid")
        except Exception:  # noqa: BLE001
            pass

        available = {font.name for font in font_manager.fontManager.ttflist}
        _PLOT_FONT_NAME = next((name for name in _CJK_FONT_CANDIDATES if name in available), "")
        families = ([_PLOT_FONT_NAME] if _PLOT_FONT_NAME else []) + ["DejaVu Sans"]
        rcParams["font.sans-serif"] = families
        rcParams["font.family"] = "sans-serif"
        rcParams["axes.unicode_minus"] = False  # 解决负号显示为方块的问题
        rcParams["figure.dpi"] = 110
        rcParams["savefig.bbox"] = "tight"
        rcParams["axes.grid"] = True
        rcParams["grid.alpha"] = 0.25
    except Exception:  # noqa: BLE001 - 绘图配置失败不应影响分析主流程
        get_logger(__name__).warning("matplotlib 绘图样式初始化失败，将使用默认样式", exc_info=True)
    finally:
        _PLOT_STYLE_READY = True
    return _PLOT_FONT_NAME


def new_chart_path(prefix: str = "chart", ext: str = CHART_EXT) -> Path:
    """生成唯一的图表输出路径（时间戳 + 业务前缀 + 随机后缀）。

    Args:
        prefix: 图表业务前缀，例如 ``"correlation"``。
        ext: 文件扩展名，默认 ``.png``。

    Returns:
        Path: 位于 ``output/charts`` 下的绝对路径。
    """
    ensure_dirs()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe_prefix = re.sub(r"[^0-9A-Za-z_\-]+", "_", str(prefix or "chart"))[:40] or "chart"
    return CHART_DIR / f"{stamp}_{safe_prefix}_{uuid.uuid4().hex[:6]}{ext}"


# --------------------------------------------------------------------------- #
# 四、日志
# --------------------------------------------------------------------------- #
def get_logger(name: str) -> logging.Logger:
    """获取模块级 logger；默认挂载 ``NullHandler``，由入口（app.py）统一配置级别。"""
    logger = logging.getLogger(name if name.startswith("industrial_data_agent") else f"industrial_data_agent.{name}")
    if not any(isinstance(handler, logging.NullHandler) for handler in logger.handlers) and not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def _self_check() -> dict[str, Any]:
    """模块自检：验证目录创建与结果包格式（可被单元测试或命令行直接调用）。"""
    try:
        ensure_dirs()
        sample = ok({"a": 1}, "自检通过")
        assert sample["status"] == "success" and sample["charts"] == []
        return ok({"project_root": str(PROJECT_ROOT), "font": setup_plot_style()}, "analytics 包自检通过")
    except Exception as exc:  # noqa: BLE001
        return fail("analytics 包自检失败", exc)


if __name__ == "__main__":
    print(json.dumps(_self_check(), ensure_ascii=False, indent=2))
