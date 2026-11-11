"""基础统计分析模块。

功能：

1. 自动识别 CSV 字段与数据类型（``detect_schema``）；
2. 单列 / 多列描述性统计（均值、标准差、最值、中位数、四分位、偏度、峰度等）；
3. 频次统计、缺失值统计、直方图绘制。

约定：所有对外函数（不带下划线前缀）统一返回 :func:`analytics.make_result`
结果包，并且内部捕获全部异常，绝不向上抛出运行时异常。
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import fail, get_logger, new_chart_path, ok, resolve_column, safe_float, setup_plot_style

__all__ = [
    "detect_schema",
    "describe_column",
    "describe_columns",
    "value_counts",
    "missing_report",
    "plot_histogram",
    "convert_numeric",
]

logger = get_logger(__name__)

#: 字段类型分类标签
_KIND_NUMERIC = "numeric"
_KIND_DATETIME = "datetime"
_KIND_BOOLEAN = "boolean"
_KIND_CATEGORY = "categorical"
_KIND_EMPTY = "empty"


# --------------------------------------------------------------------------- #
# 内部工具函数
# --------------------------------------------------------------------------- #
def _require_dataframe(df: Any, func_name: str) -> pd.DataFrame:
    """校验入参是否为 DataFrame，并返回副本切片（避免污染调用方数据）。"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{func_name} 的 df 参数必须是 pandas.DataFrame，实际为 {type(df).__name__}")
    if df.empty:
        raise ValueError(f"{func_name} 收到空 DataFrame（无任何行或列）")
    return df


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    """返回全部数值型字段名。"""
    try:
        return df.select_dtypes(include=[np.number]).columns.astype(str).tolist()
    except Exception:  # noqa: BLE001
        return []


def _kind_of(series: pd.Series) -> str:
    """推断单个字段的数据类型分类。"""
    try:
        if series.isna().all():
            return _KIND_EMPTY
        if pd.api.types.is_bool_dtype(series):
            return _KIND_BOOLEAN
        if pd.api.types.is_numeric_dtype(series):
            return _KIND_NUMERIC
        if pd.api.types.is_datetime64_any_dtype(series):
            return _KIND_DATETIME
        # 尝试按时间格式解析（工业数据中时间列常被读成字符串）
        sample = series.dropna().astype(str).head(50)
        if not sample.empty:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            if parsed.notna().mean() >= 0.9:
                return _KIND_DATETIME
        return _KIND_CATEGORY
    except Exception:  # noqa: BLE001
        return _KIND_CATEGORY


def _profile_one(series: pd.Series, sample_size: int, max_sample_values: int) -> dict[str, Any]:
    """生成单个字段的画像信息（类型、缺失、去重数、样例值等）。"""
    profile: dict[str, Any] = {
        "name": str(series.name),
        "dtype": str(series.dtype),
        "kind": _kind_of(series),
        "n_total": int(series.shape[0]),
        "n_missing": int(series.isna().sum()),
        "n_unique": int(series.nunique(dropna=True)),
        "samples": [],
    }
    profile["missing_rate"] = round(profile["n_missing"] / profile["n_total"], 6) if profile["n_total"] else 0.0
    try:
        profile["samples"] = [str(item) for item in series.dropna().head(max(0, sample_size)).tolist()]
    except Exception:  # noqa: BLE001
        profile["samples"] = []
    profile["is_low_cardinality"] = bool(profile["n_unique"] <= max(1, max_sample_values))
    return profile


# --------------------------------------------------------------------------- #
# 对外 API
# --------------------------------------------------------------------------- #
def detect_schema(df: Any, sample_size: int = 5, max_sample_values: int = 8) -> dict[str, Any]:
    """自动识别数据文件的字段与数据类型。

    Args:
        df: 数据表。
        sample_size: 每个字段返回的样例值个数。
        max_sample_values: 判定「低基数字段」（近似分类字段）的去重阈值。

    Returns:
        dict: 结果包，``data`` 结构为::

            {
                "n_rows": int, "n_columns": int,
                "numeric_columns": [...], "datetime_columns": [...],
                "categorical_columns": [...],
                "columns": [ {name, dtype, kind, n_missing, missing_rate, n_unique, samples}, ... ]
            }
    """
    try:
        frame = _require_dataframe(df, "detect_schema")
        columns = [_profile_one(frame[col], sample_size, max_sample_values) for col in frame.columns]
        data = {
            "n_rows": int(frame.shape[0]),
            "n_columns": int(frame.shape[1]),
            "numeric_columns": [item["name"] for item in columns if item["kind"] == _KIND_NUMERIC],
            "datetime_columns": [item["name"] for item in columns if item["kind"] == _KIND_DATETIME],
            "categorical_columns": [
                item["name"] for item in columns if item["kind"] in (_KIND_CATEGORY, _KIND_BOOLEAN)
            ],
            "columns": columns,
        }
        message = f"识别到 {data['n_rows']} 行 × {data['n_columns']} 列，字段画像生成完成"
        return ok(data, message)
    except Exception as exc:  # noqa: BLE001 - 统一兜底，禁止异常外泄
        logger.exception("detect_schema 执行失败")
        return fail("字段识别失败", exc)


def describe_column(df: Any, column: str) -> dict[str, Any]:
    """计算单列的描述性统计指标。

    Args:
        df: 数据表。
        column: 目标列名。

    Returns:
        dict: 结果包，``data`` 包含 count/missing/mean/std/min/q1/median/q3/max/
        iqr/range/cv/skew/kurtosis/sum/unique 等指标；非数值列仅返回计数类指标。
    """
    try:
        frame = _require_dataframe(df, "describe_column")
        column = resolve_column(frame, column)

        series = frame[column]
        data: dict[str, Any] = {
            "column": str(column),
            "kind": _kind_of(series),
            "count": int(series.notna().sum()),
            "missing": int(series.isna().sum()),
            "missing_rate": round(float(series.isna().mean()), 6),
            "unique": int(series.nunique(dropna=True)),
        }
        if _kind_of(series) != _KIND_NUMERIC:
            data["message"] = "非数值字段，仅返回计数类指标"
            return ok(data, f"字段 {column} 为 {data['kind']} 类型，仅统计计数指标")

        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            raise ValueError(f"字段 {column} 无法转换为数值，无有效样本")

        q1, median, q3 = (safe_float(v) for v in numeric.quantile([0.25, 0.5, 0.75]).tolist())
        std = safe_float(numeric.std(ddof=1), 0.0)
        mean = safe_float(numeric.mean())
        data.update(
            {
                "mean": mean,
                "std": std,
                "min": safe_float(numeric.min()),
                "max": safe_float(numeric.max()),
                "median": median,
                "q1": q1,
                "q3": q3,
                "iqr": safe_float(q3 - q1),
                "range": safe_float(numeric.max() - numeric.min()),
                "sum": safe_float(numeric.sum()),
                "cv": safe_float(std / mean, float("nan")) if abs(mean) > 1e-12 else None,
                "skew": safe_float(numeric.skew(), 0.0),
                "kurtosis": safe_float(numeric.kurtosis(), 0.0),
            }
        )
        message = (
            f"字段 {column}：均值={data['mean']:.4f}，标准差={data['std']:.4f}，"
            f"中位数={data['median']:.4f}，范围=[{data['min']:.4f}, {data['max']:.4f}]"
        )
        return ok(data, message)
    except Exception as exc:  # noqa: BLE001
        logger.exception("describe_column 执行失败")
        return fail(f"字段 {column} 描述性统计失败", exc)


def describe_columns(df: Any, columns: Sequence[str] | None = None) -> dict[str, Any]:
    """批量计算多列描述性统计。

    Args:
        df: 数据表。
        columns: 目标列列表；传 ``None`` 表示全部数值型字段。

    Returns:
        dict: 结果包，``data`` 结构为 ``{"columns": [...], "statistics": {列名: 指标字典}}``。
    """
    try:
        frame = _require_dataframe(df, "describe_columns")
        target_columns = list(columns) if columns else _numeric_columns(frame)
        if not target_columns:
            raise ValueError("未找到可统计的数值型字段，请显式指定 columns")

        statistics: dict[str, Any] = {}
        failed: dict[str, str] = {}
        for column in target_columns:
            single = describe_column(frame, column)
            if single["status"] == "success":
                statistics[str(column)] = single["data"]
            else:
                failed[str(column)] = single["error"] or single["message"]

        if not statistics:
            raise ValueError(f"全部字段统计失败：{failed}")
        data = {"columns": list(statistics.keys()), "statistics": statistics, "failed": failed}
        message = f"已完成 {len(statistics)} 个字段的描述性统计" + (f"，{len(failed)} 个字段失败" if failed else "")
        return ok(data, message)
    except Exception as exc:  # noqa: BLE001
        logger.exception("describe_columns 执行失败")
        return fail("批量描述性统计失败", exc)


def value_counts(df: Any, column: str, top_n: int = 10) -> dict[str, Any]:
    """统计指定字段的取值频次（Top-N）。"""
    try:
        frame = _require_dataframe(df, "value_counts")
        column = resolve_column(frame, column)
        series = frame[column]
        counts = series.astype("string").fillna("<缺失>").value_counts()
        top_n = max(1, int(top_n))
        data = {
            "column": str(column),
            "n_unique": int(series.nunique(dropna=True)),
            "total": int(series.shape[0]),
            "items": [
                {"value": str(index), "count": int(value), "ratio": round(float(value) / max(1, len(series)), 6)}
                for index, value in counts.head(top_n).items()
            ],
        }
        return ok(data, f"字段 {column} 共 {data['n_unique']} 个取值，已返回 Top-{top_n}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("value_counts 执行失败")
        return fail(f"字段 {column} 频次统计失败", exc)


def missing_report(df: Any, chart: bool = True) -> dict[str, Any]:
    """缺失值统计报告（含缺失率柱状图）。"""
    charts: list[str] = []
    try:
        frame = _require_dataframe(df, "missing_report")
        total = int(frame.shape[0])
        items = [
            {
                "column": str(col),
                "missing": int(frame[col].isna().sum()),
                "missing_rate": round(float(frame[col].isna().mean()), 6),
                "dtype": str(frame[col].dtype),
            }
            for col in frame.columns
        ]
        items.sort(key=lambda item: item["missing_rate"], reverse=True)
        missing_items = [item for item in items if item["missing"] > 0]

        if chart and missing_items:
            setup_plot_style()
            import matplotlib.pyplot as plt

            names = [item["column"] for item in missing_items][:30]
            rates = [item["missing_rate"] * 100 for item in missing_items][:30]
            fig, ax = plt.subplots(figsize=(max(6.0, len(names) * 0.45), 4.0))
            ax.bar(names, rates, color="#c0504d")
            ax.set_ylabel("缺失率 (%)")
            ax.set_title("各字段缺失率分布")
            ax.tick_params(axis="x", rotation=45)
            path = new_chart_path("missing_report")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            charts.append(str(path))

        data = {
            "n_rows": total,
            "n_columns": int(frame.shape[1]),
            "n_columns_with_missing": len(missing_items),
            "columns": items,
        }
        message = (
            f"共 {len(missing_items)} 个字段存在缺失值（总行数 {total}）"
            if missing_items
            else "数据完整，无缺失值"
        )
        return ok(data, message, charts=charts)
    except Exception as exc:  # noqa: BLE001
        logger.exception("missing_report 执行失败")
        return fail("缺失值统计失败", exc, charts=charts)


def plot_histogram(df: Any, column: str, bins: int = 30, kde: bool = True) -> dict[str, Any]:
    """绘制指定数值字段的直方图（含核密度曲线与均值/中位数参考线）。"""
    charts: list[str] = []
    try:
        frame = _require_dataframe(df, "plot_histogram")
        column = resolve_column(frame, column)
        numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
        if numeric.empty:
            raise ValueError(f"字段 {column} 无有效数值样本")

        setup_plot_style()
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        try:
            sns.histplot(numeric, bins=max(2, int(bins)), kde=bool(kde), ax=ax, color="#4f81bd")
        except Exception:  # noqa: BLE001 - seaborn 异常时退化为纯 matplotlib
            ax.hist(numeric, bins=max(2, int(bins)), color="#4f81bd", alpha=0.75)
        ax.axvline(float(numeric.mean()), color="#c0504d", linestyle="--", label=f"均值 {numeric.mean():.3f}")
        ax.axvline(float(numeric.median()), color="#9bbb59", linestyle="-.", label=f"中位数 {numeric.median():.3f}")
        ax.set_title(f"字段 [{column}] 分布直方图")
        ax.set_xlabel(str(column))
        ax.set_ylabel("频次")
        ax.legend()
        path = new_chart_path(f"hist_{column}")
        fig.savefig(path, dpi=150)
        plt.close(fig)

        charts.append(str(path))
        data = {
            "column": str(column),
            "bins": int(bins),
            "n_samples": int(numeric.shape[0]),
            "mean": safe_float(numeric.mean()),
            "median": safe_float(numeric.median()),
        }
        return ok(data, f"已生成字段 {column} 的分布直方图", charts=charts)
    except Exception as exc:  # noqa: BLE001
        logger.exception("plot_histogram 执行失败")
        return fail(f"字段 {column} 直方图绘制失败", exc, charts=charts)


def convert_numeric(df: Any, columns: Sequence[str] | None = None) -> dict[str, Any]:
    """将指定列（或全部列）安全转换为数值型，并返回实际转换结果。

    该函数返回转换后的数据（``data["dataframe"]``），用于上游在分析前做数据清洗。

    Args:
        df: 数据表。
        columns: 待转换列；``None`` 表示仅尝试转换「看起来像数字的 object 列」。
    """
    try:
        frame = _require_dataframe(df, "convert_numeric").copy()
        targets = list(columns) if columns else [col for col in frame.columns if frame[col].dtype == object]
        converted: list[str] = []
        failed: dict[str, str] = {}
        for column in targets:
            try:
                column = resolve_column(frame, column)
            except KeyError as exc:
                failed[str(column)] = str(exc)
                continue
            try:
                before = frame[column].notna().sum()
                numeric = pd.to_numeric(frame[column], errors="coerce")
                valid_ratio = float(numeric.notna().sum()) / max(1, int(before))
                if valid_ratio < 0.5:
                    failed[str(column)] = f"数值化有效率仅 {valid_ratio:.1%}，已跳过"
                    continue
                frame[column] = numeric
                converted.append(str(column))
            except Exception as exc:  # noqa: BLE001
                failed[str(column)] = str(exc)
        return ok(
            {"converted": converted, "failed": failed, "dataframe": frame},
            f"完成 {len(converted)} 个字段的数值化转换",
            json_safe=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("convert_numeric 执行失败")
        return fail("字段数值化转换失败", exc)
