"""时序时滞（Lead-Lag）分析模块。

工业过程中大量变量之间存在「先动后动」的传导关系（例如：入口温度变化若干分钟后，
出口质量指标才响应）。本模块通过互相关函数（CCF）量化这种时滞关系。

符号约定（非常重要）：

* ``lag > 0``：表示 **x 领先 y** ``lag`` 个采样周期；
* ``lag < 0``：表示 **y 领先 x** ``|lag|`` 个采样周期；
* ``lag = 0``：同期相关最强，不存在明显时滞。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import fail, get_logger, new_chart_path, ok, resolve_column, safe_float, setup_plot_style

__all__ = [
    "cross_correlation",
    "time_lag_analysis",
    "SUPPORTED_LAG_METHODS",
]

logger = get_logger(__name__)

SUPPORTED_LAG_METHODS: tuple[str, ...] = ("pearson", "spearman", "kendall")


# --------------------------------------------------------------------------- #
# 内部工具函数
# --------------------------------------------------------------------------- #
def _prepare_series(
    df: Any,
    x_column: str,
    y_column: str,
    time_column: str | None,
    function_name: str,
) -> tuple[pd.Series, pd.Series, Any, Any | None]:
    """取出 x/y 两列数值序列；若提供时间列则先按时间升序排序（保证时滞方向正确）。

    Returns:
        (x_series, y_series, x_label, time_label)

    Raises:
        TypeError / KeyError / ValueError: 参数非法，由调用方转换为失败结果包。
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{function_name} 的 df 参数必须是 pandas.DataFrame，实际为 {type(df).__name__}")
    if df.empty:
        raise ValueError(f"{function_name} 收到空 DataFrame")

    x_label = resolve_column(df, x_column)
    y_label = resolve_column(df, y_column)
    if x_label == y_label:
        raise ValueError("x 列与 y 列不能为同一字段，时滞分析需要两个不同变量")

    frame = df
    time_label: Any | None = None
    if time_column:
        time_label = resolve_column(df, time_column)
        try:
            frame = df.sort_values(by=time_label, kind="mergesort")
        except Exception as exc:  # noqa: BLE001 - 排序失败时保持原始顺序
            logger.warning("按时间列 %s 排序失败，将使用原始行顺序：%s", time_column, exc)
            time_label = None

    pair = frame[[x_label, y_label]].apply(pd.to_numeric, errors="coerce")
    if pair.shape[0] < 3:
        raise ValueError("有效样本不足 3 条，无法进行时滞分析")
    return pair[x_label], pair[y_label], x_label, time_label


def _validate_method(method: str) -> str:
    """校验相关性方法名。"""
    normalized = str(method or "pearson").strip().lower()
    if normalized not in SUPPORTED_LAG_METHODS:
        raise ValueError(f"不支持的相关性方法：{method}；可选值：{list(SUPPORTED_LAG_METHODS)}")
    return normalized


def _lag_curve(x: pd.Series, y: pd.Series, max_lag: int, method: str) -> list[dict[str, Any]]:
    """计算 lag ∈ [-max_lag, max_lag] 的互相关系数序列。

    计算方式：对 ``lag = k`` 计算 ``corr(x.shift(k), y)``，
    即比较 ``t-k`` 时刻的 x 与 ``t`` 时刻的 y。
    """
    curve: list[dict[str, Any]] = []
    for lag in range(-max_lag, max_lag + 1):
        shifted = x if lag == 0 else x.shift(lag)
        mask = shifted.notna() & y.notna()
        n_samples = int(mask.sum())
        correlation: float | None = None
        if n_samples >= 3:
            left = shifted[mask]
            right = y[mask]
            if left.std(ddof=1) > 0 and right.std(ddof=1) > 0:
                correlation = safe_float(left.corr(right, method=method), 0.0)
        curve.append(
            {
                "lag": int(lag),
                "correlation": None if correlation is None else round(correlation, 6),
                "n_samples": n_samples,
            }
        )
    return curve


# --------------------------------------------------------------------------- #
# 对外 API
# --------------------------------------------------------------------------- #
def cross_correlation(
    df: Any,
    x_column: str,
    y_column: str,
    max_lag: int = 10,
    method: str = "pearson",
    time_column: str | None = None,
) -> dict[str, Any]:
    """计算 x、y 之间的互相关曲线，并给出最优时滞（不含绘图）。

    Args:
        df: 数据表。
        x_column: 自变量（通常是过程变量，如入口温度）。
        y_column: 因变量（通常是质量/产量指标）。
        max_lag: 最大滞后步数（以采样点为单位的正整数）。
        method: 相关性方法，支持 pearson / spearman / kendall。
        time_column: 时间列名；提供后先按时间排序再分析，避免行顺序错乱导致时滞反向。

    Returns:
        dict: 结果包，``data`` 含 ``lags``（互相关曲线）、``best_lag``、``best_correlation``。
    """
    try:
        method = _validate_method(method)
        max_lag = int(max_lag)
        if not 1 <= max_lag <= 5000:
            raise ValueError(f"max_lag 必须位于 [1, 5000] 区间，实际为 {max_lag}")

        x, y, x_label, time_label = _prepare_series(df, x_column, y_column, time_column, "cross_correlation")
        curve = _lag_curve(x, y, max_lag, method)
        valid = [item for item in curve if item["correlation"] is not None]
        if not valid:
            raise ValueError("所有滞后步的相关性均无法计算（有效样本不足或序列方差为 0）")

        best = max(valid, key=lambda item: abs(item["correlation"]))
        data = {
            "x": str(x_label),
            "y": str(y_column),
            "method": method,
            "max_lag": max_lag,
            "time_column": None if time_label is None else str(time_label),
            "best_lag": best["lag"],
            "best_correlation": best["correlation"],
            "lags": curve,
        }
        return ok(data, f"互相关曲线计算完成，最优时滞 lag={best['lag']}（r={best['correlation']:.3f}）")
    except Exception as exc:  # noqa: BLE001
        logger.exception("cross_correlation 执行失败")
        return fail(f"字段 {x_column} 与 {y_column} 的互相关计算失败", exc)


def time_lag_analysis(
    df: Any,
    x_column: str,
    y_column: str,
    max_lag: int = 10,
    method: str = "pearson",
    time_column: str | None = None,
    sample_interval: float | None = None,
    chart: bool = True,
) -> dict[str, Any]:
    """时滞分析（互相关曲线 + 最优时滞 + 工程结论 + 时滞曲线图）。

    Args:
        df: 数据表。
        x_column: 自变量字段。
        y_column: 因变量字段。
        max_lag: 最大滞后步数。
        method: 相关性方法。
        time_column: 时间列名（可选）。
        sample_interval: 采样间隔（单位由调用方统一，如秒/分钟），用于把步数换算为时间。
        chart: 是否绘制时滞曲线图。

    Returns:
        dict: 结果包，``data`` 在 :func:`cross_correlation` 基础上增加
        ``best_lag_seconds``、``conclusion``、``top_lags``。
    """
    charts: list[str] = []
    try:
        core = cross_correlation(df, x_column, y_column, max_lag=max_lag, method=method, time_column=time_column)
        if core["status"] != "success":
            return fail(core["message"], core["error"], charts=charts)

        data = dict(core["data"])
        best_lag = int(data["best_lag"])
        best_corr = safe_float(data["best_correlation"], 0.0)
        interval = safe_float(sample_interval, float("nan")) if sample_interval else float("nan")
        data["sample_interval"] = None if np.isnan(interval) else interval
        data["best_lag_seconds"] = None if np.isnan(interval) else round(abs(best_lag) * interval, 6)

        magnitude = abs(best_corr)
        strength = "极强" if magnitude >= 0.8 else "强" if magnitude >= 0.6 else "中等" if magnitude >= 0.4 else "弱"
        time_hint = ""
        if not np.isnan(interval) and best_lag != 0:
            time_hint = f"（约 {abs(best_lag) * interval:.3f} 个时间单位，采样间隔 {interval}）"
        if best_lag > 0:
            conclusion = (
                f"{data['x']} 领先 {data['y']} 约 {best_lag} 个采样周期{time_hint}，"
                f"最大相关系数 {best_corr:.3f}（{strength}{'正' if best_corr >= 0 else '负'}相关）。"
                f"说明 {data['x']} 的变化需经过约 {best_lag} 个采样周期才在 {data['y']} 上体现。"
            )
        elif best_lag < 0:
            conclusion = (
                f"{data['y']} 领先 {data['x']} 约 {abs(best_lag)} 个采样周期{time_hint}，"
                f"最大相关系数 {best_corr:.3f}（{strength}{'正' if best_corr >= 0 else '负'}相关）。"
            )
        else:
            conclusion = (
                f"同期相关最强（lag=0，r={best_corr:.3f}，{strength}{'正' if best_corr >= 0 else '负'}相关），"
                "未检测到明显时滞，建议检查采样频率是否足以捕捉动态响应。"
            )
        data["conclusion"] = conclusion
        data["top_lags"] = sorted(
            [item for item in data["lags"] if item["correlation"] is not None],
            key=lambda item: abs(item["correlation"]),
            reverse=True,
        )[:5]

        if chart:
            setup_plot_style()
            import matplotlib.pyplot as plt

            lags = [item["lag"] for item in data["lags"]]
            values = [0.0 if item["correlation"] is None else item["correlation"] for item in data["lags"]]
            fig, ax = plt.subplots(figsize=(8.0, 4.2))
            ax.plot(lags, values, marker="o", markersize=4, color="#4f81bd", linewidth=1.4)
            ax.axhline(0, color="#7f7f7f", linewidth=0.8, linestyle="--")
            ax.axvline(0, color="#7f7f7f", linewidth=0.8, linestyle=":")
            ax.scatter(
                [best_lag],
                [best_corr],
                color="#c0504d",
                zorder=5,
                s=60,
                label=f"最优时滞 lag={best_lag}，r={best_corr:.3f}",
            )
            ax.set_xlabel("滞后步数 lag（>0 表示 x 领先 y）")
            ax.set_ylabel(f"相关系数（{data['method']}）")
            ax.set_title(f"{data['x']} 与 {data['y']} 的时滞互相关曲线")
            ax.legend(fontsize=9)
            path = new_chart_path(f"timelag_{data['x']}_{data['y']}")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            charts.append(str(path))

        return ok(data, conclusion, charts=charts)
    except Exception as exc:  # noqa: BLE001
        logger.exception("time_lag_analysis 执行失败")
        return fail(f"字段 {x_column} 与 {y_column} 的时滞分析失败", exc, charts=charts)
