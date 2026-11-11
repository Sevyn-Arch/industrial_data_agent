"""相关性分析模块。

功能：皮尔逊/斯皮尔曼/肯德尔相关性矩阵、强相关特征对挖掘、与目标变量的相关性排序，
以及相关性热力图、散点拟合图绘制。
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import fail, get_logger, new_chart_path, ok, resolve_column, safe_float, setup_plot_style

__all__ = [
    "correlation_matrix",
    "strong_correlations",
    "correlation_with_target",
    "plot_scatter",
]

logger = get_logger(__name__)

#: 支持的相关性计算方法
SUPPORTED_METHODS: tuple[str, ...] = ("pearson", "spearman", "kendall")


# --------------------------------------------------------------------------- #
# 内部工具函数
# --------------------------------------------------------------------------- #
def _prepare_numeric(
    df: Any,
    columns: Sequence[str] | None,
    function_name: str,
    min_columns: int = 2,
) -> pd.DataFrame:
    """筛选并对齐数值型字段，返回仅含数值列的新 DataFrame。

    Raises:
        TypeError: df 不是 DataFrame。
        KeyError: 指定字段不存在。
        ValueError: 可用数值字段数量不足。
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{function_name} 的 df 参数必须是 pandas.DataFrame，实际为 {type(df).__name__}")
    if df.empty:
        raise ValueError(f"{function_name} 收到空 DataFrame")

    if columns:
        resolved: list[Any] = []
        missing: list[str] = []
        for col in columns:
            try:
                resolved.append(resolve_column(df, col))
            except KeyError:
                missing.append(str(col))
        if missing:
            raise KeyError(f"字段不存在：{missing}；可用字段：{list(map(str, df.columns))}")
        subset = df[resolved]
    else:
        subset = df.select_dtypes(include=[np.number])

    numeric = subset.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all")
    if numeric.shape[1] < min_columns:
        raise ValueError(
            f"可用数值字段数量为 {numeric.shape[1]}，至少需要 {min_columns} 个；"
            "请检查字段类型或显式指定数值列"
        )
    return numeric


def _validate_method(method: str) -> str:
    """校验相关性方法名并归一化为小写。"""
    normalized = str(method or "pearson").strip().lower()
    if normalized not in SUPPORTED_METHODS:
        raise ValueError(f"不支持的相关性方法：{method}；可选值：{list(SUPPORTED_METHODS)}")
    return normalized


def _strength_label(abs_value: float) -> str:
    """将相关系数绝对值映射为可读的强弱等级。"""
    if abs_value >= 0.8:
        return "极强"
    if abs_value >= 0.6:
        return "强"
    if abs_value >= 0.4:
        return "中等"
    if abs_value >= 0.2:
        return "弱"
    return "极弱"


# --------------------------------------------------------------------------- #
# 对外 API
# --------------------------------------------------------------------------- #
def correlation_matrix(
    df: Any,
    columns: Sequence[str] | None = None,
    method: str = "pearson",
    chart: bool = True,
    annotate: bool = True,
) -> dict[str, Any]:
    """计算相关性矩阵，并（可选）绘制热力图。

    Args:
        df: 数据表。
        columns: 参与计算的字段；``None`` 表示自动使用全部数值型字段。
        method: 相关性方法，支持 pearson / spearman / kendall。
        chart: 是否生成热力图。
        annotate: 热力图上是否标注数值（字段较多时自动关闭以避免拥挤）。

    Returns:
        dict: 结果包，``data`` 含 ``matrix``（嵌套字典，便于前端渲染）、
        ``columns``、``method``、``strongest_pair``。
    """
    charts: list[str] = []
    try:
        method = _validate_method(method)
        numeric = _prepare_numeric(df, columns, "correlation_matrix")
        matrix = numeric.corr(method=method)
        matrix = matrix.fillna(0.0)

        # 找出绝对值最大的特征对（排除对角线）
        pairs: list[tuple[str, str, float]] = []
        labels: list[Any] = list(matrix.columns)
        names = [str(col) for col in labels]
        for i, left in enumerate(labels):
            for right in labels[i + 1:]:
                pairs.append((str(left), str(right), safe_float(matrix.loc[left, right], 0.0)))
        strongest = max(pairs, key=lambda item: abs(item[2]), default=None)

        if chart and len(names) >= 2:
            setup_plot_style()
            import matplotlib.pyplot as plt

            import seaborn as sns

            size = max(5.0, min(14.0, len(names) * 0.75))
            fig, ax = plt.subplots(figsize=(size, size * 0.85))
            sns.heatmap(
                matrix,
                annot=bool(annotate) and len(names) <= 15,
                fmt=".2f",
                cmap="RdBu_r",
                vmin=-1,
                vmax=1,
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "相关系数"},
                ax=ax,
            )
            ax.set_title(f"相关性热力图（{method}，{len(names)} 个字段）")
            ax.tick_params(axis="x", rotation=45)
            path = new_chart_path("correlation_heatmap")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            charts.append(str(path))

        data = {
            "method": method,
            "columns": names,
            "n_samples": int(numeric.dropna().shape[0]),
            "matrix": {
                str(row): {str(col): safe_float(matrix.loc[row, col], 0.0) for col in labels} for row in labels
            },
            "strongest_pair": (
                {
                    "feature_a": strongest[0],
                    "feature_b": strongest[1],
                    "correlation": round(strongest[2], 6),
                    "strength": _strength_label(abs(strongest[2])),
                    "direction": "正相关" if strongest[2] >= 0 else "负相关",
                }
                if strongest
                else None
            ),
        }
        message = f"已计算 {len(names)} 个字段的 {method} 相关性矩阵"
        if data["strongest_pair"]:
            pair = data["strongest_pair"]
            message += f"，最强相关对：{pair['feature_a']} 与 {pair['feature_b']}（r={pair['correlation']:.3f}）"
        return ok(data, message, charts=charts)
    except Exception as exc:  # noqa: BLE001
        logger.exception("correlation_matrix 执行失败")
        return fail("相关性矩阵计算失败", exc, charts=charts)


def strong_correlations(
    df: Any,
    columns: Sequence[str] | None = None,
    method: str = "pearson",
    threshold: float = 0.5,
    top_n: int = 20,
) -> dict[str, Any]:
    """筛选强相关特征对（|r| ≥ threshold），按相关性强度排序。

    Args:
        df: 数据表。
        columns: 参与计算的字段。
        method: 相关性方法。
        threshold: 相关系数绝对值阈值，取值范围 [0, 1]。
        top_n: 最多返回的特征对数量。
    """
    try:
        method = _validate_method(method)
        threshold = abs(safe_float(threshold, 0.5))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold 必须位于 [0, 1] 区间，实际为 {threshold}")

        numeric = _prepare_numeric(df, columns, "strong_correlations")
        matrix = numeric.corr(method=method).fillna(0.0)
        labels: list[Any] = list(matrix.columns)
        names = [str(col) for col in labels]

        pairs: list[dict[str, Any]] = []
        for i, left in enumerate(labels):
            for right in labels[i + 1:]:
                value = safe_float(matrix.loc[left, right], 0.0)
                if abs(value) >= threshold:
                    pairs.append(
                        {
                            "feature_a": str(left),
                            "feature_b": str(right),
                            "correlation": round(value, 6),
                            "abs_correlation": round(abs(value), 6),
                            "strength": _strength_label(abs(value)),
                            "direction": "正相关" if value >= 0 else "负相关",
                        }
                    )
        pairs.sort(key=lambda item: item["abs_correlation"], reverse=True)
        pairs = pairs[: max(1, int(top_n))]

        data = {
            "method": method,
            "threshold": threshold,
            "columns": names,
            "n_pairs": len(pairs),
            "pairs": pairs,
        }
        message = (
            f"共发现 {len(pairs)} 组 |r| ≥ {threshold} 的特征对（{method}）"
            if pairs
            else f"未发现 |r| ≥ {threshold} 的强相关特征对（{method}）"
        )
        return ok(data, message)
    except Exception as exc:  # noqa: BLE001
        logger.exception("strong_correlations 执行失败")
        return fail("强相关特征挖掘失败", exc)


def correlation_with_target(
    df: Any,
    target_column: str,
    feature_columns: Sequence[str] | None = None,
    method: str = "pearson",
) -> dict[str, Any]:
    """计算各特征与目标变量的相关性，并按绝对值降序排列。

    Args:
        df: 数据表。
        target_column: 目标变量（如产量、质量指标）。
        feature_columns: 候选特征；``None`` 表示使用除目标列外的全部数值字段。
        method: 相关性方法。
    """
    try:
        method = _validate_method(method)
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df 参数必须是 pandas.DataFrame")
        target_label = resolve_column(df, target_column)
        if feature_columns:
            resolved_features: list[Any] = []
            missing: list[str] = []
            for col in feature_columns:
                try:
                    resolved_features.append(resolve_column(df, col))
                except KeyError:
                    missing.append(str(col))
            if missing:
                raise KeyError(f"特征字段不存在：{missing}")
            targets = [col for col in resolved_features if col != target_label]
        else:
            targets = [col for col in df.select_dtypes(include=[np.number]).columns if col != target_label]

        numeric = _prepare_numeric(df, [target_label] + targets, "correlation_with_target")
        items: list[dict[str, Any]] = []
        for label in [col for col in numeric.columns if col != target_label]:
            pair = numeric[[label, target_label]].dropna()
            if pair.shape[0] < 3 or pair[label].std(ddof=1) == 0 or pair[target_label].std(ddof=1) == 0:
                items.append({"feature": str(label), "correlation": None, "note": "样本不足或方差为 0"})
                continue
            value = safe_float(pair[label].corr(pair[target_label], method=method), 0.0)
            items.append(
                {
                    "feature": str(label),
                    "correlation": round(value, 6),
                    "abs_correlation": round(abs(value), 6),
                    "strength": _strength_label(abs(value)),
                    "direction": "正相关" if value >= 0 else "负相关",
                    "n_samples": int(pair.shape[0]),
                }
            )
        items.sort(key=lambda item: item["abs_correlation"] if item["correlation"] is not None else -1, reverse=True)

        data = {"target": str(target_column), "method": method, "n_features": len(items), "items": items}
        return ok(data, f"已完成 {len(items)} 个特征与目标 {target_column} 的相关性分析（{method}）")
    except Exception as exc:  # noqa: BLE001
        logger.exception("correlation_with_target 执行失败")
        return fail(f"与目标 {target_column} 的相关性分析失败", exc)


def plot_scatter(
    df: Any,
    x_column: str,
    y_column: str,
    hue_column: str | None = None,
    fit_line: bool = True,
) -> dict[str, Any]:
    """绘制两变量的散点图（可选一元线性拟合线与相关系数标注）。"""
    charts: list[str] = []
    try:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df 参数必须是 pandas.DataFrame")
        x_column = resolve_column(df, x_column)
        y_column = resolve_column(df, y_column)
        hue_label = resolve_column(df, hue_column) if hue_column else None

        pair = df[[x_column, y_column]].apply(pd.to_numeric, errors="coerce").dropna()
        if pair.shape[0] < 3:
            raise ValueError("有效样本不足 3 条，无法绘制散点图")

        setup_plot_style()
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        if hue_label is not None:
            groups = df.loc[pair.index, hue_label].astype("string").fillna("<缺失>")
            for name, subset in pair.groupby(groups.values):
                ax.scatter(subset[x_column], subset[y_column], s=22, alpha=0.75, label=str(name))
            ax.legend(title=str(hue_column), fontsize=8)
        else:
            ax.scatter(pair[x_column], pair[y_column], s=22, alpha=0.75, color="#4f81bd")

        r_value = safe_float(pair[x_column].corr(pair[y_column]), float("nan"))
        if fit_line:
            try:
                slope, intercept = np.polyfit(pair[x_column].to_numpy(dtype=float), pair[y_column].to_numpy(dtype=float), 1)
                xs = np.linspace(float(pair[x_column].min()), float(pair[x_column].max()), 100)
                ax.plot(xs, slope * xs + intercept, color="#c0504d", linewidth=1.6,
                        label=f"拟合: y={slope:.4f}x{intercept:+.4f}")
                ax.legend(fontsize=8)
            except Exception:  # noqa: BLE001 - 拟合失败不影响散点展示
                logger.debug("散点拟合线计算失败", exc_info=True)

        ax.set_xlabel(str(x_column))
        ax.set_ylabel(str(y_column))
        ax.set_title(f"{x_column} vs {y_column}（r = {r_value:.3f}）")
        path = new_chart_path(f"scatter_{x_column}_{y_column}")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        charts.append(str(path))

        data = {
            "x": str(x_column),
            "y": str(y_column),
            "n_samples": int(pair.shape[0]),
            "correlation": round(r_value, 6) if not np.isnan(r_value) else None,
        }
        return ok(data, f"已生成 {x_column} 与 {y_column} 的散点图（r={r_value:.3f}）", charts=charts)
    except Exception as exc:  # noqa: BLE001
        logger.exception("plot_scatter 执行失败")
        return fail(f"散点图绘制失败（{x_column} vs {y_column}）", exc, charts=charts)
