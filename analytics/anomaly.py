"""异常检测模块（基于 3σ 原则，并提供 MAD 稳健阈值作为对照）。

工业数据常存在离群点：仪表跳变、传感器故障、工况切换等。3σ 原则假设数据近似正态分布，
当分布存在重尾或偏态时误报率较高，因此本模块同时给出基于「中位数 ± k×1.4826×MAD」的
稳健阈值（MAD × 1.4826 是正态分布下 σ 的稳健估计），便于工程人员对照判断。
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import fail, get_logger, new_chart_path, ok, resolve_column, safe_float, setup_plot_style

__all__ = [
    "detect_3sigma",
    "detect_3sigma_multi",
    "add_anomaly_flag",
]

logger = get_logger(__name__)

_MAD_SCALE = 1.4826  # 正态分布下 MAD -> σ 的一致化系数


# --------------------------------------------------------------------------- #
# 内部工具函数
# --------------------------------------------------------------------------- #
def _resolve_sides(side: str) -> str:
    """校验异常判定方向：both（双侧）/ upper（仅高值）/ lower（仅低值）。"""
    normalized = str(side or "both").strip().lower()
    if normalized not in ("both", "upper", "lower"):
        raise ValueError(f"side 参数仅支持 both/upper/lower，实际为 {side}")
    return normalized


def _apply_rule(values: pd.Series, lower: float, upper: float, side: str) -> np.ndarray:
    """按阈值与判定方向生成布尔异常掩码。"""
    if side == "upper":
        return (values > upper).to_numpy()
    if side == "lower":
        return (values < lower).to_numpy()
    return ((values > upper) | (values < lower)).to_numpy()


def _summarize_anomalies(
    values: pd.Series,
    mask: np.ndarray,
    mean: float,
    std: float,
    lower: float,
    upper: float,
    threshold: float,
    robust_lower: float,
    robust_upper: float,
    top_n: int = 10,
) -> dict[str, Any]:
    """汇总异常检测结果（通用逻辑，供单列/多列复用）。"""
    n_total = int(values.shape[0])
    n_anomalies = int(mask.sum())
    deviation = (values - mean).abs()
    anomaly_positions = np.flatnonzero(mask)
    if n_anomalies:
        order = np.argsort(-deviation.to_numpy(dtype=float)[mask])
        top_positions = anomaly_positions[order][:top_n]
    else:
        top_positions = []
    return {
        "n_total": n_total,
        "n_anomalies": n_anomalies,
        "anomaly_rate": round(n_anomalies / n_total, 6) if n_total else 0.0,
        "mean": round(float(mean), 6),
        "std": round(float(std), 6),
        "threshold": float(threshold),
        "lower_bound": round(float(lower), 6),
        "upper_bound": round(float(upper), 6),
        "n_high": int((values > upper).sum()),
        "n_low": int((values < lower).sum()),
        "robust_lower_bound": round(float(robust_lower), 6),
        "robust_upper_bound": round(float(robust_upper), 6),
        "anomaly_indices": [int(pos) for pos in anomaly_positions.tolist()],
        "anomaly_values": [round(safe_float(value), 6) for value in values.to_numpy(dtype=float)[mask].tolist()],
        "top_anomalies": [
            {
                "row_index": int(pos),
                "value": round(safe_float(values.iloc[pos]), 6),
                "deviation_sigma": round(safe_float(deviation.iloc[pos]) / std, 4) if std > 1e-12 else None,
            }
            for pos in top_positions
        ],
    }


def _plot_series_with_anomalies(
    values: pd.Series,
    mask: np.ndarray,
    lower: float,
    upper: float,
    mean: float,
    title: str,
    prefix: str,
) -> str:
    """绘制「时序曲线 + 阈值线 + 异常点」图，返回图片绝对路径。"""
    setup_plot_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.plot(np.arange(values.shape[0]), values.to_numpy(dtype=float), color="#4f81bd", linewidth=1.1, label="观测值")
    ax.axhline(mean, color="#7f7f7f", linestyle="--", linewidth=0.9, label=f"均值 {mean:.3f}")
    ax.axhline(upper, color="#c0504d", linestyle="--", linewidth=0.9, label=f"上界 {upper:.3f}")
    ax.axhline(lower, color="#c0504d", linestyle="--", linewidth=0.9, label=f"下界 {lower:.3f}")
    if mask.any():
        positions = np.flatnonzero(mask)
        ax.scatter(
            positions,
            values.to_numpy(dtype=float)[mask],
            color="#e36c0a",
            s=28,
            zorder=5,
            label=f"异常点（{int(mask.sum())} 个）",
        )
    ax.set_xlabel("样本序号")
    ax.set_ylabel("数值")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    path = new_chart_path(prefix)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


# --------------------------------------------------------------------------- #
# 对外 API
# --------------------------------------------------------------------------- #
def detect_3sigma(
    df: Any,
    column: str,
    threshold: float = 3.0,
    side: str = "both",
    chart: bool = True,
) -> dict[str, Any]:
    """基于 3σ 原则对单列进行异常检测。

    Args:
        df: 数据表。
        column: 目标数值字段。
        threshold: σ 倍数阈值，常用 3（工业上亦常用 2 或 3.5），必须为正数。
        side: 判定方向，both / upper / lower。
        chart: 是否绘制异常分布图。

    Returns:
        dict: 结果包，``data`` 含均值、标准差、上下界、异常点数量/比例、异常点索引与数值，
        以及基于 MAD 的稳健阈值作为对照。
    """
    charts: list[str] = []
    try:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df 参数必须是 pandas.DataFrame，实际为 {type(df).__name__}")
        if df.empty:
            raise ValueError("收到空 DataFrame")
        label = resolve_column(df, column)
        side = _resolve_sides(side)
        threshold = safe_float(threshold, 3.0)
        if threshold <= 0:
            raise ValueError(f"threshold 必须为正数，实际为 {threshold}")

        numeric = pd.to_numeric(df[label], errors="coerce")
        values = numeric[numeric.notna()].reset_index(drop=True)
        if values.shape[0] < 3:
            raise ValueError(f"字段 {column} 有效样本不足 3 条，无法进行 3σ 检测")

        mean = float(values.mean())
        std = float(values.std(ddof=1))
        if std <= 1e-12:
            raise ValueError(f"字段 {column} 标准差近似为 0（数据无波动），3σ 检测不适用")
        median = float(values.median())
        mad = float((values - median).abs().median())

        lower = mean - threshold * std
        upper = mean + threshold * std
        robust_lower = median - threshold * _MAD_SCALE * mad
        robust_upper = median + threshold * _MAD_SCALE * mad

        mask = _apply_rule(values, lower, upper, side)
        data = _summarize_anomalies(values, mask, mean, std, lower, upper, threshold, robust_lower, robust_upper)
        data["column"] = str(label)
        data["side"] = side
        data["note"] = "anomaly_indices 为「剔除缺失值后」的样本序号；robust_* 为 MAD 稳健阈值对照"

        if chart:
            charts.append(
                _plot_series_with_anomalies(
                    values,
                    mask,
                    lower,
                    upper,
                    mean,
                    f"字段 [{label}] 3σ 异常检测（阈值 {threshold}σ，{side}）",
                    f"anomaly_{label}",
                )
            )

        message = (
            f"字段 {label} 共检测到 {data['n_anomalies']} 个异常点"
            f"（占比 {data['anomaly_rate']:.2%}），有效样本 {data['n_total']} 个；"
            f"3σ 阈值区间 = [{lower:.4f}, {upper:.4f}]"
        )
        return ok(data, message, charts=charts)
    except Exception as exc:  # noqa: BLE001
        logger.exception("detect_3sigma 执行失败")
        return fail(f"字段 {column} 的 3σ 异常检测失败", exc, charts=charts)


def detect_3sigma_multi(
    df: Any,
    columns: Sequence[str] | None = None,
    threshold: float = 3.0,
    side: str = "both",
    chart: bool = True,
    max_charts: int = 4,
) -> dict[str, Any]:
    """对多个数值字段批量执行 3σ 异常检测（最多为前 ``max_charts`` 个字段生成图表）。"""
    charts: list[str] = []
    try:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df 参数必须是 pandas.DataFrame")
        if columns:
            labels = [resolve_column(df, col) for col in columns]
        else:
            labels = list(df.select_dtypes(include=[np.number]).columns)
        if not labels:
            raise ValueError("未找到可检测的数值型字段，请显式指定 columns")

        results: dict[str, Any] = {}
        failed: dict[str, str] = {}
        for index, label in enumerate(labels):
            single = detect_3sigma(df, label, threshold=threshold, side=side, chart=chart and index < max_charts)
            if single["status"] == "success":
                results[str(label)] = single["data"]
                charts.extend(single["charts"])
            else:
                failed[str(label)] = single["error"] or single["message"]

        if not results:
            raise ValueError(f"全部字段异常检测失败：{failed}")
        total_anomalies = int(sum(item["n_anomalies"] for item in results.values()))
        data = {
            "threshold": safe_float(threshold, 3.0),
            "side": str(side),
            "n_columns": len(results),
            "total_anomalies": total_anomalies,
            "columns": results,
            "failed": failed,
        }
        return ok(
            data,
            f"已完成 {len(results)} 个字段的 3σ 异常检测，共发现 {total_anomalies} 个异常点",
            charts=charts,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("detect_3sigma_multi 执行失败")
        return fail("批量 3σ 异常检测失败", exc, charts=charts)


def add_anomaly_flag(
    df: Any,
    column: str,
    threshold: float = 3.0,
    side: str = "both",
    flag_column: str | None = None,
) -> dict[str, Any]:
    """在原数据基础上追加异常标记列，返回增强后的 DataFrame（便于导出清洗结果）。

    Args:
        df: 数据表。
        column: 目标数值字段。
        threshold: σ 倍数阈值。
        side: 判定方向。
        flag_column: 标记列名，默认 ``{column}_is_anomaly``。
    """
    try:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df 参数必须是 pandas.DataFrame")
        label = resolve_column(df, column)
        column_name = str(flag_column) if flag_column else f"{label}_is_anomaly"
        single = detect_3sigma(df, label, threshold=threshold, side=side, chart=False)
        if single["status"] != "success":
            return fail(single["message"], single["error"])

        enhanced = df.copy()
        flags = pd.Series(False, index=enhanced.index, dtype=bool)
        numeric = pd.to_numeric(enhanced[label], errors="coerce")
        positions = numeric.dropna().index[single["data"]["anomaly_indices"]]
        flags.loc[positions] = True
        enhanced[column_name] = flags

        data = {
            "column": str(label),
            "flag_column": column_name,
            "n_anomalies": int(flags.sum()),
            "dataframe": enhanced,
        }
        return ok(
            data,
            f"已新增异常标记列 {column_name}，共标记 {int(flags.sum())} 行为异常",
            json_safe=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("add_anomaly_flag 执行失败")
        return fail(f"字段 {column} 异常标记失败", exc)
