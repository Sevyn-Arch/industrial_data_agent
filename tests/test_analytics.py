"""分析算法库单元测试（基础统计 / 相关性 / 时滞 / 异常 / 机器学习）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analytics import anomaly, basic_stats, correlation, ml_models, time_lag  # noqa: E402

ENVELOPE_KEYS = {"status", "data", "message", "error", "charts"}


def _assert_envelope(result: dict) -> None:
    """所有对外函数必须返回统一结构。"""
    assert isinstance(result, dict)
    assert ENVELOPE_KEYS.issubset(set(result.keys())), f"缺少统一字段：{result.keys()}"
    assert result["status"] in ("success", "error")


# --------------------------------------------------------------------------- #
# 基础统计
# --------------------------------------------------------------------------- #
def test_detect_schema_identifies_types(sample_frame: pd.DataFrame) -> None:
    result = basic_stats.detect_schema(sample_frame)
    _assert_envelope(result)
    assert result["status"] == "success"
    data = result["data"]
    assert data["n_rows"] == sample_frame.shape[0]
    assert "inlet_temp_c" in data["numeric_columns"]
    assert "timestamp" in data["datetime_columns"]
    assert "grade" in data["categorical_columns"]
    kinds = {item["name"]: item["kind"] for item in data["columns"]}
    assert kinds["steam_flow_t_h"] == "numeric"
    assert kinds["grade"] == "categorical"


def test_describe_column_matches_pandas(sample_frame: pd.DataFrame) -> None:
    result = basic_stats.describe_column(sample_frame, "inlet_temp_c")
    _assert_envelope(result)
    series = sample_frame["inlet_temp_c"]
    assert result["status"] == "success"
    assert result["data"]["mean"] == pytest.approx(series.mean(), rel=1e-9)
    assert result["data"]["std"] == pytest.approx(series.std(ddof=1), rel=1e-9)
    assert result["data"]["median"] == pytest.approx(series.median(), rel=1e-9)
    assert result["data"]["min"] == pytest.approx(series.min())
    assert result["data"]["max"] == pytest.approx(series.max())


def test_describe_columns_and_value_counts(sample_frame: pd.DataFrame) -> None:
    stats = basic_stats.describe_columns(sample_frame)
    assert stats["status"] == "success"
    assert "inlet_temp_c" in stats["data"]["statistics"]
    counts = basic_stats.value_counts(sample_frame, "grade", top_n=3)
    _assert_envelope(counts)
    assert counts["status"] == "success"
    assert sum(item["count"] for item in counts["data"]["items"]) > 0


def test_missing_report_detects_missing(sample_frame: pd.DataFrame) -> None:
    result = basic_stats.missing_report(sample_frame)
    assert result["status"] == "success"
    assert result["data"]["n_columns_with_missing"] >= 1
    assert len(result["charts"]) == 1
    assert Path(result["charts"][0]).exists()


def test_histogram_chart_created(sample_frame: pd.DataFrame) -> None:
    result = basic_stats.plot_histogram(sample_frame, "yield_pct")
    assert result["status"] == "success"
    assert Path(result["charts"][0]).exists()


def test_unknown_column_returns_error_envelope(sample_frame: pd.DataFrame) -> None:
    result = basic_stats.describe_column(sample_frame, "不存在的列")
    _assert_envelope(result)
    assert result["status"] == "error"
    assert result["error"]


def test_empty_dataframe_returns_error() -> None:
    result = basic_stats.detect_schema(pd.DataFrame())
    assert result["status"] == "error"
    assert "空" in result["message"] or "空" in (result["error"] or "")


# --------------------------------------------------------------------------- #
# 相关性
# --------------------------------------------------------------------------- #
def test_correlation_matrix_and_strong_pairs(sample_frame: pd.DataFrame) -> None:
    matrix = correlation.correlation_matrix(sample_frame)
    _assert_envelope(matrix)
    assert matrix["status"] == "success"
    assert matrix["data"]["matrix"]["inlet_temp_c"]["outlet_temp_c"] > 0.5
    assert len(matrix["charts"]) == 1

    strong = correlation.strong_correlations(sample_frame, threshold=0.5)
    assert strong["status"] == "success"
    pairs = {(item["feature_a"], item["feature_b"]) for item in strong["data"]["pairs"]}
    assert ("inlet_temp_c", "outlet_temp_c") in pairs or ("outlet_temp_c", "inlet_temp_c") in pairs


def test_correlation_with_target_orders_by_strength(sample_frame: pd.DataFrame) -> None:
    result = correlation.correlation_with_target(sample_frame, "yield_pct")
    assert result["status"] == "success"
    correlations = [item["abs_correlation"] for item in result["data"]["items"] if item["correlation"] is not None]
    assert correlations == sorted(correlations, reverse=True)


def test_invalid_correlation_method_rejected(sample_frame: pd.DataFrame) -> None:
    result = correlation.correlation_matrix(sample_frame, method="not-a-method")
    assert result["status"] == "error"
    assert "不支持" in f"{result['message']}{result['error']}"


# --------------------------------------------------------------------------- #
# 时滞分析
# --------------------------------------------------------------------------- #
def test_time_lag_recovers_known_lag(sample_frame: pd.DataFrame) -> None:
    result = time_lag.time_lag_analysis(
        sample_frame,
        x_column="inlet_temp_c",
        y_column="outlet_temp_c",
        max_lag=8,
        time_column="timestamp",
        sample_interval=300,
    )
    _assert_envelope(result)
    assert result["status"] == "success"
    assert result["data"]["best_lag"] == 3
    assert result["data"]["best_lag_seconds"] == pytest.approx(900.0)
    assert "领先" in result["data"]["conclusion"]
    assert len(result["charts"]) == 1


def test_cross_correlation_requires_distinct_columns(sample_frame: pd.DataFrame) -> None:
    result = time_lag.cross_correlation(sample_frame, "inlet_temp_c", "inlet_temp_c")
    assert result["status"] == "error"
    assert "同一字段" in f"{result['message']}{result['error']}"


def test_cross_correlation_lag_curve_shape(sample_frame: pd.DataFrame) -> None:
    result = time_lag.cross_correlation(sample_frame, "inlet_temp_c", "outlet_temp_c", max_lag=4)
    assert result["status"] == "success"
    assert len(result["data"]["lags"]) == 9  # -4 .. +4


# --------------------------------------------------------------------------- #
# 异常检测
# --------------------------------------------------------------------------- #
def test_3sigma_detects_injected_spike(sample_frame: pd.DataFrame) -> None:
    result = anomaly.detect_3sigma(sample_frame, "inlet_temp_c", threshold=3.0)
    _assert_envelope(result)
    assert result["status"] == "success"
    assert 50 in result["data"]["anomaly_indices"]
    assert result["data"]["upper_bound"] > result["data"]["mean"]
    assert len(result["charts"]) == 1


def test_3sigma_multi_and_flag_column(sample_frame: pd.DataFrame) -> None:
    multi = anomaly.detect_3sigma_multi(sample_frame, columns=["inlet_temp_c", "outlet_temp_c"])
    assert multi["status"] == "success"
    assert multi["data"]["n_columns"] == 2

    flagged = anomaly.add_anomaly_flag(sample_frame, "inlet_temp_c")
    assert flagged["status"] == "success"
    frame = flagged["data"]["dataframe"]
    assert isinstance(frame, pd.DataFrame)
    flag_column = flagged["data"]["flag_column"]
    assert flag_column in frame.columns
    assert frame.loc[50, flag_column]


def test_anomaly_rejects_constant_series() -> None:
    frame = pd.DataFrame({"flat": [1.0] * 20})
    result = anomaly.detect_3sigma(frame, "flat")
    assert result["status"] == "error"
    assert ("波动" in f"{result['message']}{result['error']}") or ("标准差" in f"{result['message']}{result['error']}")


# --------------------------------------------------------------------------- #
# 机器学习
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model_type", ["linear_regression", "decision_tree", "random_forest"])
def test_train_model_all_types(tmp_path: Path, sample_frame: pd.DataFrame, model_type: str) -> None:
    result = ml_models.train_model(
        sample_frame,
        target_column="yield_pct",
        model_type=model_type,
        cross_validation=3,
    )
    _assert_envelope(result)
    assert result["status"] == "success", result["error"]
    metrics = result["data"]["metrics"]
    for key in ("mse", "rmse", "mae", "r2", "n_test"):
        assert key in metrics
    assert metrics["r2"] > 0.1, f"{model_type} 拟合优度过低：{metrics['r2']}"
    assert result["data"]["feature_importance"]
    assert len(result["charts"]) >= 2
    assert result["data"]["model_path"] and Path(result["data"]["model_path"]).exists()


def test_model_save_load_predict_roundtrip(sample_frame: pd.DataFrame) -> None:
    trained = ml_models.train_model(
        sample_frame,
        target_column="yield_pct",
        feature_columns=["inlet_temp_c", "steam_flow_t_h", "outlet_temp_c"],
        model_type="random_forest",
        cross_validation=0,
        model_name="unit_test_model",
    )
    assert trained["status"] == "success"
    model_path = trained["data"]["model_path"]

    loaded = ml_models.load_model(model_path)
    _assert_envelope(loaded)
    assert loaded["status"] == "success"
    assert loaded["data"]["metadata"]["target_column"] == "yield_pct"

    predicted = ml_models.predict(model_path, sample_frame)
    _assert_envelope(predicted)
    assert predicted["status"] == "success"
    # 含缺失值的行会被跳过（fixture 中 steam_flow_t_h 存在 2 个缺失）
    assert predicted["data"]["n_predicted"] == predicted["data"]["n_input_rows"] - predicted["data"]["n_skipped"]
    assert predicted["data"]["n_predicted"] > 0
    assert predicted["data"]["prediction_stats"]["mean"] == pytest.approx(
        sample_frame["yield_pct"].mean(), rel=0.25
    )


def test_train_model_rejects_unknown_target(sample_frame: pd.DataFrame) -> None:
    result = ml_models.train_model(sample_frame, target_column="不存在的目标")
    assert result["status"] == "error"


def test_train_model_rejects_unknown_model_type(sample_frame: pd.DataFrame) -> None:
    result = ml_models.train_model(sample_frame, target_column="yield_pct", model_type="xgboost")
    assert result["status"] == "error"
    assert "不支持" in f"{result['message']}{result['error']}"


def test_list_models_envelope() -> None:
    result = ml_models.list_models()
    assert result["status"] == "success"
    assert len(result["data"]["models"]) == 3


# --------------------------------------------------------------------------- #
# 图表字体（跨平台 / 中文 / 负号）
# --------------------------------------------------------------------------- #
def test_font_candidates_follow_platform_priority() -> None:
    """Linux 优先 Noto CJK（云端），Windows 优先微软雅黑，macOS 优先苹方。"""
    from analytics import font_candidates

    assert font_candidates("linux")[0] == "Noto Sans CJK SC"
    assert font_candidates("win32")[0] == "Microsoft YaHei"
    assert font_candidates("darwin")[0] == "PingFang SC"
    assert font_candidates("linux")[-1] == "DejaVu Sans"


def test_configure_chart_fonts_sets_global_rcparams() -> None:
    """全局字体函数应设置 sans-serif 字体族，并关闭 unicode_minus（修复负号方块）。"""
    from matplotlib import rcParams

    from analytics import configure_chart_fonts

    configured = configure_chart_fonts(force=True)
    assert isinstance(configured, str)
    assert rcParams["axes.unicode_minus"] is False
    assert list(rcParams["font.family"]) == ["sans-serif"]
    assert len(rcParams["font.sans-serif"]) >= 1


def test_apply_font_to_figure_overrides_existing_text() -> None:
    """已创建的中文标题应被强制切换到全局中文字体（防止生成代码自行设错字体）。"""
    import matplotlib.pyplot as plt

    from analytics import apply_font_to_figure, configure_chart_fonts

    chosen = configure_chart_fonts(force=True)
    figure, axis = plt.subplots()
    try:
        axis.set_title("中文标题", fontfamily="DejaVu Sans")
        axis.set_xlabel("负值 -1")
        apply_font_to_figure(figure)
        if chosen:
            assert chosen in axis.title.get_fontfamily()
            assert chosen in axis.xaxis.label.get_fontfamily()
    finally:
        plt.close(figure)
