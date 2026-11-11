"""机器学习建模模块。

支持三类回归模型：

* ``linear_regression``：线性回归（可读性强，用于解释变量影响方向与幅度）；
* ``decision_tree``：决策树回归（可解释规则，易过拟合）；
* ``random_forest``：随机森林回归（抗过拟合、支持特征重要性）。

提供训练、评估（MSE / RMSE / MAE / R² / 交叉验证）、保存（joblib）、加载与推理能力。
所有对外函数返回统一结果包，且内部捕获全部异常。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import MODEL_DIR, fail, get_logger, new_chart_path, ok, resolve_column, safe_float, setup_plot_style

__all__ = [
    "SUPPORTED_MODELS",
    "list_models",
    "train_model",
    "evaluate_model",
    "save_model",
    "load_model",
    "predict",
    "feature_importance",
]

logger = get_logger(__name__)

#: 模型类型 -> 中文名称
SUPPORTED_MODELS: dict[str, str] = {
    "linear_regression": "线性回归 (LinearRegression)",
    "decision_tree": "决策树回归 (DecisionTreeRegressor)",
    "random_forest": "随机森林回归 (RandomForestRegressor)",
}

#: 常见别名 -> 规范模型名
MODEL_ALIASES: dict[str, str] = {
    "linear": "linear_regression",
    "linear_regression": "linear_regression",
    "lr": "linear_regression",
    "ols": "linear_regression",
    "线性回归": "linear_regression",
    "decision_tree": "decision_tree",
    "decisiontree": "decision_tree",
    "dt": "decision_tree",
    "tree": "decision_tree",
    "决策树": "decision_tree",
    "random_forest": "random_forest",
    "randomforest": "random_forest",
    "rf": "random_forest",
    "forest": "random_forest",
    "随机森林": "random_forest",
}

_META_SUFFIX = ".meta.json"


# --------------------------------------------------------------------------- #
# 内部工具函数
# --------------------------------------------------------------------------- #
def _import_sklearn() -> dict[str, Any]:
    """惰性导入 scikit-learn 组件（未安装时给出可执行的修复提示）。"""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import KFold, cross_val_score, train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeRegressor
    except ImportError as exc:  # pragma: no cover - 环境缺失依赖时才会触发
        raise RuntimeError(
            "未检测到 scikit-learn，请先执行：python -m pip install -r requirements.txt"
        ) from exc
    return {
        "LinearRegression": LinearRegression,
        "DecisionTreeRegressor": DecisionTreeRegressor,
        "RandomForestRegressor": RandomForestRegressor,
        "StandardScaler": StandardScaler,
        "Pipeline": Pipeline,
        "KFold": KFold,
        "cross_val_score": cross_val_score,
        "train_test_split": train_test_split,
        "mean_squared_error": mean_squared_error,
        "mean_absolute_error": mean_absolute_error,
        "r2_score": r2_score,
    }


def _import_joblib() -> Any:
    """惰性导入 joblib（模型持久化）。"""
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("未检测到 joblib，请先执行：python -m pip install -r requirements.txt") from exc
    return joblib


def normalize_model_type(model_type: str) -> str:
    """将用户输入的模型名（含别名/中文名）归一化为规范模型名。

    Raises:
        ValueError: 不支持的模型类型。
    """
    key = str(model_type or "").strip().lower().replace(" ", "_").replace("-", "_")
    normalized = MODEL_ALIASES.get(key) or MODEL_ALIASES.get(key.replace("_", ""))
    if normalized is None:
        raise ValueError(f"不支持的模型类型：{model_type}；可选值：{list(SUPPORTED_MODELS)}")
    return normalized


def _default_params(model_type: str) -> dict[str, Any]:
    """返回各模型的默认超参数（随机种子固定，保证结果可复现）。"""
    if model_type == "linear_regression":
        return {}
    if model_type == "decision_tree":
        return {"max_depth": 8, "min_samples_leaf": 2, "random_state": 42}
    return {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1, "random_state": 42, "n_jobs": -1}


def _build_estimator(model_type: str, params: dict[str, Any]) -> Any:
    """按模型类型与超参数实例化估计器（未知参数会抛出明确错误）。"""
    sklearn = _import_sklearn()
    merged = {**_default_params(model_type), **{key: value for key, value in params.items() if value is not None}}
    if model_type == "linear_regression":
        allowed = {"fit_intercept", "positive"}
        unknown = set(merged) - allowed
        if unknown:
            raise ValueError(f"线性回归不支持的超参数：{sorted(unknown)}")
        return sklearn["LinearRegression"](**merged)
    if model_type == "decision_tree":
        allowed = {
            "max_depth", "min_samples_leaf", "min_samples_split", "max_features",
            "random_state", "ccp_alpha", "min_impurity_decrease",
        }
        unknown = set(merged) - allowed
        if unknown:
            raise ValueError(f"决策树不支持的超参数：{sorted(unknown)}")
        return sklearn["DecisionTreeRegressor"](**merged)
    allowed = {
        "n_estimators", "max_depth", "min_samples_leaf", "min_samples_split", "max_features",
        "random_state", "n_jobs", "max_samples", "bootstrap",
    }
    unknown = set(merged) - allowed
    if unknown:
        raise ValueError(f"随机森林不支持的超参数：{sorted(unknown)}")
    return sklearn["RandomForestRegressor"](**merged)


def _resolve_features(df: pd.DataFrame, target_label: Any, feature_columns: Sequence[str] | None) -> list[Any]:
    """解析建模特征列：显式指定优先，否则使用「除目标列外的全部数值字段」。"""
    if feature_columns:
        labels: list[Any] = []
        for column in feature_columns:
            label = resolve_column(df, column)
            if label == target_label:
                continue
            if label not in labels:
                labels.append(label)
        if not labels:
            raise ValueError("特征列解析结果为空，请检查 feature_columns 参数")
        return labels
    labels = [col for col in df.select_dtypes(include=[np.number]).columns if col != target_label]
    if not labels:
        raise ValueError("未找到可用的数值型特征列，请显式指定 feature_columns")
    return labels


def _prepare_dataset(
    df: pd.DataFrame,
    target_label: Any,
    features: list[Any],
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """构造建模用的 X、y，并返回清洗统计信息。"""
    subset = df[[*features, target_label]].copy()
    for column in [*features, target_label]:
        subset[column] = pd.to_numeric(subset[column], errors="coerce")
    before = int(subset.shape[0])
    subset = subset.dropna(subset=[target_label])
    subset = subset.dropna(subset=features, how="any")
    after = int(subset.shape[0])
    if after < 10:
        raise ValueError(
            f"清洗后有效样本仅 {after} 条（原始 {before} 条），至少需要 10 条才能建模；请检查缺失值与字段类型"
        )
    return subset[features], subset[target_label].astype(float), {"n_rows_before": before, "n_rows_after": after}


def _compute_metrics(sklearn: dict[str, Any], y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """计算回归评估指标（MSE / RMSE / MAE / R² / MAPE）。"""
    mse = safe_float(sklearn["mean_squared_error"](y_true, y_pred))
    mae = safe_float(sklearn["mean_absolute_error"](y_true, y_pred))
    r2 = safe_float(sklearn["r2_score"](y_true, y_pred))
    nonzero = np.asarray(y_true, dtype=float) != 0
    if nonzero.any():
        mape = safe_float(np.mean(np.abs((np.asarray(y_true, dtype=float)[nonzero] - np.asarray(y_pred, dtype=float)[nonzero])
                                       / np.asarray(y_true, dtype=float)[nonzero])) * 100.0)
    else:
        mape = None
    return {
        "mse": round(mse, 6),
        "rmse": round(float(np.sqrt(mse)), 6),
        "mae": round(mae, 6),
        "r2": round(r2, 6),
        "mape": None if mape is None else round(mape, 6),
        "n_samples": int(len(np.asarray(y_true))),
    }


def _extract_importance(model: Any, feature_names: list[str]) -> list[dict[str, Any]]:
    """提取特征重要性（树模型）或回归系数（线性模型）。"""
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    values: np.ndarray | None = None
    kind = ""
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        kind = "importance"
    elif hasattr(estimator, "coef_"):
        values = np.asarray(estimator.coef_, dtype=float).ravel()
        kind = "coefficient"
    if values is None or values.shape[0] != len(feature_names):
        return []
    items = [
        {"feature": feature_names[index], kind: round(safe_float(value), 6), "abs_value": round(abs(safe_float(value)), 6)}
        for index, value in enumerate(values)
    ]
    items.sort(key=lambda item: item["abs_value"], reverse=True)
    return items


def _plot_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, target: str, prefix: str) -> str:
    """绘制「预测值 vs 实测值」散点图（含 y=x 参考线）。"""
    setup_plot_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.scatter(y_true, y_pred, s=24, alpha=0.7, color="#4f81bd")
    limits = [float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))]
    ax.plot(limits, limits, color="#c0504d", linestyle="--", linewidth=1.3, label="理想预测 y=x")
    ax.set_xlabel(f"实测值 {target}")
    ax.set_ylabel(f"预测值 {target}")
    ax.set_title(f"{target} 预测值 vs 实测值")
    ax.legend(fontsize=9)
    path = new_chart_path(prefix)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, target: str, prefix: str) -> str:
    """绘制残差分布直方图。"""
    setup_plot_style()
    import matplotlib.pyplot as plt

    residual = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.hist(residual, bins=min(40, max(8, int(np.sqrt(residual.size)))), color="#9bbb59", alpha=0.85)
    ax.axvline(0.0, color="#c0504d", linestyle="--", linewidth=1.2, label="零残差")
    ax.set_xlabel("残差（实测 - 预测）")
    ax.set_ylabel("频次")
    ax.set_title(f"{target} 残差分布（均值 {residual.mean():.4f}）")
    ax.legend(fontsize=9)
    path = new_chart_path(prefix)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _plot_importance(importance: list[dict[str, Any]], target: str) -> str | None:
    """绘制特征重要性 / 回归系数柱状图。"""
    if not importance:
        return None
    setup_plot_style()
    import matplotlib.pyplot as plt

    items = importance[:20][::-1]  # 横向柱状图自下而上排布
    metric_key = "coefficient" if "coefficient" in items[0] else "importance"
    names = [item["feature"] for item in items]
    values = [item[metric_key] for item in items]
    fig, ax = plt.subplots(figsize=(7.5, max(3.0, len(items) * 0.36)))
    colors = ["#4f81bd" if value >= 0 else "#c0504d" for value in values]
    ax.barh(names, values, color=colors)
    ax.axvline(0.0, color="#7f7f7f", linewidth=0.8)
    ax.set_xlabel("回归系数" if metric_key == "coefficient" else "重要性")
    ax.set_title(f"{target} 影响因素排序（Top {len(items)}）")
    path = new_chart_path("feature_importance")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


# --------------------------------------------------------------------------- #
# 对外 API
# --------------------------------------------------------------------------- #
def list_models() -> dict[str, Any]:
    """列出系统支持的全部模型类型。"""
    try:
        models = [
            {"model_type": key, "name": value, "default_params": _default_params(key)}
            for key, value in SUPPORTED_MODELS.items()
        ]
        return ok({"models": models, "default": "random_forest"}, f"共支持 {len(models)} 类回归模型")
    except Exception as exc:  # noqa: BLE001
        logger.exception("list_models 执行失败")
        return fail("模型清单获取失败", exc)


def train_model(
    df: Any,
    target_column: str,
    feature_columns: Sequence[str] | None = None,
    model_type: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
    cross_validation: int = 5,
    scale_features: bool = False,
    save: bool = True,
    model_name: str | None = None,
    **model_params: Any,
) -> dict[str, Any]:
    """训练回归模型并输出评估指标、特征重要性与可视化图表。

    Args:
        df: 训练数据表。
        target_column: 目标变量（被预测列）。
        feature_columns: 特征列；``None`` 表示使用除目标列外的全部数值字段。
        model_type: 模型类型，见 :data:`SUPPORTED_MODELS`。
        test_size: 测试集比例，取值 (0, 1)。
        random_state: 随机种子（保证可复现）。
        cross_validation: K 折交叉验证折数，<3 表示跳过交叉验证。
        scale_features: 是否对特征做标准化（线性模型建议开启）。
        save: 是否保存模型到 ``output/models``。
        model_name: 模型文件名（不含扩展名），默认自动生成。
        **model_params: 模型超参数，例如 ``n_estimators=500``、``max_depth=10``。

    Returns:
        dict: 结果包，``data`` 含 ``metrics``（MSE/RMSE/MAE/R²/MAPE/交叉验证得分）、
        ``feature_importance``、``model_path``、``features``、``charts`` 等。
    """
    charts: list[str] = []
    try:
        sklearn = _import_sklearn()
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df 参数必须是 pandas.DataFrame，实际为 {type(df).__name__}")
        if df.empty:
            raise ValueError("收到空 DataFrame，无法建模")

        model_type = normalize_model_type(model_type)
        target_label = resolve_column(df, target_column)
        features = _resolve_features(df, target_label, feature_columns)
        test_size = safe_float(test_size, 0.2)
        if not 0.05 <= test_size <= 0.5:
            raise ValueError(f"test_size 建议位于 [0.05, 0.5] 区间，实际为 {test_size}")

        X, y, clean_info = _prepare_dataset(df, target_label, features)
        feature_names = [str(name) for name in features]
        X.columns = feature_names

        estimator = _build_estimator(model_type, model_params)
        if scale_features:
            pipeline = sklearn["Pipeline"]([("scaler", sklearn["StandardScaler"]()), ("model", estimator)])
        else:
            pipeline = estimator

        X_train, X_test, y_train, y_test = sklearn["train_test_split"](
            X, y, test_size=test_size, random_state=int(random_state), shuffle=True
        )
        pipeline.fit(X_train, y_train)
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)

        metrics = _compute_metrics(sklearn, y_test.to_numpy(dtype=float), np.asarray(y_pred_test, dtype=float))
        metrics["train_r2"] = round(safe_float(sklearn["r2_score"](y_train, y_pred_train)), 6)
        metrics["n_train"] = int(X_train.shape[0])
        metrics["n_test"] = int(X_test.shape[0])

        cv_scores: list[float] | None = None
        fold = int(cross_validation) if cross_validation else 0
        if fold >= 3 and X.shape[0] >= fold * 5:
            try:
                cv = sklearn["KFold"](n_splits=fold, shuffle=True, random_state=int(random_state))
                scores = sklearn["cross_val_score"](pipeline, X, y, cv=cv, scoring="r2", n_jobs=None)
                cv_scores = [round(safe_float(score), 6) for score in np.asarray(scores, dtype=float).tolist()]
                metrics["cv_r2_mean"] = round(float(np.mean(cv_scores)), 6)
                metrics["cv_r2_std"] = round(float(np.std(cv_scores)), 6)
                metrics["cv_folds"] = fold
            except Exception as exc:  # noqa: BLE001 - 交叉验证失败不影响主流程
                logger.warning("交叉验证失败，已跳过：%s", exc)
                metrics["cv_error"] = str(exc)
        else:
            metrics["cv_note"] = f"样本量 {X.shape[0]} 或折数 {fold} 不足，已跳过交叉验证"

        importance = _extract_importance(pipeline, feature_names)

        # 可视化：预测 vs 实测、残差分布、特征重要性
        try:
            charts.append(
                _plot_pred_vs_actual(y_test.to_numpy(dtype=float), np.asarray(y_pred_test, dtype=float),
                                     str(target_label), f"ml_pred_{model_type}")
            )
            charts.append(
                _plot_residuals(y_test.to_numpy(dtype=float), np.asarray(y_pred_test, dtype=float),
                                str(target_label), f"ml_residual_{model_type}")
            )
            importance_chart = _plot_importance(importance, str(target_label))
            if importance_chart:
                charts.append(importance_chart)
        except Exception:  # noqa: BLE001 - 图表失败不影响建模结果
            logger.warning("建模图表生成失败", exc_info=True)

        model_path: str | None = None
        if save:
            saved = save_model(
                pipeline,
                model_name=model_name,
                metadata={
                    "model_type": model_type,
                    "target_column": str(target_label),
                    "feature_columns": feature_names,
                    "scale_features": bool(scale_features),
                    "test_size": test_size,
                    "random_state": int(random_state),
                    "metrics": metrics,
                },
            )
            if saved["status"] == "success":
                model_path = saved["data"]["model_path"]
            else:
                logger.warning("模型保存失败：%s", saved["error"])

        data = {
            "model_type": model_type,
            "model_name": SUPPORTED_MODELS[model_type],
            "target": str(target_label),
            "features": feature_names,
            "n_features": len(feature_names),
            "metrics": metrics,
            "feature_importance": importance,
            "model_path": model_path,
            "data_cleaning": clean_info,
            "params": _default_params(model_type) | {key: value for key, value in model_params.items() if value is not None},
        }
        message = (
            f"{SUPPORTED_MODELS[model_type]} 训练完成：测试集 R²={metrics['r2']:.4f}，"
            f"MSE={metrics['mse']:.4f}，RMSE={metrics['rmse']:.4f}，MAE={metrics['mae']:.4f}"
            + (f"，{fold} 折交叉验证 R²={metrics['cv_r2_mean']:.4f}" if cv_scores else "")
        )
        return ok(data, message, charts=charts)
    except Exception as exc:  # noqa: BLE001
        logger.exception("train_model 执行失败")
        return fail(f"模型训练失败（target={target_column}）", exc, charts=charts)


def evaluate_model(model: Any, X: Any, y: Any) -> dict[str, Any]:
    """对已训练模型在给定数据集上评估（不含绘图与持久化）。"""
    try:
        sklearn = _import_sklearn()
        if model is None:
            raise ValueError("model 不能为空")
        X_array = X.to_numpy(dtype=float) if hasattr(X, "to_numpy") else np.asarray(X, dtype=float)
        y_array = y.to_numpy(dtype=float) if hasattr(y, "to_numpy") else np.asarray(y, dtype=float)
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError(f"样本数不一致：X={X_array.shape[0]}，y={y_array.shape[0]}")
        predictions = np.asarray(model.predict(X_array), dtype=float)
        metrics = _compute_metrics(sklearn, y_array, predictions)
        return ok({"metrics": metrics}, f"评估完成：R²={metrics['r2']:.4f}，MSE={metrics['mse']:.4f}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("evaluate_model 执行失败")
        return fail("模型评估失败", exc)


def save_model(model: Any, model_name: str | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """使用 joblib 保存模型，并写入同名 ``.meta.json`` 元数据文件。

    Args:
        model: 已训练模型或 Pipeline。
        model_name: 文件名（不含扩展名）；默认 ``model_<时间戳>``。
        metadata: 推理时需要的元信息（特征列、目标列、指标等）。
    """
    try:
        joblib = _import_joblib()
        if model is None:
            raise ValueError("model 不能为空")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(model_name or "")) or (
            f"model_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        model_path = MODEL_DIR / f"{safe_name}.joblib"
        joblib.dump(model, model_path)

        meta_path = MODEL_DIR / f"{safe_name}{_META_SUFFIX}"
        payload = {"model_file": model_path.name, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        payload.update(metadata or {})
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)

        data = {"model_path": str(model_path), "meta_path": str(meta_path), "metadata": payload}
        return ok(data, f"模型已保存：{model_path.name}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("save_model 执行失败")
        return fail("模型保存失败", exc)


def load_model(model_path: str | Path) -> dict[str, Any]:
    """加载 joblib 模型（若存在同名 ``.meta.json`` 则一并返回元数据）。"""
    try:
        joblib = _import_joblib()
        path = Path(model_path)
        if not path.is_absolute():
            path = MODEL_DIR / path
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在：{path}")
        if path.suffix != ".joblib":
            raise ValueError(f"仅支持 .joblib 模型文件，实际为 {path.suffix}")

        model = joblib.load(path)
        meta_path = path.with_name(f"{path.stem}{_META_SUFFIX}")
        metadata: dict[str, Any] = {}
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
            except Exception as exc:  # noqa: BLE001 - 元数据损坏不影响模型加载
                logger.warning("模型元数据读取失败：%s", exc)
        # 注意：模型对象不可 JSON 序列化，必须关闭自动转换，否则模型会被字符串化
        return ok(
            {"model": model, "metadata": metadata, "model_path": str(path)},
            f"模型加载成功：{path.name}",
            json_safe=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("load_model 执行失败")
        return fail("模型加载失败", exc)


def predict(
    model_path: str | Path,
    df: Any,
    feature_columns: Sequence[str] | None = None,
    max_rows: int = 1000,
) -> dict[str, Any]:
    """使用已保存模型对新数据进行推理。

    Args:
        model_path: 模型文件路径（.joblib）。
        df: 待预测数据表。
        feature_columns: 特征列；``None`` 表示使用模型元数据中记录的特征顺序。
        max_rows: 结果包中最多返回的预测条数（防止前端卡顿）。
    """
    try:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df 参数必须是 pandas.DataFrame")
        loaded = load_model(model_path)
        if loaded["status"] != "success":
            return fail(loaded["message"], loaded["error"])
        model = loaded["data"]["model"]
        metadata = loaded["data"]["metadata"]

        names = list(feature_columns) if feature_columns else list(metadata.get("feature_columns") or [])
        if not names:
            names = [str(col) for col in df.select_dtypes(include=[np.number]).columns]
        if not names:
            raise ValueError("未指定特征列，且数据中不存在数值型字段")

        labels = [resolve_column(df, name) for name in names]
        subset = df[labels].apply(pd.to_numeric, errors="coerce")
        valid = subset.dropna()
        if valid.empty:
            raise ValueError("待预测数据在指定特征列上全部为缺失值，无法推理")

        predictions = np.asarray(model.predict(valid), dtype=float)
        max_rows = max(1, int(max_rows))
        data = {
            "model_path": str(loaded["data"]["model_path"]),
            "model_type": metadata.get("model_type"),
            "target_column": metadata.get("target_column"),
            "features": [str(name) for name in names],
            "n_input_rows": int(df.shape[0]),
            "n_predicted": int(predictions.shape[0]),
            "n_skipped": int(df.shape[0] - predictions.shape[0]),
            "prediction_stats": {
                "mean": round(float(predictions.mean()), 6),
                "min": round(float(predictions.min()), 6),
                "max": round(float(predictions.max()), 6),
            },
            "predictions": [round(safe_float(value), 6) for value in predictions[:max_rows].tolist()],
            "row_indices": [int(index) for index in valid.index[:max_rows].tolist()],
            "truncated": bool(predictions.shape[0] > max_rows),
        }
        return ok(data, f"推理完成：共预测 {data['n_predicted']} 条（均值 {data['prediction_stats']['mean']:.4f}）")
    except Exception as exc:  # noqa: BLE001
        logger.exception("predict 执行失败")
        return fail("模型推理失败", exc)


def feature_importance(model: Any, feature_columns: Sequence[str] | None = None) -> dict[str, Any]:
    """查看模型的特征重要性（树模型）或回归系数（线性模型）。"""
    try:
        if model is None:
            raise ValueError("model 不能为空")
        estimator = model.steps[-1][1] if hasattr(model, "steps") else model
        names = [str(name) for name in (feature_columns or [])]
        if not names:
            n_features = getattr(estimator, "n_features_in_", None)
            if n_features is None:
                raise ValueError("无法推断特征数量，请显式传入 feature_columns")
            names = [f"feature_{index}" for index in range(int(n_features))]
        items = _extract_importance(estimator, names)
        if not items:
            return fail("该模型不支持特征重要性输出（仅线性模型与树模型支持）")
        return ok({"feature_importance": items}, f"已输出 {len(items)} 个特征的重要性排序")
    except Exception as exc:  # noqa: BLE001
        logger.exception("feature_importance 执行失败")
        return fail("特征重要性提取失败", exc)
