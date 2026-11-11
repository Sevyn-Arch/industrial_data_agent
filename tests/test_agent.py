"""Agent 主流程单元测试（数据管理 / 自然语言分析闭环 / 工具箱 / 历史与状态）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.agent_core import IndustrialDataAgent  # noqa: E402
from agent.code_generator import INTENT_NAMES  # noqa: E402


# --------------------------------------------------------------------------- #
# 数据文件管理
# --------------------------------------------------------------------------- #
def test_list_datasets(agent: IndustrialDataAgent) -> None:
    result = agent.list_datasets()
    assert result["status"] == "success"
    names = [item["file_name"] for item in result["data"]["datasets"]]
    assert "process_data.csv" in names
    assert result["data"]["n_files"] >= 1


def test_describe_dataset(agent: IndustrialDataAgent) -> None:
    result = agent.describe_dataset("process_data.csv", preview_rows=3)
    assert result["status"] == "success"
    data = result["data"]
    assert data["n_rows"] > 100
    assert "inlet_temp_c" in data["schema"]["numeric_columns"]
    assert len(data["preview"]) == 3
    assert "inlet_temp_c" in data["preview_columns"]


def test_delete_dataset(agent: IndustrialDataAgent) -> None:
    assert agent.delete_dataset("process_data.csv")["status"] == "success"
    assert agent.describe_dataset("process_data.csv")["status"] == "error"


def test_save_uploaded_bytes(agent: IndustrialDataAgent, sample_frame: pd.DataFrame) -> None:
    payload = sample_frame.head(20).to_csv(index=False).encode("utf-8")
    ok_result = agent.save_uploaded_bytes("uploaded.csv", payload, max_mb=10)
    assert ok_result["status"] == "success"
    assert ok_result["data"]["file_name"] == "uploaded.csv"
    assert (agent.data_dir / "uploaded.csv").exists()

    assert agent.save_uploaded_bytes("bad.txt", payload)["status"] == "error"
    assert agent.save_uploaded_bytes("empty.csv", b"")["status"] == "error"
    assert agent.save_uploaded_bytes("big.csv", b"0" * (2 * 1024 * 1024), max_mb=1)["status"] == "error"


def test_resolve_data_path_blocks_traversal(agent: IndustrialDataAgent) -> None:
    result = agent.load_dataframe("../../../etc/passwd.csv")
    assert result["status"] == "error"
    assert agent.load_dataframe("not_exists.csv")["status"] == "error"


# --------------------------------------------------------------------------- #
# 自然语言分析闭环（离线模板模式，真实走沙箱）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "question, expected_intent",
    [
        ("请统计 inlet_temp_c 的均值和标准差", "describe_stats"),
        ("分析各变量之间的相关性", "correlation"),
        ("inlet_temp_c 与 outlet_temp_c 之间是否存在时滞", "time_lag"),
        ("用 3σ 原则检测 inlet_temp_c 的异常点", "anomaly"),
        ("训练随机森林模型预测 yield_pct", "ml_train"),
        ("检查数据缺失情况", "missing_values"),
    ],
)
def test_analyze_end_to_end(
    agent: IndustrialDataAgent,
    question: str,
    expected_intent: str,
) -> None:
    result = agent.analyze(question, data_file="process_data.csv", use_llm=False)
    assert result["status"] == "success", result["error"]
    data = result["data"]
    assert data["intent"] == expected_intent
    assert data["code_source"] == "template"
    assert data["execution"]["status"] == "success"
    assert data["report"].startswith("# 工业数据分析报告")
    assert Path(data["report_path"]).exists()
    assert Path(data["report_path"]).with_suffix(".html").exists()
    assert any(step["step"] == "自动生成报告" for step in data["steps"])
    payload = data["execution"]["result"]
    assert isinstance(payload, dict) and payload.get("intent") == expected_intent
    assert payload.get("intent_name"), "结构化结果应包含中文意图名"
    # 报告中的分析类型必须展示中文名称（曾出现只显示机器意图键的缺陷）
    assert f"**分析类型**：{INTENT_NAMES[expected_intent]}" in data["report"]


def test_analyze_records_history(agent: IndustrialDataAgent) -> None:
    agent.analyze("统计 inlet_temp_c 的均值", data_file="process_data.csv", use_llm=False)
    history = agent.read_history(limit=5)
    assert history["status"] == "success"
    assert history["data"]["records"]
    record = history["data"]["records"][0]
    assert record["data_file"] == "process_data.csv"
    assert record["status"] == "success"
    assert (agent.output_dir / "agent_history.json").exists()


def test_analyze_generates_charts(agent: IndustrialDataAgent) -> None:
    result = agent.analyze("统计 inlet_temp_c 的均值并给出分布直方图", data_file="process_data.csv", use_llm=False)
    assert result["status"] == "success"
    assert result["charts"], "本次分析应产出至少一张图表"
    assert all(Path(path).exists() for path in result["charts"])


def test_analyze_without_question_fails_gracefully(agent: IndustrialDataAgent) -> None:
    result = agent.analyze("   ")
    assert result["status"] == "error"
    assert "不能为空" in result["message"]


def test_analyze_with_missing_file_fails_gracefully(agent: IndustrialDataAgent) -> None:
    result = agent.analyze("统计一下数据", data_file="ghost.csv", use_llm=False)
    assert result["status"] == "error"
    assert "ghost.csv" in f"{result['message']}{result['error']}"


def test_analyze_without_report(agent: IndustrialDataAgent) -> None:
    result = agent.analyze("统计 inlet_temp_c 的均值", data_file="process_data.csv", with_report=False)
    assert result["status"] == "success"
    assert result["data"]["report"] == ""
    assert result["data"]["report_path"] is None


# --------------------------------------------------------------------------- #
# 算法工具箱
# --------------------------------------------------------------------------- #
def test_toolbox_basic_stats(agent: IndustrialDataAgent) -> None:
    result = agent.run_basic_stats("process_data.csv", columns=["inlet_temp_c", "outlet_temp_c"])
    assert result["status"] == "success"
    assert set(result["data"]["statistics"]) == {"inlet_temp_c", "outlet_temp_c"}


def test_toolbox_correlation(agent: IndustrialDataAgent) -> None:
    result = agent.run_correlation("process_data.csv", threshold=0.4)
    assert result["status"] == "success"
    assert result["data"]["matrix"]["strongest_pair"] is not None
    assert isinstance(result["data"]["strong_pairs"], list)
    assert result["charts"]


def test_toolbox_time_lag(agent: IndustrialDataAgent) -> None:
    result = agent.run_time_lag(
        "process_data.csv",
        x_column="inlet_temp_c",
        y_column="outlet_temp_c",
        max_lag=6,
        time_column="timestamp",
        sample_interval=300,
    )
    assert result["status"] == "success"
    assert result["data"]["best_lag"] == 3
    assert result["data"]["best_lag_seconds"] == pytest.approx(900.0)


def test_toolbox_anomaly(agent: IndustrialDataAgent) -> None:
    result = agent.run_anomaly("process_data.csv", columns=["inlet_temp_c"])
    assert result["status"] == "success"
    assert result["data"]["total_anomalies"] >= 1


def test_toolbox_ml_train_and_predict(agent: IndustrialDataAgent) -> None:
    trained = agent.run_ml_train(
        "process_data.csv",
        target_column="yield_pct",
        model_type="random_forest",
        cross_validation=3,
    )
    assert trained["status"] == "success", trained["error"]
    assert trained["data"]["metrics"]["r2"] > 0.3

    predicted = agent.run_ml_predict("process_data.csv", model_path=trained["data"]["model_path"])
    assert predicted["status"] == "success"
    assert predicted["data"]["n_predicted"] > 0

    auto = agent.run_ml_predict("process_data.csv")
    assert auto["status"] == "success"


def test_toolbox_unknown_file_fails_gracefully(agent: IndustrialDataAgent) -> None:
    assert agent.run_basic_stats("ghost.csv")["status"] == "error"
    assert agent.run_time_lag("ghost.csv", "a", "b")["status"] == "error"


# --------------------------------------------------------------------------- #
# 系统状态
# --------------------------------------------------------------------------- #
def test_system_status(agent: IndustrialDataAgent) -> None:
    result = agent.system_status()
    assert result["status"] == "success"
    data = result["data"]
    assert "sandbox" in data and data["sandbox"]["timeout"] > 0
    assert data["stats"]["n_datasets"] >= 1
    assert data["settings"]["code_model"]
    assert "api_key_masked" in data["settings"]
