"""Streamlit 界面冒烟测试（使用 Streamlit 官方 AppTest 无头运行）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_FILE = PROJECT_ROOT / "app.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st_testing = pytest.importorskip("streamlit.testing.v1", reason="需要 Streamlit 1.28+ 的测试框架")
AppTest = st_testing.AppTest

#: 冒烟测试使用的数据文件（项目 data 目录下的示例数据）
SAMPLE_DATA = "sample_process_data.csv"


def _new_app() -> "AppTest":
    """构建 AppTest 实例（放宽超时以适配沙箱执行）。"""
    return AppTest.from_file(str(APP_FILE), default_timeout=300)


def test_first_render_has_no_exception() -> None:
    """首次渲染不应抛出任何异常，且四个页签都在。"""
    app = _new_app().run()
    assert not app.exception, [str(item.value) for item in app.exception]
    assert app.title, "页面标题缺失"
    assert len(app.tabs) == 4
    assert any("工业数据分析 Agent 系统" in item.value for item in app.title)


def test_data_tab_and_toolbox_render() -> None:
    """数据管理页签与工具箱页签的关键控件应正确渲染。"""
    app = _new_app()
    app.session_state["data_view"] = SAMPLE_DATA
    app.run()
    assert not app.exception, [str(item.value) for item in app.exception]
    assert any(SAMPLE_DATA in option for select in app.selectbox for option in (select.options or []))


def test_natural_language_analysis_end_to_end() -> None:
    """在界面上发起一次真实分析：意图解析 -> 沙箱执行 -> 报告生成。"""
    app = _new_app()
    app.session_state["analysis_file"] = SAMPLE_DATA
    app.run()
    assert not app.exception, [str(item.value) for item in app.exception]

    app.text_area(key="question_input").set_value("请统计 inlet_temp_c 的均值和标准差")
    app.button(key="btn_run_analysis").click()
    app.run()

    assert not app.exception, [str(item.value) for item in app.exception]
    success_texts = [item.value for item in app.success]
    assert any("分析完成" in text for text in success_texts), success_texts
    markdown_blob = "\n".join(item.value for item in app.markdown)
    assert "分析结论" in markdown_blob or "工业数据分析报告" in markdown_blob
    metric_labels = [item.label for item in app.metric]
    assert "识别意图" in metric_labels
