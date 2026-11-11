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
    captions = [item.value for item in app.caption]
    assert any("所有生成代码均在独立沙箱中执行" in text for text in captions)


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


@pytest.mark.parametrize(
    "error_type, expected_text",
    [
        ("security", "代码包含危险操作，已被安全沙箱拦截，禁止执行"),
        ("runtime", "分析代码执行失败，请检查字段名或调整分析需求"),
        ("timeout", "分析任务执行超时，已强制终止，请简化分析需求"),
    ],
)
def test_sandbox_error_types_render_specific_messages(error_type: str, expected_text: str) -> None:
    """前端应按后端 error_type 展示三类不同提示，而不是统一的"未知错误"。"""
    app = _new_app()
    app.session_state["analysis_result"] = {
        "status": "error",
        "message": expected_text,
        "error": "第 1 行：禁止导入模块 os",
        "charts": [],
        "data": {
            "question": "测试失败提示",
            "intent_name": "基础描述性统计",
            "code_source": "template",
            "generation_strategy": "template-first",
            "template_reason": "标准场景，命中内置模板",
            "duration_seconds": 0.5,
            "steps": [],
            "code": "import os\n",
            "report": "",
            "charts": [],
            "execution": {
                "status": "error",
                "error_type": error_type,
                "message": expected_text,
                "error": "第 1 行：禁止导入模块 os",
                "violations": ["第 1 行：禁止导入模块 os（不在白名单内）"],
                "exit_code": None,
                "timed_out": error_type == "timeout",
                "duration_seconds": 0.1,
            },
        },
    }
    app.run()
    assert not app.exception, [str(item.value) for item in app.exception]
    errors = [item.value for item in app.error]
    assert any(expected_text in text for text in errors), errors
