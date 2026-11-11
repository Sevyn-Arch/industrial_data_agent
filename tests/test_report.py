"""报告生成器单元测试（事实整理 / 模板报告 / Markdown 转 HTML / 文件落盘）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from report.generator import ReportGenerator  # noqa: E402


def _fake_execution(status: str = "success") -> dict:
    """构造一个与沙箱执行结果同构的测试数据。"""
    return {
        "status": status,
        "data": {
            "status": status,
            "exit_code": 0 if status == "success" else 1,
            "timed_out": False,
            "duration_seconds": 1.234,
            "stdout": "开始分析\n完成\n",
            "stderr": "" if status == "success" else "Traceback ... ValueError",
            "error": None if status == "success" else "执行失败：字段不存在",
            "violations": [],
            "code_path": "output/.sandbox/run_x/analysis_code.py",
            "result": {
                "intent": "time_lag",
                "x": "inlet_temp_c",
                "y": "outlet_temp_c",
                "best_lag": 3,
                "best_correlation": 0.9123456,
                "conclusion": "inlet_temp_c 领先 outlet_temp_c 约 3 个采样周期，最大相关系数 0.912。",
                "top_lags": [{"lag": 3, "correlation": 0.9123}],
            },
        },
        "message": "代码执行成功",
        "error": None,
        "charts": [],
    }


def test_build_facts_extracts_fields() -> None:
    generator = ReportGenerator(llm_client=None)
    facts = generator.build_facts("分析时滞", _fake_execution(), schema={"n_rows": 300, "n_columns": 6,
                                                                       "numeric_columns": ["a", "b"]})
    assert facts["question"] == "分析时滞"
    assert facts["status"] == "success"
    assert facts["result"]["intent"] == "time_lag"
    assert facts["schema"]["n_rows"] == 300
    assert facts["duration_seconds"] == 1.234


def test_render_template_report_contains_sections() -> None:
    generator = ReportGenerator(llm_client=None)
    execution = _fake_execution()
    execution["data"]["result"]["intent_name"] = "时序时滞分析"
    facts = generator.build_facts("分析时滞", execution)
    markdown = generator.render_template_report(facts)
    for section in ("# 工业数据分析报告", "## 一、分析结论", "## 二、关键指标",
                    "## 四、执行环境与安全检查", "## 五、风险提示与优化建议"):
        assert section in markdown
    assert "inlet_temp_c 领先 outlet_temp_c" in markdown
    assert "| best_lag | 3 |" in markdown
    assert "**分析类型**：时序时滞分析" in markdown


def test_render_template_report_handles_failure() -> None:
    generator = ReportGenerator(llm_client=None)
    facts = generator.build_facts("分析时滞", _fake_execution(status="error"))
    markdown = generator.render_template_report(facts)
    assert "未能成功完成" in markdown
    assert "执行失败" in markdown
    assert "程序标准错误" in markdown


def test_generate_and_save_writes_files(tmp_path: Path) -> None:
    generator = ReportGenerator(llm_client=None)
    result = generator.generate("分析时滞", _fake_execution())
    assert result["status"] == "success"
    data = result["data"]
    assert data["source"] == "template"
    assert Path(data["md_path"]).exists()
    assert Path(data["html_path"]).exists()
    assert "<!DOCTYPE html>" in Path(data["html_path"]).read_text(encoding="utf-8")


def test_markdown_to_html_renders_table_list_and_code() -> None:
    generator = ReportGenerator(llm_client=None)
    markdown = (
        "# 标题\n\n"
        "- 项目一\n- 项目二\n\n"
        "| 指标 | 数值 |\n| --- | --- |\n| R² | 0.9123 |\n\n"
        "```text\nsome output\n```\n\n"
        "**加粗** 与 `代码`\n"
    )
    html = generator.markdown_to_html(markdown, [])
    assert "<h1>标题</h1>" in html
    assert "<ul>" in html and "<li>项目一</li>" in html
    assert "<table>" in html and "<th>指标</th>" in html
    assert "<pre>some output</pre>" in html
    assert "<strong>加粗</strong>" in html and "<code>代码</code>" in html


def test_markdown_to_html_embeds_chart(tmp_path: Path) -> None:
    """图表应以内嵌 base64 形式写入 HTML，保证单文件可分享。"""
    from analytics import CHART_DIR

    chart_path = CHART_DIR / "pytest_embed_check.png"
    # 生成一张最小合法 PNG（1×1 像素）
    chart_path.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
        )
    )
    try:
        generator = ReportGenerator(llm_client=None)
        html = generator.markdown_to_html("![图 1](charts/pytest_embed_check.png)", [str(chart_path)])
        assert "data:image/png;base64," in html
    finally:
        chart_path.unlink(missing_ok=True)


def test_llm_branch_is_used_when_client_available() -> None:
    class _StubReportClient:
        is_available = True

        def generate_report(self, prompt, system_prompt=None, **kwargs):
            return {
                "status": "success",
                "data": {"content": "## 一、分析结论\n\n由 LLM 撰写的结论。", "model": "stub-report-model"},
                "message": "ok",
                "error": None,
                "charts": [],
            }

    generator = ReportGenerator(llm_client=_StubReportClient())
    result = generator.generate("分析时滞", _fake_execution())
    assert result["status"] == "success"
    assert result["data"]["source"] == "llm"
    assert "由 LLM 撰写的结论" in result["data"]["markdown"]


def test_llm_failure_falls_back_to_template() -> None:
    class _FailingClient:
        is_available = True

        def generate_report(self, prompt, system_prompt=None, **kwargs):
            return {"status": "error", "data": None, "message": "接口异常", "error": "502 Bad Gateway", "charts": []}

    generator = ReportGenerator(llm_client=_FailingClient())
    result = generator.generate("分析时滞", _fake_execution())
    assert result["status"] == "success"
    assert result["data"]["source"] == "template"
    assert "## 一、分析结论" in result["data"]["markdown"]
