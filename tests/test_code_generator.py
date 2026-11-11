"""代码生成器单元测试：意图解析、代码抽取、模板安全性与可执行性、LLM 分支。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.code_generator import (  # noqa: E402
    CodeGenerator,
    INTENT_NAMES,
    analyze_intent,
    build_template_code,
    check_template_coverage,
    detect_template_model_type,
    extract_code,
    should_ask_router,
)
from analytics import basic_stats  # noqa: E402
from sandbox.executor import SandboxPolicy, execute_code, validate_code  # noqa: E402


class _StubLLMClient:
    """最小化 LLM 桩对象：用于验证 LLM 分支（不产生真实网络请求）。"""

    def __init__(self, code: str, router_use_template: bool = True, router_error: bool = False) -> None:
        self.is_available = True
        self.code_model = "stub-code-model"
        self._code = code
        self.calls = 0          # generate_code（代码生成）调用次数
        self.router_calls = 0   # chat（路由复核）调用次数
        self._router_use_template = router_use_template
        self._router_error = router_error

    def generate_code(self, prompt: str, system_prompt: str | None = None, **kwargs) -> dict:
        self.calls += 1
        return {
            "status": "success",
            "data": {"content": f"```python\n{self._code}\n```", "model": "stub-code-model"},
            "message": "ok",
            "error": None,
            "charts": [],
        }

    def chat(self, messages, model=None, temperature=0.0, max_tokens=None, response_format=None, **kwargs) -> dict:
        """模拟路由复核调用（JSON 模式返回二选一判定）。"""
        self.router_calls += 1
        if self._router_error:
            return {"status": "error", "data": None, "message": "复核失败", "error": "500", "charts": []}
        content = json.dumps(
            {"use_template": self._router_use_template, "reason": "桩服务判定理由"},
            ensure_ascii=False,
        )
        return {
            "status": "success",
            "data": {"content": content, "model": model or self.code_model},
            "message": "ok",
            "error": None,
            "charts": [],
        }


# --------------------------------------------------------------------------- #
# 意图解析
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "question, expected",
    [
        ("请计算 inlet_temp_c 的均值和标准差", "describe_stats"),
        ("分析各变量之间的相关性并给出热力图", "correlation"),
        ("入口温度与出口温度之间是否存在时滞", "time_lag"),
        ("用 3σ 原则检测数据中的异常点", "anomaly"),
        ("训练一个随机森林模型预测 yield_pct", "ml_train"),
        ("检查一下数据缺失情况", "missing_values"),
        ("数据里有哪些字段，做个总览", "overview"),
    ],
)
def test_analyze_intent(question: str, expected: str) -> None:
    result = analyze_intent(question)
    assert result["status"] == "success"
    assert result["data"]["intent"] == expected
    assert result["data"]["intent_name"] == INTENT_NAMES[expected]


def test_analyze_intent_empty_question() -> None:
    result = analyze_intent("   ")
    assert result["status"] == "error"


# --------------------------------------------------------------------------- #
# 代码抽取
# --------------------------------------------------------------------------- #
def test_extract_code_from_fence() -> None:
    text = "说明文字\n```python\nimport pandas as pd\nprint(1)\n```\n结束"
    assert extract_code(text).startswith("import pandas")


def test_extract_code_without_fence() -> None:
    text = "以下是代码：\nimport numpy as np\nvalue = np.mean([1, 2])\n"
    assert extract_code(text).startswith("import numpy")


def test_extract_code_prefers_longest_block() -> None:
    text = "```python\nx = 1\n```\n说明\n```python\nimport pandas as pd\ny = pd.Series([1, 2])\n```"
    assert "pandas" in extract_code(text)


# --------------------------------------------------------------------------- #
# 模板代码：安全性 + 可执行性（端到端离线验证）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("intent", list(INTENT_NAMES))
def test_template_code_is_safe(intent: str) -> None:
    code = build_template_code(intent, "测试问题", schema=None, data_file=None)
    is_valid, violations = validate_code(code)
    assert is_valid, f"模板 {intent} 未通过安全校验：{violations}"


#: 各意图对应的自然语言提问（模板代码据此匹配字段）
_INTENT_QUESTIONS = {
    "describe_stats": "统计 inlet_temp_c 的均值与标准差",
    "correlation": "分析 inlet_temp_c 与 outlet_temp_c 的相关性",
    "time_lag": "inlet_temp_c 与 outlet_temp_c 之间的时滞",
    "anomaly": "检测 inlet_temp_c 的异常点",
    "ml_train": "训练模型预测 yield_pct",
    "ml_predict": "模型推理预测",
    "missing_values": "检查缺失值",
    "overview": "数据总览",
}


@pytest.mark.parametrize("intent", list(INTENT_NAMES))
def test_template_code_executes_end_to_end(intent: str, project_data_file: Path, sample_frame) -> None:
    """每个意图的模板代码都必须在沙箱中真实跑通，并产出结构化结果。"""
    schema = basic_stats.detect_schema(sample_frame)["data"]
    code = build_template_code(intent, _INTENT_QUESTIONS[intent], schema, project_data_file.name)
    result = execute_code(
        code,
        policy=SandboxPolicy(timeout=120, keep_workspace=False),
        context={"data_file": project_data_file.name},
    )
    assert result["status"] == "success", f"{intent} 执行失败：{result['data']['stderr']}"
    payload = result["data"]["result"]
    assert isinstance(payload, dict)
    assert payload.get("intent") == intent
    assert payload.get("error") is None, f"{intent} 业务层返回错误：{payload.get('error')}"


# --------------------------------------------------------------------------- #
# 生成器：模板模式与 LLM 模式
# --------------------------------------------------------------------------- #
def test_generator_falls_back_to_template_without_llm() -> None:
    generator = CodeGenerator(llm_client=None)
    result = generator.generate("计算 inlet_temp_c 的平均值")
    assert result["status"] == "success"
    assert result["data"]["source"] == "template"
    assert result["data"]["intent"] == "describe_stats"
    assert result["data"]["violations"] == []


def test_generator_uses_llm_when_available() -> None:
    stub = _StubLLMClient(
        "import pandas as pd\n"
        "frame = load_data()\n"
        "emit_result({'rows': int(frame.shape[0])})\n"
    )
    generator = CodeGenerator(llm_client=stub)
    # 该需求含"筛选 + 大于"自定义逻辑，超出自定义模板覆盖范围 -> 走 LLM
    result = generator.generate("筛选 inlet_temp_c 大于 245 的数据并统计行数")
    assert result["status"] == "success"
    assert result["data"]["source"] == "llm"
    assert result["data"]["model"] == "stub-code-model"
    assert stub.calls == 1
    assert result["data"]["template_covered"] is False


def test_generator_repairs_unsafe_llm_code() -> None:
    """LLM 首次给出不安全代码时，生成器应触发修复并最终得到安全代码。"""

    class _RepairingStub(_StubLLMClient):
        def generate_code(self, prompt, system_prompt=None, **kwargs):
            self.calls += 1
            if self.calls == 1:
                body = "import os\nprint(os.getcwd())\nemit_result({'x': 1})\n"
            else:
                body = "import pandas as pd\nemit_result({'x': 1})\n"
            return {
                "status": "success",
                "data": {"content": f"```python\n{body}\n```", "model": "stub"},
                "message": "ok",
                "error": None,
                "charts": [],
            }

    stub = _RepairingStub("")
    generator = CodeGenerator(llm_client=stub, max_repair=2)
    result = generator.generate("按 grade 分组分别统计 inlet_temp_c 的均值")  # 分组逻辑 -> 走 LLM
    assert result["status"] == "success"
    assert result["data"]["repair_attempts"] >= 1
    assert "import os" not in result["data"]["code"]


def test_generator_template_when_llm_fails() -> None:
    """LLM 调用失败时必须自动回落模板，而不是整体失败。"""

    class _FailingStub:
        is_available = True

        def generate_code(self, prompt, system_prompt=None, **kwargs):
            return {"status": "error", "data": None, "message": "接口不可用", "error": "502", "charts": []}

    generator = CodeGenerator(llm_client=_FailingStub())
    # 含"筛选"自定义逻辑 -> 尝试调用 LLM，LLM 失败后回落模板
    result = generator.generate("筛选 reactor_pressure_mpa 大于 3.2 的数据并分析相关性")
    assert result["status"] == "success"
    assert result["data"]["source"] == "template"
    assert result["data"]["intent"] == "correlation"
    assert result["data"]["generation_strategy"] == "template-fallback"


# --------------------------------------------------------------------------- #
# 模板优先策略（template-first）回归测试
# --------------------------------------------------------------------------- #
#: 必须优先命中内置模板的标准场景（对应需求中的 5 类标准分析）
_STANDARD_SCENARIOS = (
    ("计算出口温度的均值、标准差、最大值和最小值", "describe_stats"),
    ("分析各变量之间的相关性，找出强相关特征对", "correlation"),
    ("分析 inlet_temp_c 与 outlet_temp_c 之间的时滞", "time_lag"),
    ("用 3σ 原则检测 inlet_temp_c 的异常点", "anomaly"),
    ("训练线性回归模型预测 yield_pct", "ml_train"),
    ("训练决策树模型预测 yield_pct", "ml_train"),
    ("训练随机森林模型预测 yield_pct", "ml_train"),
    ("检查数据的缺失值情况", "missing_values"),
)


@pytest.mark.parametrize("question, expected_intent", _STANDARD_SCENARIOS)
def test_standard_scenarios_use_local_template(question: str, expected_intent: str) -> None:
    """标准场景必须优先命中本地模板，且完全不调用 LLM（不消耗 API 额度）。"""
    stub = _StubLLMClient("import pandas as pd\nemit_result({'llm': True})\n")
    generator = CodeGenerator(llm_client=stub)
    result = generator.generate(question)

    assert result["status"] == "success"
    assert result["data"]["source"] == "template", f"{question} 应命中模板，实际 {result['data']['source']}"
    assert result["data"]["intent"] == expected_intent
    assert result["data"]["template_covered"] is True
    assert result["data"]["llm_skipped"] is True
    assert stub.calls == 0, "标准场景不应调用 LLM，避免消耗 API 额度"
    assert stub.router_calls == 0, "标准场景不应触发智能路由复核，避免无谓调用"
    assert result["data"]["router_used"] is False
    is_valid, violations = validate_code(result["data"]["code"])
    assert is_valid, f"模板代码未通过安全校验：{violations}"


def test_router_not_triggered_for_app_sample_questions() -> None:
    """界面上给出的 6 个示例问题都不应触发路由复核（保证零额外额度）。"""
    samples = (
        "请统计 inlet_temp_c 的均值、标准差、最大值和最小值",
        "分析各变量之间的相关性，找出强相关特征对",
        "inlet_temp_c 与 outlet_temp_c 之间是否存在时滞，滞后多少步",
        "用 3σ 原则检测 inlet_temp_c 和 outlet_temp_c 的异常点",
        "训练随机森林模型预测 yield_pct，并输出 R² 与特征重要性",
        "检查数据的缺失值情况并评估数据质量",
    )
    for question in samples:
        stub = _StubLLMClient("import pandas as pd\nemit_result({'llm': True})\n")
        result = CodeGenerator(llm_client=stub).generate(question)
        assert result["data"]["source"] == "template", question
        assert stub.router_calls == 0, f"示例问题不应触发路由复核：{question}"


@pytest.mark.parametrize(
    "question, expect_trigger",
    [
        ("把数据切成若干段，统计每段的均值", True),      # 可疑动词「切」
        ("帮我看看这批数据", False),
        ("计算 inlet_temp_c 的均值、标准差和最大值", False),
        ("分析各变量之间的相关性", False),
    ],
)
def test_should_ask_router_triggers(question: str, expect_trigger: bool) -> None:
    """路由触发判定：可疑动词触发，标准问法不触发，避免无谓的额度消耗。"""
    need, reason = should_ask_router(question)
    assert need is expect_trigger, f"{question} -> {need}（{reason}）"


def test_router_downgrades_ambiguous_request_to_llm() -> None:
    """规则判定命中模板但句子可疑时，大模型复核判定为定制需求 -> 改走 LLM。"""
    stub = _StubLLMClient(
        "import pandas as pd\nemit_result({'segments': 8})\n",
        router_use_template=False,
    )
    generator = CodeGenerator(llm_client=stub)
    result = generator.generate("把数据切成若干段，统计每段的均值")

    assert result["status"] == "success"
    assert result["data"]["source"] == "llm"
    assert result["data"]["router_used"] is True
    assert "大模型路由复核" in result["data"]["template_reason"]
    assert stub.router_calls == 1 and stub.calls == 1


def test_router_confirms_template_when_llm_says_ok() -> None:
    """复核判定模板可完成时，仍使用本地模板（只花一次约 100 token 的判定调用）。"""
    stub = _StubLLMClient("import pandas as pd\nemit_result({'llm': True})\n", router_use_template=True)
    generator = CodeGenerator(llm_client=stub)
    result = generator.generate("把数据切成若干段，统计每段的均值")

    assert result["data"]["source"] == "template"
    assert result["data"]["router_used"] is True
    assert stub.router_calls == 1
    assert stub.calls == 0, "复核确认模板可用时不应调用代码生成"


def test_router_failure_keeps_rule_decision() -> None:
    """路由复核调用失败时，保守沿用规则判定（走本地模板），不额外消耗代码生成额度。"""
    stub = _StubLLMClient("import pandas as pd\nemit_result({'llm': True})\n", router_error=True)
    generator = CodeGenerator(llm_client=stub)
    result = generator.generate("把数据切成若干段，统计每段的均值")

    assert result["data"]["source"] == "template"
    assert result["data"]["router_used"] is False
    assert stub.calls == 0


def test_router_can_be_disabled() -> None:
    """关闭智能路由后回到纯规则判定，不产生任何复核调用。"""
    stub = _StubLLMClient("import pandas as pd\nemit_result({'llm': True})\n", router_use_template=False)
    generator = CodeGenerator(llm_client=stub, template_router=False)
    result = generator.generate("把数据切成若干段，统计每段的均值")

    assert result["data"]["source"] == "template"
    assert result["data"]["router_used"] is False
    assert stub.router_calls == 0 and stub.calls == 0


@pytest.mark.parametrize(
    "content, expected",
    [
        ('{"use_template": true, "reason": "标准统计"}', True),
        ('{"use_template": false, "reason": "需分批"}', False),
        ('说明\n{"use_template": "false", "reason": "需分批"}\n结束', False),
    ],
)
def test_parse_router_json(content: str, expected: bool) -> None:
    """路由复核返回值解析应容忍外层文字与字符串布尔值。"""
    parsed = CodeGenerator._parse_router_json(content)
    assert parsed is not None and parsed["use_template"] is expected


def test_parse_router_json_rejects_garbage() -> None:
    """无法解析的返回应判定为未获得结论（沿用规则结果）。"""
    assert CodeGenerator._parse_router_json("抱歉，我无法判断") is None
    assert CodeGenerator._parse_router_json("") is None


@pytest.mark.parametrize(
    "question, expected_limitation",
    [
        ("筛选反应器压力大于3.2mpa的数据，统计出口温度均值", "数据筛选/过滤条件"),
        ("按 grade 分组统计 inlet_temp_c 的均值", "分组聚合"),
        ("把 inlet_temp_c 归一化后再做统计", "派生字段/特征工程"),
        ("预测未来 7 天的 yield_pct 趋势", "未来预测（模板仅做训练与评估）"),
        ("用 2.5σ 阈值检测异常点并剔除异常行", "数据筛选/过滤条件"),
        ("按天统计 inlet_temp_c 的均值", "窗口/批次聚合（按行数或时间分批统计）"),
        ("每 100 条数据计算一次蒸汽流量均值", "窗口/批次聚合（按行数或时间分批统计）"),
        ("画出 yield_pct 的箱线图", "图表类型超出模板产出"),
        ("计算 inlet_temp_c 与 outlet_temp_c 的温差", "派生字段/特征工程"),
        ("训练分类模型预测 product_grade 等级", "算法类型超出模板范围"),
        ("对数据做聚类分析", "算法类型超出模板范围"),
    ],
)
def test_custom_requests_fall_back_to_llm(question: str, expected_limitation: str) -> None:
    """含自定义逻辑（筛选/分组/派生字段等）的需求应判定为超出模板覆盖范围。"""
    coverage = check_template_coverage(question)
    assert coverage["covered"] is False
    assert expected_limitation in str(coverage["limitation"])
    assert expected_limitation in coverage["reason"]


def test_batched_scatter_request_is_routed_to_llm() -> None:
    """按行窗口分批 + 两变量散点图的需求：意图应识别为相关性分析，且判定为超出模板覆盖范围。

    该场景既可以按批次聚合，又要求模板不产出的散点图，属于典型的「需要 LLM 定制生成」需求。
    """
    question = "按每50行数据为一个批次，计算每个批次的产率平均值和蒸汽流量平均值，做散点图看两者的关系"
    intent = analyze_intent(question)
    coverage = check_template_coverage(question, intent)
    assert intent["data"]["intent"] == "correlation", "「散点图/关系」应识别为相关性分析意图"
    assert coverage["covered"] is False
    assert coverage["limitation"] == "窗口/批次聚合（按行数或时间分批统计）"

    stub = _StubLLMClient("import pandas as pd\nemit_result({'batches': 15, 'r': 0.62})\n")
    result = CodeGenerator(llm_client=stub).generate(question)
    assert result["data"]["source"] == "llm"
    assert stub.calls == 1


def test_custom_request_uses_llm_code_generation() -> None:
    """超出模板覆盖范围时，实际生成代码应来自 LLM。"""
    stub = _StubLLMClient("import pandas as pd\nemit_result({'filtered': 387})\n")
    generator = CodeGenerator(llm_client=stub)
    result = generator.generate("筛选反应器压力大于3.2mpa的数据，统计出口温度均值")

    assert result["status"] == "success"
    assert result["data"]["source"] == "llm"
    assert result["data"]["template_covered"] is False
    assert result["data"]["generation_strategy"] == "llm-fallback"
    assert stub.calls == 1


def test_explicit_llm_request_overrides_template_priority() -> None:
    """用户在提问中显式要求用大模型生成时，即使标准场景也应走 LLM。"""
    stub = _StubLLMClient("import pandas as pd\nemit_result({'llm': True})\n")
    generator = CodeGenerator(llm_client=stub)
    result = generator.generate("计算出口温度的均值，必须调用大语言模型生成代码，不使用内置模板")

    assert result["status"] == "success"
    assert result["data"]["source"] == "llm"
    assert result["data"]["force_llm"] is True
    assert stub.calls == 1


def test_unmatched_intent_goes_to_llm() -> None:
    """未命中任何模板关键词（模板匹配失败）时应交由 LLM 处理。"""
    stub = _StubLLMClient("import pandas as pd\nemit_result({'ok': 1})\n")
    generator = CodeGenerator(llm_client=stub)
    result = generator.generate("帮我看看这批数据")  # 未命中关键词 -> 置信度 0

    assert result["status"] == "success"
    assert result["data"]["template_covered"] is False
    assert result["data"]["source"] == "llm"


@pytest.mark.parametrize(
    "question, expected_model_type",
    [
        ("训练线性回归模型预测 yield_pct", "linear_regression"),
        ("用决策树预测 yield_pct", "decision_tree"),
        ("训练随机森林模型预测 yield_pct", "random_forest"),
        ("训练模型预测 yield_pct", "random_forest"),  # 未指定时默认随机森林
    ],
)
def test_template_model_type_is_detected(question: str, expected_model_type: str) -> None:
    """标准建模场景应识别出用户指定的模型类型，并写入模板代码。"""
    assert detect_template_model_type(question) == expected_model_type
    code = build_template_code("ml_train", question)
    assert f'TEMPLATE_MODEL_TYPE = "{expected_model_type}"' in code
