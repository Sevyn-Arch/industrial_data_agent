"""代码生成器：自然语言意图解析 + Python 分析代码生成（含静态校验与自修复）。

两种工作模式
========================================================================

* **LLM 模式**：调用 DeepSeek ``CODE_MODEL``，按严格约束生成分析代码；结果先经过沙箱
  静态安全校验，若校验失败或执行报错，则把错误信息回灌给 LLM 自动修复（默认最多 2 轮）。
* **本地模板模式**：未配置 API Key 或 LLM 调用失败时，使用内置规则模板生成等价代码，
  保证系统在离线/无密钥环境下依旧可以完整跑通「提问 -> 执行 -> 报告」闭环。
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Mapping

from analytics import fail, get_logger, ok
from sandbox.executor import SandboxPolicy, check_code_safety, validate_code

__all__ = [
    "CodeGenerator",
    "analyze_intent",
    "check_template_coverage",
    "should_ask_router",
    "detect_template_model_type",
    "extract_code",
    "generate_analysis_code",
    "build_template_code",
    "INTENT_NAMES",
    "TEMPLATE_INTENTS",
    "TEMPLATE_LIMITATIONS",
    "TEMPLATE_CAPABILITY_SUMMARY",
    "ROUTER_SYSTEM_PROMPT",
    "SYSTEM_PROMPT",
    "ANALYTICS_API_SUMMARY",
]

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# 一、意图体系
# --------------------------------------------------------------------------- #
INTENT_NAMES: dict[str, str] = {
    "describe_stats": "基础描述性统计",
    "correlation": "相关性分析",
    "time_lag": "时序时滞分析",
    "anomaly": "异常检测（3σ）",
    "ml_train": "机器学习建模",
    "ml_predict": "模型推理预测",
    "missing_values": "缺失值质量检查",
    "overview": "数据总览",
}

#: 内置模板覆盖的意图（这些意图的「标准用法」优先走本地模板，不消耗 API 额度）
TEMPLATE_INTENTS: frozenset[str] = frozenset(
    {
        "describe_stats",   # 基础描述性统计
        "correlation",      # Pearson 相关性分析
        "time_lag",         # 时序时滞分析
        "anomaly",          # 3σ 异常检测
        "ml_train",         # 标准机器学习建模（线性回归 / 决策树 / 随机森林）
        "ml_predict",       # 使用最新模型推理
        "missing_values",   # 缺失值检查
        "overview",         # 数据总览
    }
)

#: 「模板不支持、需要 LLM 定制生成」的需求特征（命中任意一条即降级调用 LLM）
TEMPLATE_LIMITATIONS: tuple[tuple[str, str], ...] = (
    (r"筛选|过滤|剔除|排除|只看|限定|where|filter", "数据筛选/过滤条件"),
    (r"大于|小于|超过|低于|高于|不低于|不高于|不少于|不多于|介于|区间|范围在|!=|>=|<=|==", "数值比较条件"),
    (r"按.{1,8}(分组|拆分|分类)|分组统计|分类统计|分别统计|透视|交叉表|group\s*by|groupby", "分组聚合"),
    (
        r"批次|分批|分箱|分块|按每\s*\d+\s*(行|条|个|天|小时|分钟|秒)|每\s*\d+\s*(行|条|个)"
        r"|滑窗|滑动窗口|滚动窗口|窗口|window|逐行|逐条|逐日|逐月|逐小时|每天|每日|每小时"
        r"|按天|按日|按月|按小时|按班次|按周",
        "窗口/批次聚合（按行数或时间分批统计）",
    ),
    (r"排序|排名|前\s*\d+|top\s*\d+|最大的\s*\d+|最小的\s*\d+|抽样|采样|随机抽取", "排序/抽样"),
    (
        r"新增列|计算字段|衍生|派生|归一化|标准化|取对数|平方|开方|取倒数|比值|占比计算|自定义公式|构造特征"
        r"|差值|之差|乘积|之和|温差|压差|之比|除以|乘以|相加|相减",
        "派生字段/特征工程",
    ),
    (r"如果.+就|条件判断|分段|分箱|阈值判断", "条件分支逻辑"),
    (r"移动平均|滑动|滚动|rolling|同比|环比|重采样|差分|平滑|插值|填充缺失|缺失值填充", "时序变换/缺失处理"),
    (r"调参|网格搜索|超参数|n_estimators|max_depth|min_samples|对比模型|多个模型", "自定义建模参数"),
    (r"采样间隔|时间列|时间戳列|以.{1,6}为时间", "自定义时间轴参数"),
    (r"预测未来|未来\s*\d|明天|下周|下个月|趋势预测|外推", "未来预测（模板仅做训练与评估）"),
    (r"合并|join|关联.{0,4}表|多表", "多表关联"),
    (
        r"散点图|scatter|箱线图|折线图|饼图|柱状图|条形图|面积图|雷达图|甘特图|趋势图",
        "图表类型超出模板产出（模板仅输出分布直方图/相关性热力图/时滞曲线/异常分布/建模评估图）",
    ),
    (
        r"分类模型|分类预测|聚类|时间序列预测|arima|lstm|神经网络|深度学习|xgboost|lightgbm|梯度提升|贝叶斯",
        "算法类型超出模板范围（模板仅支持回归类建模）",
    ),
)

#: 用户显式要求「必须用大模型生成」时，跳过模板优先策略
FORCE_LLM_PATTERN = re.compile(
    r"必须(调用|使用|用|由).{0,12}(大语言模型|大模型|llm)"
    r"|不使用(内置)?模板|不要用(内置)?模板|必须生成代码|用\s*(llm|大模型|大语言模型)\s*生成",
    re.IGNORECASE,
)

#: 模板支持的建模类型识别（用于「标准机器学习建模」场景）
TEMPLATE_MODEL_TYPE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("linear_regression", r"线性回归|linear\s*regression|\blr\b|最小二乘"),
    ("decision_tree", r"决策树|decision\s*tree|\bdt\b|\btree\b"),
    ("random_forest", r"随机森林|random\s*forest|\brf\b|森林"),
)

# --------------------------------------------------------------------------- #
# 大模型路由复核（长尾表述兜底）
# --------------------------------------------------------------------------- #
#: 各模板的能力边界简述（供路由判定器参考，避免"想当然"判断）
TEMPLATE_CAPABILITY_SUMMARY: dict[str, str] = {
    "describe_stats": "对指定列/全部数值列计算描述性统计并绘制分布直方图；不支持筛选、分批、分组、派生列",
    "correlation": "计算多列相关性矩阵、挖掘强相关特征对并绘制热力图；不支持只对指定两列做自定义散点分析，不支持筛选后分析",
    "time_lag": "对两个指定列做互相关时滞分析；不支持筛选、分批、重采样后再分析",
    "anomaly": "按 3σ 原则（默认阈值 3）对指定列做异常检测并高亮异常点；不支持自定义阈值筛选、不支持剔除异常后导出",
    "ml_train": "用线性回归/决策树/随机森林对目标列做标准建模与评估；不支持调参搜索、不支持分类/聚类",
    "ml_predict": "用已保存的最新模型对当前数据做批量推理；不支持指定模型文件、不支持特征改写",
    "missing_values": "统计各列缺失数量与缺失率并绘图；不支持缺失值填充",
    "overview": "输出数据规模、字段类型与缺失概况；不做具体分析",
}

#: 路由触发用「可疑动词」：出现这类词说明可能要做模板之外的变换
_ROUTER_SUSPICIOUS_PATTERN = re.compile(
    r"切|拆|割|汇总|归集|换算|组合|拼接|对齐|累计|累积|编号|打标|重塑|展开|透视|滑动|滚动"
    r"|挑选|挑出|对比|比较|排名|映射|归并|分片|若干|任意|自定义|按需|逐个|逐条"
)

#: 路由触发用「已知词汇」：这些词属于标准模板可表达的范围，判定可疑度时应剔除
_ROUTER_KNOWN_WORDS: tuple[str, ...] = (
    "请", "帮我", "一下", "看看", "看", "分别", "同时", "然后", "并且", "并", "再", "以及", "和", "与", "及", "或",
    "的", "了", "在", "为", "对", "把", "给", "出", "上", "中", "里",
    "计算", "统计", "分析", "绘制", "画出", "画", "做", "给出", "输出", "生成", "列出", "展示", "查看", "检查",
    "评估", "了解", "判断", "识别", "检测", "找出", "得到", "获取",
    "均值", "平均", "平均值", "标准差", "方差", "中位数", "最大值", "最小值", "极值", "四分位", "分位数", "描述",
    "指标", "统计量", "概况", "总览", "情况", "数据", "数据集", "字段", "列", "行", "记录", "样本",
    "各", "之间", "强", "弱", "极强", "中等", "单一", "多",
    "相关", "相关性", "相关关系", "相关系数", "关联", "热力图", "矩阵", "散点图", "关系", "两者", "变量", "特征",
    "特征重要性", "重要性", "时滞", "滞后", "领先", "超前", "延时", "步", "步数", "多少", "是否", "存在", "周期",
    "异常", "异常点", "离群", "离群点", "跳变", "偏离", "原则", "倍", "σ", "3σ", "sigma",
    "模型", "训练", "预测", "建模", "回归", "线性", "决策树", "随机森林", "森林", "交叉验证", "折", "拟合", "评估",
    "缺失", "缺失值", "缺失率", "质量", "完整", "完整性", "空值", "nan",
    "产率", "蒸汽流量", "出口温度", "入口温度", "压力", "流量", "温度", "产量", "质量指标",
    "单位", "秒", "分钟", "小时", "天", "℃，", "mpa", "kg", "t", "h", "℃", "秒）",
)

#: 路由判定提示词（只做二选一判定，不生成代码，约 100 token）
ROUTER_SYSTEM_PROMPT: str = "\n".join(
    [
        "你是数据分析需求路由判定器。给定一条用户需求，判断它能否由系统已有的标准模板完整完成。",
        "判定原则：只要需求包含模板做不到的步骤（如按行数/时间分批、筛选、分组、派生新列、",
        "指定两列自定义散点分析、指定模板不产出的图表类型、调参或分类聚类等），就必须判定为 false。",
        "",
        "严格只输出 JSON，格式：{\"use_template\": true 或 false, \"reason\": \"不超过 40 字的中文理由\"}",
    ]
)

#: 意图识别关键词（命中即累加权重，取分数最高者）
_INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "ml_train": (
        "建模", "训练", "模型", "回归", "预测模型", "随机森林", "决策树", "线性回归",
        "random forest", "decision tree", "linear regression", "train", "fit",
    ),
    "ml_predict": ("推理", "预测新数据", "调用模型", "加载模型", "predict", "inference"),
    "time_lag": ("时滞", "滞后", "领先", "延时", "lag", "cross correlation", "互相关", "超前"),
    "anomaly": ("异常", "离群", "outlier", "3σ", "3sigma", "跳变", "偏离"),
    "correlation": (
        "相关", "相关性", "热力图", "correlation", "关联",
        "散点图", "scatter", "关系",  # 「看两者的关系 / 做散点图」属相关性分析范畴
    ),
    "missing_values": ("缺失", "空值", "缺测", "nan", "missing", "数据质量"),
    "describe_stats": (
        "统计", "均值", "平均", "标准差", "方差", "中位数", "最大值", "最小值", "分位数",
        "描述", "summary", "statistics", "max", "min", "median", "mean", "std",
    ),
    "overview": ("总览", "概况", "字段", "数据类型", "有哪些列", "overview", "schema"),
}


def analyze_intent(question: str) -> dict[str, Any]:
    """基于规则解析自然语言分析意图（无需 LLM，结果确定可复现）。

    Args:
        question: 用户自然语言问题。

    Returns:
        dict: 统一结果包，``data`` 含 ``intent``、``intent_name``、``matched_keywords``、``confidence``。
    """
    try:
        text = str(question or "").strip()
        if not text:
            return fail("问题为空，无法解析分析意图")
        lowered = text.lower()

        scores: dict[str, int] = {}
        matched: dict[str, list[str]] = {}
        for intent, keywords in _INTENT_KEYWORDS.items():
            hits = [keyword for keyword in keywords if keyword.lower() in lowered]
            if hits:
                scores[intent] = len(hits)
                matched[intent] = hits

        if not scores:
            data = {
                "intent": "overview",
                "intent_name": INTENT_NAMES["overview"],
                "matched_keywords": [],
                "confidence": 0.0,
            }
            return ok(data, "未命中明确意图关键词，默认执行「数据总览」")

        order = list(_INTENT_KEYWORDS)
        intent = max(scores, key=lambda key: (scores[key], -order.index(key)))
        total = sum(scores.values())
        data = {
            "intent": intent,
            "intent_name": INTENT_NAMES[intent],
            "matched_keywords": matched[intent],
            "confidence": round(scores[intent] / max(1, total), 4),
            "all_scores": scores,
        }
        return ok(data, f"解析意图为「{INTENT_NAMES[intent]}」（命中关键词：{'、'.join(matched[intent][:5])}）")
    except Exception as exc:  # noqa: BLE001
        logger.exception("analyze_intent 执行失败")
        return fail("意图解析失败", exc)


# --------------------------------------------------------------------------- #
# 二、代码抽取与提示词素材
# --------------------------------------------------------------------------- #
_CODE_FENCE_PATTERN = re.compile(r"```(?:python|py|python3)?\s*(.*?)```", re.DOTALL)


def detect_template_model_type(question: str, default: str = "random_forest") -> str:
    """识别问题中指定的建模类型（线性回归 / 决策树 / 随机森林），命中不到时返回默认值。"""
    try:
        text = str(question or "").lower()
        for model_type, pattern in TEMPLATE_MODEL_TYPE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return model_type
        return default
    except Exception:  # noqa: BLE001
        return default


def check_template_coverage(question: str, intent_result: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """判断需求是否落在内置模板的覆盖范围内（决定「模板优先」还是「LLM 生成」）。

    判定规则（按优先级）：

    1. 用户显式要求由大模型生成（如"必须调用大语言模型生成代码、不使用内置模板"）→ 不覆盖；
    2. 意图不在 :data:`TEMPLATE_INTENTS`（无对应模板）→ 不覆盖；
    3. 意图置信度为 0（未命中任何模板关键词，说明模板匹配失败）→ 不覆盖；
    4. 命中 :data:`TEMPLATE_LIMITATIONS` 中任一条（筛选、比较、分组、派生字段、自定义参数等
       模板无法表达的逻辑）→ 不覆盖。

    Args:
        question: 用户自然语言需求。
        intent_result: :func:`analyze_intent` 的结果包；``None`` 时内部自行解析。

    Returns:
        dict: ``{"covered": bool, "reason": str, "limitation": str | None,
        "force_llm": bool, "intent": str, "confidence": float}``。
    """
    outcome: dict[str, Any] = {
        "covered": True,
        "reason": "标准场景，命中内置模板",
        "limitation": None,
        "force_llm": False,
        "intent": None,
        "confidence": None,
    }
    try:
        text = str(question or "").strip()
        if not text:
            outcome.update({"covered": False, "reason": "需求为空，无法匹配模板"})
            return outcome

        forced = FORCE_LLM_PATTERN.search(text)
        if forced:
            outcome.update(
                {
                    "covered": False,
                    "force_llm": True,
                    "reason": f"用户显式要求由大模型生成代码（命中「{forced.group(0)}」）",
                }
            )
            return outcome

        if intent_result is None:
            intent_result = analyze_intent(text)
        data = intent_result.get("data") or {}
        intent = str(data.get("intent") or "")
        confidence = data.get("confidence")
        outcome["intent"] = intent
        outcome["confidence"] = confidence

        if intent not in TEMPLATE_INTENTS:
            outcome.update({"covered": False, "reason": f"意图「{intent or '未知'}」没有对应内置模板"})
            return outcome

        # 先看是否命中「模板明确不支持」的需求特征，这样给出的原因更具体
        for pattern, label in TEMPLATE_LIMITATIONS:
            matched = re.search(pattern, text, re.IGNORECASE)
            if matched:
                outcome.update(
                    {
                        "covered": False,
                        "limitation": label,
                        "reason": f"检测到模板不支持的需求：{label}（命中「{matched.group(0)}」）",
                    }
                )
                return outcome

        if confidence == 0:
            outcome.update({"covered": False, "reason": "未命中任何模板关键词，模板匹配失败"})
            return outcome
        return outcome
    except Exception as exc:  # noqa: BLE001 - 判定异常时保守交由 LLM 处理
        outcome.update({"covered": False, "reason": f"模板覆盖度判定异常：{exc}"})
        return outcome


def should_ask_router(question: str, schema: Mapping[str, Any] | None = None) -> tuple[bool, str]:
    """判断是否需要请大模型复核「该需求能否由标准模板完成」。

    仅对「规则判定为命中模板、但句子看起来可疑」的需求触发复核——因为漏判会让模板
    静默答非所问，而误触发只是多一次约 100 token 的判定请求，代价可控：

    1. 命中可疑动词（切/拆/汇总/换算/自定义/若干/逐个…）→ 复核；
    2. 剔除已知词汇与字段名后，剩余中文实词 ≥ 5 个字符（出现模板词表外的名词/动词）→ 复核；
    3. 需求长度 ≥ 45 字符（长复合需求）→ 复核。

    Args:
        question: 用户自然语言需求。
        schema: 字段画像（提供后会把字段名从"未知内容"中剔除）。

    Returns:
        tuple[bool, str]: ``(是否需要复核, 触发原因)``。
    """
    try:
        text = str(question or "").strip()
        if not text:
            return False, ""

        suspicious = _ROUTER_SUSPICIOUS_PATTERN.search(text)
        if suspicious:
            return True, f"出现模板词表外的动作词「{suspicious.group(0)}」"

        residue = text
        # 长词优先匹配，避免「相关性」被更短的「相关」先吃掉而残留「性」
        for word in sorted(_ROUTER_KNOWN_WORDS, key=len, reverse=True):
            residue = residue.replace(word, " ")
        if isinstance(schema, Mapping):
            for column in (schema.get("numeric_columns") or []) + (schema.get("datetime_columns") or []) + (
                schema.get("categorical_columns") or []
            ):
                residue = residue.replace(str(column), " ")
            for item in schema.get("columns") or []:
                if isinstance(item, Mapping):
                    residue = residue.replace(str(item.get("name") or ""), " ")
        residue = re.sub(r"[0-9A-Za-z_%.。，、,;；:：!！?？()（）\[\]【】\s\-/／+*=~≈×·σ′″°℃]+", "", residue)
        unknown = "".join(char for char in residue if "\u4e00" <= char <= "\u9fff")
        if len(unknown) >= 5:
            return True, f"出现模板词表外的内容「{unknown[:8]}」"

        if len(text) >= 45:
            return True, "需求较长（复合型需求，可能存在多步操作）"
        return False, ""
    except Exception:  # noqa: BLE001 - 触发判定异常时不额外消耗额度
        return False, ""


def _read_router_env(default: bool = True) -> bool:
    """读取环境变量 ``TEMPLATE_ROUTER``（1/true/yes/on 视为开启）。"""
    try:
        raw = str(os.environ.get("TEMPLATE_ROUTER", str(default))).strip().lower()
        if raw in ("1", "true", "yes", "y", "on"):
            return True
        if raw in ("0", "false", "no", "n", "off"):
            return False
        return bool(default)
    except Exception:  # noqa: BLE001
        return bool(default)


def extract_code(text: str) -> str:
    """从 LLM 回复中抽取 Python 代码（优先取最长代码块，否则从首个 import 处截取）。"""
    try:
        content = str(text or "")
        if not content.strip():
            return ""
        blocks = [block.strip() for block in _CODE_FENCE_PATTERN.findall(content) if block.strip()]
        if blocks:
            return max(blocks, key=len)
        lines = content.splitlines()
        start = 0
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")) or stripped.startswith(("df =", "data =", "frame =")):
                start = index
                break
        return "\n".join(lines[start:]).strip()
    except Exception:  # noqa: BLE001
        logger.exception("extract_code 执行失败")
        return ""


def _match_columns(question: str, schema: Mapping[str, Any] | None) -> list[str]:
    """从问题文本中匹配出数据表里真实存在的字段名（模板代码选择字段时使用）。"""
    try:
        if not schema:
            return []
        columns = [str(item.get("name")) for item in (schema.get("columns") or [])]
        lowered = str(question or "").lower()
        return [name for name in columns if name and name.lower() in lowered]
    except Exception:  # noqa: BLE001
        return []


def _schema_brief(schema: Mapping[str, Any] | None) -> str:
    """把字段画像压缩成便于提示词消费的简短文本。"""
    try:
        if not schema:
            return "（未提供字段信息，请先调用 basic_stats.detect_schema(df) 自行探测）"
        lines = [f"数据规模：{schema.get('n_rows')} 行 × {schema.get('n_columns')} 列"]
        if schema.get("numeric_columns"):
            lines.append(f"数值字段：{', '.join(map(str, schema['numeric_columns']))}")
        if schema.get("datetime_columns"):
            lines.append(f"时间字段：{', '.join(map(str, schema['datetime_columns']))}")
        if schema.get("categorical_columns"):
            lines.append(f"分类字段：{', '.join(map(str, schema['categorical_columns']))}")
        lines.append("字段明细：")
        for item in (schema.get("columns") or [])[:40]:
            lines.append(
                f"  - {item.get('name')}（{item.get('kind')}，缺失率 {item.get('missing_rate')}，"
                f"样例 {item.get('samples')}）"
            )
        return "\n".join(lines)
    except Exception:  # noqa: BLE001
        return "（字段信息解析失败，请使用 basic_stats.detect_schema(df) 自行探测）"


def build_code_prompt(question: str, schema: Mapping[str, Any] | None, data_file: str | None) -> str:
    """组装代码生成提示词。"""
    return (
        f"【数据文件】{data_file or '（未指定，请使用 load_data() 读取默认文件）'}\n\n"
        f"【数据结构】\n{_schema_brief(schema)}\n\n"
        f"【用户分析需求】{question}\n\n"
        "请生成完整可运行的 Python 分析代码。"
    )


def _template_columns(schema: Mapping[str, Any] | None, question: str) -> tuple[list[str], list[str]]:
    """返回 ``(匹配字段, 数值字段)``，用于模板代码的字段选择。"""
    numeric = [str(item) for item in ((schema or {}).get("numeric_columns") or [])]
    matched = _match_columns(question, schema)
    return matched, numeric


#: 供 LLM 参考的项目算法库 API 摘要（显著提升生成代码的可用性）
ANALYTICS_API_SUMMARY: str = "\n".join(
    [
        "所有函数返回统一结果包：status / data / message / error / charts。",
        "- from analytics import basic_stats",
        "  * basic_stats.detect_schema(df) -> data.numeric_columns、data.columns 字段画像",
        "  * basic_stats.describe_column(df, column) -> data.mean/std/min/max/median/q1/q3/skew/kurtosis",
        "  * basic_stats.describe_columns(df, columns=None) -> data.statistics{列名: 指标}",
        "  * basic_stats.value_counts(df, column, top_n=10)",
        "  * basic_stats.missing_report(df) -> data.columns[].missing_rate（含缺失率柱状图）",
        "  * basic_stats.plot_histogram(df, column, bins=30) -> charts 列表",
        "- from analytics import correlation",
        "  * correlation.correlation_matrix(df, columns=None, method='pearson')",
        "  * correlation.strong_correlations(df, columns=None, threshold=0.5)",
        "  * correlation.correlation_with_target(df, target_column, feature_columns=None)",
        "  * correlation.plot_scatter(df, x_column, y_column)",
        "- from analytics import time_lag",
        "  * time_lag.time_lag_analysis(df, x_column, y_column, max_lag=10, time_column=None, sample_interval=None)",
        "    -> data.best_lag（>0 表示 x 领先 y）、data.conclusion",
        "- from analytics import anomaly",
        "  * anomaly.detect_3sigma(df, column, threshold=3.0, side='both')",
        "  * anomaly.detect_3sigma_multi(df, columns=None, threshold=3.0)",
        "- from analytics import ml_models",
        "  * ml_models.train_model(df, target_column, feature_columns=None, model_type='random_forest',",
        "        test_size=0.2, cross_validation=5, scale_features=False)",
        "    -> data.metrics{mse,rmse,mae,r2,cv_r2_mean}、data.feature_importance、data.model_path",
        "  * ml_models.predict(model_path, df, feature_columns=None)",
        "  * ml_models.list_models()",
    ]
)


SYSTEM_PROMPT: str = "\n".join(
    [
        "你是一名工业数据分析代码生成专家，为「工业数据分析 Agent 系统」编写 Python 分析代码。",
        "",
        "【运行环境（系统已自动注入，直接使用即可）】",
        "- 已导入：pandas as pd、numpy as np、matplotlib.pyplot as plt、seaborn as sns，绘图后端已设为 Agg；",
        "- DATA_DIR / CHART_DIR / OUTPUT_DIR / MODEL_DIR：数据、图表、输出、模型目录（pathlib.Path）；",
        "- DATA_FILE：本次分析绑定的数据文件路径（可能为 None）；",
        "- load_data(name=None) -> DataFrame：读取 data 目录下的 CSV，name 省略时读取 DATA_FILE；",
        "- save_chart(name='chart.png', fig=None) -> 图片路径字符串：保存图表到 output/charts；",
        "- list_models() -> list：output/models 下已保存的模型路径（按修改时间升序）；",
        "- emit_result(payload)：输出结构化结果（必须调用一次，payload 需可 JSON 序列化）；",
        "- CONTEXT：本次运行上下文（含 data_file、question）。",
        "",
        "【硬性约束（违反将被安全沙箱拒绝执行）】",
        "1. 只允许 import：pandas、numpy、scipy、sklearn、matplotlib、seaborn、joblib、analytics、",
        "   math、statistics、json、datetime、re、itertools、functools、collections、pathlib；",
        "   严禁 import os / sys / subprocess / socket / shutil / importlib / pickle / ctypes / glob；",
        "2. 禁止使用 exec、eval、compile、__import__、open、input、getattr、setattr 等函数；",
        "3. 禁止访问任何双下划线（魔术）属性；禁止 while True 无界循环；禁止 assert；",
        "4. 禁止出现 data/ 与 output/ 之外的绝对路径；读写统一用 load_data / save_chart / joblib；",
        "5. 代码必须自包含、可直接运行、中文注释、PEP8 风格，长度控制在 150 行以内；",
        "6. 最后必须调用 emit_result({...}) 输出核心指标（数值请用 float()/int() 转换）；",
        "7. 图表请用 save_chart('有意义的名字.png') 保存，标题与坐标轴使用中文。",
        "",
        "【可用算法库（优先复用，避免重复实现）】",
        ANALYTICS_API_SUMMARY,
        "",
        "【输出格式】",
        "只输出一段 Python 代码（使用 ```python 代码块），不要输出任何解释性文字。",
    ]
)


# --------------------------------------------------------------------------- #
# 三、本地模板代码（离线可用，覆盖全部意图）
# --------------------------------------------------------------------------- #
_TEMPLATE_BODIES: dict[str, str] = {
    "describe_stats": """
columns = MATCHED_COLUMNS if MATCHED_COLUMNS else NUMERIC_COLUMNS[:8]
if not columns:
    emit_result({"intent": "describe_stats", "error": "数据中不存在数值型字段"})
else:
    stats = basic_stats.describe_columns(df, columns)
    charts = []
    for column in columns[:3]:
        hist = basic_stats.plot_histogram(df, column)
        if hist["status"] == "success":
            charts.extend(hist["charts"])
    emit_result({
        "intent": "describe_stats",
        "status": stats["status"],
        "message": stats["message"],
        "n_rows": int(df.shape[0]),
        "columns": columns[:8],
        "statistics": stats["data"].get("statistics", {}) if stats["status"] == "success" else {},
        "charts": charts,
    })
""",
    "correlation": """
columns = MATCHED_COLUMNS if len(MATCHED_COLUMNS) >= 2 else NUMERIC_COLUMNS
if len(columns) < 2:
    emit_result({"intent": "correlation", "error": "数值字段不足 2 个，无法做相关性分析"})
else:
    columns = columns[:12]
    matrix = correlation.correlation_matrix(df, columns=columns, method="pearson")
    strong = correlation.strong_correlations(df, columns=columns, threshold=0.5)
    emit_result({
        "intent": "correlation",
        "status": matrix["status"],
        "message": matrix["message"],
        "columns": columns,
        "strongest_pair": matrix["data"].get("strongest_pair") if matrix["status"] == "success" else None,
        "strong_pairs": strong["data"].get("pairs", [])[:10] if strong["status"] == "success" else [],
        "charts": matrix["charts"],
    })
""",
    "time_lag": """
x_col = MATCHED_COLUMNS[0] if MATCHED_COLUMNS else (NUMERIC_COLUMNS[0] if NUMERIC_COLUMNS else None)
y_col = MATCHED_COLUMNS[1] if len(MATCHED_COLUMNS) >= 2 else (NUMERIC_COLUMNS[1] if len(NUMERIC_COLUMNS) > 1 else None)
if x_col is None or y_col is None or x_col == y_col:
    emit_result({"intent": "time_lag", "error": "至少需要两个不同的数值字段才能做时滞分析"})
else:
    result = time_lag.time_lag_analysis(df, x_column=x_col, y_column=y_col, max_lag=10)
    payload = result["data"] if result["status"] == "success" else {}
    emit_result({
        "intent": "time_lag",
        "status": result["status"],
        "message": result["message"],
        "x": x_col,
        "y": y_col,
        "best_lag": payload.get("best_lag"),
        "best_correlation": payload.get("best_correlation"),
        "conclusion": payload.get("conclusion", result["message"]),
        "top_lags": payload.get("top_lags", []),
        "charts": result["charts"],
    })
""",
    "anomaly": """
columns = MATCHED_COLUMNS if MATCHED_COLUMNS else NUMERIC_COLUMNS[:4]
if not columns:
    emit_result({"intent": "anomaly", "error": "数据中不存在数值型字段，无法检测异常"})
else:
    result = anomaly.detect_3sigma_multi(df, columns=columns, threshold=3.0)
    summary = []
    if result["status"] == "success":
        for name, item in result["data"]["columns"].items():
            summary.append({
                "column": name,
                "n_anomalies": item["n_anomalies"],
                "anomaly_rate": item["anomaly_rate"],
                "lower_bound": item["lower_bound"],
                "upper_bound": item["upper_bound"],
                "top_anomalies": item["top_anomalies"][:5],
            })
    emit_result({
        "intent": "anomaly",
        "status": result["status"],
        "message": result["message"],
        "threshold": 3.0,
        "total_anomalies": result["data"].get("total_anomalies") if result["status"] == "success" else None,
        "columns": summary,
        "charts": result["charts"],
    })
""",
    "ml_train": """
target = MATCHED_COLUMNS[-1] if MATCHED_COLUMNS else (NUMERIC_COLUMNS[-1] if NUMERIC_COLUMNS else None)
pool = MATCHED_COLUMNS[:-1] if len(MATCHED_COLUMNS) > 1 else NUMERIC_COLUMNS
features = [name for name in pool if name != target]
if target is None or not features:
    emit_result({"intent": "ml_train", "error": "缺少可用的目标字段或数值特征，无法建模"})
else:
    result = ml_models.train_model(
        df,
        target_column=target,
        feature_columns=features,
        model_type=TEMPLATE_MODEL_TYPE,
        cross_validation=5,
    )
    payload = result["data"] if result["status"] == "success" else {}
    emit_result({
        "intent": "ml_train",
        "status": result["status"],
        "message": result["message"],
        "model_type": payload.get("model_type", TEMPLATE_MODEL_TYPE),
        "target": target,
        "features": features,
        "metrics": payload.get("metrics"),
        "feature_importance": payload.get("feature_importance", [])[:10],
        "model_path": payload.get("model_path"),
        "charts": result["charts"],
    })
""",
    "ml_predict": """
models = list_models()
if not models:
    emit_result({"intent": "ml_predict", "error": "未找到已保存模型，请先完成一次建模训练"})
else:
    target_model = models[-1]
    result = ml_models.predict(str(target_model), df)
    payload = result["data"] if result["status"] == "success" else {}
    emit_result({
        "intent": "ml_predict",
        "status": result["status"],
        "message": result["message"],
        "model_path": str(target_model),
        "n_predicted": payload.get("n_predicted"),
        "prediction_stats": payload.get("prediction_stats"),
        "predictions": payload.get("predictions", [])[:50],
    })
""",
    "missing_values": """
report = basic_stats.missing_report(df)
payload = report["data"] if report["status"] == "success" else {}
emit_result({
    "intent": "missing_values",
    "status": report["status"],
    "message": report["message"],
    "n_rows": payload.get("n_rows"),
    "n_columns_with_missing": payload.get("n_columns_with_missing"),
    "top_missing": payload.get("columns", [])[:10],
    "charts": report["charts"],
})
""",
    "overview": """
schema = basic_stats.detect_schema(df)
missing = basic_stats.missing_report(df, chart=False)
schema_payload = schema["data"] if schema["status"] == "success" else {}
missing_payload = missing["data"] if missing["status"] == "success" else {}
emit_result({
    "intent": "overview",
    "status": schema["status"],
    "message": schema["message"],
    "n_rows": schema_payload.get("n_rows"),
    "n_columns": schema_payload.get("n_columns"),
    "numeric_columns": schema_payload.get("numeric_columns", []),
    "datetime_columns": schema_payload.get("datetime_columns", []),
    "columns": schema_payload.get("columns", []),
    "n_columns_with_missing": missing_payload.get("n_columns_with_missing"),
    "charts": schema["charts"],
})
""",
}


def build_template_code(
    intent: str,
    question: str,
    schema: Mapping[str, Any] | None = None,
    data_file: str | None = None,
) -> str:
    """根据意图生成内置模板代码（本地模板模式使用，全部模板均通过单元测试）。

    Args:
        intent: 已解析的分析意图。
        question: 原始问题（用于匹配字段名）。
        schema: 字段画像（``basic_stats.detect_schema`` 的 ``data``）。
        data_file: 数据文件名。

    Returns:
        str: 可直接在沙箱中执行的 Python 代码。
    """
    matched, numeric = _template_columns(schema, question)
    # 建模类型：由问题文本决定（线性回归 / 决策树 / 随机森林），未指定时默认随机森林
    model_type = detect_template_model_type(question) if str(intent) == "ml_train" else "random_forest"
    header = (
        "# 本代码由系统内置模板生成（本地模板模式，未使用 LLM）\n"
        "from analytics import anomaly, basic_stats, correlation, ml_models, time_lag\n"
        "\n"
        f"QUESTION = {json.dumps(str(question), ensure_ascii=False)}\n"
        f"DATA_FILE_NAME = {json.dumps(str(data_file or ''), ensure_ascii=False)}\n"
        f"TEMPLATE_MODEL_TYPE = {json.dumps(model_type, ensure_ascii=False)}\n"
        "df = load_data(DATA_FILE_NAME or None)\n"
        f"NUMERIC_COLUMNS = [name for name in {json.dumps(numeric, ensure_ascii=False)} if name in df.columns] "
        'or df.select_dtypes(include=["number"]).columns.tolist()\n'
        f"MATCHED_COLUMNS = [name for name in {json.dumps(matched, ensure_ascii=False)} if name in df.columns]\n"
        "\n"
    )
    body = _TEMPLATE_BODIES.get(str(intent), _TEMPLATE_BODIES["describe_stats"])
    return header + body.lstrip("\n")


# --------------------------------------------------------------------------- #
# 四、代码生成器
# --------------------------------------------------------------------------- #
class CodeGenerator:
    """分析代码生成器（本地模板优先：标准场景直接命中模板，超出覆盖范围才调用 LLM）。"""

    def __init__(
        self,
        llm_client: Any = None,
        max_repair: int = 2,
        policy: SandboxPolicy | None = None,
        template_router: bool | None = None,
    ) -> None:
        """初始化生成器。

        Args:
            llm_client: :class:`agent.llm_client.LLMClient` 实例；``None`` 表示纯模板模式。
            max_repair: LLM 生成代码的自动修复轮数上限。
            policy: 沙箱策略（影响静态校验规则，例如是否允许 while True）。
            template_router: 是否启用「大模型路由复核」。开启后，规则判定为命中模板但句子可疑时，
                会额外发一次约 100 token 的 JSON 判定请求，确认该需求是否真能由模板完成；
                ``None`` 表示读取环境变量 ``TEMPLATE_ROUTER``（默认开启）。关闭后回到纯规则零额度模式。
        """
        self.llm_client = llm_client
        self.max_repair = max(0, int(max_repair))
        self.policy = policy or SandboxPolicy.from_env()
        if template_router is None:
            template_router = _read_router_env()
        self.template_router = bool(template_router)

    # ------------------------------ 生成 ------------------------------ #
    def generate(
        self,
        question: str,
        schema: Mapping[str, Any] | None = None,
        data_file: str | None = None,
        use_llm: bool | None = None,
        intent: str | None = None,
    ) -> dict[str, Any]:
        """生成分析代码。

        生成策略（模板优先）：

        1. 若需求落在内置模板覆盖范围内（见 :func:`check_template_coverage`），
           直接使用本地模板代码，**不调用 LLM**；
        2. 若需求超出模板覆盖范围（无对应模板 / 模板匹配失败 / 含筛选、比较、分组等
           模板无法表达的自定义逻辑），且允许调用 LLM，则调用 LLM 生成代码；
        3. LLM 未启用或调用失败时，仍回落到本地模板，保证系统始终能产出可执行代码。

        Args:
            question: 用户自然语言问题。
            schema: 字段画像（``basic_stats.detect_schema`` 的 ``data``）。
            data_file: 数据文件名。
            use_llm: 是否允许调用 LLM；``None`` 表示「有可用客户端时允许」。
                （注意：允许调用并不代表一定调用，标准场景仍优先走本地模板。）
            intent: 已解析意图；``None`` 时自动解析。

        Returns:
            dict: 统一结果包，``data`` 含 ``code``、``intent``、``source``（llm/template）、
            ``template_covered``、``template_reason``、``violations``、``repair_attempts``、``safety``。
        """
        try:
            text = str(question or "").strip()
            if not text:
                return fail("分析需求不能为空")

            intent_result = analyze_intent(text)
            resolved_intent = intent or (
                intent_result["data"]["intent"] if intent_result["status"] == "success" else "overview"
            )
            intent_name = INTENT_NAMES.get(str(resolved_intent), str(resolved_intent))
            allow_llm = (
                self.llm_client is not None and bool(getattr(self.llm_client, "is_available", False))
                if use_llm is None
                else bool(use_llm)
            )

            # ---------- 策略第一步：模板优先匹配 ----------
            coverage = check_template_coverage(text, intent_result)
            coverage.update({"intent": str(resolved_intent), "intent_name": intent_name})
            router_used = False
            router_reason = ""
            if coverage["covered"] and self.template_router and allow_llm:
                # 长尾兜底：规则判定命中模板，但句子可疑（模板词表外的动作词/名词，或长复合需求）
                # 时，请大模型做一次「能否用标准模板完成」的二选一判定（约 100 token，不生成代码）。
                need_router, trigger = should_ask_router(text, schema)
                if need_router:
                    decision = self._router_decision(text, str(resolved_intent), schema, trigger)
                    if decision["used"]:
                        router_used = True
                        router_reason = decision["reason"]
                        if not decision["use_template"]:
                            # 大模型判定模板无法完整表达 -> 降级为 LLM 定制生成
                            coverage = dict(coverage)
                            coverage["covered"] = False
                            coverage["limitation"] = coverage["limitation"] or "大模型路由判定为定制需求"
                            coverage["reason"] = (
                                f"大模型路由复核认为需要定制生成（触发原因：{trigger}；"
                                f"判定理由：{router_reason or '标准模板无法完整表达该需求'}）"
                            )
                            logger.info("路由复核：%s -> LLM（%s）", text[:32], router_reason)
            if coverage["covered"]:
                code = build_template_code(str(resolved_intent), text, schema, data_file)
                safety = check_code_safety(code, self.policy)
                data = {
                    "code": code,
                    "intent": str(resolved_intent),
                    "intent_name": intent_name,
                    "source": "template",
                    "model": None,
                    "repair_attempts": 0,
                    "violations": safety["data"]["violations"] if safety["status"] == "success" else [],
                    "safety": safety["data"] if safety["status"] == "success" else None,
                    "question": text,
                    "data_file": data_file,
                    "template_covered": True,
                    "template_reason": coverage["reason"],
                    "template_limitation": coverage["limitation"],
                    "force_llm": False,
                    "generation_strategy": "template-first",
                    "llm_skipped": True,
                    "router_used": router_used,
                    "router_reason": router_reason,
                }
                message = f"已命中内置模板生成分析代码（{intent_name}），未消耗代码生成额度"
                if router_used:
                    message += f"；已由大模型路由复核确认（{router_reason or '模板可完整表达'}）"
                return ok(data, message)

            # ---------- 策略第二步：超出模板覆盖范围 -> 调用 LLM ----------
            notes: list[str] = []
            notes.append(f"未命中内置模板：{coverage['reason']}")
            if allow_llm:
                llm_result = self._generate_with_llm(text, schema, data_file)
                if llm_result["status"] == "success":
                    code = llm_result["data"]["code"]
                    safety = check_code_safety(code, self.policy)
                    data = {
                        "code": code,
                        "intent": str(resolved_intent),
                        "intent_name": intent_name,
                        "source": "llm",
                        "model": llm_result["data"].get("model"),
                        "repair_attempts": int(llm_result["data"].get("repair_attempts", 0)),
                        "violations": safety["data"]["violations"] if safety["status"] == "success" else [],
                        "safety": safety["data"] if safety["status"] == "success" else None,
                        "question": text,
                        "data_file": data_file,
                        "template_covered": False,
                        "template_reason": coverage["reason"],
                        "template_limitation": coverage["limitation"],
                        "force_llm": bool(coverage.get("force_llm")),
                        "generation_strategy": "llm-fallback",
                        "llm_skipped": False,
                        "router_used": router_used,
                        "router_reason": router_reason,
                    }
                    return ok(
                        data,
                        f"已由 LLM 生成分析代码（意图：{intent_name}，修复 "
                        f"{data['repair_attempts']} 次，模型 {data['model']}）",
                    )
                notes.append(f"LLM 代码生成失败（{llm_result['error']}），已回落到本地模板")
                logger.warning("LLM 代码生成失败，回落到本地模板：%s", llm_result["error"])
            else:
                notes.append("未启用 LLM（未配置 API Key 或已显式关闭），按最接近的模板执行")

            # ---------- 策略第三步：模板兜底（保证始终有可执行代码） ----------
            code = build_template_code(str(resolved_intent), text, schema, data_file)
            safety = check_code_safety(code, self.policy)
            data = {
                "code": code,
                "intent": str(resolved_intent),
                "intent_name": intent_name,
                "source": "template",
                "model": None,
                "repair_attempts": 0,
                "violations": safety["data"]["violations"] if safety["status"] == "success" else [],
                "safety": safety["data"] if safety["status"] == "success" else None,
                "question": text,
                "data_file": data_file,
                "template_covered": False,
                "template_reason": coverage["reason"],
                "template_limitation": coverage["limitation"],
                "force_llm": bool(coverage.get("force_llm")),
                "generation_strategy": "template-fallback",
                "llm_skipped": False,
                "router_used": router_used,
                "router_reason": router_reason,
            }
            return ok(data, "；".join(notes) or "已由本地模板生成分析代码")
        except Exception as exc:  # noqa: BLE001
            logger.exception("代码生成失败")
            return fail("代码生成失败", exc)

    # ------------------------------ 修复 ------------------------------ #
    def repair(
        self,
        code: str,
        error_message: str,
        question: str,
        schema: Mapping[str, Any] | None = None,
        data_file: str | None = None,
    ) -> dict[str, Any]:
        """把静态校验或运行期错误回灌给 LLM，请求修复代码。"""
        try:
            if self.llm_client is None or not bool(getattr(self.llm_client, "is_available", False)):
                return fail("未启用 LLM，无法自动修复代码")
            prompt = (
                f"{build_code_prompt(question, schema, data_file)}\n\n"
                "【上一次生成的代码】\n```python\n"
                f"{code}\n```\n\n"
                f"【执行或校验报错信息】\n{error_message}\n\n"
                "请修正上述错误并重新输出完整代码（仍须遵守全部硬性约束，且必须调用 emit_result）。"
            )
            response = self.llm_client.generate_code(prompt, system_prompt=SYSTEM_PROMPT)
            if response["status"] != "success":
                return fail("LLM 修复调用失败", response["error"])
            repaired = extract_code(response["data"]["content"])
            if not repaired:
                return fail("LLM 修复结果为空，未能抽取到代码")
            return ok({"code": repaired, "model": response["data"]["model"]}, "LLM 已返回修复后的代码")
        except Exception as exc:  # noqa: BLE001
            logger.exception("代码修复失败")
            return fail("代码自动修复失败", exc)

    # --------------------------- 生成-执行闭环 --------------------------- #
    def generate_and_run(
        self,
        question: str,
        runner: Any,
        schema: Mapping[str, Any] | None = None,
        data_file: str | None = None,
        use_llm: bool | None = None,
    ) -> dict[str, Any]:
        """「生成 -> 校验/执行 -> 自动修复」闭环。

        Args:
            question: 用户问题。
            runner: 可调用对象 ``runner(code) -> 统一结果包``（通常是沙箱执行器）。
            schema: 字段画像。
            data_file: 数据文件名。
            use_llm: 是否允许调用 LLM。

        Returns:
            dict: 统一结果包，``data`` 含 ``code``、``execution``、``source``、``repair_attempts``。
        """
        try:
            generated = self.generate(question, schema, data_file, use_llm=use_llm)
            if generated["status"] != "success":
                return generated

            code = generated["data"]["code"]
            source = generated["data"]["source"]
            attempts = 0
            execution = runner(code)

            # 仅对 LLM 生成的代码做自动修复；模板代码已通过单元测试，失败通常源于数据本身
            while (
                execution["status"] != "success"
                and source == "llm"
                and attempts < self.max_repair
                and self.llm_client is not None
                and bool(getattr(self.llm_client, "is_available", False))
            ):
                attempts += 1
                problem = execution.get("error") or execution.get("message") or "未知错误"
                fixed = self.repair(code, problem, question, schema, data_file)
                if fixed["status"] != "success":
                    break
                code = fixed["data"]["code"]
                execution = runner(code)

            data = dict(generated["data"])
            data.update({"code": code, "execution": execution, "repair_attempts": attempts})
            if execution["status"] == "success":
                return ok(data, execution["message"], charts=execution["charts"])
            return fail(
                f"代码执行失败：{execution['error']}",
                execution["error"],
                data=data,
                charts=execution["charts"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("生成-执行闭环失败")
            return fail("生成-执行闭环失败", exc)

    # ------------------------------ 内部 ------------------------------ #
    def _router_decision(
        self,
        question: str,
        intent: str,
        schema: Mapping[str, Any] | None,
        trigger: str = "",
    ) -> dict[str, Any]:
        """请大模型判定「该需求能否由标准模板完成」。

        只做二选一判定，不生成代码（约 100 token）。任何异常（调用失败、返回不可解析）
        都视为"未获得判定"，由调用方保守沿用规则结果，避免额外消耗与误判。

        Returns:
            dict: ``{"used": bool, "use_template": bool, "reason": str}``。
        """
        outcome: dict[str, Any] = {"used": False, "use_template": True, "reason": ""}
        try:
            if self.llm_client is None or not bool(getattr(self.llm_client, "is_available", False)):
                return outcome
            capability = TEMPLATE_CAPABILITY_SUMMARY.get(str(intent), "标准分析模板")
            columns = ""
            if isinstance(schema, Mapping):
                names = list(schema.get("numeric_columns") or [])
                columns = "、".join(map(str, names[:12]))
            prompt = (
                f"【数据字段】{columns or '（未提供）'}\n"
                f"【已匹配到的标准模板能力】{capability}\n"
                f"【可疑点】{trigger or '（未提供）'}\n"
                f"【用户需求】{question}\n\n"
                "该需求能否由上述标准模板完整完成？请只输出 JSON。"
            )
            response = self.llm_client.chat(
                [
                    {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                model=getattr(self.llm_client, "code_model", None),
                temperature=0.0,
                max_tokens=200,
                response_format="json_object",
            )
            if response.get("status") != "success":
                logger.warning("路由复核调用失败（沿用规则判定）：%s", response.get("error"))
                return outcome
            content = str((response.get("data") or {}).get("content") or "").strip()
            parsed = self._parse_router_json(content)
            if parsed is None:
                logger.warning("路由复核返回无法解析（沿用规则判定）：%s", content[:80])
                return outcome
            outcome.update({"used": True, **parsed})
            return outcome
        except Exception as exc:  # noqa: BLE001 - 复核异常一律沿用规则判定
            logger.warning("路由复核异常（沿用规则判定）：%s", exc)
            return outcome

    @staticmethod
    def _parse_router_json(content: str) -> dict[str, Any] | None:
        """解析路由复核返回的 JSON（容忍外层文字、布尔字符串等写法）。"""
        try:
            text = str(content or "").strip()
            if not text:
                return None
            match = re.search(r"\{.*\}", text, re.DOTALL)
            payload = json.loads(match.group(0) if match else text)
            if not isinstance(payload, dict) or "use_template" not in payload:
                return None
            raw = payload.get("use_template")
            if isinstance(raw, str):
                use_template = raw.strip().lower() in ("true", "yes", "1", "是")
            else:
                use_template = bool(raw)
            return {"use_template": use_template, "reason": str(payload.get("reason") or "").strip()[:80]}
        except Exception:  # noqa: BLE001
            return None

    def _generate_with_llm(
        self,
        question: str,
        schema: Mapping[str, Any] | None,
        data_file: str | None,
    ) -> dict[str, Any]:
        """调用 LLM 生成代码，并在静态校验失败时自动修复。"""
        try:
            prompt = build_code_prompt(question, schema, data_file)
            response = self.llm_client.generate_code(prompt, system_prompt=SYSTEM_PROMPT)
            if response["status"] != "success":
                return fail("LLM 调用失败", response["error"])

            code = extract_code(response["data"]["content"])
            if not code:
                return fail("LLM 回复中未抽取到 Python 代码")

            repair_attempts = 0
            is_valid, violations = validate_code(code, self.policy)
            while not is_valid and repair_attempts < self.max_repair:
                repair_attempts += 1
                fixed = self.repair(code, "安全校验未通过：" + "；".join(violations), question, schema, data_file)
                if fixed["status"] != "success":
                    return fail(f"代码静态校验失败且自动修复未成功：{'；'.join(violations[:3])}")
                code = fixed["data"]["code"]
                is_valid, violations = validate_code(code, self.policy)

            if not is_valid:
                return fail(f"代码未通过安全校验：{'；'.join(violations[:3])}")
            return ok(
                {"code": code, "model": response["data"]["model"], "repair_attempts": repair_attempts},
                "LLM 代码生成完成",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("LLM 代码生成异常")
            return fail("LLM 代码生成异常", exc)


def generate_analysis_code(
    question: str,
    schema: Mapping[str, Any] | None = None,
    data_file: str | None = None,
    llm_client: Any = None,
    use_llm: bool | None = None,
) -> dict[str, Any]:
    """便捷函数：直接生成分析代码（内部创建 :class:`CodeGenerator`）。"""
    try:
        return CodeGenerator(llm_client=llm_client).generate(question, schema, data_file, use_llm=use_llm)
    except Exception as exc:  # noqa: BLE001
        logger.exception("generate_analysis_code 执行失败")
        return fail("代码生成失败", exc)


if __name__ == "__main__":
    _samples = (
        "计算 inlet_temp 的均值和标准差",
        "分析各变量之间的相关性",
        "检测数据中的异常点",
        "训练随机森林模型预测 yield_pct",
        "查看数据缺失情况",
    )
    for _text in _samples:
        _result = analyze_intent(_text)
        print(_text, "->", json.dumps(_result["data"], ensure_ascii=False))
