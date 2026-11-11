"""报告生成器：把执行结果转成结构化中文分析报告（Markdown + 自包含 HTML）。

关键设计
========================================================================

1. **事实优先**：先把沙箱执行结果整理成「事实字典」（指标、图表、执行状态、沙箱信息），
   再交给 LLM 撰写叙述性结论，避免模型编造数据；
2. **模板兜底**：LLM 不可用时，使用确定性模板渲染同样结构的报告，保证离线可用；
3. **双格式输出**：``output/reports/*.md``（便于二次编辑）与 ``*.html``
   （图表以 base64 内嵌，单文件可直接分享）。
"""

from __future__ import annotations

import base64
import html as html_lib
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence, TYPE_CHECKING

from analytics import CHART_DIR, REPORT_DIR, fail, get_logger, ok, truncate_text

if TYPE_CHECKING:  # 仅用于类型标注，避免模块级循环依赖
    from agent.llm_client import LLMClient

__all__ = ["ReportGenerator", "REPORT_SYSTEM_PROMPT"]

logger = get_logger(__name__)

REPORT_SYSTEM_PROMPT = "\n".join(
    [
        "你是流程工业数据分析报告撰写专家。请根据给定的分析事实，输出结构化中文分析报告。",
        "",
        "要求：",
        "1. 使用 Markdown 语法，一级标题从「## 一、分析结论」开始，不要重复报告总标题；",
        "2. 章节固定为：一、分析结论；二、关键指标解读；三、图表说明；四、风险提示与优化建议；",
        "3. 所有数字必须来自给定事实，不得编造；引用数值时保留 3~4 位有效数字；",
        "4. 结论要面向工艺含义（例如时滞意味着调节提前量、异常点意味着仪表或工况异常）；",
        "5. 不要输出代码块，不要输出图片语法（图表由系统自动附加），全文控制在 800 字以内；",
        "6. 若执行失败或存在异常信息，需在结论中明确指出失败原因与排查建议。",
    ]
)

#: 报告中「关键指标」表格最多渲染的条目数
_MAX_METRIC_ROWS = 40

#: 有序列表行前缀（形如 ``1. ``）
_ORDERED_LIST_PATTERN = re.compile(r"^\d+\.\s")
#: 行内粗体 / 行内代码 / 图片 语法
_BOLD_PATTERN = re.compile(r"\*\*(.+?)\*\*")
_INLINE_CODE_PATTERN = re.compile(r"`([^`]+)`")
_IMAGE_PATTERN = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")


class ReportGenerator:
    """分析报告生成器。"""

    def __init__(self, llm_client: Any = None, max_facts_chars: int = 8000) -> None:
        """初始化。

        Args:
            llm_client: :class:`agent.llm_client.LLMClient` 实例；``None`` 表示纯模板模式。
            max_facts_chars: 送入 LLM 的事实文本最大长度（防止超长上下文）。
        """
        self.llm_client = llm_client
        self.max_facts_chars = max(1000, int(max_facts_chars))

    # ------------------------------------------------------------------ #
    # 一、主流程
    # ------------------------------------------------------------------ #
    def generate(
        self,
        question: str,
        execution: Mapping[str, Any],
        schema: Mapping[str, Any] | None = None,
        data_file: str | None = None,
        allow_llm: bool | None = None,
    ) -> dict[str, Any]:
        """生成分析报告（LLM 优先，失败自动回落模板），并保存为 md / html。

        Args:
            question: 用户原始分析需求。
            execution: 沙箱执行结果包（``execute_code`` 的返回值）。
            schema: 数据字段画像（可选）。
            data_file: 数据文件名（可选）。
            allow_llm: 是否允许调用 LLM；``None`` 表示「有可用客户端时调用」。

        Returns:
            dict: 统一结果包，``data`` 含 ``markdown``、``charts``、``md_path``、
            ``html_path``、``source``（llm/template）、``facts``。
        """
        try:
            facts = self.build_facts(question, execution, schema, data_file)
            charts = list(facts.get("charts") or [])
            use_llm = (
                self.llm_client is not None and bool(getattr(self.llm_client, "is_available", False))
                if allow_llm is None
                else bool(allow_llm)
            )

            markdown = ""
            source = "template"
            notes: list[str] = []
            if use_llm:
                llm_result = self._generate_with_llm(facts)
                if llm_result["status"] == "success" and llm_result["data"]["markdown"].strip():
                    markdown = llm_result["data"]["markdown"].strip()
                    source = "llm"
                else:
                    notes.append(f"LLM 报告生成失败（{llm_result['error']}），已使用内置模板")
                    logger.warning("LLM 报告生成失败，回落模板：%s", llm_result["error"])
            else:
                notes.append("未启用 LLM，使用内置报告模板")

            if not markdown:
                markdown = self.render_template_report(facts)

            saved = self.save(markdown, charts)
            if saved["status"] != "success":
                return fail(saved["message"], saved["error"], data={"markdown": markdown, "charts": charts})

            data = {
                "markdown": markdown,
                "charts": charts,
                "md_path": saved["data"]["md_path"],
                "html_path": saved["data"]["html_path"],
                "source": source,
                "facts": facts,
            }
            message = (
                f"报告生成完成（{'LLM 撰写' if source == 'llm' else '内置模板'}）："
                f"{Path(saved['data']['md_path']).name}"
            )
            if notes:
                message += "；" + "；".join(notes)
            return ok(data, message, charts=charts)
        except Exception as exc:  # noqa: BLE001
            logger.exception("报告生成失败")
            return fail("报告生成失败", exc)

    # ------------------------------------------------------------------ #
    # 二、事实整理
    # ------------------------------------------------------------------ #
    def build_facts(
        self,
        question: str,
        execution: Mapping[str, Any],
        schema: Mapping[str, Any] | None = None,
        data_file: str | None = None,
    ) -> dict[str, Any]:
        """把执行结果整理为报告事实字典（供 LLM 与模板共用）。"""
        try:
            payload = dict(execution or {})
            inner = payload.get("data") if isinstance(payload.get("data"), Mapping) else {}
            result_payload = inner.get("result") if isinstance(inner, Mapping) else None
            if result_payload is None:
                result_payload = payload.get("result")

            facts: dict[str, Any] = {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "question": str(question or "").strip(),
                "data_file": data_file,
                "status": inner.get("status") or payload.get("status"),
                "exit_code": inner.get("exit_code"),
                "timed_out": bool(inner.get("timed_out")),
                "duration_seconds": inner.get("duration_seconds"),
                "error": inner.get("error") or payload.get("error"),
                "violations": list(inner.get("violations") or payload.get("violations") or []),
                "stdout": truncate_text(inner.get("stdout") or "", 4000),
                "stderr": truncate_text(inner.get("stderr") or "", 3000),
                "code_path": inner.get("code_path"),
                "result": result_payload,
                "charts": [str(item) for item in (payload.get("charts") or inner.get("charts") or [])],
                "schema": None,
            }

            if isinstance(schema, Mapping):
                facts["schema"] = {
                    "n_rows": schema.get("n_rows"),
                    "n_columns": schema.get("n_columns"),
                    "numeric_columns": list(schema.get("numeric_columns") or []),
                    "datetime_columns": list(schema.get("datetime_columns") or []),
                    "categorical_columns": list(schema.get("categorical_columns") or []),
                }
            return facts
        except Exception as exc:  # noqa: BLE001
            logger.exception("事实整理失败")
            return {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "question": str(question or ""),
                "status": "error",
                "error": f"事实整理失败：{exc}",
                "charts": [],
                "result": None,
                "violations": [],
            }

    @staticmethod
    def _flatten(payload: Any, prefix: str = "", max_items: int = 400) -> list[tuple[str, Any]]:
        """把嵌套结果扁平化为 ``(路径, 标量值)`` 列表，便于渲染指标表。"""
        rows: list[tuple[str, Any]] = []

        def _walk(node: Any, path: str) -> None:
            if len(rows) >= max_items:
                return
            if isinstance(node, Mapping):
                for key, value in node.items():
                    _walk(value, f"{path}.{key}" if path else str(key))
            elif isinstance(node, (list, tuple)):
                if all(not isinstance(item, (Mapping, list, tuple)) for item in node):
                    rendered = ", ".join(str(item) for item in node[:10])
                    if len(node) > 10:
                        rendered += f" …（共 {len(node)} 项）"
                    rows.append((path, rendered))
                else:
                    for index, item in enumerate(node[:20]):
                        _walk(item, f"{path}[{index}]")
            else:
                rows.append((path, node))

        try:
            _walk(payload, prefix)
        except Exception:  # noqa: BLE001
            logger.warning("结果扁平化失败", exc_info=True)
        return rows

    def _facts_to_text(self, facts: Mapping[str, Any]) -> str:
        """把事实字典转成紧凑文本（送入 LLM）。"""
        try:
            lines = [
                f"用户需求：{facts.get('question')}",
                f"数据文件：{facts.get('data_file') or '（未指定）'}",
                f"执行状态：{facts.get('status')}（退出码 {facts.get('exit_code')}，"
                f"耗时 {facts.get('duration_seconds')} 秒，超时 {facts.get('timed_out')}）",
            ]
            if facts.get("schema"):
                schema = facts["schema"]
                lines.append(
                    f"数据结构：{schema.get('n_rows')} 行 × {schema.get('n_columns')} 列；"
                    f"数值字段：{', '.join(map(str, schema.get('numeric_columns') or []))}"
                )
            if facts.get("error"):
                lines.append(f"错误信息：{facts['error']}")
            if facts.get("violations"):
                lines.append(f"安全校验问题：{'；'.join(map(str, facts['violations'][:5]))}")
            rows = self._flatten(facts.get("result"))
            if rows:
                lines.append("分析结果明细：")
                lines.extend(f"  - {key} = {value}" for key, value in rows[:120])
            if facts.get("stdout"):
                lines.append(f"程序标准输出（节选）：\n{truncate_text(facts['stdout'], 1500)}")
            if facts.get("stderr"):
                lines.append(f"程序标准错误（节选）：\n{truncate_text(facts['stderr'], 1000)}")
            lines.append(f"生成图表数量：{len(facts.get('charts') or [])}")
            return truncate_text("\n".join(lines), self.max_facts_chars)
        except Exception as exc:  # noqa: BLE001
            logger.exception("事实文本构造失败")
            return f"事实构造失败：{exc}"

    # ------------------------------------------------------------------ #
    # 三、LLM 报告
    # ------------------------------------------------------------------ #
    def _generate_with_llm(self, facts: Mapping[str, Any]) -> dict[str, Any]:
        """调用 LLM 撰写报告正文。"""
        try:
            if self.llm_client is None:
                return fail("未配置 LLM 客户端")
            prompt = (
                "以下是本次工业数据分析的执行事实，请据此撰写分析报告：\n\n"
                f"{self._facts_to_text(facts)}\n\n"
                "请按系统提示词要求输出 Markdown 报告正文。"
            )
            response = self.llm_client.generate_report(prompt, system_prompt=REPORT_SYSTEM_PROMPT)
            if response["status"] != "success":
                return fail("LLM 报告调用失败", response["error"])
            content = str(response["data"]["content"] or "").strip()
            return ok({"markdown": content, "model": response["data"]["model"]}, "LLM 报告生成完成")
        except Exception as exc:  # noqa: BLE001
            logger.exception("LLM 报告生成异常")
            return fail("LLM 报告生成异常", exc)

    # ------------------------------------------------------------------ #
    # 四、模板报告
    # ------------------------------------------------------------------ #
    def render_template_report(self, facts: Mapping[str, Any]) -> str:
        """使用确定性模板渲染报告（离线可用，结构与 LLM 报告一致）。"""
        try:
            question = facts.get("question") or "数据分析"
            status = facts.get("status")
            result = facts.get("result") if isinstance(facts.get("result"), Mapping) else {}
            intent = str(result.get("intent") or "unknown")          # 机器意图键（用于分支判断）
            intent_label = str(result.get("intent_name") or intent)  # 展示用中文名称

            lines: list[str] = [
                f"# 工业数据分析报告：{question}",
                "",
                f"- **生成时间**：{facts.get('generated_at')}",
                f"- **数据文件**：{facts.get('data_file') or '（未指定）'}",
                f"- **分析类型**：{intent_label}",
                "- **执行状态**："
                + ("成功" if status == "success" else "失败")
                + f"（退出码 {facts.get('exit_code')}，耗时 {facts.get('duration_seconds')} 秒）",
                "",
                "## 一、分析结论",
                "",
                self._render_conclusion(facts, intent),
                "",
                "## 二、关键指标",
                "",
            ]

            rows = self._flatten(result)[:_MAX_METRIC_ROWS]
            if rows:
                lines.append("| 指标路径 | 数值 |")
                lines.append("| --- | --- |")
                lines.extend(f"| {key} | {self._fmt(value)} |" for key, value in rows)
            else:
                lines.append("本次执行未产出结构化指标（请检查生成代码是否调用了 emit_result）。")

            lines.extend(["", "## 三、图表说明", ""])
            charts: Sequence[str] = list(facts.get("charts") or [])
            if charts:
                for index, path in enumerate(charts, start=1):
                    name = Path(str(path)).name
                    lines.append(f"图 {index}：{name}")
                    lines.append("")
                    lines.append(f"![{name}](charts/{name})")
                    lines.append("")
            else:
                lines.append("本次分析未生成图表。")

            lines.extend(
                [
                    "",
                    "## 四、执行环境与安全检查",
                    "",
                    "- 沙箱机制：subprocess 子进程隔离（独立工作目录、环境变量脱敏、超时强制终止）",
                    "- 危险代码校验：" + ("通过" if not facts.get("violations") else "存在被拦截项"),
                    f"- 生成代码文件：{facts.get('code_path') or '（未记录）'}",
                ]
            )
            if facts.get("violations"):
                lines.append("- 拦截明细：" + "；".join(map(str, facts["violations"][:5])))
            if facts.get("error"):
                lines.append(f"- 执行提示：{facts['error']}")
            if facts.get("stdout"):
                lines.extend(["", "程序标准输出（节选）：", "", "```text",
                              truncate_text(facts["stdout"], 1200).strip(), "```"])
            if facts.get("stderr"):
                lines.extend(["", "程序标准错误（节选）：", "", "```text",
                              truncate_text(facts["stderr"], 800).strip(), "```"])

            lines.extend(["", "## 五、风险提示与优化建议", "", self._render_suggestions(facts, intent), ""])
            return "\n".join(lines)
        except Exception as exc:  # noqa: BLE001
            logger.exception("模板报告渲染失败")
            return f"# 工业数据分析报告\n\n模板渲染失败：{exc}\n"

    @staticmethod
    def _fmt(value: Any) -> str:
        """格式化指标值（浮点保留 4 位有效数字，表格中竖线转义）。"""
        try:
            if isinstance(value, float):
                if value != value:  # NaN 判定
                    return "NaN"
                return f"{value:.4g}"
            return str(value).replace("|", "/")
        except Exception:  # noqa: BLE001
            return str(value)

    def _render_conclusion(self, facts: Mapping[str, Any], intent: str) -> str:
        """按分析类型生成结论段落。"""
        try:
            if facts.get("status") != "success":
                return (
                    f"本次分析未能成功完成，原因：{facts.get('error') or '代码执行失败'}。"
                    "建议核查数据字段类型与样本量、确认生成代码符合沙箱安全约束后重试。"
                )
            result = facts.get("result") if isinstance(facts.get("result"), Mapping) else {}
            if result.get("error"):
                return f"代码执行成功但业务层返回错误：{result['error']}。请检查数据字段与参数设置。"
            if intent == "time_lag":
                return str(result.get("conclusion") or "时滞分析完成，请参考关键指标中的最优滞后步数。")
            if intent == "correlation":
                pair = result.get("strongest_pair") or {}
                pairs = result.get("strong_pairs") or []
                text = "已完成相关性分析。"
                if pair:
                    text += (
                        f"最强相关字段为 {pair.get('feature_a')} 与 {pair.get('feature_b')}，"
                        f"相关系数 r={self._fmt(pair.get('correlation'))}"
                        f"（{pair.get('strength')}{pair.get('direction')}）。"
                    )
                text += f"共发现 {len(pairs)} 组强相关特征对，可用于特征筛选与变量解耦分析。"
                return text
            if intent == "anomaly":
                columns = result.get("columns") or []
                total = result.get("total_anomalies")
                if columns:
                    worst = max(columns, key=lambda item: item.get("n_anomalies", 0))
                    return (
                        f"基于 3σ 原则完成异常检测，共发现 {total} 个异常点；异常最多的是字段 "
                        f"{worst.get('column')}（{worst.get('n_anomalies')} 个，占比 "
                        f"{self._fmt(worst.get('anomaly_rate'))}）。"
                        "建议结合工艺操作记录判断是仪表跳变、工况切换还是真实过程异常。"
                    )
                return f"基于 3σ 原则完成异常检测，共发现 {total} 个异常点。"
            if intent == "ml_train":
                metrics = result.get("metrics") or {}
                importance = result.get("feature_importance") or []
                top = "、".join(str(item.get("feature")) for item in importance[:3])
                text = (
                    f"建模目标为 {result.get('target')}，测试集 R²={self._fmt(metrics.get('r2'))}，"
                    f"MSE={self._fmt(metrics.get('mse'))}，RMSE={self._fmt(metrics.get('rmse'))}，"
                    f"MAE={self._fmt(metrics.get('mae'))}。"
                )
                if top:
                    text += f"影响最大的特征依次为：{top}。"
                if result.get("model_path"):
                    text += "模型已保存，可用于后续在线推理。"
                return text
            if intent == "ml_predict":
                stats = result.get("prediction_stats") or {}
                return (
                    f"使用模型 {Path(str(result.get('model_path'))).name} 完成推理，共预测 "
                    f"{result.get('n_predicted')} 条，预测均值 {self._fmt(stats.get('mean'))}，"
                    f"范围 [{self._fmt(stats.get('min'))}, {self._fmt(stats.get('max'))}]。"
                )
            if intent == "missing_values":
                return (
                    f"数据共 {result.get('n_rows')} 行，其中 {result.get('n_columns_with_missing')} 个字段"
                    "存在缺失值，建议对高缺失率字段评估补测或剔除，避免影响统计与建模结论。"
                )
            if intent == "overview":
                return (
                    f"数据规模 {result.get('n_rows')} 行 × {result.get('n_columns')} 列，数值字段 "
                    f"{len(result.get('numeric_columns') or [])} 个，时间字段 "
                    f"{len(result.get('datetime_columns') or [])} 个，"
                    f"{result.get('n_columns_with_missing')} 个字段存在缺失值。"
                )
            statistics = result.get("statistics") if isinstance(result.get("statistics"), Mapping) else {}
            if statistics:
                text = "已完成描述性统计，示例："
                for name, stats in list(statistics.items())[:3]:
                    if isinstance(stats, Mapping):
                        text += (
                            f"{name}（均值 {self._fmt(stats.get('mean'))}，标准差 "
                            f"{self._fmt(stats.get('std'))}，范围 [{self._fmt(stats.get('min'))}, "
                            f"{self._fmt(stats.get('max'))}]）"
                        )
                text += "。可结合标准差与分布图判断波动水平是否超出工艺控制范围。"
                return text
            return "分析执行成功，详细指标见下表。"
        except Exception as exc:  # noqa: BLE001
            logger.warning("结论渲染失败：%s", exc)
            return "结论渲染出现异常，请参考关键指标与图表。"

    @staticmethod
    def _render_suggestions(facts: Mapping[str, Any], intent: str) -> str:
        """生成风险提示与优化建议段落。"""
        tips: list[str] = []
        if facts.get("timed_out"):
            tips.append("本次执行触发超时保护：建议缩小数据规模或分步计算，并确认不存在无界循环。")
        if facts.get("violations"):
            tips.append("生成代码存在被拦截的不安全写法：建议检查导入白名单与路径约束后重新生成。")
        if intent == "time_lag":
            tips.append("时滞结论依赖采样频率：若采样间隔不均，应先做重采样与时间对齐。")
        if intent == "correlation":
            tips.append("相关性不等同因果：建议结合工艺机理与滞后结构（时滞分析）进一步验证。")
        if intent == "anomaly":
            tips.append("3σ 依赖近似正态假设：数据偏态或含趋势时应参照 MAD 稳健阈值对照判断。")
        if intent == "ml_train":
            tips.append("工业数据常存在时间自相关：建议改用时间序列切分（TimeSeriesSplit）验证泛化能力。")
        tips.append("长期运行建议：将沙箱部署在容器内并使用低权限账户执行，进一步收敛风险面。")
        return "\n".join(f"- {tip}" for tip in tips)

    # ------------------------------------------------------------------ #
    # 五、保存与导出
    # ------------------------------------------------------------------ #
    def save(self, markdown: str, charts: Sequence[str] | None = None, stem: str | None = None) -> dict[str, Any]:
        """保存报告为 Markdown 与自包含 HTML（图表以 base64 内嵌）。"""
        try:
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            safe_stem = re.sub(r"[^0-9A-Za-z_\-]+", "_", str(stem or ""))[:40] or (
                f"report_{time.strftime('%Y%m%d_%H%M%S')}"
            )
            md_path = REPORT_DIR / f"{safe_stem}.md"
            html_path = REPORT_DIR / f"{safe_stem}.html"
            md_path.write_text(markdown, encoding="utf-8")
            html_path.write_text(
                self.markdown_to_html(markdown, charts or [], title=f"工业数据分析报告 {safe_stem}"),
                encoding="utf-8",
            )
            return ok(
                {"md_path": str(md_path), "html_path": str(html_path), "stem": safe_stem},
                f"报告已保存：{md_path.name} 与 {html_path.name}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("报告保存失败")
            return fail("报告保存失败", exc)

    def markdown_to_html(self, markdown: str, charts: Sequence[str], title: str = "分析报告") -> str:
        """把 Markdown 报告转换为自包含 HTML（图片以 base64 内嵌，便于单文件分享）。"""
        try:
            body_lines: list[str] = []
            lines = str(markdown or "").splitlines()
            index = 0
            in_list = False
            while index < len(lines):
                line = lines[index].rstrip()
                stripped = line.strip()
                if in_list and not stripped.startswith(("- ", "* ")) and not _ORDERED_LIST_PATTERN.match(stripped):
                    body_lines.append("</ul>")
                    in_list = False
                if not stripped:
                    index += 1
                    continue
                is_table_start = (
                    stripped.startswith("|")
                    and index + 1 < len(lines)
                    and set(lines[index + 1].replace("|", "").strip()) <= {"-", ":", " "}
                    and lines[index + 1].strip()
                )
                if is_table_start:
                    table_html, index = self._render_table(lines, index)
                    body_lines.append(table_html)
                    continue
                if stripped.startswith("```"):
                    block: list[str] = []
                    index += 1
                    while index < len(lines) and not lines[index].strip().startswith("```"):
                        block.append(html_lib.escape(lines[index]))
                        index += 1
                    body_lines.append("<pre>" + "\n".join(block) + "</pre>")
                elif stripped.startswith("#"):
                    level = min(6, len(stripped) - len(stripped.lstrip("#")))
                    body_lines.append(f"<h{level}>{self._inline(stripped[level:].strip())}</h{level}>")
                elif stripped.startswith(("- ", "* ")):
                    if not in_list:
                        body_lines.append("<ul>")
                        in_list = True
                    body_lines.append(f"<li>{self._inline(stripped[2:].strip())}</li>")
                elif _ORDERED_LIST_PATTERN.match(stripped):
                    if not in_list:
                        body_lines.append("<ul>")
                        in_list = True
                    body_lines.append(f"<li>{self._inline(_ORDERED_LIST_PATTERN.sub('', stripped))}</li>")
                elif stripped.startswith("!["):
                    body_lines.append(self._render_image(stripped))
                else:
                    body_lines.append(f"<p>{self._inline(stripped)}</p>")
                index += 1
            if in_list:
                body_lines.append("</ul>")

            style = (
                "body{font-family:'Microsoft YaHei',Arial,sans-serif;max-width:960px;margin:32px auto;"
                "padding:0 20px;line-height:1.75;color:#222}"
                "h1{border-bottom:2px solid #4f81bd;padding-bottom:8px}"
                "table{border-collapse:collapse;width:100%;margin:12px 0}"
                "th,td{border:1px solid #ccc;padding:6px 10px;font-size:14px;text-align:left}"
                "th{background:#eef3f9}pre{background:#f6f8fa;padding:12px;overflow:auto;font-size:13px}"
                "img{max-width:100%;border:1px solid #ddd;margin:12px 0}li{margin:4px 0}"
            )
            return (
                '<!DOCTYPE html>\n<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n'
                f"<title>{html_lib.escape(title)}</title>\n<style>{style}</style>\n</head>\n<body>\n"
                + "\n".join(body_lines)
                + "\n</body>\n</html>\n"
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Markdown 转 HTML 失败")
            return (
                '<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
                f"<title>{html_lib.escape(title)}</title></head><body><pre>"
                f"{html_lib.escape(str(markdown))}</pre></body></html>"
            )

    @staticmethod
    def _render_table(lines: Sequence[str], start: int) -> tuple[str, int]:
        """渲染 Markdown 表格，返回 ``(html, 下一行索引)``。"""
        header = [cell.strip() for cell in lines[start].strip().strip("|").split("|")]
        index = start + 2
        rows: list[list[str]] = []
        while index < len(lines) and lines[index].strip().startswith("|"):
            rows.append([cell.strip() for cell in lines[index].strip().strip("|").split("|")])
            index += 1
        parts = ["<table><thead><tr>"]
        parts.extend(f"<th>{html_lib.escape(cell)}</th>" for cell in header)
        parts.append("</tr></thead><tbody>")
        for row in rows:
            parts.append("<tr>")
            parts.extend(f"<td>{html_lib.escape(cell)}</td>" for cell in row)
            parts.append("</tr>")
        parts.append("</tbody></table>")
        return "".join(parts), index

    @staticmethod
    def _inline(text: str) -> str:
        """处理行内 Markdown 语法（粗体、行内代码）并转义 HTML。"""
        escaped = html_lib.escape(str(text))
        escaped = _BOLD_PATTERN.sub(r"<strong>\1</strong>", escaped)
        escaped = _INLINE_CODE_PATTERN.sub(r"<code>\1</code>", escaped)
        return escaped

    @staticmethod
    def _render_image(line: str) -> str:
        """把 Markdown 图片标签渲染为 base64 内嵌图片（失败时保留说明文字）。"""
        try:
            match = _IMAGE_PATTERN.match(str(line).strip())
            if not match:
                return f"<p>{html_lib.escape(line)}</p>"
            alt = match.group("alt") or "图表"
            src = match.group("src").strip()
            path = Path(src)
            if not path.is_absolute():
                path = (CHART_DIR / path.name).resolve()
            if not path.is_file():
                return f"<p>（图表文件缺失：{html_lib.escape(str(path))}）</p>"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f'<img alt="{html_lib.escape(alt)}" src="data:image/png;base64,{encoded}">'
        except Exception:  # noqa: BLE001
            logger.warning("图表内嵌失败", exc_info=True)
            return f"<p>{html_lib.escape(line)}</p>"
