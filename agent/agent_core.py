"""Agent 主逻辑：串联「数据管理 -> 意图解析 -> 代码生成 -> 安全执行 -> 自动报告」闭环。

对外主入口为 :class:`IndustrialDataAgent`：

* 数据管理：``list_datasets`` / ``describe_dataset`` / ``delete_dataset`` / ``read_history``；
* 自然语言分析：``analyze``（核心闭环）；
* 算法工具箱：``run_basic_stats`` / ``run_correlation`` / ``run_time_lag`` /
  ``run_anomaly`` / ``run_ml_train`` / ``run_ml_predict``（可通过界面直接参数化调用）。

所有方法返回 ``analytics`` 定义的统一结果包，绝不向上抛出运行时异常。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from analytics import (
    DATA_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    fail,
    get_logger,
    ok,
    to_jsonable,
)
from analytics import anomaly as anomaly_module
from analytics import basic_stats, correlation, ml_models, time_lag
from sandbox.executor import SandboxPolicy, execute_code

from .code_generator import INTENT_NAMES, CodeGenerator, analyze_intent
from .llm_client import LLMClient, get_settings

__all__ = ["IndustrialDataAgent"]

logger = get_logger(__name__)

#: 历史记录文件（保存每次分析的摘要）
HISTORY_FILE_NAME = "agent_history.json"
#: 历史记录最大条数
MAX_HISTORY = 200


class IndustrialDataAgent:
    """工业数据分析 Agent 主控制器。"""

    def __init__(
        self,
        llm_client: Any = None,
        policy: SandboxPolicy | None = None,
        data_dir: Path | str | None = None,
        output_dir: Path | str | None = None,
        max_repair: int = 2,
        use_llm: bool = True,
        template_router: bool | None = None,
    ) -> None:
        """初始化 Agent。

        Args:
            llm_client: LLM 客户端；``None`` 时自动创建 :class:`LLMClient`。
            policy: 沙箱策略；``None`` 时读取环境变量。
            data_dir: 数据目录；默认 ``data/``。
            output_dir: 输出目录；默认 ``output/``。
            max_repair: 代码自动修复轮数上限。
            use_llm: 是否允许调用 LLM（界面可切换为纯本地模板模式）。
        """
        self.llm_client = llm_client if llm_client is not None else LLMClient()
        self.policy = policy or SandboxPolicy.from_env()
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        self.use_llm = bool(use_llm)
        self.max_repair = max(0, int(max_repair))
        self.template_router = template_router
        self.code_generator = CodeGenerator(
            self.llm_client,
            max_repair=self.max_repair,
            policy=self.policy,
            template_router=template_router,
        )
        # 报告生成器按需延迟导入，避免模块级循环依赖
        self._report_generator: Any = None
        # 分析历史（内存 + output/agent_history.json 持久化）
        self.history: list[dict[str, Any]] = []
        self._load_history()

    # ------------------------------------------------------------------ #
    # 一、数据文件管理
    # ------------------------------------------------------------------ #
    def _resolve_data_path(self, file_name: str) -> Path:
        """将文件名解析为 data 目录下的真实路径（防目录穿越）。

        Raises:
            FileNotFoundError: 文件不存在。
            ValueError: 文件名非法或超出数据目录范围。
        """
        if not file_name:
            raise ValueError("文件名不能为空")
        candidate = Path(str(file_name))
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (self.data_dir / candidate.name).resolve()
        data_root = self.data_dir.resolve()
        if not resolved.is_relative_to(data_root):
            raise ValueError(f"非法文件名：{file_name}（仅允许访问 data 目录内的文件）")
        if resolved.suffix.lower() != ".csv":
            raise ValueError(f"仅支持 CSV 文件，实际为 {resolved.suffix or '无扩展名'}")
        if not resolved.is_file():
            raise FileNotFoundError(f"数据文件不存在：{resolved.name}")
        return resolved

    def list_datasets(self) -> dict[str, Any]:
        """列出 data 目录下的全部 CSV 数据集及其概要信息。"""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            datasets: list[dict[str, Any]] = []
            for path in sorted(self.data_dir.glob("*.csv")):
                try:
                    stat = path.stat()
                    datasets.append(
                        {
                            "file_name": path.name,
                            "size_kb": round(stat.st_size / 1024, 2),
                            "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
                        }
                    )
                except Exception:  # noqa: BLE001 - 单个文件异常不影响整体列表
                    logger.warning("读取数据文件信息失败：%s", path, exc_info=True)
            return ok({"datasets": datasets, "n_files": len(datasets)}, f"共发现 {len(datasets)} 个 CSV 数据集")
        except Exception as exc:  # noqa: BLE001
            logger.exception("list_datasets 执行失败")
            return fail("数据文件列表读取失败", exc)

    def load_dataframe(self, file_name: str) -> dict[str, Any]:
        """读取数据集为 DataFrame（自动尝试 UTF-8 / GBK / 自动分隔符）。

        Returns:
            dict: 统一结果包，``data`` 含 ``dataframe``（未做 JSON 转换）、``n_rows``、
            ``n_columns``、``read_options``。
        """
        try:
            path = self._resolve_data_path(file_name)
            attempts: list[dict[str, Any]] = [
                {"encoding": "utf-8"},
                {"encoding": "gbk"},
                {"encoding": "utf-8-sig"},
                {"encoding": "utf-8", "sep": None, "engine": "python"},
            ]
            last_error: Exception | None = None
            for options in attempts:
                try:
                    frame = pd.read_csv(path, **options)
                    if frame.shape[0] == 0 or frame.shape[1] == 0:
                        raise ValueError("数据文件为空（无有效行列）")
                    data = {
                        "dataframe": frame,
                        "file_name": path.name,
                        "n_rows": int(frame.shape[0]),
                        "n_columns": int(frame.shape[1]),
                        "read_options": options,
                    }
                    message = (
                        f"已读取 {path.name}：{data['n_rows']} 行 × {data['n_columns']} 列"
                        f"（编码 {options.get('encoding')}）"
                    )
                    return ok(data, message, json_safe=False)
                except Exception as exc:  # noqa: BLE001 - 逐个编码尝试
                    last_error = exc
            return fail(f"数据文件读取失败（已尝试多种编码）：{path.name}", last_error)
        except Exception as exc:  # noqa: BLE001
            logger.exception("load_dataframe 执行失败")
            return fail(f"数据文件 {file_name} 读取失败", exc)

    def describe_dataset(self, file_name: str, preview_rows: int = 5) -> dict[str, Any]:
        """输出数据集画像（字段类型、缺失率、数据预览）。"""
        try:
            loaded = self.load_dataframe(file_name)
            if loaded["status"] != "success":
                return loaded
            frame: pd.DataFrame = loaded["data"]["dataframe"]
            schema = basic_stats.detect_schema(frame)
            preview_rows = max(1, min(int(preview_rows), 50))
            preview = frame.head(preview_rows)
            data = {
                "file_name": loaded["data"]["file_name"],
                "n_rows": loaded["data"]["n_rows"],
                "n_columns": loaded["data"]["n_columns"],
                "read_options": loaded["data"]["read_options"],
                "schema": schema["data"] if schema["status"] == "success" else None,
                "preview": to_jsonable(preview),
                "preview_columns": [str(col) for col in preview.columns],
            }
            return ok(data, f"已生成 {data['file_name']} 的数据画像（{data['n_rows']} 行）")
        except Exception as exc:  # noqa: BLE001
            logger.exception("describe_dataset 执行失败")
            return fail(f"数据画像生成失败：{file_name}", exc)

    def delete_dataset(self, file_name: str) -> dict[str, Any]:
        """删除 data 目录下的指定 CSV 文件（严格限定在数据目录内，且必须是 .csv）。"""
        try:
            path = self._resolve_data_path(file_name)
            size_kb = round(path.stat().st_size / 1024, 2)
            path.unlink()
            logger.info("已删除数据文件：%s", path)
            return ok(
                {"file_name": path.name, "size_kb": size_kb},
                f"已删除数据文件 {path.name}（{size_kb} KB），该操作不可恢复",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("delete_dataset 执行失败")
            return fail(f"数据文件删除失败：{file_name}", exc)

    def save_uploaded_bytes(self, file_name: str, content: bytes, max_mb: float = 200.0) -> dict[str, Any]:
        """保存 Streamlit 上传的 CSV 文件到 data 目录（含大小与后缀校验）。"""
        try:
            if not content:
                raise ValueError("上传内容为空")
            size_mb = len(content) / (1024 * 1024)
            if size_mb > float(max_mb):
                raise ValueError(f"文件大小 {size_mb:.2f} MB 超过上限 {max_mb} MB")
            safe_name = Path(str(file_name or "uploaded.csv")).name
            if not safe_name.lower().endswith(".csv"):
                raise ValueError("仅支持上传 .csv 文件")
            target = (self.data_dir / safe_name).resolve()
            if not target.is_relative_to(self.data_dir.resolve()):
                raise ValueError("非法文件名（不允许写入 data 目录之外的路径）")
            self.data_dir.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            return ok(
                {"file_name": target.name, "size_kb": round(size_mb * 1024, 2)},
                f"已保存数据文件 {target.name}（{size_mb * 1024:.1f} KB）",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("save_uploaded_bytes 执行失败")
            return fail(f"文件保存失败：{file_name}", exc)

    # ------------------------------------------------------------------ #
    # 二、核心闭环：自然语言分析
    # ------------------------------------------------------------------ #
    def run_code(
        self,
        code: str,
        data_file: str | None = None,
        question: str | None = None,
        validate: bool = True,
    ) -> dict[str, Any]:
        """在沙箱中执行分析代码（供 :class:`CodeGenerator` 作为 runner 使用）。"""
        try:
            context = {
                "data_file": data_file,
                "question": question,
                "generated_by": "industrial_data_agent",
                "timeout_seconds": self.policy.timeout,
            }
            return execute_code(
                code,
                policy=self.policy,
                context=context,
                validate=validate,
                data_dir=self.data_dir,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("run_code 执行失败")
            return fail("沙箱执行失败", exc)

    def analyze(
        self,
        question: str,
        data_file: str | None = None,
        use_llm: bool | None = None,
        with_report: bool = True,
        record_history: bool = True,
    ) -> dict[str, Any]:
        """执行完整分析闭环：意图解析 -> 代码生成 -> 安全执行 -> 自动报告。

        Args:
            question: 用户自然语言分析需求。
            data_file: 数据文件名；``None`` 时若 data 目录只有一个 CSV 则自动选择。
            use_llm: 是否允许调用 LLM；``None`` 时使用 Agent 默认设置。
            with_report: 是否生成分析报告。
            record_history: 是否写入历史记录。

        Returns:
            dict: 统一结果包，``data`` 含 ``question``、``intent``、``code``、``execution``、
            ``report``、``report_path``、``charts``、``steps``、``duration_seconds``。

        代码生成采用「本地模板优先」策略：标准场景直接命中内置模板，只有需求超出模板
        覆盖范围（或大模型路由复核判定为定制需求）时才调用 LLM 生成代码；
        ``template_router`` 为 ``None`` 时读取环境变量 ``TEMPLATE_ROUTER``（默认开启）。
        """
        started = time.perf_counter()
        steps: list[dict[str, Any]] = []
        try:
            text = str(question or "").strip()
            if not text:
                return fail("分析需求不能为空")

            # 步骤 1：确定数据文件
            file_name = self._pick_data_file(data_file)
            steps.append({"step": "确定数据文件", "detail": file_name or "未指定"})

            schema: dict[str, Any] | None = None
            if file_name:
                loaded = self.load_dataframe(file_name)
                if loaded["status"] != "success":
                    return fail(loaded["message"], loaded["error"])
                schema_result = basic_stats.detect_schema(loaded["data"]["dataframe"])
                schema = schema_result["data"] if schema_result["status"] == "success" else None
                steps.append(
                    {
                        "step": "读取数据并识别字段",
                        "detail": f"{loaded['data']['n_rows']} 行 × {loaded['data']['n_columns']} 列",
                    }
                )
            else:
                steps.append({"step": "读取数据并识别字段", "detail": "未找到数据文件，代码将依赖自身逻辑"})

            # 步骤 2：意图解析
            intent_result = analyze_intent(text)
            intent = intent_result["data"]["intent"] if intent_result["status"] == "success" else "overview"
            steps.append({"step": "解析分析意图", "detail": INTENT_NAMES.get(intent, intent)})

            # 步骤 3~4：代码生成 + 沙箱执行（含自动修复）
            allow_llm = self.use_llm if use_llm is None else bool(use_llm)
            closure = self.code_generator.generate_and_run(
                question=text,
                runner=lambda code: self.run_code(code, data_file=file_name, question=text),
                schema=schema,
                data_file=file_name,
                use_llm=allow_llm,
            )
            code = (closure.get("data") or {}).get("code", "")
            execution = (closure.get("data") or {}).get("execution") or {
                "status": closure["status"],
                "error": closure["error"],
                "charts": closure.get("charts", []),
                "data": {},
                "message": closure["message"],
            }
            # 为结构化结果补充中文意图名（供报告与前端展示，不改变原始指标）
            # 注意：结果包中顶层 result 与 data.result 是两份内容相同的对象（data 侧经过 JSON 安全转换），
            # 两者都要同步，否则报告读到的仍是未补充的副本。
            if isinstance(execution.get("result"), dict):
                enriched_result = execution["result"]
                enriched_result.setdefault("intent_name", INTENT_NAMES.get(intent, intent))
                if isinstance(execution.get("data"), dict):
                    execution["data"]["result"] = enriched_result
            source = (closure.get("data") or {}).get("source", "template")
            template_reason = str((closure.get("data") or {}).get("template_reason") or "")
            if source == "llm":
                strategy_detail = f"来源：LLM 生成（{template_reason or '需求超出内置模板覆盖范围'}）"
            else:
                strategy_detail = f"来源：本地模板（{template_reason or '标准场景直接匹配'}）"
            if (closure.get("data") or {}).get("router_used"):
                router_note = str((closure.get("data") or {}).get("router_reason") or "")
                strategy_detail += f"；大模型路由复核：{'模板可完成' if source == 'template' else '需定制生成'}"
                if router_note:
                    strategy_detail += f"（{router_note}）"
            steps.append({"step": "生成分析代码", "detail": strategy_detail})
            steps.append(
                {
                    "step": "沙箱安全执行",
                    "detail": execution.get("message") or execution.get("error") or "已执行",
                }
            )

            charts = list(execution.get("charts") or [])
            execution_ok = execution.get("status") == "success"

            # 步骤 5：报告生成
            report_markdown = ""
            report_path: str | None = None
            report_source = "skipped"
            if with_report:
                report_result = self.generate_report(
                    question=text,
                    execution=execution,
                    schema=schema,
                    data_file=file_name,
                    allow_llm=allow_llm,
                )
                if report_result["status"] == "success":
                    report_markdown = report_result["data"]["markdown"]
                    report_path = report_result["data"].get("md_path")
                    report_source = report_result["data"].get("source", "template")
                else:
                    logger.warning("报告生成失败：%s", report_result["error"])
                steps.append(
                    {
                        "step": "自动生成报告",
                        "detail": f"来源：{'LLM' if report_source == 'llm' else '内置模板'}"
                        + (f"，文件：{Path(report_path).name}" if report_path else "（失败）"),
                    }
                )

            duration = round(time.perf_counter() - started, 3)
            data = {
                "question": text,
                "data_file": file_name,
                "intent": intent,
                "intent_name": INTENT_NAMES.get(intent, intent),
                "code": code,
                "code_source": source,
                "generation_strategy": (closure.get("data") or {}).get("generation_strategy"),
                "template_reason": template_reason,
                "llm_skipped": bool((closure.get("data") or {}).get("llm_skipped")),
                "router_used": bool((closure.get("data") or {}).get("router_used")),
                "router_reason": str((closure.get("data") or {}).get("router_reason") or ""),
                "execution": {
                    "status": execution.get("status"),
                    "exit_code": execution.get("exit_code"),
                    "timed_out": execution.get("timed_out"),
                    "duration_seconds": execution.get("duration_seconds"),
                    "stdout": execution.get("stdout", ""),
                    "stderr": execution.get("stderr", ""),
                    "error": execution.get("error"),
                    "violations": execution.get("violations", []),
                    "result": execution.get("result"),
                    "code_path": execution.get("code_path"),
                    "workdir": execution.get("workdir"),
                },
                "report": report_markdown,
                "report_path": report_path,
                "report_source": report_source,
                "charts": charts,
                "steps": steps,
                "duration_seconds": duration,
                "llm_available": bool(getattr(self.llm_client, "is_available", False)),
                "llm_used": bool(source == "llm" or report_source == "llm"),
            }

            if record_history:
                self._append_history(
                    {
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "question": text,
                        "data_file": file_name,
                        "intent": INTENT_NAMES.get(intent, intent),
                        "status": execution.get("status"),
                        "code_source": source,
                        "report_path": report_path,
                        "n_charts": len(charts),
                        "duration_seconds": duration,
                        "error": execution.get("error"),
                    }
                )

            if execution_ok:
                return ok(data, f"分析完成（{INTENT_NAMES.get(intent, intent)}，耗时 {duration:.2f}s）", charts=charts)
            return fail(
                f"分析未成功完成：{execution.get('error') or '代码执行失败'}",
                execution.get("error"),
                data=data,
                charts=charts,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("analyze 执行失败")
            return fail("分析流程执行失败", exc, data={"steps": steps})

    def generate_report(
        self,
        question: str,
        execution: Mapping[str, Any],
        schema: Mapping[str, Any] | None = None,
        data_file: str | None = None,
        allow_llm: bool = True,
    ) -> dict[str, Any]:
        """调用报告生成器（延迟导入，避免模块级循环依赖）。"""
        try:
            if self._report_generator is None:
                from report.generator import ReportGenerator

                self._report_generator = ReportGenerator(
                    llm_client=self.llm_client if allow_llm else None,
                )
            return self._report_generator.generate(
                question=question,
                execution=execution,
                schema=schema,
                data_file=data_file,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("generate_report 执行失败")
            return fail("报告生成失败", exc)

    # ------------------------------------------------------------------ #
    # 三、算法工具箱（界面可直接参数化调用）
    # ------------------------------------------------------------------ #
    def _with_dataframe(self, file_name: str, function: Any) -> dict[str, Any]:
        """读取数据后执行给定分析函数（统一异常处理与结果包）。"""
        try:
            loaded = self.load_dataframe(file_name)
            if loaded["status"] != "success":
                return loaded
            return function(loaded["data"]["dataframe"])
        except Exception as exc:  # noqa: BLE001
            logger.exception("工具箱分析执行失败")
            return fail(f"分析执行失败：{file_name}", exc)

    def run_basic_stats(self, file_name: str, columns: Sequence[str] | None = None) -> dict[str, Any]:
        """工具箱：基础描述性统计。"""
        return self._with_dataframe(file_name, lambda frame: basic_stats.describe_columns(frame, columns))

    def run_correlation(
        self,
        file_name: str,
        columns: Sequence[str] | None = None,
        method: str = "pearson",
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """工具箱：相关性矩阵 + 强相关特征对。"""
        return self._with_dataframe(
            file_name,
            lambda frame: self._combine(
                correlation.correlation_matrix(frame, columns=columns, method=method),
                correlation.strong_correlations(frame, columns=columns, method=method, threshold=threshold),
            ),
        )

    def run_time_lag(
        self,
        file_name: str,
        x_column: str,
        y_column: str,
        max_lag: int = 10,
        method: str = "pearson",
        time_column: str | None = None,
        sample_interval: float | None = None,
    ) -> dict[str, Any]:
        """工具箱：时序时滞分析。"""
        return self._with_dataframe(
            file_name,
            lambda frame: time_lag.time_lag_analysis(
                frame,
                x_column=x_column,
                y_column=y_column,
                max_lag=max_lag,
                method=method,
                time_column=time_column,
                sample_interval=sample_interval,
            ),
        )

    def run_anomaly(
        self,
        file_name: str,
        columns: Sequence[str] | None = None,
        threshold: float = 3.0,
        side: str = "both",
    ) -> dict[str, Any]:
        """工具箱：3σ 异常检测。"""
        return self._with_dataframe(
            file_name,
            lambda frame: anomaly_module.detect_3sigma_multi(
                frame, columns=columns, threshold=threshold, side=side
            ),
        )

    def run_ml_train(
        self,
        file_name: str,
        target_column: str,
        feature_columns: Sequence[str] | None = None,
        model_type: str = "random_forest",
        test_size: float = 0.2,
        cross_validation: int = 5,
        scale_features: bool = False,
        **model_params: Any,
    ) -> dict[str, Any]:
        """工具箱：机器学习建模。"""
        return self._with_dataframe(
            file_name,
            lambda frame: ml_models.train_model(
                frame,
                target_column=target_column,
                feature_columns=feature_columns,
                model_type=model_type,
                test_size=test_size,
                cross_validation=cross_validation,
                scale_features=scale_features,
                **model_params,
            ),
        )

    def run_ml_predict(
        self,
        file_name: str,
        model_path: str | None = None,
        feature_columns: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """工具箱：模型推理（未指定模型时使用最新训练的模型）。"""
        try:
            if not model_path:
                # 兼容两种场景：自定义输出目录、以及 analytics 默认模型目录 output/models
                candidates: list[Path] = []
                for directory in {self.output_dir / "models", MODEL_DIR}:
                    if directory.exists():
                        candidates.extend(directory.glob("*.joblib"))
                candidates = sorted(candidates, key=lambda item: item.stat().st_mtime)
                if not candidates:
                    return fail("未找到已保存模型，请先完成一次建模训练")
                model_path = str(candidates[-1])
            return self._with_dataframe(
                file_name,
                lambda frame: ml_models.predict(str(model_path), frame, feature_columns=feature_columns),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("run_ml_predict 执行失败")
            return fail("模型推理失败", exc)

    @staticmethod
    def _combine(primary: Mapping[str, Any], secondary: Mapping[str, Any]) -> dict[str, Any]:
        """合并两个分析结果包（用于相关性矩阵 + 强相关对这类组合输出）。"""
        try:
            charts = list(primary.get("charts") or []) + list(secondary.get("charts") or [])
            if primary["status"] != "success":
                return fail(primary["message"], primary["error"], charts=charts)
            data = {
                "matrix": primary["data"],
                "strong_pairs": secondary["data"].get("pairs", []) if secondary["status"] == "success" else [],
                "strong_pairs_message": secondary["message"],
            }
            return ok(data, primary["message"], charts=charts)
        except Exception as exc:  # noqa: BLE001
            return fail("结果合并失败", exc)

    # ------------------------------------------------------------------ #
    # 四、系统状态与历史记录
    # ------------------------------------------------------------------ #
    def system_status(self) -> dict[str, Any]:
        """汇总系统运行状态（配置、沙箱策略、数据与产物统计）。"""
        try:
            settings = get_settings()
            datasets = self.list_datasets()
            charts = list((self.output_dir / "charts").glob("*.png")) if (self.output_dir / "charts").exists() else []
            reports = list((self.output_dir / "reports").glob("*.md")) if (self.output_dir / "reports").exists() else []
            models = list((self.output_dir / "models").glob("*.joblib")) if (self.output_dir / "models").exists() else []
            last_errors = [item.get("error") for item in self.history[-5:] if item.get("error")]
            data = {
                "settings": settings["data"] if settings["status"] == "success" else None,
                "sandbox": {
                    "timeout": self.policy.timeout,
                    "max_output_chars": self.policy.max_output_chars,
                    "memory_limit_mb": self.policy.memory_limit_mb,
                    "allow_while_true": self.policy.allow_while_true,
                },
                "llm": self.llm_client.describe() if hasattr(self.llm_client, "describe") else {},
                "stats": {
                    "n_datasets": datasets["data"].get("n_files", 0) if datasets["status"] == "success" else 0,
                    "n_charts": len(charts),
                    "n_reports": len(reports),
                    "n_models": len(models),
                    "n_history": len(self.history),
                },
                "recent_errors": last_errors,
            }
            return ok(data, "系统状态读取完成")
        except Exception as exc:  # noqa: BLE001
            logger.exception("system_status 执行失败")
            return fail("系统状态读取失败", exc)

    def read_history(self, limit: int = 20) -> dict[str, Any]:
        """读取最近的分析历史记录。"""
        try:
            limit = max(1, int(limit))
            records = list(reversed(self.history[-limit:]))
            return ok({"records": records, "n_records": len(records)}, f"已返回最近 {len(records)} 条分析记录")
        except Exception as exc:  # noqa: BLE001
            logger.exception("read_history 执行失败")
            return fail("历史记录读取失败", exc)

    def _append_history(self, record: Mapping[str, Any]) -> None:
        """追加一条历史记录并持久化到 ``output/agent_history.json``。"""
        try:
            self.history.append(dict(record))
            self.history = self.history[-MAX_HISTORY:]
            self.output_dir.mkdir(parents=True, exist_ok=True)
            history_path = self.output_dir / HISTORY_FILE_NAME
            with open(history_path, "w", encoding="utf-8") as handle:
                json.dump(self.history, handle, ensure_ascii=False, indent=2, default=str)
        except Exception:  # noqa: BLE001 - 历史记录失败不影响主流程
            logger.warning("历史记录写入失败", exc_info=True)

    def _load_history(self) -> None:
        """启动时加载历史记录（文件损坏时静默重置）。"""
        try:
            history_path = self.output_dir / HISTORY_FILE_NAME
            if history_path.exists():
                with open(history_path, "r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                if isinstance(loaded, list):
                    self.history = [item for item in loaded if isinstance(item, dict)][-MAX_HISTORY:]
        except Exception:  # noqa: BLE001
            logger.warning("历史记录加载失败，已重置", exc_info=True)
            self.history = []

    def _pick_data_file(self, data_file: str | None) -> str | None:
        """选择数据文件：显式指定优先；否则在仅有单一数据集时自动选择。"""
        try:
            if data_file:
                return Path(str(data_file)).name
            listing = self.list_datasets()
            if listing["status"] == "success" and listing["data"]["n_files"] == 1:
                return listing["data"]["datasets"][0]["file_name"]
            return None
        except Exception:  # noqa: BLE001
            return None
