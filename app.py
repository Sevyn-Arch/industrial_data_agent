"""工业数据分析 Agent 系统 —— Streamlit 主程序入口。

启动方式::

    cd industrial_data_agent
    streamlit run app.py

界面分为四个页签：

1. 数据文件管理：CSV 上传、字段与类型识别、预览、删除；
2. 自然语言分析：输入分析需求 -> 自动生成代码 -> 沙箱执行 -> 自动生成报告；
3. 算法工具箱：不依赖 LLM，按参数直接调用统计/相关性/时滞/异常/建模算法；
4. 系统状态：模型与沙箱配置、运行统计、安全说明与分析历史。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# 保证以 `streamlit run app.py` 直接启动时能正确导入项目包
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.agent_core import IndustrialDataAgent  # noqa: E402
from agent.code_generator import INTENT_NAMES  # noqa: E402
from agent.llm_client import get_settings  # noqa: E402
from analytics import OUTPUT_DIR, REPORT_DIR, ensure_dirs  # noqa: E402
from sandbox.executor import ERROR_MESSAGES as SANDBOX_ERROR_MESSAGES  # noqa: E402
from sandbox.executor import SandboxPolicy  # noqa: E402

from analytics import basic_stats  # noqa: E402
from analytics import correlation, time_lag  # noqa: E402
from analytics import ml_models  # noqa: E402

st.set_page_config(
    page_title="工业数据分析 Agent 系统",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

_LOGGING_READY = False

#: 示例问题（一键填充）
SAMPLE_QUESTIONS: tuple[str, ...] = (
    "请统计 inlet_temp_c 的均值、标准差、最大值和最小值",
    "分析各变量之间的相关性，找出强相关特征对",
    "inlet_temp_c 与 outlet_temp_c 之间是否存在时滞，滞后多少步",
    "用 3σ 原则检测 inlet_temp_c 和 outlet_temp_c 的异常点",
    "训练随机森林模型预测 yield_pct，并输出 R² 与特征重要性",
    "检查数据的缺失值情况并评估数据质量",
)


# --------------------------------------------------------------------------- #
# 一、公共工具
# --------------------------------------------------------------------------- #
def configure_logging(level: str = "INFO") -> None:
    """配置根日志（只执行一次），使各模块日志可输出到控制台。"""
    global _LOGGING_READY
    if _LOGGING_READY:
        return
    try:
        logging.basicConfig(
            level=getattr(logging, str(level).upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    except Exception:  # noqa: BLE001 - 日志配置失败不影响主流程
        pass
    _LOGGING_READY = True


@st.cache_resource(show_spinner=False)
def get_agent(use_llm: bool, sandbox_timeout: float, max_repair: int, template_router: bool) -> IndustrialDataAgent:
    """创建（并缓存）Agent 实例；参数为可哈希的基础类型，便于 Streamlit 缓存。"""
    policy = SandboxPolicy(
        timeout=float(sandbox_timeout),
        max_output_chars=20000,
        keep_workspace=True,
        max_workspaces=10,
    )
    ensure_dirs()
    return IndustrialDataAgent(
        policy=policy,
        max_repair=int(max_repair),
        use_llm=bool(use_llm),
        template_router=bool(template_router),
    )


def show_result_message(result: dict[str, Any], success_text: str = "操作成功") -> bool:
    """按统一结果包渲染成功/失败提示，返回是否成功。"""
    try:
        if result.get("status") == "success":
            st.success(result.get("message") or success_text)
            return True
        detail = result.get("error")
        st.error(result.get("message") or "操作失败")
        if detail:
            st.caption(f"错误详情：{detail}")
        return False
    except Exception as exc:  # noqa: BLE001
        st.error(f"结果渲染失败：{exc}")
        return False


def render_charts(charts: list[str], caption_prefix: str = "图表") -> None:
    """以网格形式展示本次分析产出的图表。"""
    try:
        valid = [path for path in charts or [] if Path(path).is_file()]
        if not valid:
            return
        st.markdown(f"**{caption_prefix}（{len(valid)} 张）**")
        columns = st.columns(2)
        for index, path in enumerate(valid):
            with columns[index % 2]:
                st.image(path, caption=Path(path).name, use_column_width=True)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"图表展示失败：{exc}")


def download_buttons(markdown: str, charts: list[str], base_name: str) -> None:
    """提供报告 Markdown / HTML 下载按钮。"""
    try:
        from report.generator import ReportGenerator

        columns = st.columns(2)
        with columns[0]:
            st.download_button(
                "⬇️ 下载 Markdown 报告",
                data=str(markdown).encode("utf-8"),
                file_name=f"{base_name}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with columns[1]:
            html_content = ReportGenerator(llm_client=None).markdown_to_html(markdown, charts, title=base_name)
            st.download_button(
                "⬇️ 下载 HTML 报告（图表内嵌）",
                data=html_content.encode("utf-8"),
                file_name=f"{base_name}.html",
                mime="text/html",
                use_container_width=True,
            )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"报告下载按钮生成失败：{exc}")


def column_options(agent: IndustrialDataAgent, file_name: str | None) -> dict[str, list[str]]:
    """读取数据集字段清单，供工具箱各类参数选择使用。"""
    try:
        if not file_name:
            return {"all": [], "numeric": [], "datetime": [], "categorical": []}
        described = agent.describe_dataset(file_name, preview_rows=1)
        if described["status"] != "success":
            st.warning(described["message"])
            return {"all": [], "numeric": [], "datetime": [], "categorical": []}
        schema = described["data"]["schema"] or {}
        return {
            "all": [item["name"] for item in schema.get("columns", [])],
            "numeric": list(schema.get("numeric_columns") or []),
            "datetime": list(schema.get("datetime_columns") or []),
            "categorical": list(schema.get("categorical_columns") or []),
        }
    except Exception as exc:  # noqa: BLE001
        st.warning(f"字段清单读取失败：{exc}")
        return {"all": [], "numeric": [], "datetime": [], "categorical": []}


# --------------------------------------------------------------------------- #
# 二、页签一：数据文件管理
# --------------------------------------------------------------------------- #
def render_data_tab(agent: IndustrialDataAgent, settings: dict[str, Any]) -> None:
    """数据文件管理页签：上传、查看、字段识别、删除。"""
    try:
        st.subheader("数据文件管理")
        max_upload_mb = float(settings.get("max_upload_mb") or 200.0)
        uploaded = st.file_uploader(
            f"上传 CSV 数据文件（单个文件不超过 {max_upload_mb:.0f} MB）",
            type=["csv"],
            accept_multiple_files=False,
        )
        if uploaded is not None and st.button("保存到 data 目录", type="primary", key="btn_save_upload"):
            with st.spinner("正在校验并保存文件..."):
                saved = agent.save_uploaded_bytes(uploaded.name, uploaded.getvalue(), max_mb=max_upload_mb)
            show_result_message(saved, "文件已保存")
            st.rerun()

        listing = agent.list_datasets()
        if listing["status"] != "success":
            show_result_message(listing)
            return
        datasets = listing["data"]["datasets"]
        if not datasets:
            st.info("data 目录暂无 CSV 数据文件，请先上传或把 CSV 放入 data 目录。")
            return

        frame = pd.DataFrame(datasets)
        st.dataframe(frame, use_container_width=True, hide_index=True)
        st.caption(listing["message"])

        selected = st.selectbox("选择要查看的数据文件", [item["file_name"] for item in datasets], key="data_view")
        preview_rows = st.slider("预览行数", min_value=3, max_value=50, value=8, step=1)

        described = agent.describe_dataset(selected, preview_rows=preview_rows)
        if not show_result_message(described):
            return
        payload = described["data"]
        schema = payload["schema"] or {}

        metrics = st.columns(4)
        metrics[0].metric("行数", payload["n_rows"])
        metrics[1].metric("列数", payload["n_columns"])
        metrics[2].metric("数值字段", len(schema.get("numeric_columns") or []))
        metrics[3].metric("时间字段", len(schema.get("datetime_columns") or []))

        st.markdown("**字段与数据类型识别结果**")
        st.dataframe(pd.DataFrame(schema.get("columns") or []), use_container_width=True, hide_index=True)

        st.markdown("**数据预览**")
        st.dataframe(pd.DataFrame(payload["preview"]), use_container_width=True, hide_index=True)

        with st.expander("缺失值统计（含缺失率柱状图）"):
            if st.button("运行缺失值分析", key="btn_missing"):
                loaded = agent.load_dataframe(selected)
                if loaded["status"] == "success":
                    report = basic_stats.missing_report(loaded["data"]["dataframe"])
                    show_result_message(report)
                    render_charts(report["charts"], "缺失率图表")
                    st.dataframe(pd.DataFrame(report["data"]["columns"]), use_container_width=True, hide_index=True)
                else:
                    show_result_message(loaded)

        with st.expander("删除数据文件（不可恢复，请谨慎操作）"):
            st.warning("删除后无法通过本系统恢复，请确认文件已备份。")
            confirm = st.checkbox(f"我确认删除 {selected}", key="confirm_delete")
            if st.button("删除该文件", type="secondary", disabled=not confirm):
                deleted = agent.delete_dataset(selected)
                show_result_message(deleted, "文件已删除")
                st.rerun()
    except Exception as exc:  # noqa: BLE001 - 界面层兜底，禁止异常外泄
        st.error(f"数据管理页签渲染失败：{exc}")


# --------------------------------------------------------------------------- #
# 三、页签二：自然语言分析
# --------------------------------------------------------------------------- #
def render_analysis_tab(agent: IndustrialDataAgent, settings: dict[str, Any]) -> None:
    """自然语言分析页签：提问 -> 代码生成 -> 沙箱执行 -> 自动报告。"""
    try:
        st.subheader("自然语言分析（提问 → 自动执行 → 自动报告）")
        listing = agent.list_datasets()
        datasets = listing["data"]["datasets"] if listing["status"] == "success" else []
        options = ["（不指定，使用目录中唯一文件）"] + [item["file_name"] for item in datasets]

        top = st.columns([3, 2])
        with top[0]:
            data_choice = st.selectbox("选择数据文件", options, key="analysis_file")
        with top[1]:
            llm_available = bool(getattr(agent.llm_client, "is_available", False))
            st.markdown(
                "**代码/报告生成模式**：" + ("LLM（DeepSeek）" if llm_available else "本地模板（未配置 API Key）")
            )
            st.caption(
                f"代码模型：{settings.get('code_model')}｜报告模型：{settings.get('report_model')}"
            )

        st.markdown("**示例问题（点击填充）**")
        sample_columns = st.columns(3)
        for index, sample in enumerate(SAMPLE_QUESTIONS):
            with sample_columns[index % 3]:
                if st.button(sample[:26] + ("…" if len(sample) > 26 else ""), key=f"sample_{index}"):
                    st.session_state["question_input"] = sample

        question = st.text_area(
            "请输入分析需求（自然语言）",
            key="question_input",
            height=110,
            placeholder="例如：inlet_temp_c 与 outlet_temp_c 之间是否存在时滞？滞后多少步？",
        )
        controls = st.columns(4)
        with controls[0]:
            with_report = st.checkbox("生成分析报告", value=True)
        with controls[1]:
            use_llm = st.checkbox(
                "允许调用 LLM",
                value=llm_available,
                disabled=not llm_available,
                help=(
                    "模板优先策略：基础统计、相关性、时滞、3σ 异常检测、标准建模等场景直接使用"
                    "内置模板（不消耗 API 额度）；仅当需求超出模板覆盖范围（如筛选、比较、分组等"
                    "自定义逻辑）时才调用大模型生成代码。若确需对标准场景强制走大模型，"
                    "可在提问中写明「必须调用大语言模型生成代码」。"
                ),
            )
        with controls[2]:
            show_code = st.checkbox("展示生成代码", value=True)
        with controls[3]:
            run_now = st.button("开始分析", type="primary", use_container_width=True, key="btn_run_analysis")

        if run_now:
            if not question.strip():
                st.warning("请先输入分析需求。")
                st.caption("🛡️ 所有生成代码均在独立沙箱中执行，系统会拦截危险操作，保障服务器安全")
                return
            file_name = None if data_choice.startswith("（不指定") else data_choice
            with st.spinner("正在解析意图、生成并执行分析代码..."):
                result = agent.analyze(
                    question=question,
                    data_file=file_name,
                    use_llm=use_llm,
                    with_report=with_report,
                )
            st.session_state["analysis_result"] = result

        result = st.session_state.get("analysis_result")
        if not result:
            st.info("等待执行：输入分析需求后点击「开始分析」。")
            st.caption("🛡️ 所有生成代码均在独立沙箱中执行，系统会拦截危险操作，保障服务器安全")
            return

        data = result.get("data") or {}
        execution = data.get("execution") or {}
        error_type = execution.get("error_type")
        st.divider()
        if error_type:
            # 前端按后端 error_type 展示统一文案（与安全沙箱的三种失败类型一一对应）
            st.error(SANDBOX_ERROR_MESSAGES.get(error_type, SANDBOX_ERROR_MESSAGES["runtime"]))
        else:
            show_result_message(result, "分析完成")

        info = st.columns(4)
        info[0].metric("识别意图", data.get("intent_name") or "-")
        info[1].metric("代码来源", "LLM" if data.get("code_source") == "llm" else "本地模板")
        info[2].metric("执行状态", "成功" if (data.get("execution") or {}).get("status") == "success" else "失败")
        info[3].metric("总耗时", f"{data.get('duration_seconds', 0)} 秒")

        strategy = data.get("generation_strategy") or ""
        if strategy == "template-first":
            st.caption(f"代码生成策略：模板优先 ｜ {data.get('template_reason') or '标准场景直接命中内置模板'}（未调用 LLM）")
        elif strategy:
            st.caption(
                f"代码生成策略：LLM 兜底 ｜ {data.get('template_reason') or '需求超出模板覆盖范围'}"
            )
        if data.get("router_used"):
            st.caption(
                f"智能路由复核：已启用 ｜ 判定理由：{data.get('router_reason') or '模板可完整表达该需求'}"
            )

        st.markdown("**执行流程**")
        for step in data.get("steps") or []:
            st.markdown(f"- **{step.get('step')}**：{step.get('detail')}")

        if execution.get("violations"):
            st.markdown("**安全校验拦截项**")
            st.code("\n".join(f"- {item}" for item in execution["violations"]), language="text")

        if show_code and data.get("code"):
            st.markdown("**生成的 Python 分析代码**")
            st.code(data["code"], language="python")
            st.caption(f"代码文件：{execution.get('code_path') or '（未记录）'}")

        with st.expander("沙箱执行详情（安全验证证据）", expanded=execution.get("status") != "success"):
            details = st.columns(4)
            details[0].metric("退出码", execution.get("exit_code"))
            details[1].metric("执行耗时", f"{execution.get('duration_seconds')} 秒")
            details[2].metric("是否超时", "是" if execution.get("timed_out") else "否")
            details[3].metric("工作目录", "独立沙箱目录")
            if error_type:
                # 顶部错误横幅已按 error_type 展示统一文案，这里补充类型标签便于排查
                labels = {"security": "安全拦截", "runtime": "代码运行错误", "timeout": "执行超时"}
                st.markdown(f"**错误类型**：{labels.get(error_type, '执行错误')}")
            if execution.get("error"):
                st.caption(f"详细错误：{execution['error']}")
            if execution.get("stdout"):
                st.markdown("**标准输出**")
                st.code(execution["stdout"], language="text")
            if execution.get("stderr"):
                st.markdown("**标准错误**")
                st.code(execution["stderr"], language="text")
            if execution.get("result") is not None:
                st.markdown("**结构化结果（emit_result 输出）**")
                st.json(execution["result"])

        render_charts(list(data.get("charts") or []), "分析图表")

        if with_report and data.get("report"):
            st.markdown("**自动生成的分析报告**")
            st.markdown(data["report"])
            base_name = Path(str(data.get("report_path") or "report")).stem
            download_buttons(data["report"], list(data.get("charts") or []), base_name)

        st.caption("🛡️ 所有生成代码均在独立沙箱中执行，系统会拦截危险操作，保障服务器安全")
    except Exception as exc:  # noqa: BLE001
        st.error(f"分析页签渲染失败：{exc}")


# --------------------------------------------------------------------------- #
# 四、页签三：算法工具箱（不依赖 LLM）
# --------------------------------------------------------------------------- #
def render_toolbox_tab(agent: IndustrialDataAgent) -> None:
    """算法工具箱：按参数直接调用统计 / 相关性 / 时滞 / 异常 / 建模算法。"""
    try:
        st.subheader("算法工具箱（规则化参数调用，可离线运行）")
        listing = agent.list_datasets()
        datasets = listing["data"]["datasets"] if listing["status"] == "success" else []
        if not datasets:
            st.info("请先在「数据文件管理」上传 CSV 数据。")
            return

        file_names = [item["file_name"] for item in datasets]
        selected_file = st.selectbox("选择数据文件", file_names, key="toolbox_file")
        options = column_options(agent, selected_file)
        numeric_columns = options["numeric"]

        tool = st.selectbox(
            "选择分析工具",
            (
                "基础统计（均值/标准差/最值/中位数）",
                "相关性分析",
                "时序时滞分析",
                "3σ 异常检测",
                "机器学习建模",
                "模型推理预测",
            ),
            key="toolbox_tool",
        )

        result: dict[str, Any] | None = None
        charts: list[str] = []

        if tool.startswith("基础统计"):
            columns = st.multiselect("统计字段（留空表示全部数值字段）", numeric_columns, key="tb_stats_cols")
            if st.button("运行基础统计", type="primary"):
                with st.spinner("正在计算描述性统计..."):
                    result = agent.run_basic_stats(selected_file, columns or None)
                if result["status"] == "success":
                    statistics = result["data"]["statistics"]
                    st.dataframe(pd.DataFrame(statistics).T, use_container_width=True)
                charts = result.get("charts") or []

        elif tool.startswith("相关性"):
            columns = st.multiselect("参与计算的字段（留空表示全部数值字段）", numeric_columns, key="tb_corr_cols")
            method = st.selectbox("相关性方法", ("pearson", "spearman", "kendall"), key="tb_corr_method")
            threshold = st.slider("强相关阈值 |r|", min_value=0.1, max_value=0.95, value=0.5, step=0.05)
            if st.button("运行相关性分析", type="primary"):
                with st.spinner("正在计算相关性矩阵..."):
                    result = agent.run_correlation(selected_file, columns or None, method, threshold)
                if result["status"] == "success":
                    matrix = pd.DataFrame(result["data"]["matrix"]["matrix"])
                    st.dataframe(matrix, use_container_width=True)
                    pairs = result["data"]["strong_pairs"]
                    if pairs:
                        st.markdown("**强相关特征对**")
                        st.dataframe(pd.DataFrame(pairs), use_container_width=True, hide_index=True)
                    else:
                        st.info("未发现超过阈值的强相关特征对。")
                charts = result.get("charts") or []

        elif tool.startswith("时序时滞"):
            if len(numeric_columns) < 2:
                st.warning("时滞分析至少需要两个数值字段。")
            else:
                controls = st.columns(2)
                with controls[0]:
                    x_column = st.selectbox("自变量 x（领先变量候选）", numeric_columns, key="tb_lag_x")
                with controls[1]:
                    y_column = st.selectbox("因变量 y", numeric_columns, index=1, key="tb_lag_y")
                controls2 = st.columns(3)
                with controls2[0]:
                    max_lag = st.slider("最大滞后步数", 1, 60, 10)
                with controls2[1]:
                    time_column = st.selectbox(
                        "时间列（可选，用于排序）",
                        ["（不使用）"] + options["all"],
                        key="tb_lag_time",
                    )
                with controls2[2]:
                    sample_interval = st.number_input("采样间隔（秒，0 表示不换算）", min_value=0.0, value=300.0)
                if st.button("运行时滞分析", type="primary"):
                    with st.spinner("正在计算互相关曲线..."):
                        result = agent.run_time_lag(
                            selected_file,
                            x_column=x_column,
                            y_column=y_column,
                            max_lag=int(max_lag),
                            time_column=None if time_column.startswith("（") else time_column,
                            sample_interval=None if sample_interval <= 0 else float(sample_interval),
                        )
                    if result["status"] == "success":
                        st.info(result["data"]["conclusion"])
                        st.dataframe(pd.DataFrame(result["data"]["top_lags"]), use_container_width=True, hide_index=True)
                    charts = result.get("charts") or []

        elif tool.startswith("3σ"):
            columns = st.multiselect("检测字段（留空表示前 4 个数值字段）", numeric_columns, key="tb_anom_cols")
            controls = st.columns(2)
            with controls[0]:
                threshold = st.slider("σ 倍数阈值", 1.0, 5.0, 3.0, 0.1)
            with controls[1]:
                side = st.selectbox("判定方向", ("both", "upper", "lower"), key="tb_anom_side")
            if st.button("运行异常检测", type="primary"):
                with st.spinner("正在执行 3σ 检测..."):
                    result = agent.run_anomaly(
                        selected_file, columns or None, threshold=float(threshold), side=side
                    )
                if result["status"] == "success":
                    summary = [
                        {
                            "字段": name,
                            "异常点数": item["n_anomalies"],
                            "异常比例": item["anomaly_rate"],
                            "下界": item["lower_bound"],
                            "上界": item["upper_bound"],
                        }
                        for name, item in result["data"]["columns"].items()
                    ]
                    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
                charts = result.get("charts") or []

        elif tool.startswith("机器学习"):
            controls = st.columns(2)
            with controls[0]:
                target = st.selectbox("目标字段（被预测变量）", numeric_columns, index=len(numeric_columns) - 1 if numeric_columns else 0, key="tb_ml_target")
            with controls[1]:
                model_type = st.selectbox(
                    "模型类型",
                    tuple(ml_models.SUPPORTED_MODELS.keys()),
                    format_func=lambda key: ml_models.SUPPORTED_MODELS[key],
                    key="tb_ml_type",
                )
            feature_candidates = [name for name in numeric_columns if name != target]
            features = st.multiselect("特征字段（留空表示其余全部数值字段）", feature_candidates, key="tb_ml_features")
            controls2 = st.columns(3)
            with controls2[0]:
                test_size = st.slider("测试集比例", 0.1, 0.5, 0.2, 0.05)
            with controls2[1]:
                cv_folds = st.slider("交叉验证折数（1 表示跳过）", 1, 10, 5)
            with controls2[2]:
                scale_features = st.checkbox("特征标准化（线性模型建议开启）", value=False)
            if st.button("开始训练模型", type="primary"):
                with st.spinner("正在训练模型并评估..."):
                    result = agent.run_ml_train(
                        selected_file,
                        target_column=target,
                        feature_columns=features or None,
                        model_type=model_type,
                        test_size=float(test_size),
                        cross_validation=0 if int(cv_folds) < 2 else int(cv_folds),
                        scale_features=bool(scale_features),
                    )
                if result["status"] == "success":
                    metrics = result["data"]["metrics"]
                    cards = st.columns(4)
                    cards[0].metric("R²", metrics.get("r2"))
                    cards[1].metric("MSE", metrics.get("mse"))
                    cards[2].metric("RMSE", metrics.get("rmse"))
                    cards[3].metric("MAE", metrics.get("mae"))
                    importance = result["data"]["feature_importance"]
                    if importance:
                        st.dataframe(pd.DataFrame(importance), use_container_width=True, hide_index=True)
                    if result["data"].get("model_path"):
                        st.caption(f"模型已保存：{result['data']['model_path']}")
                charts = result.get("charts") or []

        else:  # 模型推理预测
            controls = st.columns(2)
            with controls[0]:
                model_path = st.text_input("模型文件路径（留空自动使用最新模型）", value="")
            with controls[1]:
                features = st.multiselect("推理特征字段（留空使用模型元数据）", numeric_columns, key="tb_predict_features")
            if st.button("执行推理", type="primary"):
                with st.spinner("正在加载模型并推理..."):
                    result = agent.run_ml_predict(
                        selected_file,
                        model_path=model_path.strip() or None,
                        feature_columns=features or None,
                    )
                if result["status"] == "success":
                    payload = result["data"]
                    cards = st.columns(3)
                    cards[0].metric("预测条数", payload.get("n_predicted"))
                    cards[1].metric("预测均值", (payload.get("prediction_stats") or {}).get("mean"))
                    cards[2].metric("预测最大值", (payload.get("prediction_stats") or {}).get("max"))
                    preview = pd.DataFrame(
                        {"行号": payload.get("row_indices", []), "预测值": payload.get("predictions", [])}
                    )
                    st.dataframe(preview, use_container_width=True, hide_index=True)
                charts = result.get("charts") or []

        if result is not None:
            st.divider()
            show_result_message(result)
            if result["status"] == "error" and result.get("data"):
                st.json(result["data"])
            render_charts(charts, "本轮产出图表")
    except Exception as exc:  # noqa: BLE001
        st.error(f"工具箱页签渲染失败：{exc}")


# --------------------------------------------------------------------------- #
# 五、页签四：系统状态
# --------------------------------------------------------------------------- #
def render_status_tab(agent: IndustrialDataAgent, settings: dict[str, Any]) -> None:
    """系统状态：配置、沙箱策略、运行统计、安全说明与历史记录。"""
    try:
        st.subheader("系统状态与安全说明")
        status = agent.system_status()
        if status["status"] != "success":
            show_result_message(status)
            return
        payload = status["data"]

        left, right = st.columns(2)
        with left:
            st.markdown("**LLM 配置**")
            st.json(payload.get("llm") or {})
            st.caption(f"配置文件：{settings.get('env_file')}（存在：{settings.get('env_file_exists')}）")
        with right:
            st.markdown("**沙箱策略**")
            st.json(payload.get("sandbox") or {})

        stats = payload.get("stats") or {}
        cards = st.columns(5)
        cards[0].metric("数据集", stats.get("n_datasets", 0))
        cards[1].metric("图表", stats.get("n_charts", 0))
        cards[2].metric("报告", stats.get("n_reports", 0))
        cards[3].metric("模型", stats.get("n_models", 0))
        cards[4].metric("历史记录", stats.get("n_history", 0))

        st.markdown("**安全机制说明（生成代码执行路径）**")
        st.markdown(
            "\n".join(
                [
                    "- 危险代码校验：AST 白名单（仅允许 pandas / numpy / scipy / sklearn / matplotlib / seaborn / "
                    "joblib / analytics 等），禁用 `exec`、`eval`、`open`、`__import__`、魔术属性与 `while True`；",
                    "- 目录权限限制：子进程 cwd 为 `output/.sandbox/run_<uuid>` 独立空目录，"
                    "代码中的绝对路径必须位于 `data/` 或 `output/` 之内；",
                    "- 执行超时：默认 60 秒，超时后 `taskkill /T`（Windows）或 `os.killpg`（POSIX）杀掉整个进程树；",
                    "- 环境变量脱敏：子进程仅透传 PATH/SYSTEMROOT 等系统变量，`DEEPSEEK_API_KEY` 等敏感信息不会外泄；",
                    "- 执行方式：`python -I -X utf8 <script>` 子进程执行，绝不使用 `exec()`；",
                    "- 残余风险：subprocess 并非强隔离边界，生产环境建议再加一层容器与低权限账户。",
                ]
            )
        )

        if payload.get("recent_errors"):
            st.markdown("**最近错误摘要**")
            st.write(list(payload["recent_errors"]))

        st.markdown("**分析历史（最近 20 条）**")
        history = agent.read_history(limit=20)
        if history["status"] == "success" and history["data"]["records"]:
            st.dataframe(pd.DataFrame(history["data"]["records"]), use_container_width=True, hide_index=True)
        else:
            st.caption("暂无历史记录。")

        st.markdown("**产物目录**")
        st.caption(f"输出根目录：{OUTPUT_DIR}；报告目录：{REPORT_DIR}")
        if st.button("刷新状态"):
            st.rerun()
    except Exception as exc:  # noqa: BLE001
        st.error(f"系统状态页签渲染失败：{exc}")


# --------------------------------------------------------------------------- #
# 六、主入口
# --------------------------------------------------------------------------- #
def main() -> None:
    """渲染整站界面。"""
    try:
        settings_result = get_settings()
        settings = settings_result["data"] if settings_result["status"] == "success" else {}
        configure_logging(str(settings.get("log_level") or "INFO"))
        ensure_dirs()

        st.title("🏭 工业数据分析 Agent 系统")
        st.caption("自然语言提问 → 自动生成分析代码 → 沙箱安全执行 → 自动输出图文报告")

        with st.sidebar:
            st.header("运行设置")
            use_llm = st.checkbox(
                "启用 LLM（DeepSeek）",
                value=bool(settings.get("has_api_key")),
                disabled=not bool(settings.get("has_api_key")),
                help="未配置 API Key 时自动使用本地模板模式，功能仍可完整演示。",
            )
            sandbox_timeout = st.slider("沙箱执行超时（秒）", 5, 300, int(settings.get("sandbox_timeout") or 60))
            max_repair = st.slider("代码自动修复轮数", 0, 3, 2)
            template_router = st.checkbox(
                "智能路由复核",
                value=bool(settings.get("template_router", True)),
                help=(
                    "开启后：规则判定为「标准场景」但句子可疑（出现模板词表外的动作词或长复合需求）时，"
                    "额外发一次约 100 token 的 JSON 判定请求，请大模型复核该需求能否由模板完成，"
                    "避免长尾表述被模板静默误判；关闭后回到纯规则判定（零额外额度）。"
                ),
            )
            st.divider()
            if settings.get("has_api_key"):
                st.success(f"API Key 已配置：{settings.get('api_key_masked')}")
            else:
                st.warning("未检测到 DEEPSEEK_API_KEY，当前为本地模板模式。")
            st.caption(
                f"SDK 可用：{settings.get('sdk_available')}｜传输通道："
                f"{'openai SDK' if settings.get('sdk_available') else '内置 HTTP（urllib）'}"
            )
            st.caption(f"代码模型：{settings.get('code_model')}")
            st.caption(f"报告模型：{settings.get('report_model')}")
            st.caption(f"模型降级备选：{settings.get('fallback_model')}")
            st.divider()
            if st.button("清除界面缓存并重载", use_container_width=True):
                st.cache_resource.clear()
                st.session_state.pop("analysis_result", None)
                st.rerun()

        agent = get_agent(use_llm, float(sandbox_timeout), int(max_repair), bool(template_router))
        tabs = st.tabs(["📁 数据文件管理", "🤖 自然语言分析", "🧰 算法工具箱", "⚙️ 系统状态"])
        with tabs[0]:
            render_data_tab(agent, settings)
        with tabs[1]:
            render_analysis_tab(agent, settings)
        with tabs[2]:
            render_toolbox_tab(agent)
        with tabs[3]:
            render_status_tab(agent, settings)

        st.divider()
        st.caption(
            "提示：生成代码在隔离子进程中执行，并经过导入白名单、路径约束、超时与输出上限等多重校验；"
            "所有分析结果与报告均保存在 output/ 目录下。"
        )
    except Exception as exc:  # noqa: BLE001 - 入口兜底
        st.error(f"应用启动失败：{exc}")
        st.exception(exc)


if __name__ == "__main__":
    main()
else:
    # `streamlit run app.py` 会以模块方式导入本文件，需要主动触发界面渲染
    main()
