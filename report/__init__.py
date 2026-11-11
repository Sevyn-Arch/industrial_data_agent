"""报告生成包。

对外暴露 :class:`report.generator.ReportGenerator`：把沙箱执行结果（结构化 JSON + 标准输出）
交给 LLM 生成中文分析报告；LLM 不可用时使用内置模板渲染同结构报告。
"""

from __future__ import annotations

from . import generator

__all__ = ["generator"]

__version__ = "1.0.0"
