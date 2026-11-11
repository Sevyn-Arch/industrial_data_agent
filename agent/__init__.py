"""Agent 核心逻辑包。

构成：

* :mod:`agent.llm_client` —— DeepSeek（OpenAI 协议）客户端封装与全局配置读取；
* :mod:`agent.code_generator` —— 自然语言意图解析与 Python 分析代码生成（含自修复）；
* :mod:`agent.agent_core` —— Agent 主流程：意图解析 -> 代码生成 -> 安全执行 -> 报告生成。

依赖方向：``agent`` 依赖 ``analytics``、``sandbox``、``report``；不反向依赖 ``app``。
"""

from __future__ import annotations

from . import agent_core, code_generator, llm_client

__all__ = ["llm_client", "code_generator", "agent_core"]

__version__ = "1.0.0"
