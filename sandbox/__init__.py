"""代码安全沙箱包。

对外暴露 :mod:`sandbox.executor` 中的核心能力：

* :func:`sandbox.executor.execute_code` —— 在隔离子进程中执行生成代码；
* :func:`sandbox.executor.check_code_safety` —— 静态安全校验（供前端展示明细）；
* :class:`sandbox.executor.SandboxPolicy` —— 沙箱策略（超时、内存、输出上限等）。
"""

from __future__ import annotations

from . import executor

__all__ = ["executor"]

__version__ = "1.0.0"
