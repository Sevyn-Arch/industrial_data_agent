"""沙箱执行器单元测试。

覆盖：合法代码执行、结果与图表回收、项目算法库复用、危险代码拦截（导入/路径/删除/
无界循环/动态执行）、运行时异常捕获、超时终止、敏感环境变量隔离。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sandbox.executor import (  # noqa: E402
    SandboxPolicy,
    _build_env,
    check_code_safety,
    execute_code,
    validate_code,
)


@pytest.fixture(scope="module")
def policy() -> SandboxPolicy:
    """测试用沙箱策略：缩短超时并关闭工作目录保留。"""
    return SandboxPolicy(timeout=60, keep_workspace=False, max_workspaces=3)


def _run(code: str, policy: SandboxPolicy, **kwargs) -> dict:
    """统一执行封装，便于断言。"""
    return execute_code(code, policy=policy, **kwargs)


# --------------------------------------------------------------------------- #
# 一、正常路径
# --------------------------------------------------------------------------- #
def test_valid_code_executes_and_emits_result(policy: SandboxPolicy) -> None:
    """合法代码应执行成功，且 emit_result 的结构化结果可被回收。"""
    code = (
        "import pandas as pd\n"
        "values = pd.Series([1, 2, 3, 4, 5])\n"
        "emit_result({'mean': float(values.mean()), 'count': int(values.size)})\n"
        "print('done')\n"
    )
    result = _run(code, policy)
    assert result["status"] == "success"
    assert result["data"]["exit_code"] == 0
    assert result["data"]["result"]["count"] == 5
    assert abs(result["data"]["result"]["mean"] - 3.0) < 1e-9
    assert "done" in result["data"]["stdout"]
    assert result["data"]["timed_out"] is False


def test_chart_is_collected(policy: SandboxPolicy) -> None:
    """生成代码保存的图表应被自动回收。"""
    code = (
        "plt.figure(figsize=(4, 3))\n"
        "plt.plot([1, 3, 2])\n"
        "path = save_chart('unit_test_chart.png')\n"
        "emit_result({'chart': path})\n"
    )
    result = _run(code, policy)
    assert result["status"] == "success"
    assert len(result["charts"]) >= 1
    assert all(Path(item).exists() for item in result["charts"])


def test_generated_code_can_reuse_project_library(policy: SandboxPolicy) -> None:
    """生成代码应能复用项目内已测试的分析算法库（analytics）。"""
    code = (
        "from analytics.ml_models import list_models\n"
        "info = list_models()\n"
        "emit_result({'status': info['status'], 'n_models': len(info['data']['models'])})\n"
    )
    result = _run(code, policy)
    assert result["status"] == "success"
    assert result["data"]["result"]["n_models"] == 3


# --------------------------------------------------------------------------- #
# 二、危险代码拦截
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "code, keyword",
    [
        ("import os\nprint(os.getcwd())\n", "白名单"),
        ("import subprocess\nsubprocess.run(['echo', 'hi'])\n", "白名单"),
        ("import socket\n", "白名单"),
        ("data = eval('1+1')\n", "动态执行"),
        ("data = __import__('os')\n", "危险模式"),
        ("obj = ().__class__.__bases__\n", "魔术"),
        ("while True:\n    pass\n", "无界循环"),
        ("assert 1 == 1\n", "assert"),
    ],
)
def test_dangerous_code_is_rejected(code: str, keyword: str) -> None:
    """危险写法必须被静态校验拦截，且不会进入执行阶段。"""
    is_valid, violations = validate_code(code)
    assert is_valid is False
    assert violations, "必须给出违规说明"
    assert any(keyword in item for item in violations), f"未命中关键字 {keyword}：{violations}"

    result = execute_code(code, policy=SandboxPolicy(timeout=10, keep_workspace=False))
    assert result["status"] == "error"
    assert result["data"]["violations"]
    assert result["data"]["exit_code"] is None  # 未启动子进程


def test_path_escape_is_rejected() -> None:
    """读取允许目录之外的绝对路径必须被拒绝。"""
    # 跨平台选择系统目录外的绝对路径：Windows 用系统目录，Linux 用 /etc
    outside = r"C:\Windows\win.ini" if os.name == "nt" else "/etc/passwd"
    code = (
        "import pandas as pd\n"
        f"frame = pd.read_csv(r'{outside}')\n"
        "emit_result({'ok': True})\n"
    )
    result = execute_code(code, policy=SandboxPolicy(timeout=10, keep_workspace=False))
    assert result["status"] == "error"
    assert any("超出允许目录范围" in item for item in result["data"]["violations"])


def test_file_deletion_is_rejected() -> None:
    """文件删除类方法必须被拒绝。"""
    code = "import pathlib\npathlib.Path('x.txt').unlink()\nemit_result({'ok': True})\n"
    result = execute_code(code, policy=SandboxPolicy(timeout=10, keep_workspace=False))
    assert result["status"] == "error"
    assert any("unlink" in item for item in result["data"]["violations"])


def test_syntax_error_reported() -> None:
    """语法错误应在静态校验阶段被明确指出。"""
    is_valid, violations = validate_code("def broken(:\n    pass\n")
    assert is_valid is False
    assert any("语法错误" in item for item in violations)


def test_safety_check_report_shape() -> None:
    """安全校验报告应包含白名单等可解释信息，便于前端展示。"""
    report = check_code_safety("import pandas as pd\nemit_result({'n': 1})\n")
    assert report["status"] == "success"
    assert report["data"]["is_safe"] is True
    assert "pandas" in report["data"]["allowed_imports"]
    assert "os" in report["data"]["forbidden_builtins"] or "open" in report["data"]["forbidden_builtins"]


# --------------------------------------------------------------------------- #
# 三、运行时异常与超时
# --------------------------------------------------------------------------- #
def test_runtime_error_is_captured(policy: SandboxPolicy) -> None:
    """运行期异常应被捕获为失败结果包，而非向上抛出。"""
    code = "import pandas as pd\nframe = pd.read_csv('不存在的文件.csv')\nemit_result({'ok': True})\n"
    result = _run(code, policy)
    assert result["status"] == "error"
    assert result["data"]["exit_code"] != 0
    assert result["data"]["stderr"]


def test_timeout_kills_process() -> None:
    """超时后应强制终止子进程，并标记 timed_out。"""
    strict = SandboxPolicy(timeout=3, keep_workspace=False, max_workspaces=2)
    started = time.perf_counter()
    result = execute_code("import time\ntime.sleep(60)\nemit_result({'ok': True})\n", policy=strict)
    elapsed = time.perf_counter() - started
    assert result["status"] == "error"
    assert result["data"]["timed_out"] is True
    assert elapsed < 45, f"超时终止耗时过长：{elapsed:.1f}s"


# --------------------------------------------------------------------------- #
# 四、环境变量隔离
# --------------------------------------------------------------------------- #
def test_secrets_are_not_passed_to_child_process(tmp_path: Path) -> None:
    """DEEPSEEK_API_KEY 等敏感变量不得传入子进程环境。"""
    os.environ["DEEPSEEK_API_KEY"] = "unit-test-secret-value"
    env = _build_env(tmp_path, tmp_path / "result.json", {"data_file": "demo.csv"})
    assert "DEEPSEEK_API_KEY" not in env
    assert all("unit-test-secret-value" not in str(value) for value in env.values())
    assert env["SANDBOX_CONTEXT"] == '{"data_file": "demo.csv"}'
