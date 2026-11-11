"""LLM 客户端单元测试。

不依赖外网：HTTPS 通道用本地 stub HTTP 服务模拟，SDK 通道用内存中的假 openai 模块模拟。
覆盖：配置加载、密钥脱敏、HTTP 通道成功/失败、模型自动降级、重试、未配置密钥时的优雅降级。
"""

from __future__ import annotations

import importlib.machinery
import json
import os
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.llm_client import (  # noqa: E402
    LLMClient,
    _load_streamlit_secrets,
    get_settings,
    load_env,
    mask_secret,
)


# --------------------------------------------------------------------------- #
# 一、本地 stub 服务（模拟 DeepSeek / OpenAI 协议）
# --------------------------------------------------------------------------- #
class _StubHandler(BaseHTTPRequestHandler):
    """极简 chat/completions 桩实现，行为由类属性 ``mode`` 控制。"""

    mode = "success"
    calls: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except Exception:  # noqa: BLE001
            body = {}
        _StubHandler.calls.append({"path": self.path, "body": body, "auth": self.headers.get("Authorization")})

        model = str(body.get("model", ""))
        if self.mode == "supported_list" and model not in ("supported-a", "supported-b"):
            self._send(
                400,
                {
                    "error": {
                        "message": (
                            "The supported API model names are supported-a, supported-b, "
                            f"but you passed {model}."
                        ),
                        "type": "invalid_request_error",
                    }
                },
            )
            return
        if self.mode == "empty_content" and model == "empty-model":
            self._send(
                200,
                {
                    "model": model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}}],
                    "usage": {"total_tokens": 7},
                },
            )
            return
        if self.mode == "model_missing" and model != "fallback-model":
            self._send(400, {"error": {"message": "Model Not Exist", "type": "invalid_request_error"}})
            return
        if self.mode == "transient" and len(_StubHandler.calls) <= 2:
            self._send(503, {"error": {"message": "Service temporarily unavailable"}})
            return
        if self.mode == "server_error":
            self._send(500, {"error": {"message": "internal error"}})
            return
        self._send(
            200,
            {
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "桩服务回复内容"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
            },
        )

    def _send(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:  # 静默日志，保持测试输出整洁
        return


@pytest.fixture()
def stub_server():
    """启动本地 stub 服务，返回 ``(base_url, handler_class)``。"""
    _StubHandler.calls = []
    _StubHandler.mode = "success"
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", _StubHandler
    finally:
        server.shutdown()
        server.server_close()


# --------------------------------------------------------------------------- #
# 二、配置与脱敏
# --------------------------------------------------------------------------- #
def test_mask_secret() -> None:
    assert mask_secret("") == "(未配置)"
    assert mask_secret("short") == "*" * 5
    masked = mask_secret("sk-1234567890abcdef")
    assert masked.startswith("sk-1") and masked.endswith("cdef") and "*" in masked


def test_load_env_from_custom_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# 注释行\nDEEPSEEK_BASE_URL=http://127.0.0.1:9/v1\nCODE_MODEL=demo-code-model\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("CODE_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    result = load_env(env_file)
    assert result["status"] == "success"
    assert result["data"]["exists"] is True

    settings = get_settings()["data"]
    assert settings["code_model"] == "demo-code-model"
    assert settings["base_url"].startswith("http://127.0.0.1:9")
    assert settings["api_key_masked"]


def test_missing_env_file_is_reported(tmp_path: Path) -> None:
    result = load_env(tmp_path / "not_exists.env")
    assert result["status"] == "success"
    assert result["data"]["exists"] is False
    assert "未找到配置文件" in result["message"]


# --------------------------------------------------------------------------- #
# 三、HTTP 通道（urllib）
# --------------------------------------------------------------------------- #
def test_http_transport_success(stub_server, monkeypatch: pytest.MonkeyPatch) -> None:
    base_url, handler = stub_server
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-unit-test")
    client = LLMClient(base_url=base_url, code_model="demo-model", transport="http")
    result = client.chat([{"role": "user", "content": "你好"}], model="demo-model")
    assert result["status"] == "success", result["error"]
    assert result["data"]["content"] == "桩服务回复内容"
    assert result["data"]["usage"]["total_tokens"] == 12
    assert result["data"]["transport"] == "http"
    assert handler.calls[0]["path"].endswith("/chat/completions")
    assert handler.calls[0]["auth"] == "Bearer sk-unit-test"


def test_http_transport_model_fallback(stub_server, monkeypatch: pytest.MonkeyPatch) -> None:
    """服务端返回「模型不存在」时，应自动降级到 FALLBACK_MODEL。"""
    base_url, handler = stub_server
    handler.mode = "model_missing"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-unit-test")
    client = LLMClient(
        base_url=base_url,
        code_model="not-exist-model",
        fallback_model="fallback-model",
        transport="http",
    )
    result = client.chat([{"role": "user", "content": "你好"}], model="not-exist-model")
    assert result["status"] == "success", result["error"]
    assert result["data"]["model"] == "fallback-model"
    assert result["data"]["fallback_used"] is True
    # 诊断信息在新版本中给出更明确的说明（不被服务端支持 -> 自动改用备用模型）
    assert any("自动改用" in item for item in client.last_diagnostics)
    assert [call["body"]["model"] for call in handler.calls] == ["not-exist-model", "fallback-model"]


def test_http_transport_retries_transient_failure(stub_server, monkeypatch: pytest.MonkeyPatch) -> None:
    """瞬时错误（503）应自动重试并最终成功。"""
    base_url, handler = stub_server
    handler.mode = "transient"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-unit-test")
    monkeypatch.setenv("CODE_TEMPERATURE", "0")
    client = LLMClient(base_url=base_url, code_model="demo-model", max_retries=3, transport="http")
    result = client.chat([{"role": "user", "content": "重试测试"}])
    assert result["status"] == "success", result["error"]
    assert result["data"]["attempts"] >= 3
    assert len(handler.calls) >= 3


def test_http_transport_permanent_failure(stub_server, monkeypatch: pytest.MonkeyPatch) -> None:
    """持续 500 错误应重试到上限后返回失败，而不是抛异常。"""
    base_url, handler = stub_server
    handler.mode = "server_error"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-unit-test")
    client = LLMClient(base_url=base_url, code_model="demo-model", transport="http", max_retries=1)
    result = client.chat([{"role": "user", "content": "失败测试"}])
    assert result["status"] == "error"
    assert result["error"]
    assert len(handler.calls) == 2  # 首次 + 1 次重试


def test_unreachable_endpoint_is_graceful(monkeypatch: pytest.MonkeyPatch) -> None:
    """无法连接的服务地址应返回失败结果包（不抛异常）。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-unit-test")
    client = LLMClient(base_url="http://127.0.0.1:9/v1", transport="http", max_retries=1, timeout=2)
    result = client.chat([{"role": "user", "content": "连接失败测试"}])
    assert result["status"] == "error"
    assert result["error"]


def test_missing_api_key_degrades_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client = LLMClient(api_key="", base_url="http://127.0.0.1:9/v1", transport="http")
    assert client.is_available is False
    result = client.chat([{"role": "user", "content": "无密钥"}])
    assert result["status"] == "error"
    assert "DEEPSEEK_API_KEY" in result["message"]
    health = client.health_check()
    assert health["status"] == "error"
    described = client.describe()
    assert described["available"] is False and described["api_key"] == "(未配置)"


def test_empty_messages_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-unit-test")
    client = LLMClient(transport="http")
    result = client.chat([])
    assert result["status"] == "error"
    assert "不能为空" in result["message"]


# --------------------------------------------------------------------------- #
# 四、SDK 通道（内存假 openai 模块）
# --------------------------------------------------------------------------- #
def _install_fake_openai(monkeypatch: pytest.MonkeyPatch) -> dict:
    """注入假 openai 模块，返回用于断言的调用记录。"""
    record: dict = {"calls": []}

    class _Message:
        def __init__(self, content: str) -> None:
            self.content = content

    class _Choice:
        def __init__(self, content: str) -> None:
            self.message = _Message(content)

    class _Usage:
        def model_dump(self) -> dict:
            return {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}

    class _Response:
        def __init__(self, model: str) -> None:
            self.model = model
            self.choices = [_Choice("SDK 回复内容")]
            self.usage = _Usage()

    class _Completions:
        def create(self, **kwargs):
            record["calls"].append(kwargs)
            return _Response(str(kwargs.get("model")))

    class _Chat:
        def __init__(self) -> None:
            self.completions = _Completions()

    class _FakeOpenAI:
        def __init__(self, api_key=None, base_url=None, timeout=None, max_retries=None) -> None:
            record["init"] = {"api_key": api_key, "base_url": base_url, "timeout": timeout}
            self.chat = _Chat()

    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    fake_module.__spec__ = importlib.machinery.ModuleSpec("openai", loader=None)
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    return record


def test_sdk_transport_success(monkeypatch: pytest.MonkeyPatch) -> None:
    record = _install_fake_openai(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-unit-test")
    client = LLMClient(base_url="https://api.example.com/v1", code_model="demo-model", transport="sdk")
    result = client.chat(
        [{"role": "user", "content": "用 SDK"}],
        model="demo-model",
        temperature=0.0,
        max_tokens=64,
        response_format="json_object",
    )
    assert result["status"] == "success", result["error"]
    assert result["data"]["content"] == "SDK 回复内容"
    assert result["data"]["transport"] == "sdk"
    assert record["init"]["base_url"] == "https://api.example.com/v1"
    assert record["calls"][0]["response_format"] == {"type": "json_object"}
    assert record["calls"][0]["stream"] is False


def test_sdk_transport_error_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """SDK 抛出异常时应被规范化为失败结果包（含模型降级判定）。"""

    class _BrokenCompletions:
        def create(self, **kwargs):
            raise RuntimeError("Error code: 400 - {'error': {'message': 'Model Not Exist'}}")

    class _Chat:
        def __init__(self) -> None:
            self.completions = _BrokenCompletions()

    class _FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            self.chat = _Chat()

    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    fake_module.__spec__ = importlib.machinery.ModuleSpec("openai", loader=None)
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-unit-test")

    client = LLMClient(
        base_url="https://api.example.com/v1",
        code_model="not-exist-model",
        fallback_model="not-exist-model",  # 与主模型同名 -> 无法降级，应返回失败
        transport="sdk",
        max_retries=0,
    )
    result = client.chat([{"role": "user", "content": "报错"}], model="not-exist-model")
    assert result["status"] == "error"
    assert "Model Not Exist" in str(result["error"])


def test_generate_code_and_report_use_expected_models(stub_server, monkeypatch: pytest.MonkeyPatch) -> None:
    """代码生成使用 CODE_MODEL，报告生成使用 REPORT_MODEL。"""
    base_url, handler = stub_server
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-unit-test")
    client = LLMClient(
        base_url=base_url,
        code_model="demo-code-model",
        report_model="demo-report-model",
        transport="http",
    )
    code_result = client.generate_code("写代码")
    report_result = client.generate_report("写报告")
    assert code_result["status"] == "success" and report_result["status"] == "success"
    models = [call["body"]["model"] for call in handler.calls]
    assert models == ["demo-code-model", "demo-report-model"]
    assert code_result["data"]["model"] == "demo-code-model"


# --------------------------------------------------------------------------- #
# 五、openai 与 httpx 版本不兼容（proxies 参数）的回归测试
# --------------------------------------------------------------------------- #
def test_sdk_proxies_incompatibility_is_worked_around(monkeypatch: pytest.MonkeyPatch) -> None:
    """SDK 因 ``proxies`` 参数构造失败时，应改用显式 http_client 构造并成功调用。"""
    pytest.importorskip("httpx", reason="兼容分支需要 httpx（openai 的依赖）")
    record = {"http_client_used": False}

    class _Completions:
        def create(self, **kwargs):
            message = types.SimpleNamespace(content="兼容模式回复")
            choice = types.SimpleNamespace(message=message)
            return types.SimpleNamespace(model=kwargs.get("model"), choices=[choice], usage=None)

    class _Chat:
        def __init__(self) -> None:
            self.completions = _Completions()

    class _CompatOpenAI:
        """模拟 openai<=1.55：不带 http_client 时抛 proxies 报错，带上则正常工作。"""

        def __init__(self, **kwargs) -> None:
            if kwargs.get("http_client") is None:
                raise TypeError("Client.__init__() got an unexpected keyword argument 'proxies'")
            record["http_client_used"] = True
            self.chat = _Chat()

    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = _CompatOpenAI  # type: ignore[attr-defined]
    fake_module.__spec__ = importlib.machinery.ModuleSpec("openai", loader=None)
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-unit-test")

    client = LLMClient(base_url="https://api.example.com/v1", code_model="demo-model", transport="sdk")
    result = client.chat([{"role": "user", "content": "你好"}], model="demo-model")

    assert result["status"] == "success", result["error"]
    assert result["data"]["content"] == "兼容模式回复"
    assert record["http_client_used"] is True, "兼容分支应通过 http_client 重新构造客户端"
    assert any("版本不兼容" in item for item in client.last_diagnostics)


def test_sdk_incompatibility_switches_to_http_transport(
    stub_server, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SDK 完全不可用时，应自动切换到内置 HTTP 通道，而不是回落到本地模板模式。"""
    base_url, handler = stub_server

    class _BrokenOpenAI:
        def __init__(self, **kwargs) -> None:
            raise TypeError("Client.__init__() got an unexpected keyword argument 'proxies'")

    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = _BrokenOpenAI  # type: ignore[attr-defined]
    fake_module.__spec__ = importlib.machinery.ModuleSpec("openai", loader=None)
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-unit-test")

    client = LLMClient(base_url=base_url, code_model="demo-model", transport="sdk", max_retries=0)
    result = client.chat([{"role": "user", "content": "通道切换测试"}])

    assert result["status"] == "success", result["error"]
    assert client.transport == "http", "SDK 不可用时应切换到内置 HTTP 通道"
    assert result["data"]["transport"] == "http"
    assert result["data"]["content"] == "桩服务回复内容"
    assert any("HTTP 通道" in item for item in client.last_diagnostics)
    assert handler.calls, "切换后应真实发起 HTTP 请求"


# --------------------------------------------------------------------------- #
# 六、模型名不被服务端支持 / 模型返回空内容的自愈能力
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "error_text, expected",
    [
        (
            "The supported API model names are deepseek-flash, deepseek-v4-pro, but you passed deepseek-coder-v2.",
            ["deepseek-flash", "deepseek-v4-pro"],
        ),
        (
            "The supported API model names are deepseek-flash and deepseek-v4-pro, but you passed deepseek-v2.5.",
            ["deepseek-flash", "deepseek-v4-pro"],
        ),
        ("Model Not Exist", []),
    ],
)
def test_parse_supported_models(error_text: str, expected: list) -> None:
    """应从服务端 400 报错中解析出受支持的模型名，且不混入被拒绝的模型。"""
    assert LLMClient._parse_supported_models(error_text) == expected


def test_unsupported_model_switches_to_supported_one(stub_server, monkeypatch: pytest.MonkeyPatch) -> None:
    """模型名不被服务端支持时，应自动改用服务端声明支持的模型，而不是回落到本地模板。"""
    base_url, handler = stub_server
    handler.mode = "supported_list"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-unit-test")
    client = LLMClient(
        base_url=base_url,
        code_model="deepseek-coder-v2",   # 服务端不支持
        fallback_model="supported-b",     # 服务端支持的模型
        transport="http",
        max_retries=0,
    )
    result = client.chat([{"role": "user", "content": "生成代码"}])

    assert result["status"] == "success", result["error"]
    assert result["data"]["model"] == "supported-b"
    assert [call["body"]["model"] for call in handler.calls] == ["deepseek-coder-v2", "supported-b"]
    assert any("不被服务端支持" in item for item in client.last_diagnostics)
    assert any("supported-a、supported-b" in item for item in client.last_diagnostics)


def test_empty_model_response_switches_to_next_model(stub_server, monkeypatch: pytest.MonkeyPatch) -> None:
    """模型返回空内容（推理型模型）时应自动换模型重试，避免"调用成功但无产出"。"""
    base_url, handler = stub_server
    handler.mode = "empty_content"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-unit-test")
    client = LLMClient(
        base_url=base_url,
        code_model="empty-model",         # 返回 200 但 content 为空
        fallback_model="fallback-model",  # 正常返回内容
        transport="http",
        max_retries=0,
    )
    result = client.chat([{"role": "user", "content": "生成代码"}])

    assert result["status"] == "success", result["error"]
    assert result["data"]["model"] == "fallback-model"
    assert result["data"]["content"] == "桩服务回复内容"
    assert any("返回内容为空" in item for item in client.last_diagnostics)


# --------------------------------------------------------------------------- #
# 七、Streamlit Community Cloud Secrets 桥接（云端部署适配）
# --------------------------------------------------------------------------- #
def test_streamlit_secrets_are_injected_into_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """云端 Secrets 应被注入 os.environ，使统一通过 os.getenv 读取的配置在云端同样生效。"""
    fake_streamlit = types.ModuleType("streamlit")
    fake_streamlit.secrets = {  # type: ignore[attr-defined]
        "DEEPSEEK_API_KEY": "sk-from-secrets",
        "CODE_MODEL": "cloud-model",
        "NESTED_SECTION": {"A": 1},  # 嵌套配置不是标量，应跳过
    }
    fake_runtime = types.ModuleType("streamlit.runtime")
    fake_runtime.exists = lambda: True  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    monkeypatch.setitem(sys.modules, "streamlit.runtime", fake_runtime)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("CODE_MODEL", raising=False)
    monkeypatch.delenv("NESTED_SECTION", raising=False)

    injected = _load_streamlit_secrets()

    assert injected == 2
    assert os.environ["DEEPSEEK_API_KEY"] == "sk-from-secrets"
    assert os.environ["CODE_MODEL"] == "cloud-model"
    assert "NESTED_SECTION" not in os.environ


def test_streamlit_secrets_do_not_override_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """已存在的系统环境变量优先，Secrets 不得覆盖（便于临时用环境变量覆盖云端配置）。"""
    fake_streamlit = types.ModuleType("streamlit")
    fake_streamlit.secrets = {"DEEPSEEK_API_KEY": "sk-from-secrets"}  # type: ignore[attr-defined]
    fake_runtime = types.ModuleType("streamlit.runtime")
    fake_runtime.exists = lambda: True  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    monkeypatch.setitem(sys.modules, "streamlit.runtime", fake_runtime)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-system-env")

    _load_streamlit_secrets()

    assert os.environ["DEEPSEEK_API_KEY"] == "sk-from-system-env"


def test_streamlit_secrets_bridge_is_safe_outside_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """非 Streamlit 运行环境（CLI / 单元测试）下应静默返回 0，不抛异常。"""
    monkeypatch.delitem(sys.modules, "streamlit", raising=False)
    assert _load_streamlit_secrets() >= 0  # 不抛异常即可（若本机装了 Streamlit，返回 0）


def test_env_file_missing_falls_back_to_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """云端没有 config/.env 时，应可直接从环境变量读取配置（模拟 Cloud 部署）。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-cloud-env")
    monkeypatch.setenv("CODE_MODEL", "cloud-code-model")
    result = load_env(tmp_path / "no_such.env")
    assert result["status"] == "success"
    assert result["data"]["exists"] is False

    client = LLMClient(code_model=None, env_file=tmp_path / "no_such.env")
    assert client.api_key == "sk-cloud-env"
    assert client.code_model == "cloud-code-model"
