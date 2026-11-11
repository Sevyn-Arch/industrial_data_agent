"""LLM 客户端封装（DeepSeek API，兼容 OpenAI SDK 协议）。

设计要点
========================================================================

1. **统一配置来源**：所有密钥与模型名从 ``config/.env`` 读取，使用 python-dotenv；
   若依赖缺失则退化为内置的极简 ``KEY=VALUE`` 解析器，保证配置能力始终可用。
2. **双通道传输**：优先使用官方 ``openai`` SDK；当环境未安装（工业内网常见）时，
   自动回退到标准库 ``urllib`` 直连同一 HTTP 接口，避免整条链路不可用。
3. **模型自动降级**：需求指定 ``deepseek-coder-v2`` / ``deepseek-v2.5``；若服务端返回
   「模型不存在」，自动改用 ``FALLBACK_MODEL``（默认 ``deepseek-chat``）重试并记录提示。
4. **永不抛出异常**：所有公开方法返回统一结果包；未配置密钥时返回明确提示，
   上层据此切换到「本地模板模式」，保证系统在无网络/无密钥时仍可完整演示。
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

from analytics import ENV_FILE, fail, get_logger, ok

__all__ = [
    "LLMClient",
    "LLMError",
    "load_env",
    "get_settings",
    "mask_secret",
    "DEFAULT_BASE_URL",
]

logger = get_logger(__name__)

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"

#: 触发重试的瞬时错误关键字
_TRANSIENT_HINTS = (
    "rate limit", "too many requests", "timeout", "timed out", "connection",
    "temporarily", "overloaded", "502", "503", "504", "500",
)
#: 表示「模型不存在」的错误关键字（用于触发模型降级）
_MODEL_MISSING_HINTS = (
    "model not exist", "model_not_found", "does not exist", "invalid model",
    "unknown model", "no such model", "not found", "model is not available",
    # DeepSeek 实际返回的措辞：
    # The supported API model names are deepseek-flash, deepseek-v4-pro, but you passed xxx.
    "supported api model names", "but you passed",
    "unsupported model", "model is not supported", "is not supported",
)

#: 从错误信息中提取服务端「实际支持的模型名」列表
_SUPPORTED_MODELS_PATTERN = re.compile(
    r"supported\s+api\s+model\s+names?\s+are\s*(?P<names>[^.\n]+)", re.IGNORECASE
)


class LLMError(RuntimeError):
    """LLM 调用异常（内部使用，携带错误分类信息以便重试与降级决策）。"""

    def __init__(
        self,
        message: str,
        *,
        model_missing: bool = False,
        transient: bool = False,
        sdk_incompatible: bool = False,
        supported_models: Sequence[str] | None = None,
        empty_response: bool = False,
    ) -> None:
        super().__init__(message)
        self.model_missing = model_missing
        self.transient = transient
        #: 标记 SDK 客户端无法构造（例如 openai 与 httpx 版本不兼容），上层据此切换到内置 HTTP 通道
        self.sdk_incompatible = sdk_incompatible
        #: 服务端声明支持的模型名列表（从 400 错误信息中解析得到）
        self.supported_models: list[str] = [str(item).strip() for item in (supported_models or [])]
        #: 标记模型返回内容为空（推理型模型或 max_tokens 过小时出现），上层据此更换模型重试
        self.empty_response = empty_response


# --------------------------------------------------------------------------- #
# 一、配置加载
# --------------------------------------------------------------------------- #
def _manual_parse_env(path: Path, override: bool = False) -> int:
    """极简 ``.env`` 解析器（python-dotenv 缺失时的备用实现）。

    Returns:
        int: 成功写入的环境变量条数（解析失败返回 0）。
    """
    written = 0
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if not key:
                continue
            if override or key not in os.environ:
                os.environ[key] = value
                written += 1
    except Exception:  # noqa: BLE001
        logger.warning("内置 .env 解析失败：%s", path, exc_info=True)
    return written


def _load_streamlit_secrets() -> int:
    """把 Streamlit Community Cloud 的 Secrets 注入进程环境变量。

    背景：Streamlit Cloud 后台填写的 Secrets 只通过 ``st.secrets`` 暴露，**不会**自动
    出现在 ``os.environ`` 中；而本项目统一通过 ``os.getenv`` 读取配置。这里在 Streamlit
    运行时把 Secrets 中的标量项注入环境变量（不覆盖同名系统环境变量），从而在云端与本地
    保持同一套配置读取契约。

    Returns:
        int: 实际注入的环境变量条数；不在 Streamlit 运行时或无法读取时返回 0。
    """
    try:
        import streamlit as st  # 惰性导入：非 Web 场景（CLI/测试）不依赖 Streamlit

        try:
            from streamlit.runtime import exists as runtime_exists

            if not runtime_exists():
                return 0
        except Exception:  # noqa: BLE001 - 旧版本无该 API 时继续尝试
            pass

        injected = 0
        for key in st.secrets:
            value = st.secrets[key]
            if isinstance(value, (str, int, float, bool)) and str(key) not in os.environ:
                os.environ[str(key)] = str(value)
                injected += 1
        return injected
    except Exception:  # noqa: BLE001 - 无 Secrets 文件 / 非运行时场景，静默跳过
        return 0


def load_env(env_file: Path | str | None = None, override: bool = False) -> dict[str, Any]:
    """加载配置到进程环境变量（幂等）。

    配置优先级：系统环境变量 > Streamlit Cloud Secrets > ``config/.env``（默认不覆盖已有值）。
    所有配置最终统一通过 ``os.getenv`` 读取，业务代码不直接依赖配置来源。

    Args:
        env_file: 自定义 .env 路径；默认使用 ``config/.env``。
        override: 是否覆盖已存在的同名环境变量（默认不覆盖，便于用系统环境变量覆盖配置）。

    Returns:
        dict: 统一结果包，``data`` 含 ``env_file``、``exists``、``loaded``、
        ``secrets_injected`` 信息。
    """
    try:
        secrets_injected = _load_streamlit_secrets()
        path = Path(env_file) if env_file else ENV_FILE
        if not path.exists():
            return ok(
                {"env_file": str(path), "exists": False, "loaded": 0, "secrets_injected": secrets_injected},
                f"未找到配置文件 {path}，将使用系统环境变量 / Streamlit Secrets",
            )
        try:
            from dotenv import load_dotenv  # type: ignore

            load_dotenv(dotenv_path=str(path), override=override, encoding="utf-8")
            loaded = -1  # python-dotenv 不返回条数，-1 表示由 dotenv 处理
        except ImportError:
            loaded = _manual_parse_env(path, override=override)
        return ok(
            {"env_file": str(path), "exists": True, "loaded": loaded, "secrets_injected": secrets_injected},
            f"已加载配置文件 {path.name}",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("环境变量加载失败")
        return fail("环境变量加载失败", exc)


def _env(key: str, default: str = "") -> str:
    """读取环境变量并去除首尾空白（内部工具，永不抛异常）。"""
    try:
        return str(os.environ.get(key, default) or "").strip()
    except Exception:  # noqa: BLE001
        return default


def _env_float(key: str, default: float) -> float:
    """读取浮点型环境变量，解析失败返回默认值。"""
    try:
        return float(_env(key, str(default)))
    except Exception:  # noqa: BLE001
        return default


def _env_int(key: str, default: int) -> int:
    """读取整型环境变量，解析失败返回默认值。"""
    try:
        return int(float(_env(key, str(default))))
    except Exception:  # noqa: BLE001
        return default


def _env_bool(key: str, default: bool) -> bool:
    """读取布尔型环境变量（1/true/yes/on 视为真），解析失败返回默认值。"""
    try:
        raw = _env(key, "true" if default else "false").lower()
        if raw in ("1", "true", "yes", "y", "on"):
            return True
        if raw in ("0", "false", "no", "n", "off"):
            return False
        return bool(default)
    except Exception:  # noqa: BLE001
        return bool(default)


def mask_secret(secret: str | None) -> str:
    """对密钥做脱敏展示（仅保留首尾各 4 位）。"""
    text = str(secret or "")
    if not text:
        return "(未配置)"
    if len(text) <= 8:
        return "*" * len(text)
    return f"{text[:4]}{'*' * 6}{text[-4:]}"


def get_settings() -> dict[str, Any]:
    """读取全部运行时配置（供前端「系统状态」面板展示）。

    Returns:
        dict: 统一结果包，``data`` 含密钥脱敏值、模型名、超时与沙箱参数等。
    """
    try:
        env_result = load_env()
        api_key = _env("DEEPSEEK_API_KEY")
        data = {
            "has_api_key": bool(api_key),
            "api_key_masked": mask_secret(api_key),
            "base_url": _env("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL) or DEFAULT_BASE_URL,
            "code_model": _env("CODE_MODEL", "deepseek-coder-v2"),
            "report_model": _env("REPORT_MODEL", "deepseek-v2.5"),
            "fallback_model": _env("FALLBACK_MODEL", "deepseek-chat"),
            "timeout": _env_float("LLM_TIMEOUT", 120.0),
            "max_retries": _env_int("LLM_MAX_RETRIES", 3),
            "code_temperature": _env_float("CODE_TEMPERATURE", 0.0),
            "report_temperature": _env_float("REPORT_TEMPERATURE", 0.3),
            "sandbox_timeout": _env_float("SANDBOX_TIMEOUT", 60.0),
            "sandbox_max_output_chars": _env_int("SANDBOX_MAX_OUTPUT_CHARS", 20000),
            "sandbox_memory_mb": _env_int("SANDBOX_MEMORY_MB", 2048),
            "log_level": _env("APP_LOG_LEVEL", "INFO").upper(),
            "max_upload_mb": _env_float("MAX_UPLOAD_MB", 200.0),
            "template_router": _env_bool("TEMPLATE_ROUTER", True),
            "env_file": str(ENV_FILE),
            "env_file_exists": ENV_FILE.exists(),
            "env_load_message": env_result["message"],
            "sdk_available": _sdk_available(),
        }
        message = "配置读取完成" + ("" if data["has_api_key"] else "（未配置 API Key，将使用本地模板模式）")
        return ok(data, message)
    except Exception as exc:  # noqa: BLE001
        logger.exception("配置读取失败")
        return fail("配置读取失败", exc)


def _sdk_available() -> bool:
    """探测 openai SDK 是否可用（不触发网络请求）。"""
    try:
        import importlib.util

        return importlib.util.find_spec("openai") is not None
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# 二、LLM 客户端
# --------------------------------------------------------------------------- #
class LLMClient:
    """DeepSeek LLM 客户端（OpenAI 协议兼容）。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        code_model: str | None = None,
        report_model: str | None = None,
        fallback_model: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        transport: str | None = None,
        env_file: Path | str | None = None,
    ) -> None:
        """初始化客户端。

        Args:
            api_key: API Key；``None`` 时读取 ``DEEPSEEK_API_KEY``。
            base_url: 服务地址；``None`` 时读取 ``DEEPSEEK_BASE_URL``。
            code_model: 代码生成模型；``None`` 时读取 ``CODE_MODEL``。
            report_model: 报告生成模型；``None`` 时读取 ``REPORT_MODEL``。
            fallback_model: 模型降级备用名；``None`` 时读取 ``FALLBACK_MODEL``。
            timeout: 单次请求超时（秒）。
            max_retries: 瞬时错误重试次数。
            transport: 强制指定传输方式（``"sdk"`` / ``"http"``）；``None`` 表示自动选择。
            env_file: 自定义 .env 路径。
        """
        load_env(env_file)
        self.api_key = (api_key if api_key is not None else _env("DEEPSEEK_API_KEY")).strip()
        self.base_url = (base_url or _env("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL) or DEFAULT_BASE_URL).rstrip("/")
        self.code_model = code_model or _env("CODE_MODEL", "deepseek-coder-v2")
        self.report_model = report_model or _env("REPORT_MODEL", "deepseek-v2.5")
        self.fallback_model = fallback_model or _env("FALLBACK_MODEL", "deepseek-chat")
        self.timeout = float(timeout if timeout is not None else _env_float("LLM_TIMEOUT", 120.0))
        self.max_retries = int(max_retries if max_retries is not None else _env_int("LLM_MAX_RETRIES", 3))
        self.transport = (transport or "").strip().lower() or ("sdk" if _sdk_available() else "http")
        self.last_diagnostics: list[str] = []

    # ------------------------------ 属性 ------------------------------ #
    @property
    def is_available(self) -> bool:
        """是否具备调用条件（已配置密钥）。"""
        return bool(self.api_key)

    def describe(self) -> dict[str, Any]:
        """返回客户端当前配置摘要（密钥脱敏）。"""
        return {
            "available": self.is_available,
            "api_key": mask_secret(self.api_key),
            "base_url": self.base_url,
            "code_model": self.code_model,
            "report_model": self.report_model,
            "fallback_model": self.fallback_model,
            "transport": self.transport,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

    # --------------------------- 对外主方法 --------------------------- #
    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: str | None = None,
    ) -> dict[str, Any]:
        """发起一次对话补全请求（含重试与模型降级）。

        Args:
            messages: OpenAI 格式消息列表，如 ``[{"role": "user", "content": "..."}]``。
            model: 模型名；``None`` 时使用代码模型。
            temperature: 采样温度。
            max_tokens: 最大生成 token 数。
            response_format: 传 ``"json_object"`` 时要求模型返回 JSON。

        Returns:
            dict: 统一结果包，``data`` 含 ``content``、``model``、``usage``、
            ``elapsed_seconds``、``transport``、``attempts``。
        """
        started = time.perf_counter()
        attempts = 0
        try:
            if not messages:
                return fail("消息列表不能为空")
            if not self.is_available:
                return fail(
                    "未配置 DeepSeek API Key（请在 config/.env 中设置 DEEPSEEK_API_KEY）",
                    data={"available": False, "diagnostics": list(self.last_diagnostics)},
                )

            current_model = model or self.code_model
            fallback_used = False
            last_error = "未知错误"
            tried_models: set[str] = {current_model}
            supported_models: list[str] = []

            while attempts <= max(0, self.max_retries):
                attempts += 1
                try:
                    content, usage, used_model = self._send(
                        messages, current_model, temperature, max_tokens, response_format
                    )
                    data = {
                        "content": content,
                        "model": used_model,
                        "usage": usage,
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "transport": self.transport,
                        "attempts": attempts,
                        "fallback_used": fallback_used,
                    }
                    return ok(data, f"LLM 调用成功（模型 {used_model}，耗时 {data['elapsed_seconds']:.2f}s）")
                except LLMError as exc:
                    last_error = str(exc)
                    if exc.sdk_incompatible and self.transport != "http":
                        # SDK 通道不可用（典型场景：openai 与 httpx 版本不兼容导致客户端无法构造）
                        # → 自动切换到内置 HTTP 通道重试，保证 DeepSeek 调用仍然可用，
                        #   而不是直接回落到本地模板模式。
                        self.last_diagnostics.append("SDK 通道不可用，已自动切换到内置 HTTP 通道重试")
                        logger.warning("SDK 通道不可用（%s），已切换到内置 HTTP 通道重试", exc)
                        self.transport = "http"
                        attempts -= 1  # 切换通道不计入重试次数
                        continue
                    if exc.supported_models:
                        supported_models = list(exc.supported_models)
                    if exc.model_missing or exc.empty_response:
                        # 模型不被服务端支持，或该模型返回空内容（推理型模型常见）
                        # -> 自动换用备用模型 / 服务端声明支持的模型重试
                        candidate = self._pick_alternative_model(current_model, supported_models, tried_models)
                        if candidate:
                            if exc.model_missing:
                                reason = f"模型 {current_model} 不被服务端支持"
                            else:
                                reason = f"模型 {current_model} 返回内容为空（多为推理型模型）"
                            self.last_diagnostics.append(f"{reason}，已自动改用 {candidate} 重试")
                            if supported_models:
                                self.last_diagnostics.append(
                                    "服务端声明支持的模型："
                                    + "、".join(supported_models)
                                    + "；建议同步修改 config/.env 中的 CODE_MODEL / REPORT_MODEL"
                                )
                            logger.warning("%s，已自动改用 %s 重试", reason, candidate)
                            tried_models.add(current_model)
                            current_model = candidate
                            fallback_used = True
                            attempts -= 1  # 换模型重试不计入重试次数
                            continue
                    if exc.transient and attempts <= self.max_retries:
                        delay = min(8.0, 2.0 ** (attempts - 1))
                        logger.warning("LLM 调用瞬时失败（第 %s 次）：%s，%.1fs 后重试", attempts, exc, delay)
                        time.sleep(delay)
                        continue
                    break

            return fail(
                f"LLM 调用失败（已尝试 {attempts} 次，模型 {current_model}）",
                last_error,
                data={
                    "available": True,
                    "diagnostics": list(self.last_diagnostics),
                    "supported_models": supported_models,
                    "tried_models": sorted(tried_models),
                },
                transport=self.transport,
                attempts=attempts,
            )
        except Exception as exc:  # noqa: BLE001 - 客户端任何异常都不得外泄
            logger.exception("LLM 客户端异常")
            return fail("LLM 调用异常", exc, data={"available": self.is_available})

    def generate_code(self, prompt: str, system_prompt: str | None = None, max_tokens: int = 4096) -> dict[str, Any]:
        """使用代码模型生成分析代码。"""
        temperature = _env_float("CODE_TEMPERATURE", 0.0)
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, model=self.code_model, temperature=temperature, max_tokens=max_tokens)

    def generate_report(self, prompt: str, system_prompt: str | None = None, max_tokens: int = 4096) -> dict[str, Any]:
        """使用报告模型生成自然语言报告。"""
        temperature = _env_float("REPORT_TEMPERATURE", 0.3)
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, model=self.report_model, temperature=temperature, max_tokens=max_tokens)

    def health_check(self) -> dict[str, Any]:
        """连通性自检（发送一条极短请求），用于「系统状态」面板。"""
        try:
            if not self.is_available:
                return fail("未配置 API Key，无法进行连通性检查")
            result = self.chat(
                [{"role": "user", "content": "ping"}],
                model=self.report_model,
                temperature=0.0,
                max_tokens=8,
            )
            if result["status"] == "success":
                return ok(
                    {
                        "model": result["data"]["model"],
                        "transport": result["data"]["transport"],
                        "elapsed_seconds": result["data"]["elapsed_seconds"],
                        "diagnostics": list(self.last_diagnostics),
                    },
                    "LLM 服务连通性正常",
                )
            return fail("LLM 服务连通性异常", result["error"], data={"diagnostics": list(self.last_diagnostics)})
        except Exception as exc:  # noqa: BLE001
            logger.exception("health_check 执行失败")
            return fail("LLM 连通性检查失败", exc)

    # --------------------------- 内部实现 --------------------------- #
    def _send(
        self,
        messages: Sequence[Mapping[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: str | None,
    ) -> tuple[str, dict[str, Any], str]:
        """按配置的传输方式发送请求，返回 ``(内容, usage, 实际模型)``。

        Raises:
            LLMError: 携带错误分类信息（模型缺失 / 瞬时错误）。
        """
        payload_messages = [{"role": str(item.get("role", "user")), "content": str(item.get("content", ""))} for item in messages]
        if self.transport == "sdk" and _sdk_available():
            try:
                content, usage, used_model = self._send_via_sdk(
                    payload_messages, model, temperature, max_tokens, response_format
                )
            except LLMError:
                raise
            except Exception as exc:  # noqa: BLE001 - SDK 异常统一分类后抛出
                raise self._classify(exc) from exc
        else:
            content, usage, used_model = self._send_via_http(
                payload_messages, model, temperature, max_tokens, response_format
            )

        # 空内容保护：推理型模型（如 deepseek-flash）会把 token 全部消耗在思考过程，
        # 返回 content 为空字符串。此处统一识别并交给上层换模型重试，
        # 避免"调用成功但没有产出"被当成正常结果继续往下走。
        if not str(content or "").strip():
            raise LLMError(
                f"模型 {model} 返回内容为空（该模型可能为推理型模型，或 max_tokens 设置过小）",
                empty_response=True,
            )
        return content, usage, used_model

    def _send_via_sdk(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: str | None,
    ) -> tuple[str, dict[str, Any], str]:
        """使用官方 openai SDK 发送请求。

        客户端统一由 :meth:`_build_sdk_client` 构造，只传 api_key / base_url / timeout
        三个必要参数，不传任何代理（proxies）相关参数。
        """
        client = self._build_sdk_client()
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        if response_format:
            kwargs["response_format"] = {"type": response_format}
        response = client.chat.completions.create(**kwargs)
        content = (response.choices[0].message.content or "").strip()
        usage = {}
        try:
            usage = dict(response.usage.model_dump()) if response.usage else {}
        except Exception:  # noqa: BLE001
            usage = {}
        used_model = str(getattr(response, "model", model) or model)
        return content, usage, used_model

    def _build_sdk_client(self) -> Any:
        """构造 openai SDK 同步客户端（仅三个必要参数，并处理 httpx 版本兼容）。

        说明：本系统不需要网络代理参数，代理应由 httpx 自行读取系统环境变量处理。
        但 ``openai<=1.55`` 会在内部**无条件**把 ``proxies`` 传给 ``httpx.Client``
        （见 openai/_base_client.py 中 ``SyncHttpxClientWrapper(..., proxies=proxies, ...)``），
        而 ``httpx>=0.28`` 已删除该参数，于是抛出
        ``TypeError: Client.__init__() got an unexpected keyword argument 'proxies'``。

        处理策略（两级兼容，均不涉及代理参数）：

        1. 首选：只传 api_key / base_url / timeout 构造客户端；
        2. 若命中上述版本不兼容：自行创建 ``httpx.Client`` 并通过 ``http_client`` 传入——
           源码显示传入 ``http_client`` 后会跳过 SDK 内部的代理参数传递，问题即可规避；
        3. 两者都失败：抛出标记为 ``sdk_incompatible`` 的 :class:`LLMError`，
           由 :meth:`chat` 自动切换到内置 HTTP 通道继续调用，避免直接回落到本地模板。

        Raises:
            LLMError: 客户端无法构造时抛出（携带 ``sdk_incompatible`` 标记）。
        """
        from openai import OpenAI  # 惰性导入：未安装时不阻断其它功能

        try:
            # 只保留三个必要参数：api_key / base_url / timeout
            return OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        except TypeError as exc:
            if "proxies" not in str(exc):
                raise LLMError(f"OpenAI 客户端构造失败：{exc}", sdk_incompatible=True) from exc

            message = (
                "检测到 openai 与 httpx 版本不兼容（httpx>=0.28 已移除 proxies 参数），"
                "已改用显式 http_client 方式构造客户端"
            )
            self.last_diagnostics.append(message)
            logger.warning("%s：%s", message, exc)
            try:
                import httpx  # openai 的依赖，随 SDK 一同安装

                return OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout,
                    http_client=httpx.Client(timeout=self.timeout, follow_redirects=True),
                )
            except Exception as inner:  # noqa: BLE001 - 兼容模式也失败：交给上层切换通道
                raise LLMError(
                    f"OpenAI 客户端构造失败（{exc}）；兼容模式亦失败：{inner}",
                    sdk_incompatible=True,
                ) from inner

    def _send_via_http(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: str | None,
    ) -> tuple[str, dict[str, Any], str]:
        """使用标准库 urllib 直连 OpenAI 协议接口（SDK 不可用时的备用通道）。"""
        url = f"{self.base_url}/chat/completions"
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        if response_format:
            body["response_format"] = {"type": response_format}
        request = urllib.request.Request(
            url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - 固定 https 接口
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                detail = ""
            raise self._classify(RuntimeError(f"HTTP {exc.code}: {detail or exc.reason}")) from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"网络连接失败：{exc.reason}", transient=True) from exc
        except Exception as exc:  # noqa: BLE001
            raise self._classify(exc) from exc

        try:
            parsed = json.loads(raw)
            content = str(parsed["choices"][0]["message"]["content"] or "").strip()
            usage = parsed.get("usage") or {}
            used_model = str(parsed.get("model") or model)
            return content, usage, used_model
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"响应解析失败：{exc}", transient=False) from exc

    @staticmethod
    def _classify(exc: Exception) -> LLMError:
        """将底层异常归类为「模型缺失 / 瞬时错误 / 永久错误」。"""
        text = f"{type(exc).__name__}: {exc}".lower()
        model_missing = any(hint in text for hint in _MODEL_MISSING_HINTS)
        transient = any(hint in text for hint in _TRANSIENT_HINTS)
        # 兜底识别 openai/httpx 版本不兼容（httpx 0.28 起移除 proxies 参数）
        sdk_incompatible = "proxies" in text and "unexpected keyword argument" in text
        return LLMError(
            str(exc),
            model_missing=model_missing,
            transient=transient,
            sdk_incompatible=sdk_incompatible,
            supported_models=LLMClient._parse_supported_models(str(exc)),
        )

    @staticmethod
    def _parse_supported_models(error_text: str) -> list[str]:
        """从服务端错误信息中解析其声明支持的模型名列表。

        例如 DeepSeek 返回：
        ``The supported API model names are deepseek-flash, deepseek-v4-pro, but you passed xxx``
        → 解析为 ``["deepseek-flash", "deepseek-v4-pro"]``。
        """
        try:
            matched = _SUPPORTED_MODELS_PATTERN.search(str(error_text))
            if not matched:
                return []
            # 截断 "..., but you passed xxx" 尾巴，避免把被拒绝的模型名混进支持列表
            raw = re.split(r"\bbut\s+you\s+passed\b", matched.group("names"), flags=re.IGNORECASE)[0]
            raw = raw.replace(" and ", ",")
            models = [item.strip().strip("'\"") for item in re.split(r"[,;、，]", raw)]
            return [item for item in models if item]
        except Exception:  # noqa: BLE001
            return []

    def _pick_alternative_model(
        self,
        failed_model: str,
        supported_models: Sequence[str],
        tried_models: set[str],
    ) -> str | None:
        """挑选替代模型：优先 FALLBACK_MODEL，其次服务端声明支持的模型。

        Args:
            failed_model: 当前失败的模型名。
            supported_models: 服务端声明支持的模型名（可能为空）。
            tried_models: 已经尝试过的模型集合，避免重复尝试造成死循环。

        Returns:
            str | None: 可尝试的模型名；无可用替代时返回 ``None``。
        """
        candidates: list[str] = []
        if self.fallback_model:
            candidates.append(self.fallback_model)
        candidates.extend(supported_models)
        for candidate in candidates:
            name = str(candidate).strip()
            if name and name != failed_model and name not in tried_models:
                return name
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(json.dumps(get_settings(), ensure_ascii=False, indent=2))
