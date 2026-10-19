"""DeepSeek chat client (OpenAI-compatible) plus the LLM interface used by the
agent executor.

The executor depends only on the ``chat(messages, tools)`` method, so tests can
inject a scripted fake without touching the network.
"""
from __future__ import annotations

import contextlib
import json
import threading
from dataclasses import dataclass, field
from typing import Any

import httpx

from app import config


class LLMError(Exception):
    """Raised when the model call fails or is misconfigured."""


class LLMCancelled(LLMError):
    """Raised when an in-flight model call is aborted by CancelToken."""


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None
    # Real token usage from the API ``usage`` object when the provider returns
    # it. ``None`` means "not provided" — the executor then falls back to an
    # explicitly-marked estimate.
    usage: dict[str, Any] | None = None

    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def parsed_tool_calls(self) -> list[tuple[str, str, dict[str, Any]]]:
        """Return ``[(call_id, name, args_dict)]`` with parsed JSON arguments."""
        out: list[tuple[str, str, dict[str, Any]]] = []
        for call in self.tool_calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            raw = fn.get("arguments", "{}")
            try:
                args = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except json.JSONDecodeError:
                args = {"_raw": raw}
            out.append((call.get("id", ""), name, args if isinstance(args, dict) else {"value": args}))
        return out


SYSTEM_PROMPT = """你是一个本地 AI Agent，负责真实地分析和修复一个代码项目。

你拥有以下工具，可以读取文件、搜索代码、运行测试/构建、检查端口和 HTTP 服务、
写入文件以及查看自己的执行日志。你必须**实际调用工具**来获取信息、执行操作并
验证结果，不能凭空猜测或编造。

工作规则：
1. 先用工具了解现状（list_files / read_file / search_code / check_port / check_http）。
2. 需要修改已有文件的局部内容时，用 edit_file 做精确替换（old_string 必须与原文一致）；
   只有创建新文件或确实要整体重写时才用 write_file。修改后运行相关测试或构建
   （run_project_command）验证修改是否生效。
   每次写入都会在写入前保存快照并返回 tx_id；若需要撤销自己的修改，用
   rollback_file 并传入该 tx_id（需要审批）。
3. 工具返回的结果是唯一的事实来源。若工具失败，根据错误信息决定重试、换一种
   方式或修复后再验证。
4. 不要输出或要求任何 API Key、密码、令牌等敏感信息。
   如果缺少完成任务所必需、且无法通过读取代码获得的信息（例如需求有歧义），
   可以调用 ask_user 向用户提一个具体、简短的问题，任务会暂停等待回复。
   不要用 ask_user 询问可以自己查到的内容。
5. 当你确定任务已完成并已通过验证时，直接给出最终总结（不要继续调用工具）。

命令与脚本：
- 运行测试时必须给出明确的测试文件或目录，例如
  run_project_command(program="python", args=["-m", "pytest", "tests/test_x.py", "-q"])。
  不带目标（如 `pytest -q`）会递归收集整个项目，属于需要审批的高风险命令。
- 不允许把普通脚本交给 pytest 执行（pytest 会把显式指定的 .py 当作模块导入），
  也不允许 `python -c`、`coverage run <脚本>`、`python -m unittest <模块>` 这类
  间接执行入口。需要调试时，请阅读代码、修改代码并用测试验证。
- 不要编写临时脚本绕过工具限制；这类命令会被安全策略拒绝。

高效执行（你的轮次预算是有限的，请务必节约）：
- 先搜索后读取：先用 search_code 定位关键函数/调用位置，再 read_file 读取相关实现，
  不要为了改几个函数而通读整个项目。
- 一次回复并行请求多个工具：可以在同一条回复里批量发起多个 read_file / search_code，
  批量调用不额外消耗轮次。
- 读取不要重叠：read_file 会返回“本次范围中有 N 行此前已读取”的提示；看到该提示
  或已读内容时，用 offset 跳到未读部分，避免重复读取同一段。
- 大文件分段读取：read_file 返回 truncated=true 时会给出 next_offset，按提示继续；
  不要为了一口气读完而反复读取已读内容。
- 尽早动手：定位到目标后先形成简短修改计划并开始修改，不必等整个项目全部读完。
- 只读分析任务不要调用 write_file / edit_file；修复任务在修改后必须运行测试/构建验证。

工具调用格式：你可以在回复中请求一个或多个工具调用，工具名和参数必须以 JSON
形式给出，参数类型要与工具定义一致。"""


class DeepSeekClient:
    """Minimal OpenAI-compatible chat client targeting the DeepSeek API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else config.DEEPSEEK_API_KEY
        self.base_url = (base_url or config.DEEPSEEK_BASE_URL).rstrip("/")
        self.model = model or config.DEEPSEEK_MODEL
        self.timeout = timeout if timeout is not None else config.LLM_TIMEOUT_SECONDS

    def _require_key(self) -> None:
        if not self.api_key:
            env_file = config.PROJECT_ROOT / ".env"
            raise LLMError(
                "DeepSeek API Key 未配置。请在 .env 文件中设置 DEEPSEEK_API_KEY "
                f"（参考 .env.example），或设置同名环境变量后重启服务。"
                f"配置查找路径：{env_file}"
            )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        cancel_token: Any = None,
    ) -> LLMResponse:
        self._require_key()
        if cancel_token is not None and getattr(cancel_token, "is_set", lambda: False)():
            raise LLMCancelled(f"任务已取消：{cancel_token.reason()}")
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        connect_s = float(getattr(config, "LLM_CONNECT_TIMEOUT_SECONDS", 10) or 10)
        read_s = float(self.timeout)
        timeout = httpx.Timeout(read_s, connect=connect_s, write=connect_s, pool=connect_s)
        force_ms = int(getattr(config, "CANCEL_FORCE_MS", 500) or 500)
        join_ms = int(getattr(config, "CANCEL_JOIN_MS", 1000) or 1000)

        # Dedicated Client per call (no shared pool across cancelled requests).
        # Worker thread + abandon flag so a late response is discarded.
        box: dict[str, Any] = {"abandoned": False}
        done = threading.Event()
        abandoned = threading.Event()

        def _close_client(client: httpx.Client | None) -> None:
            if client is None:
                return
            with contextlib.suppress(Exception):
                client.close()
            # Drop keepalives / underlying transport if still referenced.
            for attr in ("_transport", "_mounts"):
                with contextlib.suppress(Exception):
                    obj = getattr(client, attr, None)
                    if obj is None:
                        continue
                    if hasattr(obj, "close"):
                        obj.close()
                    elif isinstance(obj, dict):
                        for v in obj.values():
                            with contextlib.suppress(Exception):
                                close = getattr(v, "close", None) or getattr(
                                    getattr(v, "_pool", None), "close", None
                                )
                                if callable(close):
                                    close()

        def _do_post() -> None:
            limits = httpx.Limits(max_connections=1, max_keepalive_connections=0)
            client = httpx.Client(timeout=timeout, limits=limits)
            box["client"] = client
            if cancel_token is not None:
                cancel_token.register_http(client)
            try:
                response = client.post(url, headers=headers, json=payload)
                if abandoned.is_set() or box.get("abandoned"):
                    # Late response after cancel — discard; do not publish.
                    return
                box["response"] = response
            except Exception as exc:  # noqa: BLE001 — surfaced below if not abandoned
                if not (abandoned.is_set() or box.get("abandoned")):
                    box["error"] = exc
            finally:
                if cancel_token is not None:
                    cancel_token.unregister_http(client)
                _close_client(client)
                done.set()

        worker = threading.Thread(target=_do_post, name="llm-chat", daemon=True)
        worker.start()

        poll = max(0.05, force_ms / 1000.0)
        while not done.wait(timeout=poll):
            if cancel_token is not None and cancel_token.is_set():
                box["abandoned"] = True
                abandoned.set()
                _close_client(box.get("client"))
                # Brief wait for clean close; then abandon the daemon thread.
                done.wait(timeout=max(0.05, join_ms / 1000.0))
                raise LLMCancelled(
                    f"任务已取消（模型请求已中止）：{cancel_token.reason()}"
                )

        if cancel_token is not None and cancel_token.is_set():
            box["abandoned"] = True
            abandoned.set()
            raise LLMCancelled(f"任务已取消：{cancel_token.reason()}")

        if box.get("abandoned"):
            raise LLMCancelled("任务已取消（已丢弃迟到的模型响应）")

        if "error" in box:
            exc = box["error"]
            if isinstance(exc, httpx.HTTPError):
                raise LLMError(f"调用 DeepSeek API 网络错误：{exc}") from exc
            raise LLMError(f"调用 DeepSeek API 失败：{exc}") from exc

        response = box["response"]
        if response.status_code != 200:
            snippet = response.text[:500]
            raise LLMError(f"DeepSeek API 返回 HTTP {response.status_code}：{snippet}")

        try:
            data = response.json()
            choice = data["choices"][0]
            message = choice.get("message", {})
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"DeepSeek API 响应格式异常：{response.text[:500]}") from exc

        return LLMResponse(
            content=message.get("content"),
            tool_calls=message.get("tool_calls") or [],
            finish_reason=choice.get("finish_reason"),
            usage=data.get("usage"),
        )





def build_messages(
    goal: str,
    history: list[dict[str, Any]] | None = None,
    workspace: str | None = None,
) -> list[dict[str, Any]]:
    """Assemble the full message list for a chat completion call."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    user = f"用户目标：{goal}"
    if workspace:
        user += (
            f"\n本次任务工作区：{workspace}。"
            "你只能读取、搜索、写入该目录内的文件；绝对路径也必须位于该目录内部。"
        )
    user += "\n请开始分析并执行，最终给出已验证的总结。"
    messages.append({"role": "user", "content": user})
    if history:
        messages.extend(history)
    return messages
