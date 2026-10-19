"""Context assembly: build a compact request from raw history + task memory.

The executor keeps the *raw* ``messages`` list intact (for correctness and
resume persistence) but passes a *projected* list to the model. Projection:

* keeps the system + user prefix and the current tool exchange (the most recent
  assistant ``tool_calls`` message and everything after it) verbatim, so the
  OpenAI tool-call protocol stays valid;
* truncates older tool-message contents to ``tool_tail_chars`` — their
  conclusions are carried in the task-memory block instead;
* injects one bounded ``[任务记忆]`` user message summarizing what is still
  relevant.

This is what actually reduces input tokens; a cache hit alone does not.
"""
from __future__ import annotations

from typing import Any

from app.memory import RunMemory


def estimate_tokens(text: str) -> int:
    """Coarse token estimate (≈ 4 chars/token). Explicitly an *estimate*.

    Real token counts come from the API ``usage`` object when available; this
    fallback must never be presented as an actual count.
    """
    return max(1, (len(text) + 3) // 4)


def _message_chars(message: dict[str, Any]) -> int:
    total = len(str(message.get("content", "")) or "")
    for call in message.get("tool_calls") or []:
        fn = call.get("function", {}) if isinstance(call, dict) else {}
        total += len(str(fn.get("name", "")) or "") + len(str(fn.get("arguments", "")) or "")
    return total


def estimate_prompt_tokens(messages: list[dict[str, Any]]) -> int:
    total_chars = sum(_message_chars(m) for m in messages)
    return max(1, (total_chars + 3) // 4)


def _is_failed_tool_content(content: str) -> bool:
    """True when a serialized tool result starts with ``"ok": false``."""
    head = content[:400].replace(" ", "")
    return '"ok":false' in head


def _compress_old_tool_content(
    content: str, tail_chars: int, failure_tail_chars: int
) -> str:
    """Shrink an *old* tool result while keeping the evidence that matters.

    Successful results are cut to ``tail_chars`` (their conclusions live in the
    task-memory block). **Failed** results keep much more budget and retain both
    the beginning and the end (head + tail), because pytest/ruff error details
    and the summary line often sit at opposite ends — blindly keeping only the
    prefix would silently drop the reason a run failed.
    """
    if _is_failed_tool_content(content):
        limit = max(tail_chars, failure_tail_chars)
        if len(content) <= limit:
            return content
        head_keep = min(800, limit // 4)
        tail_keep = limit - head_keep
        return (
            content[:head_keep]
            + "\n…[历史失败输出中部已省略，结尾保留原始错误详情]…\n"
            + content[-tail_keep:]
        )
    if len(content) <= tail_chars:
        return content
    return content[:tail_chars] + "…[历史结果已压缩，结论见任务记忆]"


def assemble_messages(
    messages: list[dict[str, Any]],
    memory: RunMemory,
    *,
    tool_tail_chars: int = 400,
    memory_block_chars: int = 4000,
    failure_tail_chars: int = 4000,
) -> list[dict[str, Any]]:
    """Return the projected message list for the next model call.

    Protocol guarantee: the projection never drops an assistant ``tool_calls``
    message nor its matching ``tool`` replies, so every tool call id in the
    request still has its answer (required by the OpenAI-compatible API).
    """
    if not messages:
        return list(messages)

    # Find the most recent assistant message that issued tool calls: it and
    # everything after it are the *live* exchange and must stay intact.
    last_tool_calls = -1
    for i, m in enumerate(messages):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            last_tool_calls = i

    head = messages[:last_tool_calls] if last_tool_calls >= 0 else messages
    tail = messages[last_tool_calls:] if last_tool_calls >= 0 else []

    projected: list[dict[str, Any]] = []
    for m in head:
        if m.get("role") == "tool":
            content = m.get("content", "")
            if isinstance(content, str):
                content = _compress_old_tool_content(
                    content, tool_tail_chars, failure_tail_chars
                )
            projected.append({**m, "content": content})
        else:
            projected.append(m)

    block = memory.context_block(memory_block_chars)
    if block:
        projected.append({"role": "user", "content": block})

    projected.extend(tail)
    return projected
