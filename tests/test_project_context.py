"""Project context cache, task memory and context-assembly tests.

Covers the Phase-1 "shared cache & task memory" system:

1. repeated same-range reads hit the cache;
2. a modified file invalidates its cache;
3. two projects never share each other's content;
4. the symbol index locates functions/classes and line numbers;
5. an unparseable Python file is still text-searchable;
6. after ``write_file`` the agent reads the *latest* content;
7. task memory persists and is reloaded on the next step/resume;
8. context assembly does not re-send the same large code block;
9. path safety / protected files / existing features keep working.

All tests are offline (temp dirs + scripted LLM; the autouse fixtures isolate
the cache DB and block real network).
"""
from __future__ import annotations

import json
from typing import Any

from app.agents.agent_executor import ResumeState, run_agent
from app.agents.llm import LLMResponse
from app.context_builder import assemble_messages, estimate_prompt_tokens
from app.memory import RunMemory
from app.project_context import ProjectContext
from app.tools import tool_schemas
from app.tools.base import ToolContext
from app.tools.registry import get_tool
from tests.conftest import FakeLLM, final_answer, tool_call


class CapturingRecorder:
    """Minimal RunRecorder that keeps every persisted state dict."""

    def __init__(self, approve: bool = True):
        self.steps: list[dict[str, Any]] = []
        self.state: dict[str, Any] = {}
        self.snapshots: list[dict[str, Any]] = []
        self.approve = approve

    def save_step(self, step):
        self.steps.append(step)

    def set_status(self, status, summary=None, error=None):
        return None

    def set_state(self, state):
        self.snapshots.append(dict(state))
        self.state.update(state)

    def request_approval(self, run_id, tool_name, args, risk_reason):
        from app.engine.approvals import ApprovalDecision

        return ApprovalDecision(approval_id="a1", approved=self.approve)


def _ctx(tmp_path, pc) -> ToolContext:
    return ToolContext(run_id="r", workspace=tmp_path, project=pc)


# ---------------------------------------------------------------------------
# 1. Cache hits for repeated reads of the same file / range
# ---------------------------------------------------------------------------

def test_repeated_read_hits_cache(tmp_path):
    (tmp_path / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    pc = ProjectContext(tmp_path, db_path=tmp_path / "c.sqlite3")

    first = pc.get_file("a.py")
    assert first.cache_hit is False
    second = pc.get_file("a.py")
    assert second.cache_hit is True
    assert second.content == first.content

    stats = pc.stats()
    assert stats["read_requests"] == 2
    assert stats["disk_reads"] == 1
    assert stats["cache_hits"] == 1
    assert stats["cache_misses"] == 1


def test_same_range_repeated_read_via_tool_hits_cache(tmp_path):
    (tmp_path / "f.txt").write_text("\n".join(f"l{i}" for i in range(20)), encoding="utf-8")
    pc = ProjectContext(tmp_path, db_path=tmp_path / "c.sqlite3")
    read = get_tool("read_file")
    ctx = _ctx(tmp_path, pc)

    first = read.handler({"path": "f.txt", "offset": 1, "limit": 5}, ctx)
    second = read.handler({"path": "f.txt", "offset": 1, "limit": 5}, ctx)
    assert first["ok"] and second["ok"]
    assert first["data"]["cache_hit"] is False
    assert second["data"]["cache_hit"] is True
    # Content identical despite the cache hit (same version, same range).
    assert first["data"]["content"] == second["data"]["content"]
    assert "缓存" in second["output"]


# ---------------------------------------------------------------------------
# 2. Modification invalidates the cache
# ---------------------------------------------------------------------------

def test_modified_file_invalidates_cache(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("value = 1\n", encoding="utf-8")
    pc = ProjectContext(tmp_path, db_path=tmp_path / "c.sqlite3")
    pc.get_file("a.py")

    f.write_text("value = 2222\n", encoding="utf-8")  # different size too
    entry = pc.get_file("a.py")
    assert entry.cache_hit is False  # re-read, not served stale
    assert "2222" in entry.content and "= 1" not in entry.content


def test_write_then_read_returns_latest(tmp_path):
    pc = ProjectContext(tmp_path, db_path=tmp_path / "c.sqlite3")
    ctx = _ctx(tmp_path, pc)
    write = get_tool("write_file")
    read = get_tool("read_file")

    write.handler({"path": "f.txt", "content": "old\n"}, ctx)
    first = read.handler({"path": "f.txt"}, ctx)
    assert "old" in first["data"]["content"]

    write.handler({"path": "f.txt", "content": "new version\n"}, ctx)
    second = read.handler({"path": "f.txt"}, ctx)
    assert "new version" in second["data"]["content"]
    assert "old" not in second["data"]["content"]


# ---------------------------------------------------------------------------
# 3. Cross-project isolation (shared DB must not mix content)
# ---------------------------------------------------------------------------

def test_projects_do_not_share_content(tmp_path):
    p1 = tmp_path / "p1"
    p2 = tmp_path / "p2"
    p1.mkdir()
    p2.mkdir()
    (p1 / "same.py").write_text("answer = 1\n", encoding="utf-8")
    (p2 / "same.py").write_text("answer = 2\n", encoding="utf-8")

    db = tmp_path / "shared.sqlite3"
    pc1 = ProjectContext(p1, db_path=db)
    pc2 = ProjectContext(p2, db_path=db)

    assert "answer = 1" in pc1.get_file("same.py").content
    assert "answer = 2" in pc2.get_file("same.py").content
    # Re-read after both are cached still returns each project's own content.
    assert pc1.get_file("same.py").content != pc2.get_file("same.py").content


def test_cross_run_reuse_when_version_unchanged(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("def f():\n    return 1\n", encoding="utf-8")
    db = tmp_path / "c.sqlite3"

    pc = ProjectContext(tmp_path, db_path=db)
    pc.get_file("a.py")  # miss -> persisted

    # A *fresh* instance (a later task on the same project) reuses the cache
    # only because the version (mtime+size) is unchanged.
    pc2 = ProjectContext(tmp_path, db_path=db)
    entry = pc2.get_file("a.py")
    assert entry.cache_hit is True
    assert pc2.stats()["disk_reads"] == 0


# ---------------------------------------------------------------------------
# 4. Symbol index locates functions/classes and line numbers
# ---------------------------------------------------------------------------

def test_index_locates_function_class_and_line(tmp_path):
    (tmp_path / "m.py").write_text(
        "import os\n\n\nclass Foo:\n    def bar(self):\n        return 1\n\n\ndef top():\n    return 2\n",
        encoding="utf-8",
    )
    pc = ProjectContext(tmp_path, db_path=tmp_path / "c.sqlite3")

    funcs = pc.query_symbols("bar")
    assert any(s["name"] == "bar" and s["line"] == 5 and s["kind"] == "function" for s in funcs)

    classes = pc.query_symbols("Foo")
    assert any(s["name"] == "Foo" and s["line"] == 4 and s["kind"] == "class" for s in classes)

    tops = pc.query_symbols("top")
    assert any(s["name"] == "top" and s["line"] == 9 and s["kind"] == "function" for s in tops)
    assert pc.stats().get("index_hits", 0) >= 1


# ---------------------------------------------------------------------------
# 5. Unparseable Python still falls back to text search
# ---------------------------------------------------------------------------

def test_unparseable_python_still_searchable(tmp_path):
    (tmp_path / "bad.py").write_text(
        "def broken(:\n   syntax error ===\nneedle_marker = 1\n", encoding="utf-8"
    )
    pc = ProjectContext(tmp_path, db_path=tmp_path / "c.sqlite3")

    # Indexing must not crash on an unparseable file; it simply has no symbols.
    assert pc.query_symbols("broken") == []

    search = get_tool("search_code")
    result = search.handler({"query": "needle_marker"}, _ctx(tmp_path, pc))
    assert result["ok"] is True
    assert any(m["path"] == "bad.py" for m in result["data"]["matches"])


# ---------------------------------------------------------------------------
# 6. Agent write_file -> read_file sees latest content (end-to-end)
# ---------------------------------------------------------------------------

def test_agent_writes_then_reads_latest(tmp_path):
    (tmp_path / "src.txt").write_text("hello\n", encoding="utf-8")
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    recorder = CapturingRecorder()
    llm = FakeLLM([
        tool_call("c1", "write_file", {"path": "out.txt", "content": "changed\n"}),
        tool_call("c2", "read_file", {"path": "out.txt"}),
        tool_call(
            "c3", "run_project_command",
            {"program": "python", "args": ["-m", "pytest", "test_ok.py", "-q"], "cwd": "."},
        ),
        final_answer("已写入并验证。"),
    ])
    outcome = run_agent("run", "goal", llm, tool_schemas(), recorder, workspace=tmp_path)
    assert outcome.status == "succeeded"

    # The projected request sent for the read step must contain the *new* text.
    read_step = [m for c in llm.calls for m in c[0]
                 if m.get("role") == "tool" and "out.txt" in str(m.get("content", ""))]
    assert read_step, "read_file tool message not found in model context"
    assert "changed" in read_step[-1]["content"]


# ---------------------------------------------------------------------------
# 7. Task memory persists and is reloaded on the next step
# ---------------------------------------------------------------------------

def test_run_memory_roundtrip():
    m = RunMemory(goal="g", plan=["p1", "p2"])
    m.record_modified_file("a.py")
    m.record_command("python", ["-m", "pytest"], 0, True)
    m.record_test_evidence("python", ["-m", "pytest"], True, "1 passed")
    m.add_finding("verified fact", kind="verified")
    m.add_finding("still guessing", kind="inferred")

    m2 = RunMemory.from_dict(m.to_dict())
    assert m2.goal == "g"
    assert m2.plan == ["p1", "p2"]
    assert m2.modified_files == ["a.py"]
    assert m2.commands[0]["exit_code"] == 0
    assert m2.findings[0]["kind"] == "verified"
    assert m2.findings[1]["kind"] == "inferred"


def test_executor_persists_memory_and_token_stats(tmp_path):
    (tmp_path / "f.txt").write_text("x\n", encoding="utf-8")
    recorder = CapturingRecorder()
    llm = FakeLLM([
        tool_call("c1", "read_file", {"path": "f.txt"}),
        tool_call("c2", "write_file", {"path": "out.txt", "content": "done"}),
        final_answer("done"),
    ])
    run_agent("run", "goal", llm, tool_schemas(), recorder, workspace=tmp_path)

    mem = recorder.state.get("memory")
    assert mem is not None
    assert mem["goal"] == "goal"
    assert "out.txt" in mem["modified_files"]
    assert "f.txt" in mem["confirmed_locations"]

    stats = recorder.state.get("token_stats")
    assert stats is not None
    assert stats["model_calls"] >= 1
    # FakeLLM returns no usage -> explicitly marked as estimated.
    assert stats["estimated"] is True

    # The run logs a human-readable stats step.
    assert any(s["kind"] == "stats" for s in recorder.steps)


def test_resume_loads_memory_into_context(tmp_path):
    mem = RunMemory(goal="g", plan=["do A", "do B"])
    mem.record_modified_file("a.py")
    rec = CapturingRecorder()
    llm = FakeLLM([final_answer("ok")])
    resume = ResumeState(messages=[], pending_tools=[], memory=mem.to_dict())

    run_agent("run", "g", llm, tool_schemas(), rec, workspace=tmp_path, resume=resume)

    sent = llm.calls[0][0]
    joined = "".join(str(m.get("content", "")) for m in sent)
    assert "[任务记忆]" in joined
    assert "do A" in joined and "a.py" in joined


# ---------------------------------------------------------------------------
# 8. Context assembly does not re-send the same large code block
# ---------------------------------------------------------------------------

def test_context_assembly_compresses_old_large_outputs():
    big = "BIG_BLOCK_" * 2000  # a large tool output
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "goal"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"output": big})},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c2", "type": "function",
                         "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c2", "content": json.dumps({"output": "small"})},
    ]
    memory = RunMemory(goal="g")
    projected = assemble_messages(messages, memory, tool_tail_chars=400, memory_block_chars=4000)

    all_text = "".join(str(m.get("content", "")) for m in projected)
    # The big content is no longer present in full anywhere in the request.
    assert big not in all_text
    # The current (latest) exchange survives verbatim.
    assert any(m.get("tool_call_id") == "c2" for m in projected)
    assert '"output": "small"' in all_text


def test_estimate_prompt_tokens_is_monotonic():
    small = [{"content": "x"}]
    big = [{"content": "x" * 400}]
    assert estimate_prompt_tokens(big) > estimate_prompt_tokens(small)


def test_llm_response_carries_real_usage():
    r = LLMResponse(
        content="", usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )
    assert r.usage["prompt_tokens"] == 10
    assert r.usage["total_tokens"] == 15


# ---------------------------------------------------------------------------
# 8b. Regression: resume must accumulate stats exactly once
# ---------------------------------------------------------------------------

class _UsageLLM:
    """Returns real usage on the first call only (a mixed-source scenario)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def chat(self, messages, tools):
        self.calls += 1
        response = self.responses.pop(0)
        if self.calls == 1:
            response.usage = {
                "prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
            }
        return response


def test_resume_accumulates_token_and_cache_stats_once(tmp_path):
    (tmp_path / "f.txt").write_text("hello\n", encoding="utf-8")

    rec1 = CapturingRecorder()
    llm1 = FakeLLM([tool_call("c1", "read_file", {"path": "f.txt"}), final_answer("done")])
    run_agent("r1", "g", llm1, tool_schemas(), rec1, workspace=tmp_path)
    first = dict(rec1.state["token_stats"])
    assert first["model_calls"] == 2

    rec2 = CapturingRecorder()
    llm2 = FakeLLM([final_answer("resumed")])
    resume = ResumeState(
        messages=[], pending_tools=[],
        memory=rec1.state["memory"],
        token_stats=first,
        cache_stats=rec1.state["cache_stats"],
    )
    run_agent("r1", "g", llm2, tool_schemas(), rec2, workspace=tmp_path, resume=resume)
    second = rec2.state["token_stats"]

    # Exactly one new model call, accumulated on top of the two prior calls.
    assert second["model_calls"] == first["model_calls"] + 1
    assert second["calls_without_real_usage"] == first["calls_without_real_usage"] + 1
    assert second["prompt_tokens_estimated"] > first["prompt_tokens_estimated"]
    # Cache stats are carried forward, not reset (the resumed process re-reads
    # the file, adding a hit, but the prior counters are never lost).
    assert rec2.state["cache_stats"]["read_requests"] >= rec1.state["cache_stats"]["read_requests"]


def test_paused_approval_state_carries_cache_stats(tmp_path):
    """A run paused for approval must persist its cache counters, so resuming
    does not lose the pre-pause statistics."""
    (tmp_path / "f.txt").write_text("x\n", encoding="utf-8")
    (tmp_path / ".env").write_text("A=1\n", encoding="utf-8")
    rec = CapturingRecorder(approve=True)
    llm = FakeLLM([
        tool_call("c1", "read_file", {"path": "f.txt"}),
        tool_call("c2", "write_file", {"path": ".env", "content": "A=2\n"}),
        final_answer("done"),
    ])
    run_agent("r", "g", llm, tool_schemas(), rec, workspace=tmp_path)

    pause_snapshots = [
        s for s in rec.snapshots
        if "agent_messages" in s and "cache_stats" in s
    ]
    assert pause_snapshots, "approval pause did not persist stats"
    assert pause_snapshots[-1]["cache_stats"]["read_requests"] >= 1


def test_mixed_usage_is_not_reported_as_all_real(tmp_path):
    rec = CapturingRecorder()
    llm = _UsageLLM([
        tool_call("c1", "list_files", {}),
        LLMResponse(content="final"),  # no usage -> estimated
    ])
    run_agent("r", "g", llm, tool_schemas(), rec, workspace=tmp_path)
    ts = rec.state["token_stats"]

    assert ts["model_calls"] == 2
    assert ts["calls_with_real_usage"] == 1
    assert ts["calls_without_real_usage"] == 1
    assert ts["usage_source"] == "mixed"
    # "estimated" stays False only when EVERY call had real usage.
    assert ts["estimated"] is False
    assert ts["total_tokens"] == 110  # real total covers the one real call only
    assert ts["prompt_tokens_estimated"] > 0  # estimate covers the other call

    stats_step = [s for s in rec.steps if s["kind"] == "stats"][-1]["output"]
    assert "不可相加" in stats_step


def test_all_real_usage_marks_api_source(tmp_path):
    rec = CapturingRecorder()

    class AllUsage:
        def chat(self, messages, tools):
            return LLMResponse(
                content="final",
                usage={"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
            )

    run_agent("r", "g", AllUsage(), tool_schemas(), rec, workspace=tmp_path)
    ts = rec.state["token_stats"]
    assert ts["usage_source"] == "api"
    assert ts["estimated"] is False
    assert ts["total_tokens"] == 9
    assert ts["prompt_tokens_estimated"] == 0


# ---------------------------------------------------------------------------
# 8c. Regression: history compression must keep failure evidence + protocol
# ---------------------------------------------------------------------------

def test_compression_preserves_failed_output_head_and_tail():
    big = ("noise line\n" * 200) + "AssertionError: expected 2 got 3\n" + ("trailing\n" * 200)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "goal"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "run_project_command", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1",
         "content": json.dumps({"ok": False, "output": big}, ensure_ascii=False)},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c2", "type": "function",
                         "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c2", "content": json.dumps({"ok": True, "output": "x"})},
    ]
    projected = assemble_messages(
        messages, RunMemory(goal="g"), tool_tail_chars=400, failure_tail_chars=4000
    )
    old_failed = projected[3]["content"]
    # The tail (pytest summary region) survives, so the failure reason is intact.
    assert "trailing" in old_failed
    assert len(old_failed) > 400


def test_compression_keeps_tool_call_protocol_valid():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "goal"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"output": "a" * 9000})},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c2", "type": "function",
                         "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c2", "content": json.dumps({"output": "small"})},
    ]
    projected = assemble_messages(messages, RunMemory(goal="g"), tool_tail_chars=200)

    call_ids = {
        call["id"]
        for m in projected if m.get("role") == "assistant"
        for call in (m.get("tool_calls") or [])
    }
    replied = {m.get("tool_call_id") for m in projected if m.get("role") == "tool"}
    # Every requested tool call still has its reply (protocol intact).
    assert call_ids == replied == {"c1", "c2"}
    # The live exchange is verbatim; only the older one was compressed.
    assert projected[-1]["content"] == json.dumps({"output": "small"})


def test_executor_records_failure_in_memory(tmp_path):
    rec = CapturingRecorder()
    llm = FakeLLM([
        tool_call("c1", "read_file", {"path": "missing.py"}),
        final_answer("done"),
    ])
    run_agent("r", "g", llm, tool_schemas(), rec, workspace=tmp_path)
    pending = rec.state["memory"]["pending"]
    assert any("read_file 失败" in p for p in pending)


# ---------------------------------------------------------------------------
# 9b. Regression: external file changes must refresh the symbol index
# ---------------------------------------------------------------------------

def test_index_refreshes_after_external_edit(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("def alpha():\n    pass\n", encoding="utf-8")
    pc = ProjectContext(tmp_path, db_path=tmp_path / "c.sqlite3")
    assert pc.query_symbols("alpha")

    # External edit (not through write_file): the index must not stay stale.
    f.write_text("def beta():\n    pass\n", encoding="utf-8")
    assert pc.query_symbols("alpha") == []
    assert any(s["name"] == "beta" for s in pc.query_symbols("beta"))


def test_index_detects_externally_added_and_deleted_files(tmp_path):
    pc = ProjectContext(tmp_path, db_path=tmp_path / "c.sqlite3")
    (tmp_path / "seed.py").write_text("def seed():\n    pass\n", encoding="utf-8")
    assert any(s["name"] == "seed" for s in pc.query_symbols("seed"))

    added = tmp_path / "added.py"
    added.write_text("def gamma():\n    pass\n", encoding="utf-8")
    assert any(s["name"] == "gamma" for s in pc.query_symbols("gamma"))

    added.unlink()
    assert pc.query_symbols("gamma") == []


def test_deleted_file_drops_cache_entry(tmp_path):
    f = tmp_path / "gone.py"
    f.write_text("X = 1\n", encoding="utf-8")
    pc = ProjectContext(tmp_path, db_path=tmp_path / "c.sqlite3")
    pc.get_file("gone.py")
    assert "gone.py" in pc._files

    f.unlink()
    result = get_tool("read_file").handler(
        {"path": "gone.py"}, _ctx(tmp_path, pc)
    )
    assert result["ok"] is False
    assert "gone.py" not in pc._files



# ---------------------------------------------------------------------------
# 9. Path safety / protected files / non-regression with the cache attached
# ---------------------------------------------------------------------------

def test_protected_file_not_served_from_cache(tmp_path):
    (tmp_path / ".env").write_text("SECRET=1\n", encoding="utf-8")
    pc = ProjectContext(tmp_path, db_path=tmp_path / "c.sqlite3")
    read = get_tool("read_file")
    result = read.handler({"path": ".env"}, _ctx(tmp_path, pc))
    assert result["ok"] is False
    assert "受保护" in result["output"]


def test_path_escape_still_rejected_with_cache(tmp_path):
    outside = tmp_path / ".." / "outside.txt"
    (tmp_path / "inside.txt").write_text("inside\n", encoding="utf-8")
    pc = ProjectContext(tmp_path, db_path=tmp_path / "c.sqlite3")
    read = get_tool("read_file")
    result = read.handler({"path": "../outside.txt"}, _ctx(tmp_path, pc))
    assert result["ok"] is False
    assert "越界" in result["output"]
    assert not outside.exists()
