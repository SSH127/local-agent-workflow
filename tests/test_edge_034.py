"""v0.3.4 edge close: cancelable LLM HTTP, soft sandbox, legacy write_scope."""
from __future__ import annotations

import threading
import time

import pytest

from app import config
from app.agents.agent_executor import (
    STATUS_CANCELLED,
    STOP_CANCELLED,
    run_agent,
)
from app.agents.llm import LLMCancelled
from app.cancel import CancelToken
from app.engine import planning as P
from app.tools import tool_schemas
from app.tools.base import ToolContext
from app.tools.command_sandbox import (
    begin_session,
    check_argv_outside_workspace,
    finish_session,
    scratch_dir,
)
from app.tools.registry import get_tool
from app.write_scope import WriteScope
from tests.test_task_graph import (
    FakeExecutor,
    PlannerLLM,
    Recorder,
    plan_json,
    run,
    setup_planner,
    sub,
)


@pytest.fixture
def planner_on(monkeypatch, tmp_path):
    return setup_planner(monkeypatch, tmp_path)


# ---------------------------------------------------------------------------
# 1) Cancel interrupts in-flight LLM
# ---------------------------------------------------------------------------


class SlowLLM:
    """Blocks in chat until cancel_token is set (or timeout)."""

    def __init__(self):
        self.entered = threading.Event()
        self.calls = 0

    def chat(self, messages, tools, cancel_token=None):
        self.calls += 1
        self.entered.set()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if cancel_token is not None and cancel_token.is_set():
                raise LLMCancelled(f"cancelled:{cancel_token.reason()}")
            time.sleep(0.05)
        raise TimeoutError("SlowLLM not cancelled in time")


def test_cancel_aborts_inflight_llm_within_2s(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    token = CancelToken()
    llm = SlowLLM()

    class _Rec:
        def save_step(self, step):
            return None
        def set_status(self, *a, **k):
            return None
        def set_state(self, state):
            return None
        def request_approval(self, *a, **k):
            raise AssertionError("no approval expected")

    outcome_box: dict = {}

    def worker():
        outcome_box["o"] = run_agent(
            "run-c", "分析项目", llm, tool_schemas(),
            _Rec(), cancel_token=token,
        )

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    assert llm.entered.wait(timeout=2.0), "LLM chat never entered"
    t0 = time.monotonic()
    token.set("用户取消")
    t.join(timeout=3.0)
    elapsed = time.monotonic() - t0
    assert not t.is_alive(), "agent loop did not stop"
    assert elapsed < 2.0, f"cancel took too long: {elapsed:.2f}s"
    o = outcome_box["o"]
    assert o.status == STATUS_CANCELLED
    assert o.stop_reason == STOP_CANCELLED
    assert llm.calls == 1


def test_cancel_token_closes_registered_http():
    token = CancelToken()
    closed = []

    class FakeClient:
        def close(self):
            closed.append(True)

    c = FakeClient()
    token.register_http(c)
    token.set("x")
    assert closed == [True]


# ---------------------------------------------------------------------------
# 2) Soft sandbox snapshot for any command under write_scope
# ---------------------------------------------------------------------------


def test_soft_sandbox_detects_source_write_outside_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACT_POLICY", "allow")
    monkeypatch.setattr(config, "COMMAND_SOFT_SANDBOX", True)
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    (tmp_path / "app").mkdir()
    victim = tmp_path / "app" / "main.py"
    victim.write_text("keep\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    # Drive sandbox session directly (same finish path as run_project_command).
    session = begin_session(
        workspace=tmp_path, run_id="r1", subtask_id="t1",
        scope_patterns=["tests/**"], kind="generic",
    )
    assert "TMPDIR" in session.env or "TEMP" in session.env
    assert "PYTHONPYCACHEPREFIX" in session.env
    victim.write_text("tampered\n", encoding="utf-8")
    (tmp_path / "dist").mkdir(exist_ok=True)
    (tmp_path / "dist" / "out.js").write_text("bundle\n", encoding="utf-8")
    finish = finish_session(session)
    assert any(v.path == "app/main.py" for v in finish.violations)
    assert victim.read_text(encoding="utf-8") == "keep\n"
    assert any(a.path.startswith("dist/") for a in finish.artifacts) or (
        # newly created dist file under allow
        any(a.path == "dist/out.js" for a in finish.artifacts)
    )


def test_soft_sandbox_refuses_abs_path_outside_workspace(tmp_path):
    err = check_argv_outside_workspace(
        ["python", "/etc/passwd"], tmp_path, scratch_dir(tmp_path, "r", "s"),
    )
    assert err and "/etc/passwd" in err
    # Inside workspace is fine
    inside = str((tmp_path / "app" / "x.py").resolve())
    (tmp_path / "app").mkdir(exist_ok=True)
    assert check_argv_outside_workspace(["python", inside], tmp_path) is None


def test_run_command_always_scratch_when_scoped(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    monkeypatch.setattr(config, "COMMAND_SOFT_SANDBOX", True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_ok.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8",
    )
    scope = WriteScope(["tests/**"], subtask_id="a1")
    ctx = ToolContext(
        run_id="run-g", workspace=tmp_path, write_scope=scope, subtask_id="a1",
        approval_check=lambda n, a: True,
    )
    res = get_tool("run_project_command").handler(
        {"program": "python", "args": ["-m", "pytest", "tests", "-q"], "cwd": "."}, ctx,
    )
    data = res.get("data") or {}
    assert res.get("ok") is True or data.get("exit_code") == 0, res
    assert scratch_dir(tmp_path, "run-g", "a1").exists()


# ---------------------------------------------------------------------------
# 3) Legacy write_scope migration
# ---------------------------------------------------------------------------


def test_legacy_analyze_becomes_readonly():
    plan = {"subtasks": [
        {"id": "1", "kind": "analyze", "status": P.SUBTASK_PENDING, "title": "a"},
    ], "notes": []}
    notes = P.migrate_legacy_write_scopes(plan, strict=True)
    assert plan["subtasks"][0]["write_scope"] == []
    assert any("只读" in n for n in notes)


def test_legacy_modify_blocks_under_strict(planner_on, monkeypatch):
    monkeypatch.setattr(config, "LEGACY_WRITE_SCOPE_STRICT", True)
    monkeypatch.setattr(config, "PLAN_REQUIRES_APPROVAL", False)
    # Manually inject a legacy-shaped plan by patching request_subtasks result
    # Easier: build plan via PlannerLLM then strip write_scope keys before execute.
    llm = PlannerLLM([plan_json(sub("1", kind="modify", write_scope=["app/**"]))])
    rec = Recorder()

    # Hook store.save to strip write_scope after decompose
    original_save = rec.plan_store.save

    def strip_then_save(plan):
        for s in plan.get("subtasks") or []:
            s.pop("write_scope", None)
            s.pop("needs_write_scope", None)
        return original_save(plan)

    # Only strip once after initial decompose
    calls = {"n": 0}
    def save_once(plan):
        calls["n"] += 1
        if calls["n"] == 1:
            for s in plan.get("subtasks") or []:
                s.pop("write_scope", None)
        return original_save(plan)

    rec.plan_store.save = save_once  # type: ignore[method-assign]
    res = run("修改 a.py", llm, rec, FakeExecutor({}), monkeypatch, planner_on)
    assert res["status"] == "needs_attention", res
    assert "write_scope" in (res.get("error") or "")
    plan = rec.plan_store.load()
    assert any(s.get("needs_write_scope") for s in plan["subtasks"])


def test_legacy_modify_migrates_when_strict_false():
    plan = {"subtasks": [
        {"id": "1", "kind": "modify", "status": P.SUBTASK_PENDING, "title": "m"},
    ], "notes": []}
    notes = P.migrate_legacy_write_scopes(plan, strict=False)
    assert plan["subtasks"][0]["write_scope"] == ["**"]
    assert not plan["subtasks"][0].get("needs_write_scope")
    assert any("**" in n for n in notes)
    ev = plan["subtasks"][0]["acceptance_evidence"]
    assert any(e.get("type") == "legacy_scope_migration" for e in ev)
