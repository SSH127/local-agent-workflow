"""ask_user / waiting_for_input flow (mirrors the approval flow)."""
from __future__ import annotations

import json
import threading
import time
from typing import Any

from app import config, storage
from app.agents.agent_executor import ResumeState, run_agent
from app.engine.approvals import ApprovalDecision
from app.engine.runner import Runner, StorageRecorder
from app.schemas import InputRequest
from app.tools import tool_schemas
from tests.conftest import FakeLLM, final_answer, tool_call


class Recorder:
    def __init__(self) -> None:
        self.steps: list[dict] = []
        self.states: list[dict] = []

    def save_step(self, step):
        self.steps.append(step)

    def set_status(self, status, summary=None, error=None):
        return None

    def set_state(self, state):
        self.states.append(dict(state))

    def request_approval(self, run_id, tool_name, args, risk_reason):
        return ApprovalDecision(approval_id="x", approved=False)


class AnsweringRecorder(Recorder):
    def __init__(self, answer: str | None) -> None:
        super().__init__()
        self.answer = answer
        self.questions: list[str] = []

    def request_input(self, run_id, question):
        self.questions.append(question)
        return self.answer


def _tool_messages(llm: FakeLLM) -> list[dict[str, Any]]:
    return [m for m in llm.calls[-1][0] if m.get("role") == "tool"]


def test_ask_user_tool_is_registered():
    names = [t["function"]["name"] for t in tool_schemas()]
    assert "ask_user" in names


def test_ask_user_answer_is_returned_to_model(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    recorder = AnsweringRecorder("用 PostgreSQL")
    llm = FakeLLM([
        tool_call("c1", "ask_user", {"question": "用哪个数据库？"}),
        final_answer("好的，已记录。"),
    ])
    outcome = run_agent("run-ask", "测试任务", llm, tool_schemas(), recorder)
    assert recorder.questions == ["用哪个数据库？"]
    tool_msgs = _tool_messages(llm)
    assert len(tool_msgs) == 1
    payload = json.loads(tool_msgs[0]["content"])
    assert payload["ok"] is True
    assert payload["data"]["answer"] == "用 PostgreSQL"
    statuses = [s["status"] for s in recorder.steps if s["kind"] == "input"]
    assert statuses == ["waiting_for_input", "answered"]
    assert outcome.status == "succeeded"


def test_ask_user_unanswered_fails_cleanly(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    recorder = AnsweringRecorder(None)
    llm = FakeLLM([
        tool_call("c1", "ask_user", {"question": "继续吗？"}),
        final_answer("无法继续。"),
    ])
    run_agent("run-ask", "测试任务", llm, tool_schemas(), recorder)
    payload = json.loads(_tool_messages(llm)[0]["content"])
    assert payload["ok"] is False
    assert "未在时限内回复" in payload["output"]


def test_ask_user_without_channel(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    recorder = Recorder()  # no request_input
    llm = FakeLLM([
        tool_call("c1", "ask_user", {"question": "继续吗？"}),
        final_answer("无法继续。"),
    ])
    run_agent("run-ask", "测试任务", llm, tool_schemas(), recorder)
    payload = json.loads(_tool_messages(llm)[0]["content"])
    assert payload["ok"] is False
    assert "没有可用的用户输入通道" in payload["output"]


def test_ask_user_pauses_with_pending_state(monkeypatch, tmp_path):
    """Before blocking, the remaining batch is persisted so a restart can
    replay the question (same as approvals)."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    recorder = AnsweringRecorder("是")
    llm = FakeLLM([
        tool_call("c1", "ask_user", {"question": "继续吗？"}),
        final_answer("完成。"),
    ])
    run_agent("run-ask", "测试任务", llm, tool_schemas(), recorder)
    pending = [s for s in recorder.states if s.get("pending_tools")]
    assert pending and pending[0]["pending_tools"][0]["name"] == "ask_user"


def test_resume_answers_pending_question_without_asking(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    recorder = AnsweringRecorder("不应被调用")
    from app.agents.llm import build_messages

    messages = build_messages("测试任务", workspace=str(tmp_path))
    messages.append({
        "role": "assistant", "content": "",
        "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "ask_user", "arguments": json.dumps({"question": "端口？"})}}],
    })
    resume = ResumeState(
        messages=messages,
        pending_tools=[{"name": "ask_user", "args": {"question": "端口？"}, "call_id": "c1"}],
        user_input="8080",
    )
    llm = FakeLLM([final_answer("完成。")])
    run_agent("run-ask", "测试任务", llm, tool_schemas(), recorder, resume=resume)
    assert recorder.questions == []
    payload = json.loads(_tool_messages(llm)[0]["content"])
    assert payload["data"]["answer"] == "8080"


def _save_run(run_id: str, status: str, state: dict[str, Any] | None = None) -> None:
    storage.init_db()
    storage.save_run({
        "run_id": run_id, "goal": "测试任务", "status": status,
        "attempts": 0, "summary": "", "error": None,
        "state": state or {},
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    })


def test_live_input_resolves_blocked_question(monkeypatch):
    from app import main

    _save_run("r-in", "running")
    fresh = Runner()
    monkeypatch.setattr(main, "runner", fresh)
    recorder = StorageRecorder("r-in", fresh.gate)
    answers: list[str | None] = []
    worker = threading.Thread(
        target=lambda: answers.append(recorder.request_input("r-in", "选哪个方案？")),
    )
    worker.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        run = storage.get_run("r-in")
        if run["status"] == "waiting_for_input" and (run["state"] or {}).get("pending_input"):
            break
        time.sleep(0.01)
    run = storage.get_run("r-in")
    assert run["status"] == "waiting_for_input"
    assert run["state"]["pending_input"]["question"] == "选哪个方案？"

    resume_calls: list[str] = []
    monkeypatch.setattr(fresh, "resume", lambda run_id, goal: resume_calls.append(run_id))
    main.provide_input("r-in", InputRequest(message="方案 B"))
    worker.join(timeout=5)
    assert answers == ["方案 B"]
    assert resume_calls == []  # live waiter answered; no resume thread
    final = storage.get_run("r-in")
    assert final["status"] == "running"
    assert final["state"].get("pending_input") is None


def test_input_after_restart_resumes_with_answer(monkeypatch):
    from app import main

    _save_run("r-lost", "waiting_for_input", {
        "pending_input": {"input_id": "lost-id", "question": "端口？"},
    })
    fresh = Runner()  # empty gate: the waiter was lost
    monkeypatch.setattr(main, "runner", fresh)
    resume_calls: list[str] = []
    monkeypatch.setattr(fresh, "resume", lambda run_id, goal: resume_calls.append(run_id))
    result = main.provide_input("r-lost", InputRequest(message="8080"))
    assert resume_calls == ["r-lost"]
    assert result["status"] == "running"
    assert result["state"]["user_input"] == "8080"


def test_input_wait_times_out(monkeypatch):
    _save_run("r-to", "running")
    monkeypatch.setattr(config, "INPUT_WAIT_SECONDS", 0.05)
    recorder = StorageRecorder("r-to", Runner().gate)
    assert recorder.request_input("r-to", "在吗？") is None
    assert storage.get_run("r-to")["status"] == "running"
