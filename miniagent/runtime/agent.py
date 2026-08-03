from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any

from miniagent.ids import new_id
from miniagent.llm.base import LLMClient
from miniagent.runtime.context import ContextManager
from miniagent.runtime.parser import ActionParser
from miniagent.runtime.tool_executor import ToolExecutor
from miniagent.session.repository import SQLiteRepository
from miniagent.tools.base import ToolContext
from miniagent.tools.registry import ToolRegistry
from miniagent.tracing.logger import TraceLogger


@dataclass
class AgentResponse:
    session_id: str
    turn_id: str
    trace_id: str
    status: str
    answer: str
    stop_reason: str


class SessionBusyError(RuntimeError):
    pass


class AgentRuntime:
    def __init__(
        self,
        repo: SQLiteRepository,
        llm: LLMClient,
        registry: ToolRegistry,
        context_manager: ContextManager,
        max_steps: int = 8,
        tool_timeout_seconds: float = 10.0,
    ) -> None:
        self.repo = repo
        self.llm = llm
        self.registry = registry
        self.context_manager = context_manager
        self.max_steps = max_steps
        self.parser = ActionParser()
        self.executor = ToolExecutor(registry, timeout_seconds=tool_timeout_seconds)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def send_message(self, user_id: str, session_id: str, content: str) -> AgentResponse:
        session = self.repo.get_session(user_id, session_id)
        if not session:
            raise PermissionError("session not found or forbidden")
        lock = self._lock_for(session_id)
        if not lock.acquire(blocking=False):
            raise SessionBusyError("session is busy")
        turn_id = new_id("turn")
        trace_id = new_id("trace")
        trace = TraceLogger(self.repo, trace_id, turn_id, session_id)
        started = time.perf_counter()
        try:
            self.repo.update_session(session_id, status="running")
            self.repo.save_message(session_id, turn_id, "user", content)
            trace.event(0, "USER_MESSAGE_RECEIVED", {"message_length": len(content)})

            compression = self.context_manager.maybe_compress(session_id)
            if compression:
                trace.event(0, "CONTEXT_COMPRESSED", compression)

            seen_signatures: list[str] = []
            for step in range(1, self.max_steps + 1):
                messages, metadata = self.context_manager.build(session_id)
                trace.event(step, "CONTEXT_BUILT", metadata)
                llm_started = time.perf_counter()
                trace.event(step, "LLM_REQUEST_STARTED", {"model": getattr(self.llm, "model", "fake"), "tool_count": len(self.registry.names())})
                try:
                    llm_message = self.llm.complete(messages, self.registry.schemas())
                    latency = int((time.perf_counter() - llm_started) * 1000)
                    trace.event(
                        step,
                        "LLM_REQUEST_COMPLETED",
                        {
                            "has_tool_calls": bool(llm_message.get("tool_calls")),
                            "finish_reason": llm_message.get("_raw_finish_reason"),
                        },
                        latency_ms=latency,
                    )
                    action = self.parser.parse(llm_message)
                except Exception as exc:
                    answer = f"模型响应解析失败：{exc}"
                    self.repo.save_message(session_id, turn_id, "assistant", answer)
                    trace.event(step, "LLM_REQUEST_FAILED", {"error_type": type(exc).__name__, "message": str(exc)})
                    return self._finish(session_id, turn_id, trace_id, "failed", answer, "llm_failed", trace, started)

                if action.is_final:
                    answer = action.final_answer or ""
                    self.repo.save_message(session_id, turn_id, "assistant", answer)
                    return self._finish(session_id, turn_id, trace_id, "completed", answer, "final_answer", trace, started, step)

                if not action.tool_calls:
                    answer = "模型没有给出可执行动作。"
                    self.repo.save_message(session_id, turn_id, "assistant", answer)
                    return self._finish(session_id, turn_id, trace_id, "failed", answer, "empty_action", trace, started, step)

                self.repo.save_message(
                    session_id,
                    turn_id,
                    "assistant",
                    json.dumps(
                        {
                            "decision_summary": action.decision_summary,
                            "tool_calls": [call.__dict__ for call in action.tool_calls],
                        },
                        ensure_ascii=False,
                    ),
                    message_type="tool_call",
                )
                for call in action.tool_calls:
                    signature = self.executor.canonical_signature(call.name, call.arguments)
                    seen_signatures.append(signature)
                    if len(seen_signatures) >= 3 and seen_signatures[-3:] == [signature, signature, signature]:
                        answer = f"检测到重复工具调用，已停止：{call.name}"
                        self.repo.save_message(session_id, turn_id, "assistant", answer)
                        trace.event(step, "AGENT_FAILED", {"error_type": "repeated_tool_call", "tool_name": call.name})
                        return self._finish(session_id, turn_id, trace_id, "failed", answer, "repeated_tool_call", trace, started, step)

                    trace.event(step, "TOOL_CALL_REQUESTED", {"tool_name": call.name, "arguments": call.arguments, "tool_call_id": call.id})
                    result, tool_latency = self.executor.execute(call.name, call.arguments, ToolContext(user_id, session_id))
                    event_type = "TOOL_CALL_SUCCEEDED" if result.success else "TOOL_CALL_FAILED"
                    trace.event(
                        step,
                        event_type,
                        {"tool_name": call.name, "result_summary": result.summary(), "error": result.error},
                        latency_ms=tool_latency,
                    )
                    self.repo.save_message(
                        session_id,
                        turn_id,
                        "tool",
                        result.to_json(),
                        message_type="tool_result",
                        tool_name=call.name,
                        tool_call_id=call.id,
                    )

            answer = f"达到最大执行步数 {self.max_steps}，已停止。"
            self.repo.save_message(session_id, turn_id, "assistant", answer)
            return self._finish(session_id, turn_id, trace_id, "failed", answer, "max_steps_exceeded", trace, started, self.max_steps)
        finally:
            self.repo.update_session(session_id, status="idle")
            lock.release()

    def _finish(
        self,
        session_id: str,
        turn_id: str,
        trace_id: str,
        status: str,
        answer: str,
        stop_reason: str,
        trace: TraceLogger,
        started: float,
        step: int = 0,
    ) -> AgentResponse:
        event = "AGENT_COMPLETED" if status == "completed" else "AGENT_FAILED"
        trace.event(step, event, {"stop_reason": stop_reason, "total_latency_ms": int((time.perf_counter() - started) * 1000)})
        return AgentResponse(session_id, turn_id, trace_id, status, answer, stop_reason)

    def _lock_for(self, session_id: str) -> threading.Lock:
        with self._locks_guard:
            if session_id not in self._locks:
                self._locks[session_id] = threading.Lock()
            return self._locks[session_id]
