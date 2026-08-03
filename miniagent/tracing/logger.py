from __future__ import annotations

from typing import Any

from miniagent.session.repository import SQLiteRepository


SENSITIVE_KEYS = ("api_key", "token", "password", "secret")


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "***REDACTED***" if any(marker in key.lower() for marker in SENSITIVE_KEYS) else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


class TraceLogger:
    def __init__(self, repo: SQLiteRepository, trace_id: str, turn_id: str, session_id: str) -> None:
        self.repo = repo
        self.trace_id = trace_id
        self.turn_id = turn_id
        self.session_id = session_id

    def event(self, step: int, event_type: str, payload: dict[str, Any] | None = None, latency_ms: int | None = None) -> None:
        self.repo.save_trace(
            trace_id=self.trace_id,
            turn_id=self.turn_id,
            session_id=self.session_id,
            step=step,
            event_type=event_type,
            payload=_redact(payload or {}),
            latency_ms=latency_ms,
        )
