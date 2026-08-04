from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from miniagent.ids import new_id


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class RepositoryError(RuntimeError):
    pass


class SQLiteRepository:
    def __init__(self, database_url: str = "sqlite:///./agent.db") -> None:
        if database_url.startswith("sqlite:///"):
            path = database_url.removeprefix("sqlite:///")
        else:
            path = database_url
        self.path = path
        if path not in (":memory:", ""):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def connect(self) -> Iterable[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'idle',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, updated_at);

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tool_name TEXT,
                    tool_call_id TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session_time ON messages(session_id, created_at);

                CREATE TABLE IF NOT EXISTS todos (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    source_session_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    due_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_todos_user_status ON todos(user_id, status);

                CREATE TABLE IF NOT EXISTS trace_events (
                    id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    latency_ms INTEGER,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_trace_turn ON trace_events(turn_id, created_at);
                """
            )

    def create_session(self, user_id: str, title: str | None = None) -> dict[str, Any]:
        now = utc_now()
        session = {
            "id": new_id("sess"),
            "user_id": user_id,
            "title": title or "New session",
            "summary_json": "{}",
            "status": "idle",
            "created_at": now,
            "updated_at": now,
        }
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions(id, user_id, title, summary_json, status, created_at, updated_at)
                VALUES(:id, :user_id, :title, :summary_json, :status, :created_at, :updated_at)
                """,
                session,
            )
        return session

    def get_session(self, user_id: str, session_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id=? AND user_id=?",
                (session_id, user_id),
            ).fetchone()
        return dict(row) if row else None

    def list_sessions(self, user_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE user_id=? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def update_session(self, session_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = utc_now()
        assignments = ", ".join(f"{key}=:{key}" for key in fields)
        fields["id"] = session_id
        with self.connect() as conn:
            conn.execute(f"UPDATE sessions SET {assignments} WHERE id=:id", fields)

    def delete_session(self, user_id: str, session_id: str) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT id FROM sessions WHERE id=? AND user_id=?",
                (session_id, user_id),
            ).fetchone()
            if not row:
                return False
            conn.execute("DELETE FROM trace_events WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id=? AND user_id=?", (session_id, user_id))
        return True

    def save_message(
        self,
        session_id: str,
        turn_id: str,
        role: str,
        content: str,
        message_type: str = "text",
        tool_name: str | None = None,
        tool_call_id: str | None = None,
    ) -> dict[str, Any]:
        message = {
            "id": new_id("msg"),
            "session_id": session_id,
            "turn_id": turn_id,
            "role": role,
            "message_type": message_type,
            "content": content,
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "created_at": utc_now(),
        }
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO messages(id, session_id, turn_id, role, message_type, content, tool_name, tool_call_id, created_at)
                VALUES(:id, :session_id, :turn_id, :role, :message_type, :content, :tool_name, :tool_call_id, :created_at)
                """,
                message,
            )
        return message

    def get_messages(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM messages WHERE session_id=? ORDER BY created_at ASC, rowid ASC"
        params: tuple[Any, ...] = (session_id,)
        if limit is not None:
            sql = (
                "SELECT * FROM (SELECT *, rowid AS _rowid FROM messages WHERE session_id=? "
                "ORDER BY created_at DESC, _rowid DESC LIMIT ?) ORDER BY created_at ASC, _rowid ASC"
            )
            params = (session_id, limit)
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def count_messages(self, session_id: str) -> int:
        with self.connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM messages WHERE session_id=?", (session_id,)).fetchone()[0])

    def get_summary(self, session_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute("SELECT summary_json FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not row or not row["summary_json"]:
            return {}
        try:
            return json.loads(row["summary_json"])
        except json.JSONDecodeError:
            return {}

    def update_summary(self, session_id: str, summary: dict[str, Any]) -> None:
        self.update_session(session_id, summary_json=json.dumps(summary, ensure_ascii=False))

    def create_todo(self, user_id: str, source_session_id: str, title: str, due_at: str | None = None) -> dict[str, Any]:
        now = utc_now()
        todo = {
            "id": new_id("todo"),
            "user_id": user_id,
            "source_session_id": source_session_id,
            "title": title,
            "status": "pending",
            "due_at": due_at,
            "created_at": now,
            "updated_at": now,
        }
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO todos(id, user_id, source_session_id, title, status, due_at, created_at, updated_at)
                VALUES(:id, :user_id, :source_session_id, :title, :status, :due_at, :created_at, :updated_at)
                """,
                todo,
            )
        return todo

    def list_todos(self, user_id: str, status: str | None = None) -> list[dict[str, Any]]:
        with self.connect() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM todos WHERE user_id=? AND status=? ORDER BY created_at DESC",
                    (user_id, status),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM todos WHERE user_id=? ORDER BY created_at DESC",
                    (user_id,),
                ).fetchall()
        return [dict(row) for row in rows]

    def complete_todo(self, user_id: str, todo_id: str) -> dict[str, Any] | None:
        return self.set_todo_status(user_id, todo_id, "completed")

    def reopen_todo(self, user_id: str, todo_id: str) -> dict[str, Any] | None:
        return self.set_todo_status(user_id, todo_id, "pending")

    def set_todo_status(self, user_id: str, todo_id: str, status: str) -> dict[str, Any] | None:
        if status not in {"pending", "completed"}:
            raise ValueError("todo status must be pending or completed")
        now = utc_now()
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM todos WHERE id=? AND user_id=?", (todo_id, user_id)).fetchone()
            if not row:
                return None
            conn.execute("UPDATE todos SET status=?, updated_at=? WHERE id=?", (status, now, todo_id))
            updated = conn.execute("SELECT * FROM todos WHERE id=?", (todo_id,)).fetchone()
        return dict(updated) if updated else None

    def save_trace(
        self,
        trace_id: str,
        turn_id: str,
        session_id: str,
        step: int,
        event_type: str,
        payload: dict[str, Any],
        latency_ms: int | None = None,
    ) -> dict[str, Any]:
        event = {
            "id": new_id("evt"),
            "trace_id": trace_id,
            "turn_id": turn_id,
            "session_id": session_id,
            "step": step,
            "event_type": event_type,
            "payload_json": json.dumps(payload, ensure_ascii=False, default=str),
            "latency_ms": latency_ms,
            "created_at": utc_now(),
        }
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO trace_events(id, trace_id, turn_id, session_id, step, event_type, payload_json, latency_ms, created_at)
                VALUES(:id, :trace_id, :turn_id, :session_id, :step, :event_type, :payload_json, :latency_ms, :created_at)
                """,
                event,
            )
        return event

    def get_trace(self, turn_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trace_events WHERE turn_id=? ORDER BY created_at ASC",
                (turn_id,),
            ).fetchall()
        events = [dict(row) for row in rows]
        for event in events:
            try:
                event["payload"] = json.loads(event.pop("payload_json"))
            except json.JSONDecodeError:
                event["payload"] = {}
        return events
