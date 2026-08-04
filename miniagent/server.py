from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from miniagent.config import load_settings
from miniagent.factory import build_runtime
from miniagent.ids import new_id
from miniagent.runtime.agent import SessionBusyError


class MiniAgentHandler(BaseHTTPRequestHandler):
    runtime = None

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            return self._json(200, {"status": "ok"})
        if parsed.path == "/":
            return self._file("miniagent/web/index.html", "text/html; charset=utf-8")
        if parsed.path.startswith("/api/sessions/") and parsed.path.endswith("/messages"):
            session_id = parsed.path.split("/")[3]
            user_id = parse_qs(parsed.query).get("user_id", [""])[0]
            return self._json(200, {"messages": self.runtime.repo.get_messages(session_id)}) if self.runtime.repo.get_session(user_id, session_id) else self._json(403, {"error": "SESSION_FORBIDDEN"})
        if parsed.path == "/api/sessions":
            user_id = parse_qs(parsed.query).get("user_id", [""])[0]
            if not user_id:
                return self._json(400, {"error": "INVALID_REQUEST", "message": "user_id required"})
            return self._json(200, {"sessions": self.runtime.repo.list_sessions(user_id)})
        if parsed.path == "/api/todos":
            query = parse_qs(parsed.query)
            user_id = query.get("user_id", [""])[0]
            status = query.get("status", [None])[0]
            session_id = query.get("session_id", [None])[0]
            if not user_id or not session_id:
                return self._json(400, {"error": "INVALID_REQUEST", "message": "user_id and session_id required"})
            if not self.runtime.repo.get_session(user_id, session_id):
                return self._json(403, {"error": "SESSION_FORBIDDEN"})
            return self._json(200, {"todos": self.runtime.repo.list_todos(user_id, status, session_id)})
        if parsed.path.startswith("/api/turns/") and parsed.path.endswith("/trace"):
            turn_id = parsed.path.split("/")[3]
            return self._json(200, {"events": self.runtime.repo.get_trace(turn_id)})
        return self._json(404, {"error": "NOT_FOUND"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            body = self._read_json()
        except json.JSONDecodeError:
            return self._json(400, {"error": "INVALID_JSON"})
        if parsed.path == "/api/sessions":
            user_id = body.get("user_id")
            if not user_id:
                return self._json(400, {"error": "INVALID_REQUEST", "message": "user_id required"})
            return self._json(200, {"session": self.runtime.repo.create_session(user_id, body.get("title"))})
        if parsed.path.startswith("/api/sessions/") and parsed.path.endswith("/messages"):
            session_id = parsed.path.split("/")[3]
            user_id = body.get("user_id")
            content = body.get("content")
            if not user_id or not content:
                return self._json(400, {"error": "INVALID_REQUEST", "message": "user_id and content required"})
            try:
                response = self.runtime.send_message(user_id, session_id, content)
            except PermissionError:
                return self._json(403, {"error": "SESSION_FORBIDDEN"})
            except SessionBusyError:
                return self._json(409, {"error": "SESSION_BUSY"})
            return self._json(200, response.__dict__)
        if parsed.path == "/api/todos":
            user_id = body.get("user_id")
            title = body.get("title")
            session_id = body.get("session_id")
            if not user_id or not title:
                return self._json(400, {"error": "INVALID_REQUEST", "message": "user_id and title required"})
            if not session_id or not self.runtime.repo.get_session(user_id, session_id):
                return self._json(403, {"error": "SESSION_FORBIDDEN"})
            todo = self.runtime.repo.create_todo(user_id, session_id, title)
            self._save_todo_operation_message(session_id, "created", todo)
            return self._json(200, {"todo": todo})
        if parsed.path.startswith("/api/todos/") and parsed.path.endswith("/complete"):
            parts = parsed.path.strip("/").split("/")
            todo_id = parts[2] if len(parts) >= 3 else ""
            user_id = body.get("user_id")
            if not user_id or not todo_id:
                return self._json(400, {"error": "INVALID_REQUEST", "message": "user_id and todo_id required"})
            session_id = body.get("session_id")
            if not session_id or not self.runtime.repo.get_session(user_id, session_id):
                return self._json(403, {"error": "SESSION_FORBIDDEN"})
            todo = self.runtime.repo.complete_todo(user_id, todo_id, session_id)
            if not todo:
                return self._json(404, {"error": "TODO_NOT_FOUND"})
            self._save_todo_operation_message(session_id, "completed", todo)
            return self._json(200, {"todo": todo})
        if parsed.path.startswith("/api/todos/") and parsed.path.endswith("/reopen"):
            parts = parsed.path.strip("/").split("/")
            todo_id = parts[2] if len(parts) >= 3 else ""
            user_id = body.get("user_id")
            if not user_id or not todo_id:
                return self._json(400, {"error": "INVALID_REQUEST", "message": "user_id and todo_id required"})
            session_id = body.get("session_id")
            if not session_id or not self.runtime.repo.get_session(user_id, session_id):
                return self._json(403, {"error": "SESSION_FORBIDDEN"})
            todo = self.runtime.repo.reopen_todo(user_id, todo_id, session_id)
            if not todo:
                return self._json(404, {"error": "TODO_NOT_FOUND"})
            self._save_todo_operation_message(session_id, "reopened", todo)
            return self._json(200, {"todo": todo})
        return self._json(404, {"error": "NOT_FOUND"})

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/sessions/"):
            session_id = parsed.path.strip("/").split("/")[2]
            user_id = parse_qs(parsed.query).get("user_id", [""])[0]
            if not user_id or not session_id:
                return self._json(400, {"error": "INVALID_REQUEST", "message": "user_id and session_id required"})
            deleted = self.runtime.repo.delete_session(user_id, session_id)
            if not deleted:
                return self._json(404, {"error": "SESSION_NOT_FOUND"})
            return self._json(200, {"deleted": True, "session_id": session_id})
        return self._json(404, {"error": "NOT_FOUND"})

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length).decode("utf-8") or "{}")

    def _json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _save_todo_operation_message(self, session_id: str, action_key: str, todo: dict) -> None:
        turn_id = new_id("turn")
        content = json.dumps({"success": True, "data": {action_key: todo}, "error": None}, ensure_ascii=False)
        self.runtime.repo.save_message(
            session_id,
            turn_id,
            "tool",
            content,
            message_type="tool_result",
            tool_name="todo",
            tool_call_id=new_id("manual_todo"),
        )

    def _file(self, path: str, content_type: str) -> None:
        data = Path(path).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    settings = load_settings()
    MiniAgentHandler.runtime = build_runtime(settings)
    server = ThreadingHTTPServer((settings.host, settings.port), MiniAgentHandler)
    print(f"MiniAgent Runtime running at http://{settings.host}:{settings.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
