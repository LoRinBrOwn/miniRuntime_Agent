from __future__ import annotations

from typing import Any

from miniagent.session.repository import SQLiteRepository
from miniagent.tools.base import ToolContext, ToolResult, ToolValidationError, optional_str, require_str


class TodoTool:
    name = "todo"
    description = "Create, list, or complete todos. Todos are user scoped and record the source session."
    args_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["create", "list", "complete"], "description": "Todo operation"},
            "title": {"type": "string", "description": "Todo title, required for create"},
            "todo_id": {"type": "string", "description": "Todo id, required for complete"},
            "status": {"type": "string", "enum": ["pending", "completed"], "description": "Optional list filter"},
        },
        "required": ["action"],
        "additionalProperties": False,
    }

    def __init__(self, repo: SQLiteRepository) -> None:
        self.repo = repo

    def validate(self, raw_arguments: dict[str, Any]) -> dict[str, Any]:
        action = require_str(raw_arguments, "action")
        if action not in {"create", "list", "complete"}:
            raise ToolValidationError("action must be one of: create, list, complete")
        args: dict[str, Any] = {"action": action}
        if action == "create":
            args["title"] = require_str(raw_arguments, "title")
        if action == "complete":
            args["todo_id"] = require_str(raw_arguments, "todo_id")
        status = optional_str(raw_arguments, "status")
        if status:
            if status not in {"pending", "completed"}:
                raise ToolValidationError("status must be pending or completed")
            args["status"] = status
        return args

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        action = arguments["action"]
        if action == "create":
            todo = self.repo.create_todo(context.user_id, context.session_id, arguments["title"])
            return ToolResult(True, {"created": todo})
        if action == "list":
            todos = self.repo.list_todos(context.user_id, arguments.get("status"))
            return ToolResult(True, {"todos": todos})
        if action == "complete":
            todo = self.repo.complete_todo(context.user_id, arguments["todo_id"])
            if not todo:
                return ToolResult(False, error="todo not found")
            return ToolResult(True, {"completed": todo})
        return ToolResult(False, error="unsupported action")
