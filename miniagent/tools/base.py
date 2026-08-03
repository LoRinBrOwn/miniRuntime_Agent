from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol


class ToolValidationError(ValueError):
    pass


@dataclass
class ToolContext:
    user_id: str
    session_id: str


@dataclass
class ToolResult:
    success: bool
    data: dict[str, Any] | list[Any] | str | int | float | None = None
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps({"success": self.success, "data": self.data, "error": self.error}, ensure_ascii=False)

    def summary(self, max_len: int = 500) -> str:
        text = self.to_json()
        return text if len(text) <= max_len else text[: max_len - 3] + "..."


class BaseTool(Protocol):
    name: str
    description: str
    args_schema: dict[str, Any]

    def validate(self, raw_arguments: dict[str, Any]) -> dict[str, Any]:
        ...

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        ...


def require_str(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolValidationError(f"{key} must be a non-empty string")
    return value.strip()


def optional_str(args: dict[str, Any], key: str) -> str | None:
    value = args.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ToolValidationError(f"{key} must be a string")
    return value.strip() or None
