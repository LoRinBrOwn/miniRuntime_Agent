from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from miniagent.session.repository import SQLiteRepository
from miniagent.tools.base import ToolContext, ToolResult, ToolValidationError, optional_str, require_str


RELATIVE_DAYS = {"今天": 0, "明天": 1, "后天": 2}
CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
NUMBER_TOKEN = r"\d{1,2}|[零〇一二两三四五六七八九十]{1,3}"
TIME_PATTERN = re.compile(
    r"\s*(?P<period>凌晨|早上|上午|中午|下午|傍晚|晚上)?\s*"
    rf"(?P<hour>{NUMBER_TOKEN})"
    rf"(?:\s*(?:点|:|：)\s*(?:(?P<half>半)|(?P<minute>{NUMBER_TOKEN}))?)?"
)


def normalize_todo_due(
    title: str,
    due_at: str | None = None,
    timezone: str = "Asia/Shanghai",
    now_provider: Callable[[], datetime] | None = None,
) -> tuple[str, str | None]:
    clean_title = title.strip()
    clean_due_at = due_at.strip() if isinstance(due_at, str) and due_at.strip() else None
    match = re.search("|".join(RELATIVE_DAYS), clean_title)
    if not match:
        return clean_title, clean_due_at

    now = _current_time(timezone, now_provider)
    target_date = now.date() + timedelta(days=RELATIVE_DAYS[match.group(0)])
    time_match = TIME_PATTERN.match(clean_title[match.end() :])
    parsed_time = _parse_time(time_match)
    inferred_due_at = _format_due_at(target_date, parsed_time, now.tzinfo)
    resolved_due_at = clean_due_at or inferred_due_at
    replacement = _display_due_at(resolved_due_at)
    replace_end = match.end() + (time_match.end() if parsed_time and time_match else 0)
    normalized_title = _replace_with_spacing(clean_title, match.start(), replace_end, replacement)
    return normalized_title, resolved_due_at


def _current_time(timezone: str, now_provider: Callable[[], datetime] | None) -> datetime:
    if now_provider:
        now = now_provider()
    else:
        try:
            now = datetime.now(ZoneInfo(timezone))
        except ZoneInfoNotFoundError:
            now = datetime.now().astimezone()
    if now.tzinfo is None:
        try:
            return now.replace(tzinfo=ZoneInfo(timezone))
        except ZoneInfoNotFoundError:
            return now.astimezone()
    return now


def _parse_time(match: re.Match[str] | None) -> time | None:
    if not match:
        return None
    period = match.group("period") or ""
    hour = _parse_number(match.group("hour"))
    minute = 30 if match.group("half") else _parse_number(match.group("minute") or "0")
    if hour is None or minute is None:
        return None
    if minute > 59:
        return None
    if period in {"下午", "傍晚", "晚上"} and 1 <= hour < 12:
        hour += 12
    elif period == "中午" and 1 <= hour < 12:
        hour += 12
    elif period in {"凌晨", "早上", "上午"} and hour == 12:
        hour = 0
    if hour > 23:
        return None
    return time(hour=hour, minute=minute)


def _parse_number(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    if value == "十":
        return 10
    if value.startswith("十"):
        tail = value[1:]
        return 10 + CHINESE_DIGITS.get(tail, 0)
    if "十" in value:
        head, tail = value.split("十", 1)
        if head not in CHINESE_DIGITS:
            return None
        return CHINESE_DIGITS[head] * 10 + (CHINESE_DIGITS.get(tail, 0) if tail else 0)
    if value in CHINESE_DIGITS:
        return CHINESE_DIGITS[value]
    return None


def _format_due_at(target_date: date, parsed_time: time | None, tzinfo: Any) -> str:
    if parsed_time is None:
        return target_date.isoformat()
    return datetime.combine(target_date, parsed_time, tzinfo=tzinfo).isoformat()


def _display_due_at(due_at: str) -> str:
    if "T" in due_at:
        return due_at[:16].replace("T", " ")
    return due_at[:10]


def _replace_with_spacing(text: str, start: int, end: int, replacement: str) -> str:
    prefix = text[:start].rstrip()
    suffix = text[end:].lstrip()
    if not suffix:
        left_separator = "" if not prefix or prefix[-1] in " ，,。.;；:：" else " "
        return (prefix + left_separator + replacement).strip()
    left_separator = "" if not prefix or prefix[-1] in " ，,。.;；:：" else " "
    right_separator = "" if suffix[0] in "，,。.;；:：" else " "
    return (prefix + left_separator + replacement + right_separator + suffix).strip()


class TodoTool:
    name = "todo"
    description = "Create, list, or complete todos in the current session. Todos are isolated by user and session."
    args_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["create", "list", "complete"], "description": "Todo operation"},
            "title": {"type": "string", "description": "Todo title, required for create"},
            "due_at": {"type": "string", "description": "Optional absolute due date or datetime for create"},
            "todo_id": {"type": "string", "description": "Todo id, required for complete"},
            "status": {"type": "string", "enum": ["pending", "completed"], "description": "Optional list filter"},
        },
        "required": ["action"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        repo: SQLiteRepository,
        timezone: str = "Asia/Shanghai",
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self.repo = repo
        self.timezone = timezone
        self.now_provider = now_provider

    def validate(self, raw_arguments: dict[str, Any]) -> dict[str, Any]:
        action = require_str(raw_arguments, "action")
        if action not in {"create", "list", "complete"}:
            raise ToolValidationError("action must be one of: create, list, complete")
        args: dict[str, Any] = {"action": action}
        if action == "create":
            args["title"] = require_str(raw_arguments, "title")
            args["due_at"] = optional_str(raw_arguments, "due_at")
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
            title, due_at = normalize_todo_due(
                arguments["title"],
                arguments.get("due_at"),
                self.timezone,
                self.now_provider,
            )
            todo = self.repo.create_todo(context.user_id, context.session_id, title, due_at)
            return ToolResult(True, {"created": todo})
        if action == "list":
            todos = self.repo.list_todos(context.user_id, arguments.get("status"), context.session_id)
            return ToolResult(True, {"todos": todos})
        if action == "complete":
            todo = self.repo.complete_todo(context.user_id, arguments["todo_id"], context.session_id)
            if not todo:
                return ToolResult(False, error="todo not found")
            return ToolResult(True, {"completed": todo})
        return ToolResult(False, error="unsupported action")
