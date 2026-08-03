from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Action:
    final_answer: str | None
    tool_calls: list[ToolCall]
    decision_summary: str | None = None

    @property
    def is_final(self) -> bool:
        return self.final_answer is not None and not self.tool_calls


class ActionParser:
    def parse(self, message: dict[str, Any]) -> Action:
        native_calls = message.get("tool_calls") or []
        if native_calls:
            calls = [self._parse_native_call(call) for call in native_calls]
            return Action(None, calls, self._summary_from_content(message.get("content")) or "requested tool call")

        content = message.get("content") or ""
        if isinstance(content, list):
            content = "\n".join(str(part) for part in content)
        text = str(content).strip()
        parsed_json = self._try_json(text)
        if parsed_json:
            calls = self._parse_json_calls(parsed_json)
            if calls:
                return Action(None, calls, parsed_json.get("decision_summary"))
            if "final_answer" in parsed_json or "answer" in parsed_json:
                return Action(str(parsed_json.get("final_answer") or parsed_json.get("answer")), [], parsed_json.get("decision_summary"))
        return Action(text, [], None)

    def _parse_native_call(self, call: dict[str, Any]) -> ToolCall:
        function = call.get("function") or {}
        raw_args = function.get("arguments") or "{}"
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid tool arguments JSON: {exc}") from exc
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            raise ValueError("tool arguments must be JSON object")
        return ToolCall(id=call.get("id") or f"call_{function.get('name', 'unknown')}", name=function.get("name", ""), arguments=args)

    def _parse_json_calls(self, payload: dict[str, Any]) -> list[ToolCall]:
        raw_calls = payload.get("tool_calls") or payload.get("calls") or []
        calls = []
        for index, item in enumerate(raw_calls):
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("tool_name")
            arguments = item.get("arguments") or {}
            if name and isinstance(arguments, dict):
                calls.append(ToolCall(id=item.get("id") or f"call_json_{index}", name=name, arguments=arguments))
        return calls

    def _try_json(self, text: str) -> dict[str, Any] | None:
        if not text.startswith("{"):
            return None
        try:
            payload = json.loads(text)
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            return None

    def _summary_from_content(self, content: Any) -> str | None:
        if not content:
            return None
        text = str(content).strip()
        payload = self._try_json(text)
        if payload:
            value = payload.get("decision_summary")
            return str(value) if value else None
        return text[:160]
