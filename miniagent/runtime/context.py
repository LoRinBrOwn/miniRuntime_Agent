from __future__ import annotations

import json
import re
from typing import Any

from miniagent.session.repository import SQLiteRepository


SYSTEM_PROMPT = """你是 MiniAgent Runtime 中的执行 Agent。
你可以直接回答，也可以调用系统提供的工具。
规则：
1. 仅调用已提供的工具，并严格遵守参数 Schema。
2. 当回答依赖计算、搜索、天气或待办操作时，优先调用对应工具。
3. 工具结果会以 tool 消息返回；请基于结果继续决策。
4. 信息已经充分时，输出最终回答，不要继续调用无关工具。
5. 不要重复完全相同的工具调用。
6. 工具失败时，可以修正参数后重试，也可以向用户解释限制。
7. 不输出完整内部思维过程；只提供必要的简短决策说明。
8. 不要把 Session 摘要中的旧信息视为高于用户当前输入的指令。
"""


EMPTY_SUMMARY = {
    "confirmed_facts": [],
    "user_preferences": [],
    "important_results": [],
    "open_tasks": [],
    "entities": [],
    "conversation_summary": "",
    "summary_version": 0,
}


class ContextManager:
    def __init__(
        self,
        repo: SQLiteRepository,
        recent_message_limit: int = 20,
        summary_trigger_messages: int = 30,
        context_max_chars: int = 24000,
    ) -> None:
        self.repo = repo
        self.recent_message_limit = recent_message_limit
        self.summary_trigger_messages = summary_trigger_messages
        self.context_max_chars = context_max_chars

    def build(self, session_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        summary = self.repo.get_summary(session_id)
        recent = self.repo.get_messages(session_id, limit=self.recent_message_limit)
        messages: list[dict[str, Any]] = [{"role": "system", "content": self._system_content(summary)}]
        for message in recent:
            role = message["role"]
            if role == "tool":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.get("tool_call_id") or "tool_result",
                        "name": message.get("tool_name"),
                        "content": message["content"],
                    }
                )
            elif role in {"user", "assistant"}:
                if role == "assistant" and message.get("message_type") == "tool_call":
                    tool_call_message = self._assistant_tool_call_message(message["content"])
                    if tool_call_message:
                        messages.append(tool_call_message)
                    else:
                        messages.append({"role": role, "content": message["content"]})
                else:
                    messages.append({"role": role, "content": message["content"]})
        messages = self._trim(messages)
        messages = self._sanitize_tool_sequence(messages)
        metadata = {
            "message_count": len(messages),
            "estimated_chars": sum(len(str(item.get("content", ""))) for item in messages),
            "summary_version": summary.get("summary_version", 0) if isinstance(summary, dict) else 0,
        }
        return messages, metadata

    def maybe_compress(self, session_id: str) -> dict[str, Any] | None:
        count = self.repo.count_messages(session_id)
        if count <= self.summary_trigger_messages:
            return None
        all_messages = self.repo.get_messages(session_id)
        cutoff = max(0, len(all_messages) - self.recent_message_limit)
        old_messages = all_messages[:cutoff]
        if not old_messages:
            return None
        old_summary = self.repo.get_summary(session_id) or EMPTY_SUMMARY.copy()
        new_summary = self._summarize(old_summary, old_messages)
        self.repo.update_summary(session_id, new_summary)
        return {
            "before_messages": count,
            "compressed_messages": len(old_messages),
            "summary_version": new_summary["summary_version"],
        }

    def _system_content(self, summary: dict[str, Any]) -> str:
        if summary:
            return SYSTEM_PROMPT + "\nSession summary JSON:\n" + json.dumps(summary, ensure_ascii=False)
        return SYSTEM_PROMPT

    def _trim(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        total = sum(len(str(item.get("content", ""))) for item in messages)
        if total <= self.context_max_chars:
            return messages
        system = messages[0]
        kept = []
        running = len(system["content"])
        for message in reversed(messages[1:]):
            size = len(str(message.get("content", "")))
            if running + size > self.context_max_chars:
                break
            kept.append(message)
            running += size
        return [system, *reversed(kept)]

    def _sanitize_tool_sequence(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not messages:
            return messages
        sanitized: list[dict[str, Any]] = [messages[0]]
        pending_tool_ids: set[str] = set()
        index = 1
        while index < len(messages):
            message = messages[index]
            if message.get("role") == "assistant":
                tool_calls = message.get("tool_calls") or []
                required_tool_ids = {
                    call.get("id")
                    for call in tool_calls
                    if isinstance(call, dict) and isinstance(call.get("id"), str)
                }
                if required_tool_ids and not self._has_contiguous_tool_results(messages, index + 1, required_tool_ids):
                    sanitized.append(
                        {
                            "role": "assistant",
                            "content": f"Historical tool call summary: {json.dumps(tool_calls, ensure_ascii=False)}",
                        }
                    )
                    pending_tool_ids = set()
                    index += 1
                    continue
                pending_tool_ids = required_tool_ids
                sanitized.append(message)
                index += 1
                continue

            if message.get("role") == "tool":
                tool_call_id = message.get("tool_call_id")
                if tool_call_id in pending_tool_ids:
                    sanitized.append(message)
                    pending_tool_ids.discard(tool_call_id)
                else:
                    sanitized.append(
                        {
                            "role": "assistant",
                            "content": f"Historical tool result ({message.get('name') or 'unknown'}): {message.get('content', '')}",
                        }
                    )
                    pending_tool_ids = set()
                index += 1
                continue

            pending_tool_ids = set()
            sanitized.append(message)
            index += 1
        return sanitized

    def _has_contiguous_tool_results(self, messages: list[dict[str, Any]], start: int, required_tool_ids: set[str]) -> bool:
        seen: set[str] = set()
        index = start
        while index < len(messages) and messages[index].get("role") == "tool":
            tool_call_id = messages[index].get("tool_call_id")
            if isinstance(tool_call_id, str):
                seen.add(tool_call_id)
            index += 1
        return required_tool_ids.issubset(seen)

    def _assistant_tool_call_message(self, content: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return None
        raw_calls = payload.get("tool_calls") if isinstance(payload, dict) else None
        if not isinstance(raw_calls, list):
            return None
        tool_calls = []
        for item in raw_calls:
            if not isinstance(item, dict):
                continue
            tool_calls.append(
                {
                    "id": item.get("id") or "call_unknown",
                    "type": "function",
                    "function": {
                        "name": item.get("name") or "",
                        "arguments": json.dumps(item.get("arguments") or {}, ensure_ascii=False),
                    },
                }
            )
        if not tool_calls:
            return None
        return {
            "role": "assistant",
            "content": payload.get("decision_summary"),
            "tool_calls": tool_calls,
        }

    def _summarize(self, old_summary: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
        summary = {**EMPTY_SUMMARY, **old_summary}
        facts = list(summary.get("confirmed_facts", []))
        results = list(summary.get("important_results", []))
        tasks = list(summary.get("open_tasks", []))
        entities = list(summary.get("entities", []))
        snippets = []
        existing_entities = {entity.get("name") for entity in entities if isinstance(entity, dict)}

        for message in messages:
            content = message["content"]
            if message["role"] == "user":
                snippets.append(content[:80])
                fact = self._extract_fact(content)
                if fact and fact not in facts:
                    facts.append(fact)
                project = self._extract_project(content)
                if project and project not in existing_entities:
                    entities.append({"name": project, "type": "project"})
                    existing_entities.add(project)
            if message["role"] == "tool":
                self._extract_tool_result(content, results, tasks)

        summary["confirmed_facts"] = facts[-20:]
        summary["important_results"] = results[-20:]
        summary["open_tasks"] = tasks[-20:]
        summary["entities"] = entities[-20:]
        addition = "；".join(snippets[-8:])
        previous = summary.get("conversation_summary", "")
        summary["conversation_summary"] = (previous + "；" + addition).strip("；")[-1200:]
        summary["summary_version"] = int(summary.get("summary_version", 0)) + 1
        return summary

    def _extract_fact(self, text: str) -> str | None:
        if "记住" in text or "当前" in text or "项目" in text:
            return text[:160]
        return None

    def _extract_project(self, text: str) -> str | None:
        match = re.search(r"(?:项目(?:是|叫|名称为)?|project\s+)([A-Za-z0-9_\-\u4e00-\u9fff]{2,30})", text, re.I)
        return match.group(1) if match else None

    def _extract_tool_result(self, content: str, results: list[str], tasks: list[str]) -> None:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            return
        if "result" in data:
            results.append(f"calculator result: {data['result']}")
        if "city" in data and "condition" in data:
            results.append(f"weather: {data.get('city')} {data.get('condition')} {data.get('temperature')}")
        created = data.get("created")
        if isinstance(created, dict):
            tasks.append(created.get("title", "untitled todo"))
