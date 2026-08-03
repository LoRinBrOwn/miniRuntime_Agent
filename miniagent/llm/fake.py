from __future__ import annotations

from typing import Any


class SequenceFakeLLM:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        self.calls.append({"messages": messages, "tools": tools})
        if not self.responses:
            return {"role": "assistant", "content": "FakeLLM has no more responses."}
        return self.responses.pop(0)
