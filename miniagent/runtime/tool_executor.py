from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any

from miniagent.tools.base import ToolContext, ToolResult, ToolValidationError
from miniagent.tools.registry import ToolRegistry


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, timeout_seconds: float = 10.0) -> None:
        self.registry = registry
        self.timeout_seconds = timeout_seconds
        self.pool = ThreadPoolExecutor(max_workers=8)

    def canonical_signature(self, name: str, arguments: dict[str, Any]) -> str:
        return name + ":" + json.dumps(arguments, sort_keys=True, ensure_ascii=False)

    def execute(self, name: str, arguments: dict[str, Any], context: ToolContext) -> tuple[ToolResult, int]:
        started = time.perf_counter()
        tool = self.registry.get(name)
        if not tool:
            return ToolResult(False, error=f"unknown tool: {name}"), self._elapsed(started)
        try:
            validated = tool.validate(arguments)
        except ToolValidationError as exc:
            return ToolResult(False, error=f"invalid arguments: {exc}"), self._elapsed(started)
        except Exception as exc:
            return ToolResult(False, error=f"argument validation failed: {exc}"), self._elapsed(started)

        future = self.pool.submit(tool.execute, validated, context)
        try:
            result = future.result(timeout=self.timeout_seconds)
            return result, self._elapsed(started)
        except FutureTimeout:
            return ToolResult(False, error=f"tool timeout after {self.timeout_seconds}s"), self._elapsed(started)
        except Exception as exc:
            return ToolResult(False, error=f"tool execution failed: {exc}"), self._elapsed(started)

    def _elapsed(self, started: float) -> int:
        return int((time.perf_counter() - started) * 1000)
