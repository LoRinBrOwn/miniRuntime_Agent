from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from miniagent.tools.base import ToolContext, ToolResult, ToolValidationError, require_str


class MockSearchTool:
    name = "search"
    description = "Search local mock documents and return concise results with source ids."
    args_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "top_k": {"type": "integer", "description": "Number of results to return", "default": 3},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, data_path: str | Path = "data/search_docs.json") -> None:
        self.data_path = Path(data_path)

    def validate(self, raw_arguments: dict[str, Any]) -> dict[str, Any]:
        query = require_str(raw_arguments, "query")
        top_k = raw_arguments.get("top_k", 3)
        if not isinstance(top_k, int) or top_k < 1 or top_k > 10:
            raise ToolValidationError("top_k must be an integer between 1 and 10")
        return {"query": query, "top_k": top_k}

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        docs = json.loads(self.data_path.read_text(encoding="utf-8"))
        terms = [term.lower() for term in arguments["query"].replace("/", " ").split() if term.strip()]
        scored = []
        for doc in docs:
            haystack = f"{doc['title']} {doc['snippet']} {' '.join(doc.get('tags', []))}".lower()
            score = sum(haystack.count(term) for term in terms) if terms else 0
            if score:
                scored.append((score, doc))
        if not scored:
            scored = [(0, doc) for doc in docs[: arguments["top_k"]]]
        results = [
            {"title": doc["title"], "snippet": doc["snippet"], "source": doc["source"]}
            for _, doc in sorted(scored, key=lambda item: item[0], reverse=True)[: arguments["top_k"]]
        ]
        return ToolResult(True, {"query": arguments["query"], "results": results})
