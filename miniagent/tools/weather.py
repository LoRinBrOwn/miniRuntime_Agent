from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from miniagent.tools.base import ToolContext, ToolResult, require_str


class MockWeatherTool:
    name = "weather"
    description = "Query mock weather data for a city. Useful for weather plans and follow-up city comparisons."
    args_schema = {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name, such as 东京, 大阪, Tokyo, or Shanghai"}},
        "required": ["city"],
        "additionalProperties": False,
    }

    def __init__(self, data_path: str | Path = "data/weather.json") -> None:
        self.data_path = Path(data_path)

    def validate(self, raw_arguments: dict[str, Any]) -> dict[str, Any]:
        return {"city": require_str(raw_arguments, "city")}

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        city = arguments["city"]
        data = json.loads(self.data_path.read_text(encoding="utf-8"))
        normalized = data.get(city) or data.get(city.lower())
        if not normalized:
            aliases = data.get("_aliases", {})
            normalized = data.get(aliases.get(city, aliases.get(city.lower(), "")))
        if not normalized:
            return ToolResult(False, error=f"weather data not found for city: {city}")
        return ToolResult(True, {"city": city, **normalized})
