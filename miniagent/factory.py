from __future__ import annotations

from miniagent.config import Settings, load_settings
from miniagent.llm.openai_compatible import OpenAICompatibleClient
from miniagent.runtime.agent import AgentRuntime
from miniagent.runtime.context import ContextManager
from miniagent.session.repository import SQLiteRepository
from miniagent.tools.calculator import SafeCalculator
from miniagent.tools.registry import ToolRegistry
from miniagent.tools.search import MockSearchTool
from miniagent.tools.todo import TodoTool
from miniagent.tools.weather import MockWeatherTool


def build_registry(repo: SQLiteRepository) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(SafeCalculator())
    registry.register(MockSearchTool())
    registry.register(MockWeatherTool())
    registry.register(TodoTool(repo))
    return registry


def build_runtime(settings: Settings | None = None) -> AgentRuntime:
    settings = settings or load_settings()
    repo = SQLiteRepository(settings.database_url)
    llm = OpenAICompatibleClient(settings.llm_base_url, settings.llm_api_key, settings.llm_model, settings.llm_timeout_seconds)
    registry = build_registry(repo)
    context = ContextManager(
        repo,
        recent_message_limit=settings.recent_message_limit,
        summary_trigger_messages=settings.summary_trigger_messages,
        context_max_chars=settings.context_max_chars,
    )
    return AgentRuntime(
        repo=repo,
        llm=llm,
        registry=registry,
        context_manager=context,
        max_steps=settings.max_agent_steps,
        tool_timeout_seconds=settings.tool_timeout_seconds,
    )
