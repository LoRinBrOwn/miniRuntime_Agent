from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    database_url: str = "sqlite:///./agent.db"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    max_agent_steps: int = 8
    tool_timeout_seconds: float = 10.0
    llm_timeout_seconds: float = 60.0
    recent_message_limit: int = 20
    summary_trigger_messages: int = 30
    context_max_chars: int = 24000
    host: str = "127.0.0.1"
    port: int = 8000


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        database_url=os.getenv("DATABASE_URL", "sqlite:///./agent.db"),
        llm_base_url=os.getenv("LLM_BASE_URL", ""),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", ""),
        max_agent_steps=int(os.getenv("MAX_AGENT_STEPS", "8")),
        tool_timeout_seconds=float(os.getenv("TOOL_TIMEOUT_SECONDS", "10")),
        llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
        recent_message_limit=int(os.getenv("RECENT_MESSAGE_LIMIT", "20")),
        summary_trigger_messages=int(os.getenv("SUMMARY_TRIGGER_MESSAGES", "30")),
        context_max_chars=int(os.getenv("CONTEXT_MAX_CHARS", "24000")),
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
    )


def load_dotenv(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.lstrip("\ufeff").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not os.environ.get(key):
            os.environ[key] = value
