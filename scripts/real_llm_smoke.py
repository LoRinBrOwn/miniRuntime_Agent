from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from miniagent.config import Settings, load_dotenv
from miniagent.factory import build_runtime


def main() -> None:
    load_dotenv()
    settings = Settings(
        database_url=os.getenv("DATABASE_URL", "sqlite:///./agent_smoke.db"),
        llm_base_url=os.getenv("LLM_BASE_URL", ""),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", ""),
    )
    runtime = build_runtime(settings)
    session = runtime.repo.create_session("smoke_user", "real llm smoke")
    response = runtime.send_message("smoke_user", session["id"], "请计算 237 * 48，并告诉我结果。")
    print(response)
    print("Trace:")
    for event in runtime.repo.get_trace(response.turn_id):
        print(event["event_type"], event["payload"])


if __name__ == "__main__":
    main()
