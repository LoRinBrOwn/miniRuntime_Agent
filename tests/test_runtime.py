from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from miniagent.llm.fake import SequenceFakeLLM
from miniagent.runtime.agent import AgentRuntime
from miniagent.runtime.context import ContextManager
from miniagent.session.repository import SQLiteRepository
from miniagent.tools.calculator import SafeCalculator
from miniagent.tools.registry import ToolRegistry
from miniagent.tools.search import MockSearchTool
from miniagent.tools.todo import TodoTool
from miniagent.tools.weather import MockWeatherTool


def tool_call(name: str, args: dict, call_id: str = "call_1") -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }
        ],
    }


class RuntimeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "agent.db"
        self.repo = SQLiteRepository(f"sqlite:///{self.db_path}")
        self.registry = ToolRegistry()
        self.registry.register(SafeCalculator())
        self.registry.register(MockSearchTool())
        self.registry.register(MockWeatherTool())
        self.registry.register(TodoTool(self.repo))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def runtime(self, responses: list[dict], max_steps: int = 8) -> AgentRuntime:
        context = ContextManager(self.repo, recent_message_limit=20, summary_trigger_messages=30)
        return AgentRuntime(self.repo, SequenceFakeLLM(responses), self.registry, context, max_steps=max_steps, tool_timeout_seconds=2)

    def test_direct_answer_has_no_tool_call(self) -> None:
        session = self.repo.create_session("user_a", "chat")
        runtime = self.runtime([{"role": "assistant", "content": "你好，我是 MiniAgent。"}])
        response = runtime.send_message("user_a", session["id"], "你好")
        self.assertEqual(response.status, "completed")
        self.assertEqual(response.stop_reason, "final_answer")
        events = self.repo.get_trace(response.turn_id)
        self.assertFalse(any(event["event_type"] == "TOOL_CALL_REQUESTED" for event in events))

    def test_calculator_tool_flow(self) -> None:
        session = self.repo.create_session("user_a", "calc")
        runtime = self.runtime(
            [
                tool_call("calculator", {"expression": "123 * 456"}),
                {"role": "assistant", "content": "123 * 456 = 56088。"},
            ]
        )
        response = runtime.send_message("user_a", session["id"], "计算 123*456")
        self.assertIn("56088", response.answer)
        tool_messages = [m for m in self.repo.get_messages(session["id"]) if m["role"] == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn('"result": 56088', tool_messages[0]["content"])

    def test_calculator_rejects_dangerous_expression(self) -> None:
        tool = SafeCalculator()
        result = tool.execute({"expression": "__import__('os').system('whoami')"}, context=None)  # type: ignore[arg-type]
        self.assertFalse(result.success)
        self.assertIn("unsupported", result.error or "")

    def test_search_tool_returns_mock_sources(self) -> None:
        session = self.repo.create_session("user_a", "search")
        runtime = self.runtime(
            [
                tool_call("search", {"query": "Agent Runtime", "top_k": 2}),
                {"role": "assistant", "content": "找到了 Agent Runtime 设计资料。"},
            ]
        )
        response = runtime.send_message("user_a", session["id"], "搜索 Agent Runtime")
        self.assertEqual(response.status, "completed")
        tool_message = [m for m in self.repo.get_messages(session["id"]) if m["tool_name"] == "search"][0]
        self.assertIn("mock://agent-runtime/design", tool_message["content"])

    def test_weather_then_todo_multitool(self) -> None:
        session = self.repo.create_session("user_a", "weather todo")
        runtime = self.runtime(
            [
                tool_call("weather", {"city": "东京"}, "call_weather"),
                tool_call("todo", {"action": "create", "title": "带伞"}, "call_todo"),
                {"role": "assistant", "content": "东京有小雨，已创建带伞待办。"},
            ]
        )
        response = runtime.send_message("user_a", session["id"], "查东京天气，如果下雨就记一个带伞待办")
        self.assertEqual(response.status, "completed")
        events = [event["payload"].get("tool_name") for event in self.repo.get_trace(response.turn_id) if event["event_type"] == "TOOL_CALL_REQUESTED"]
        self.assertEqual(events, ["weather", "todo"])
        todos = self.repo.list_todos("user_a")
        self.assertEqual(todos[0]["title"], "带伞")
        self.assertEqual(todos[0]["source_session_id"], session["id"])

    def test_tool_followup_keeps_recent_context(self) -> None:
        session = self.repo.create_session("user_a", "followup")
        runtime = self.runtime(
            [
                tool_call("weather", {"city": "东京"}, "call_tokyo"),
                {"role": "assistant", "content": "东京小雨。"},
                tool_call("weather", {"city": "大阪"}, "call_osaka"),
                {"role": "assistant", "content": "大阪多云。"},
            ]
        )
        runtime.send_message("user_a", session["id"], "查东京天气")
        response = runtime.send_message("user_a", session["id"], "那大阪呢？")
        self.assertIn("大阪", response.answer)
        messages = runtime.llm.calls[-1]["messages"]  # type: ignore[attr-defined]
        self.assertTrue(any("东京" in str(message.get("content", "")) for message in messages))

    def test_session_context_is_isolated(self) -> None:
        s1 = self.repo.create_session("user_a", "window 1")
        s2 = self.repo.create_session("user_a", "window 2")
        runtime = self.runtime(
            [
                {"role": "assistant", "content": "已记住 Alpha。"},
                {"role": "assistant", "content": "当前窗口没有项目上下文。"},
            ]
        )
        runtime.send_message("user_a", s1["id"], "记住当前项目是 Alpha")
        runtime.send_message("user_a", s2["id"], "当前项目是什么？")
        second_call_messages = runtime.llm.calls[-1]["messages"]  # type: ignore[attr-defined]
        self.assertFalse(any("Alpha" in str(message.get("content", "")) for message in second_call_messages))

    def test_context_compression_keeps_fact_and_recent_window(self) -> None:
        session = self.repo.create_session("user_a", "compress")
        for index in range(35):
            text = "记住当前项目是 Alpha" if index == 0 else f"普通聊天 {index}"
            self.repo.save_message(session["id"], f"turn_{index}", "user", text)
            self.repo.save_message(session["id"], f"turn_{index}", "assistant", f"回复 {index}")
        context = ContextManager(self.repo, recent_message_limit=10, summary_trigger_messages=30)
        compression = context.maybe_compress(session["id"])
        self.assertIsNotNone(compression)
        summary = self.repo.get_summary(session["id"])
        self.assertIn("Alpha", json.dumps(summary, ensure_ascii=False))
        messages, _ = context.build(session["id"])
        self.assertTrue(any("普通聊天 34" in str(message.get("content", "")) for message in messages))

    def test_unknown_tool_is_observed_not_crashed(self) -> None:
        session = self.repo.create_session("user_a", "unknown")
        runtime = self.runtime(
            [
                tool_call("not_registered", {"x": 1}),
                {"role": "assistant", "content": "这个工具不可用。"},
            ]
        )
        response = runtime.send_message("user_a", session["id"], "调用不存在工具")
        self.assertEqual(response.status, "completed")
        tool_message = [m for m in self.repo.get_messages(session["id"]) if m["role"] == "tool"][0]
        self.assertIn("unknown tool", tool_message["content"])

    def test_max_steps_stops_controlled_loop(self) -> None:
        session = self.repo.create_session("user_a", "loop")
        runtime = self.runtime([tool_call("weather", {"city": "东京"}, f"call_{i}") for i in range(10)], max_steps=2)
        response = runtime.send_message("user_a", session["id"], "一直查天气")
        self.assertEqual(response.status, "failed")
        self.assertEqual(response.stop_reason, "max_steps_exceeded")


if __name__ == "__main__":
    unittest.main()
