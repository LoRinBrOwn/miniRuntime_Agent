from __future__ import annotations

import json
import unittest

from miniagent.runtime.parser import ActionParser


class ParserTestCase(unittest.TestCase):
    def test_native_tool_call(self) -> None:
        action = ActionParser().parse(
            {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "weather",
                            "arguments": json.dumps({"city": "东京"}, ensure_ascii=False),
                        },
                    }
                ]
            }
        )
        self.assertFalse(action.is_final)
        self.assertEqual(action.tool_calls[0].name, "weather")
        self.assertEqual(action.tool_calls[0].arguments["city"], "东京")

    def test_json_final_answer(self) -> None:
        action = ActionParser().parse({"content": '{"final_answer":"完成","decision_summary":"enough info"}'})
        self.assertTrue(action.is_final)
        self.assertEqual(action.final_answer, "完成")
        self.assertEqual(action.decision_summary, "enough info")

    def test_json_tool_call_fallback(self) -> None:
        action = ActionParser().parse({"content": '{"tool_calls":[{"name":"calculator","arguments":{"expression":"1+1"}}]}'})
        self.assertEqual(action.tool_calls[0].name, "calculator")


if __name__ == "__main__":
    unittest.main()
