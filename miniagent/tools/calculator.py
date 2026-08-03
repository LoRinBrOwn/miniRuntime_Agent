from __future__ import annotations

import ast
import operator
from typing import Any

from miniagent.tools.base import ToolContext, ToolResult, ToolValidationError, require_str


class SafeCalculator:
    name = "calculator"
    description = "Safely evaluate arithmetic expressions with +, -, *, /, parentheses, and powers."
    args_schema = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Arithmetic expression, for example: (237 * 48) / 4",
            }
        },
        "required": ["expression"],
        "additionalProperties": False,
    }

    _bin_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }
    _unary_ops = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    def validate(self, raw_arguments: dict[str, Any]) -> dict[str, Any]:
        return {"expression": require_str(raw_arguments, "expression")}

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        expression = arguments["expression"]
        try:
            parsed = ast.parse(expression, mode="eval")
            value = self._eval(parsed.body)
        except Exception as exc:
            return ToolResult(False, error=f"invalid arithmetic expression: {exc}")
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return ToolResult(True, {"result": value, "normalized_expression": expression})

    def _eval(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in self._bin_ops:
            left = self._eval(node.left)
            right = self._eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ToolValidationError("power exponent is too large")
            return self._bin_ops[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._unary_ops:
            return self._unary_ops[type(node.op)](self._eval(node.operand))
        raise ToolValidationError(f"unsupported expression node: {type(node).__name__}")
