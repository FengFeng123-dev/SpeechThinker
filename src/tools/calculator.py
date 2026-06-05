import math
from langchain_core.tools import tool


@tool
def calculate(expression: str) -> str:
    """计算数学表达式。支持加减乘除、括号、幂运算和常用数学函数（如 sqrt, sin, cos, log 等）。

    Args:
        expression: 数学表达式字符串，例如 "2 + 3 * 4"、"sqrt(16)"、"2 ** 10"
    """
    allowed_names = {
        k: v for k, v in math.__dict__.items()
        if not k.startswith("_")
    }
    allowed_names["abs"] = abs
    allowed_names["round"] = round
    allowed_names["pow"] = pow
    allowed_names["max"] = max
    allowed_names["min"] = min
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}。请检查表达式格式是否正确。"
