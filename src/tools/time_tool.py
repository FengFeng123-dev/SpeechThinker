from datetime import datetime
from langchain_core.tools import tool


@tool
def get_current_time() -> str:
    """获取当前日期和时间，包括年月日时分秒和星期几。当用户问"现在几点"、"今天几号"、"星期几"时调用此工具。"""
    now = datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekday = weekdays[now.weekday()]
    return f"{now.strftime('%Y年%m月%d日 %H:%M:%S')} {weekday}"
