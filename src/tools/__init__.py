from .time_tool import get_current_time
from .calculator import calculate
from .weather import get_weather
from .knowledge import search_knowledge
from .notes import save_note

ALL_TOOLS = [
    get_current_time,
    calculate,
    get_weather,
    search_knowledge,
    save_note,
]
