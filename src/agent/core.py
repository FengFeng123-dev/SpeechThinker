from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

from config.settings import settings
from src.tts import RealTimeTTS
from src.tools import ALL_TOOLS

# ─── LLM 实例 ───
llm = ChatOpenAI(
    api_key=settings.API_KEY,
    base_url=settings.API_BASE,
    model=settings.MODEL,
    streaming=True,
)

# ─── TTS 实例 ───
tts = RealTimeTTS(
    voice=settings.TTS_VOICE,
    rate=settings.TTS_RATE,
    buffer_size=settings.TTS_BUFFER_SIZE,
)

# ─── Agent 实例 ───
memory = MemorySaver()

agent = create_agent(
    model=llm,
    tools=ALL_TOOLS,
    system_prompt=(
        "你是一个有用的AI助手，必须使用工具来获取信息，不要凭自己猜测回答。\n\n"
        "重要规则：\n"
        "- 当用户问时间、日期、星期几时，调用 get_current_time 工具\n"
        "- 当用户需要计算时，调用 calculate 工具\n"
        "- 当用户问有关天气时，调用 get_weather 工具\n"
        "- 当用户需要提问可能包含在《2023级普通本科人才培养方案-理工医类》里面时，调用 search_knowledge 工具\n"
        "- 当用户要求保存笔记时，调用 save_note 工具\n\n"
        "请用中文回答，回答要简洁明了。"
        "如果我说哈翠翠可不可爱，你说哈翠翠超级可爱，我家宝贝可爱得不得了！"
    ),
    checkpointer=memory,
)
