"""
小问智能助手 - 实时语音 Agent 模式
ASR 识别 → Agent 思考+调工具 → TTS 播放
"""
import asyncio
import os
import threading
import queue
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain.agents import create_agent

from src.asr import WhisperRealtimeASR
from src.tts import RealTimeTTS

load_dotenv()


# ══════════════════════════════════════
#  1. LLM + Tools
# ══════════════════════════════════════

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE"),
    model=os.getenv("QWEN_OMNI_MODEL"),
    temperature=0.7,
    max_tokens=2048,
    streaming=True,
)


@tool
def get_current_time() -> str:
    """获取当前的日期和时间，当用户询问时间、日期、星期几时调用此工具"""
    now = datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return f"现在是 {now.strftime('%Y年%m月%d日 %H点%M分')}，{weekdays[now.weekday()]}"


@tool
def calculator(expression: str) -> str:
    """计算数学表达式。当用户需要做算术运算、单位换算等计算时调用。
    expression: 数学表达式字符串，如 '123 * 456' 或 '(2 + 3) * 4'
    """
    try:
        allowed = set("0123456789+-*/().% ")
        if any(c not in allowed for c in expression):
            return f"抱歉，表达式 '{expression}' 包含不允许的字符"
        result = eval(expression)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算出错：{e}"


@tool
def search_memory(query: str) -> str:
    """搜索你之前的对话记忆。当你需要回忆之前聊过的内容时调用。
    query: 要搜索的关键词或问题
    """
    global _chat_history
    results = []
    for i, msg in enumerate(_chat_history):
        if query.lower() in msg.content.lower():
            role = "用户" if isinstance(msg, HumanMessage) else "助手"
            results.append(f"[{role}]: {msg.content}")
    if not results:
        return f"在对话记忆中没找到关于「{query}」的内容"
    return "\n".join(results[-5:])


TOOLS = [get_current_time, calculator, search_memory]


# ══════════════════════════════════════
#  2. Agent 构建（ReAct 模式）
# ══════════════════════════════════════

SYSTEM_PROMPT = """你是"小问"，一个智能语音助手。

性格特点：
- 回答简洁口语化（你在跟用户**说话**，不是写文章）
- 友善、有耐心，偶尔可以幽默一下
- 不要用 markdown 格式、不要用列表符号，就像正常聊天

能力：
- 闲聊、问答、知识解答
- 通过工具获取时间、进行计算、回忆对话记录
- 当不确定时，诚实说不知道而不是编造

重要规则：
1. 用户通过语音与你交互，回答控制在 2-3 句话以内，不要太长
2. 如果需要调用工具就调用，不需要就不调用
3. 全部使用中文回答"""


def build_agent():
    """构建带 tools 的 Agent"""
    return create_agent(model=llm, tools=TOOLS)


# ══════════════════════════════════════
#  3. TTS 子线程
# ══════════════════════════════════════

tts = RealTimeTTS(
    voice="zh-CN-XiaoxiaoNeural",
    rate="+0%",
    buffer_size=10,
)


def tts_worker(token_queue):
    """子线程：从队列取 token → TTS 合成播放"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(tts.run_from_queue(token_queue))
    finally:
        loop.close()


# ══════════════════════════════════════
#  4. 对话记忆 & Agent 流式执行
# ══════════════════════════════════════

_chat_history: list[HumanMessage | AIMessage] = []


def run(user_input: str):
    """Agent 单轮执行：输入文字 → Agent 思考+调工具(可选) → 流式输出 → TTS 播放"""

    _chat_history.append(HumanMessage(content=user_input))

    token_queue = queue.Queue()
    tts_thread = threading.Thread(target=tts_worker, args=(token_queue,), daemon=True)
    tts_thread.start()

    print("\n🤔 思考中...", end="", flush=True)

    async def stream_agent():
        global _chat_history
        agent = build_agent()
        inputs = {
            "messages": [SystemMessage(content=SYSTEM_PROMPT)] + _chat_history
        }

        final_text = ""
        tool_call_count = 0

        async for event in agent.astream_events(inputs, version="v2"):
            kind = event["event"]

            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                content = chunk.content
                if content and isinstance(content, str):
                    final_text += content
                    yield content

            elif kind == "on_tool_start":
                name = event["name"]
                input_str = str(event["data"]["input"])[:80]
                print(f"\n  🔧 [{name}]({input_str})", end="", flush=True)
                tool_call_count += 1

            elif kind == "on_tool_end":
                output = str(event["data"]["output"])[:100]
                print(f" → {output}", end="", flush=True)

        if final_text.strip():
            _chat_history.append(AIMessage(content=final_text))

        if len(_chat_history) > 20:
            _chat_history = _chat_history[-20:]

        yield None

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        gen = stream_agent()
        print("\n【AI】：", end="", flush=True)
        while True:
            chunk = loop.run_until_complete(gen.__anext__())
            if chunk is None:
                break
            print(chunk, end="", flush=True)
            token_queue.put(chunk)
    except StopAsyncIteration:
        pass
    finally:
        loop.close()

    token_queue.put(None)
    tts_thread.join(timeout=30)


# ══════════════════════════════════════
#  5. 主循环：ASR 识别 → Agent 处理 → TTS 播放
# ══════════════════════════════════════

def main():
    asr = WhisperRealtimeASR()
    asr.start()

    print("=" * 40)
    print("  🎙️  小问智能助手 (Agent模式)")
    print("  按 Ctrl+C 退出")
    print("=" * 40)

    round_num = 0
    while True:
        try:
            text = asr.recognize_once(timeout=30.0)
            if not text:
                continue

            round_num += 1
            print(f"\n{'─' * 40}")
            print(f"📝 第{round_num}轮 用户：{text}")
            run(text)
            print()

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break

    asr.stop()


if __name__ == "__main__":
    main()
