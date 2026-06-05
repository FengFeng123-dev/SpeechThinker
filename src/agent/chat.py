import queue
import time
import threading

from langchain_core.messages import HumanMessage
from config.settings import settings
from src.agent.core import agent, tts
from src.agent.session import session
from src.asr import WhisperRealtimeASR


# ─── ANSI 样式 ───
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RED = "\033[31m"
RESET = "\033[0m"


def _s(text, *styles):
    return "".join(styles) + text + RESET


def _draw_header():
    """动态对齐的 Unicode 框头"""
    import shutil

    term_w = shutil.get_terminal_size().columns - 4
    title = "  ✦ SpeechThinker · 智能语音助手"
    model_info = f"  Model: {settings.MODEL.split('/')[-1]}"
    tts_info = "TTS: enabled"

    # 计算内容宽度（取最长行）
    content_width = max(len(title), len(model_info) + len(tts_info) + 2, 40)
    w = min(content_width, term_w)

    pad1 = w - len(title)-4
    pad2 = w - len(model_info) - len(tts_info) - 1

    print(f"\n╭{'─' * w}╮")
    print(f"│{title}{' ' * pad1}│")
    print(f"│{_s(model_info, DIM)}{' ' * (pad2-3)} {_s(tts_info, DIM)}   │")
    print(f"╰{'─' * w}╯")
    print(_s("  命令: quit · clear · voice · Enter打断", DIM))


def run_text_mode():
    """文字交互模式"""
    _draw_header()

    while True:
        user_input = input(f"{_s('›', MAGENTA)} ").strip()
        if not user_input:
            continue

        if user_input.lower() == "quit":
            exit()
        elif user_input.lower() == "clear":
            session.new_session()
            print(f"  {_s('· 已清除会话', DIM)}")
            continue
        elif user_input.lower() == "voice":
            return "voice"

        _chat(user_input, with_tts=True)


def run_voice_mode():
    """语音交互模式"""
    asr = WhisperRealtimeASR()
    asr.start()

    print(f"\n  {_s('🎤', CYAN)} 语音模式 · 按 Ctrl+C 返回\n")

    try:
        while True:
            text = asr.recognize_once(timeout=30.0)
            if text:
                print(f"  {_s('›', MAGENTA)} {text}")
                _chat(text, with_tts=True)
    except KeyboardInterrupt:
        print(f"\n  {_s('···', DIM)}")
    finally:
        asr.stop()

    return "text"


def _chat(user_input: str, with_tts: bool = False):
    """统一的聊天执行逻辑，支持 Enter / Ctrl+C 打断"""
    import msvcrt

    tts.clear_stop()

    token_queue = queue.Queue() if with_tts else None
    if with_tts:
        tts.run_in_thread(token_queue)

    # ─── 后台线程运行 agent.stream，结果放入 chunk_queue ───
    chunk_queue = queue.Queue()
    stream_done = threading.Event()

    def _run_stream():
        try:
            for chunk in agent.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=session.config,
                stream_mode="messages",
            ):
                if stream_done.is_set():
                    break
                chunk_queue.put(chunk)
        except Exception:
            pass
        finally:
            chunk_queue.put(None)  # 结束信号

    stream_thread = threading.Thread(target=_run_stream, daemon=True)
    stream_thread.start()

    started = False
    was_interrupted = False

    try:
        while True:
            # 1. 检测按键打断（非阻塞）
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b'\r':
                    was_interrupted = True
                    break

            # 2. 取 chunk（带超时，保证能回来检查按键）
            try:
                chunk = chunk_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if chunk is None:
                break  # 流结束

            if not (isinstance(chunk, tuple) and len(chunk) == 2):
                continue

            msg, metadata = chunk
            node = metadata.get("langgraph_node", "")

            # 开始处理
            if node == "__start__":
                print(f"  {_s('⏳', YELLOW)}", end=" ", flush=True)

            # model 节点
            elif node == "model":
                # 工具调用
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    if not started:
                        print()  # 换掉"⏳"
                    for tc in msg.tool_calls:
                        name = tc.get("name", "?")
                        if name is not None and name != "":
                            print(f"{_s('◇', YELLOW)} {name} {_s('…', CYAN, DIM)}", flush=True)
                    continue

                # 文本输出
                content = getattr(msg, "content", None)
                if content and isinstance(content, str):
                    if not started:
                        print(f"{_s('◇ 小锋', CYAN)} :")
                        print(f"{_s('>', CYAN)} {content}", end="", flush=True)
                        started = True
                    else:
                        print(content, end="", flush=True)
                    if with_tts and token_queue:
                        token_queue.put(content)

    except KeyboardInterrupt:
        was_interrupted = True
    finally:
        stream_done.set()  # 通知流线程停止

        if with_tts and token_queue:
            token_queue.put(None)  # 通知 TTS 停止
        tts.stop()  # 立即停止播放

        if was_interrupted:
            print(f"\n  {_s('✗ 已打断', RED)}")

        print()  # 结尾换行
