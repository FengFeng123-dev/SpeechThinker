import asyncio
import re
import threading
import queue as _threading_queue
import edge_tts
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import socket
import tempfile
from asyncio import Queue
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# DNS 修补（edge-tts 在部分网络下需要）
_orig = socket.getaddrinfo
socket.getaddrinfo = lambda h, p, *a, **k: (
    [r for ip in ["13.107.5.25", "204.79.197.200"]
     for r in _orig(ip, p, *a, **k)]
    if h == "speech.platform.bing.com"
    else _orig(h, p, *a, **k)
)


class RealTimeTTS:
    """实时语音合成播放器：LLM 流式输出 → 句子切割 → TTS 合成 → 队列缓冲 → 播放"""

    def __init__(
        self,
        api_key=None,
        base_url=None,
        model=None,
        voice="zh-CN-XiaoxiaoNeural",
        rate="+0%",
        buffer_size=10,
    ):
        self.api_key = api_key or os.getenv("API_KEY")
        self.base_url = base_url or os.getenv("API_BASE")
        self.model = model or os.getenv("QWEN_VL_MODEL")
        self.voice = voice
        self.rate = rate
        self.buffer_size = buffer_size
        self._mixer_initialized = False
        self._stop_flag = threading.Event()
        self._tts_thread = None

    # ─── 打断控制 ───

    def stop(self):
        """打断：立即停止合成和播放"""
        self._stop_flag.set()
        try:
            import pygame
            if self._mixer_initialized and pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
        except Exception:
            pass

    def clear_stop(self):
        """清除打断标志（下次调用前需要清除）"""
        self._stop_flag.clear()

    # ─── 文本清洗：移除不需要朗读的符号 ───
    @staticmethod
    def _clean_text(text: str) -> str:
        # 移除 emoji 及其他 Unicode 表情符号
        text = re.sub(
            r'[\U0001F600-\U0001F64F'  # 表情符号
            r'\U0001F300-\U0001F5FF'    # 杂项符号和象形文字
            r'\U0001F680-\U0001F6FF'    # 交通和地图符号
            r'\U0001F1E0-\U0001F1FF'    # 旗帜
            r'\U00002702-\U000027B0'    # 装饰符号
            r'\U000024C2-\U0001F251'    # 其他符号
            r'\U0001F900-\U0001F9FF'    # 补充表情符号
            r'\U0001FA00-\U0001FA6F'    # 扩展表情符号A
            r'\U0001FA70-\U0001FAFF'    # 扩展表情符号B
            r'\U00002600-\U000026FF'    # 杂项符号
            r'\U0000FE00-\U0000FE0F'    # 变体选择符
            r'\U0000200D'              # 零宽连接符
            r']',
            '', text
        )
        # Markdown 格式标记（需先处理，避免移除星号后匹配失效）
        text = re.sub(r'\*+([^*]+)\*+', r'\1', text)          # **粗体** / *斜体* → 纯文本
        text = re.sub(r'`{1,3}[^`]*`{1,3}', '', text)          # `code` 或 ```code``` → 删除
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # [链接](url) → 链接
        text = re.sub(r'#{1,6}\s*', '', text)                  # # 标题 → 标题
        # 移除杂项符号
        text = re.sub(r'[#*@_~^`|\\]', '', text)
        # 合并多余空白
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    # ─── 合成单句 ───
    async def synthesize(self, text):
        text = self._clean_text(text)
        if not text:
            return b""
        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
        chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        return b"".join(chunks)

    # ─── 播放音频 ───
    async def play_audio(self, audio_bytes):
        if not audio_bytes or self._stop_flag.is_set():
            return
        import pygame
        if not self._mixer_initialized:
            pygame.mixer.init()
            self._mixer_initialized = True

        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp_path = tmp.name
        tmp.close()
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            if self._stop_flag.is_set():
                pygame.mixer.music.stop()
                break
            await asyncio.sleep(0.05)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # ─── 句子切割 ───
    async def stream_sentence(self, generator):
        buffer = ""
        stops = {"。", "！", "？", "；", ".", "!", "?"}
        async for token in generator:
            if self._stop_flag.is_set():
                return
            buffer += token
            for sym in stops:
                if sym in buffer:
                    idx = buffer.index(sym)
                    s = buffer[:idx + 1].strip()
                    buffer = buffer[idx + 1:].strip()
                    if s:
                        yield s
                    break
            await asyncio.sleep(0)
        if buffer.strip() and not self._stop_flag.is_set():
            yield buffer.strip()

    # ─── LLM 流式调用 ───
    async def langchain_stream(self, prompt):
        llm = ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            streaming=True,
            temperature=0,
        )
        messages = [HumanMessage(content=prompt)]
        async for chunk in llm.astream(messages):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                yield chunk.content

    # ─── 生产者：持续合成 ───
    async def _producer(self, sentence_gen, queue):
        async for sentence in sentence_gen:
            if self._stop_flag.is_set():
                break
            audio = await self.synthesize(sentence)
            if audio and not self._stop_flag.is_set():
                await queue.put(audio)
        await queue.put(None)

    # ─── 消费者：持续播放 ───
    async def _consumer(self, queue):
        while not self._stop_flag.is_set():
            try:
                audio = await asyncio.wait_for(queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            if audio is None:
                break
            await self.play_audio(audio)

    # ─── 主流程（内置 LLM） ───
    async def run(self, prompt):
        """输入 prompt，流式合成并播放语音"""
        print("\n【AI 回答】：", end="", flush=True)

        llm_gen = self.langchain_stream(prompt)
        sentence_gen = self.stream_sentence(llm_gen)
        queue = Queue(maxsize=self.buffer_size)

        await asyncio.gather(
            self._producer(sentence_gen, queue),
            self._consumer(queue),
        )

        print("\n\n✅ 播放完成")

    # ─── 主流程（外部流） ───
    async def run_from_stream(self, token_stream):
        """接受外部的 async generator(str)，流式合成并播放语音"""
        sentence_gen = self.stream_sentence(token_stream)
        queue = Queue(maxsize=self.buffer_size)

        await asyncio.gather(
            self._producer(sentence_gen, queue),
            self._consumer(queue),
        )

        # print("\n\n✅ 播放完成")

    # ─── 主流程（从同步队列消费） ───
    async def run_from_queue(self, sync_queue):
        """从 threading.Queue(str) 消费 token，流式合成并播放语音"""
        async def _queue_reader():
            loop = asyncio.get_event_loop()
            while not self._stop_flag.is_set():
                try:
                    token = await loop.run_in_executor(
                        None, lambda: sync_queue.get(timeout=0.3)
                    )
                except _threading_queue.Empty:
                    continue
                if token is None:
                    break
                yield token

        sentence_gen = self.stream_sentence(_queue_reader())
        queue = Queue(maxsize=self.buffer_size)

        await asyncio.gather(
            self._producer(sentence_gen, queue),
            self._consumer(queue),
        )

        # print("\n\n✅ 播放完成")

    def run_in_thread(self, sync_queue):
        """启动一个子线程来运行 TTS（同步接口，适合主线程阻塞等待场景）

        Args:
            sync_queue: threading.Queue，用于接收外部传入的文本 token
        Returns:
            threading.Thread: 已启动的子线程对象，可调用 .join() 等待完成
        """
        import threading

        # 等待上一次 TTS 线程退出
        if self._tts_thread and self._tts_thread.is_alive():
            self._tts_thread.join(timeout=2.0)

        def _worker():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.run_from_queue(sync_queue))
            finally:
                loop.close()

        self._tts_thread = threading.Thread(target=_worker, daemon=True)
        self._tts_thread.start()
        return self._tts_thread

    async def run_from_queue_background(self, sync_queue):
        """在已有事件循环中异步运行 TTS（异步接口，适合 asyncio 环境）"""
        await self.run_from_queue(sync_queue)


if __name__ == "__main__":
    prompt = input("请输入问题：")
    tts = RealTimeTTS()
    asyncio.run(tts.run(prompt))
