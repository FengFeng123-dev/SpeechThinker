"""
笔记机器人 - 从屏幕截图中识别信息并整理成笔记
使用 VL 多模态模型 + 视频流捕获
"""
import subprocess
import time
import os
import edge_tts
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from src.vision import VideoStream, StreamType

load_dotenv()

# 创建实例时指定 WiFi 地址 + ipcam 模式
vs = VideoStream(
    scale=1,
    queue_size=1,
    fps=5,
    phone_usb_url="http://192.168.0.102:4747/video",
    phone_usb_mode="ipcam"
)

vs.start(StreamType.PHONE_USB)

llm = ChatOpenAI(
    base_url=os.getenv("API_BASE"),
    api_key=os.getenv("API_KEY"),
    model=os.getenv("QWEN_VL_MODEL"),
    temperature=0.5,
    max_tokens=1024,
    streaming=True,
)

NOTE_PATH = "note/note1.md"

system_prompt = """你是一个笔记机器人，负责从截图中识别到的信息整理成笔记。
要求：
1.不知道的内容请勿回答，只回复你看到且确定的答案，不要盲猜
2.你只关注有关学习的东西，不要关注其他内容，比如电脑截图中无关的组件
3.将笔记整理美观
4.结合已有笔记和截图中的新内容，重新整理笔记
5.请不要连续输出太多重复内容
6.如果截图太模糊,不进行回答并提醒(视频有点模糊,尝试重新识别...)
"""

# 等待队列有数据
while vs.queue_len == 0:
    time.sleep(0.1)

while True:
    frames_b64 = vs.get_all_frames()
    if not frames_b64:
        time.sleep(1)
        continue

    print(f"收集到 {len(frames_b64)} 张图片，一起发送给模型")
    existing_note = ""
    if os.path.exists(NOTE_PATH):
        with open(NOTE_PATH, "r", encoding="utf-8") as f:
            existing_note = f.read().strip()

    user_text = "根据图片重新整理笔记"

    content_list = [{"type": "text", "text": user_text}]
    for b64 in frames_b64:
        content_list.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
                "detail": "high"
            }
        })

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_list}
    ]

    content = ""
    text = ""
    for chunk in llm.stream(messages):
        text = chunk.content
        print(text, end='', flush=True)
        content += text

    if text.strip():
        tts = edge_tts.Communicate(
            text=text,
            voice="zh-CN-YunxiNeural",
            rate="+10%",
        )
        temp_mp3 = "temp_chunk.mp3"
        tts.save_sync(temp_mp3)
        subprocess.run(["ffplay", "-nodisp", "-autoexit", temp_mp3],
                       capture_output=True)
    print("------------------------")

vs.stop()
