<div align="center">

# ⚡ SpeechThinker

**多模态 AI 语音助手**

实时语音识别 · 视觉理解 · Agent 工具调用 · 语音合成

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.1+-1C3C3C?logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![Whisper](https://img.shields.io/badge/Whisper-Large_V3_Turbo-FF6F00?logo=openai&logoColor=white)](https://github.com/openai/whisper)
[![Edge TTS](https://img.shields.io/badge/Edge_TTS-6.1+-0078D4?logo=microsoft&logoColor=white)](https://github.com/rany2/edge-tts)

</div>

---

## ✨ 项目简介

SpeechThinker 是一个多模态 AI 语音助手，融合了**语音识别、视觉理解、智能对话和语音合成**四大核心能力。它可以在你学习时实时截取屏幕/摄像头画面，自动整理笔记；也可以通过语音与你自然对话，调用工具获取信息。

```
┌──────────┐     ┌──────────┐     ┌─────────────┐     ┌──────────┐
│  语音输入 │───▶│  Whisper │───▶│  LangGraph   │───▶│ Edge TTS │
│  视觉输入 │───▶│  VL 模型  │───▶│  Agent 核心  │───▶│ 语音播放  │
│  文字输入 │───▶│          │───▶│  + Tools     │───▶│ 文字输出  │
└──────────┘     └──────────┘     └──────────────┘    └──────────┘
```

## 🎯 核心功能

### 🧠 智能体 Agent（LangGraph）

基于 LangGraph 构建的有状态 Agent，支持会话记忆和打断输出，内置 5 个工具：

| 工具 | 功能 | 示例 |
|------|------|------|
| 🕐 `get_current_time` | 获取当前日期时间和星期 | "现在几点？" |
| 🧮 `calculate` | 数学表达式计算，支持三角函数/对数等 | "计算 2^10" |
| 🌤️ `get_weather` | 查询全国 3500+ 城市实况天气 | "北京天气怎么样" |
| 📚 `search_knowledge` | 本地知识库搜索（RAG 向量检索） | "什么是 RAG" |
| 📝 `save_note` | 保存笔记到本地文件 | "帮我记一下..." |

### 🎙️ 实时语音识别（Whisper）

- 基于 **Whisper Large V3 Turbo** 模型，通过 WASAPI 回环流捕获系统声音
- 自动静音检测，停顿超时后触发识别
- 支持长音频自动分段，超过 28 秒自动切片
- 可配置静音阈值和超时时间

### 👁️ 视觉理解（VL 多模态模型）

- 支持**屏幕截图 / 摄像头 / 手机 USB 摄像头**三种视频流输入
- 自动抽帧 → 压缩 → 编码 → 发送给 VL 模型
- 结合已有笔记与截图新内容，自动整理美观的 Markdown 笔记

### 🔊 语音合成（Edge TTS）

- 流式输出 → 句子切割 → TTS 合成 → 队列缓冲 → 实时播放
- 自动清洗 Markdown 格式标记，确保朗读自然流畅
- 多种语音角色可选（默认：晓晓）
- 支持 Enter 键打断正在播放的语音

## 📁 项目结构

```
SpeechThinker/
├── main.py                  # 主入口：文字/语音模式切换
├── model_test.py            # 笔记机器人：VL 模型 + 视频流自动整理笔记
├── realtime_ask.py          # 语音 Agent（独立版）：ASR → Agent → TTS
├── requirements.txt         # 依赖清单
├── config/
│   ├── settings.py          # 全局配置中心（LLM/ASR/TTS/路径/API Keys）
│   └── __init__.py
├── data/
│   ├── China-City-List-latest.csv   # 全国城市编码表（天气查询用）
│   └── 2023级普通本科人才培养方案-理工医类.pdf  # 知识库文档
├── chroma_db/               # ChromaDB 向量数据库持久化存储
└── src/
    ├── agent/               # Agent 核心
    │   ├── core.py          #   LLM + TTS + Agent 实例构建（MemorySaver）
    │   ├── chat.py          #   文字/语音交互模式 + 打断支持
    │   └── session.py       #   会话状态管理（thread_id）
    ├── asr/                 # 语音识别
    │   └── whisper_asr.py   #   Whisper 实时 ASR（WASAPI 回环流）
    ├── tts/                 # 语音合成
    │   └── edge_tts_engine.py  # Edge TTS 流式合成播放器
    ├── tools/               # Agent 工具集
    │   ├── time_tool.py     #   时间查询
    │   ├── calculator.py    #   数学计算
    │   ├── weather.py       #   天气查询（高德 API）
    │   ├── knowledge.py     #   知识库搜索（RAG + ChromaDB）
    │   ├── notes.py         #   笔记保存
    │   └── chroma_db/       #   ChromaDB 持久化数据
    └── vision/              # 视觉输入
        └── video_stream.py  #   多源视频流（屏幕/摄像头/手机USB）
```

## 🚀 快速开始

### 1. 环境准备

- Python 3.10+
- CUDA 环境（Whisper 模型推理加速）
- Windows 系统（WASAPI 音频捕获）
- 本地部署 LLM（要求支持工具调用）+ Whisper Large V3 Turbo

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

在项目根目录创建 `.env` 文件：

```env
# LLM 配置（OpenAI 兼容接口）
API_BASE=https://your-api-base/v1
API_KEY=your-api-key
LMSTUDIO_GEMMA_4_E4B=your-model-name
QWEN_VL_MODEL=your-vl-model-name
QWEN_OMNI_MODEL=your-omni-model-name
EMBEDDING_MODEL=your-embedding-model-name

# ASR 配置
WHISPER_MODEL_DIR=path/to/whisper-large-v3-turbo
WHISPER_DEVICE=cuda
WHISPER_SAMPLE_RATE=16000
SILENCE_TIMEOUT=4.0
SILENCE_THRESHOLD=0.05

# TTS 配置
TTS_VOICE=zh-CN-XiaoxiaoNeural
TTS_RATE=+0%
TTS_BUFFER_SIZE=10

# 天气查询 API Key（高德开放平台）
AMAP_API_KEY=your-amap-api-key
```

### 4. 运行

```bash
# 交互式 Agent（文字/语音模式，支持打断）
python main.py

# 笔记机器人（VL 模型 + 视频流自动整理笔记）
python model_test.py

# 语音 Agent 独立运行（简化版）
python realtime_ask.py
```

### 交互命令

| 命令 | 说明 |
|------|------|
| `voice` | 切换到语音模式 |
| `clear` | 清除对话记忆（开启新会话） |
| `quit` | 退出程序 |
| `Enter` | 打断当前 AI 输出/TTS 播放 |
| `Ctrl+C` | 语音模式下返回文字模式 |

## 🔧 技术栈

| 模块 | 技术 |
|------|------|
| Agent 框架 | LangGraph + LangChain |
| LLM | ChatOpenAI（兼容 Qwen/Gemma 等） |
| 语音识别 | Whisper Large V3 Turbo + PyAudioWPatch |
| 语音合成 | Edge TTS + Pygame |
| 视觉输入 | OpenCV + MSS + DroidCam/IPCam |
| 知识库 | ChromaDB + LangChain RAG |
| 天气数据 | 高德开放平台 API |
| 配置管理 | python-dotenv |

## 💡 使用场景

- **学习助手**：打开网课/电子书，自动截屏识别内容并整理笔记
- **语音问答**：语音提问，Agent 调用工具获取实时天气/时间/计算结果
- **知识检索**：本地知识库快速查找技术概念（如人才培养方案）
- **快捷笔记**：语音说"帮我记一下..."，自动保存到本地文件

## ⚙️ 视频流配置

`VideoStream` 支持三种输入源，在 `model_test.py` 中配置：

```python
from src.vision import VideoStream, StreamType

vs = VideoStream(
    scale=0.5,           # 帧压缩比例
    fps=5,               # 抽帧帧率
    phone_usb_url="http://192.168.0.102:4747/video",
    phone_usb_mode="ipcam"  # droidcam / ipcam / scrcpy
)

vs.start(StreamType.PHONE_USB)  # SCREEN / WEBCAM / PHONE_USB
```

---

<div align="center">

**让 AI 成为你的第二大脑** 🧠

</div>
