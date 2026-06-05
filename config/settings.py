import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class Settings:
    """全局配置中心"""

    # ─── LLM 配置 ───
    API_BASE: str = os.getenv("API_BASE", "")
    API_KEY: str = os.getenv("API_KEY", "")
    MODEL: str = os.getenv("LMSTUDIO_GEMMA_4_E4B", "")
    QWEN_VL_MODEL: str = os.getenv("QWEN_VL_MODEL", "")
    QWEN_OMNI_MODEL: str = os.getenv("QWEN_OMNI_MODEL", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "")

    # ─── ASR 配置 ───
    WHISPER_MODEL_DIR: str = os.getenv(
        "WHISPER_MODEL_DIR",
        r"E:\Develop\modelscope\models\openai-mirror\whisper-large-v3-turbo",
    )
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cuda")
    WHISPER_SAMPLE_RATE: int = int(os.getenv("WHISPER_SAMPLE_RATE", "16000"))
    SILENCE_TIMEOUT: float = float(os.getenv("SILENCE_TIMEOUT", "4.0"))
    SILENCE_THRESHOLD: float = float(os.getenv("SILENCE_THRESHOLD", "0.05"))

    # ─── TTS 配置 ───
    TTS_VOICE: str = os.getenv("TTS_VOICE", "zh-CN-XiaoxiaoNeural")
    TTS_RATE: str = os.getenv("TTS_RATE", "+0%")
    TTS_BUFFER_SIZE: int = int(os.getenv("TTS_BUFFER_SIZE", "10"))

    # 智能体名称
    AGENT_NAME: str = os.getenv("AGENT_NAME", "")

    # ─── 路径配置 ───
    DATA_DIR: Path = PROJECT_ROOT / "data"
    OUTPUT_DIR: Path = PROJECT_ROOT / "output"
    NOTES_DIR: Path = PROJECT_ROOT / "output" / "notes"

    # ─── API Keys ───
    AMAP_API_KEY: str = os.getenv("AMAP_API_KEY", "")


settings = Settings()
