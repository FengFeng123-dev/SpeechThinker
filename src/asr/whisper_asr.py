import numpy as np
import time
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import pyaudiowpatch as audio

from config.settings import settings


class WhisperRealtimeASR:
    """Whisper 实时语音识别类，监听 Windows 系统声音并识别

    用法:
        asr = WhisperRealtimeASR()
        asr.start()                          # 加载模型 + 打开音频流（只执行一次）
        text1 = asr.recognize_once()         # 第一次识别
        text2 = asr.recognize_once()         # 第二次识别（复用模型和音频流）
        asr.stop()                           # 释放所有资源
    """

    def __init__(
        self,
        model_dir: str = None,
        device: str = None,
        sample_rate: int = None,
        silence_timeout: float = None,
        silence_threshold: float = None,
        check_interval: float = 0.5,
        max_audio_sec: int = 28,
    ):
        self.model_dir = model_dir or settings.WHISPER_MODEL_DIR
        self.device = device or settings.WHISPER_DEVICE
        self.sample_rate = sample_rate or settings.WHISPER_SAMPLE_RATE
        self.silence_timeout = silence_timeout or settings.SILENCE_TIMEOUT
        self.silence_threshold = silence_threshold or settings.SILENCE_THRESHOLD
        self.check_interval = check_interval
        self.max_audio_sec = max_audio_sec

        # 模型
        self.processor = None
        self.model = None

        # 音频流（start 后持续保持，stop 才释放）
        self._stream = None
        self._pyaudio = None
        self._device_sr = None
        self._channels = None

        # 状态
        self._started = False

    # ── 启动 / 停止 ──────────────────────────────────────────

    def start(self):
        """加载模型并打开音频流，后续可多次调用 recognize_once()"""
        if self._started:
            print("⚠️ 已启动，无需重复调用")
            return

        # 加载模型
        print("⏳ 加载模型...")
        self.processor = WhisperProcessor.from_pretrained(self.model_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_dir, torch_dtype=torch.float16, device_map="auto"
        ).eval()
        self._zh_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="zh", task="transcribe"
        )
        print("✅ 模型就绪")

        # 打开音频流
        self._open_stream()
        self._started = True
        print("📡 就绪，可调用 recognize_once() 进行识别\n")

    def stop(self):
        """释放模型和音频流资源"""
        if not self._started:
            return
        self._close_stream()
        self.model = None
        self.processor = None
        self._started = False
        print("⏹️ 已停止，资源已释放")

    def is_started(self) -> bool:
        """是否已启动（模型和音频流就绪）"""
        return self._started

    # ── 单次识别 ──────────────────────────────────────────────

    def recognize_once(self, timeout: float = 30.0) -> str:
        """执行一次识别：等待语音 → 静默后识别 → 返回文本

        Args:
            timeout: 最长等待语音的超时秒数，超时返回空字符串

        Returns:
            识别出的文本，未检测到语音或超时返回空字符串
        """
        if not self._started:
            raise RuntimeError("❌ 请先调用 start() 初始化")

        audio_buffer = []
        last_speech_time = None
        is_speaking = False
        start_time = time.time()
        frames_per_check = int(self._device_sr * self.check_interval / 1024)

        while True:
            # 等待语音超时
            if not is_speaking and time.time() - start_time > timeout:
                print("⏱️ 等待语音超时")
                return ""

            frames = [
                self._stream.read(1024, exception_on_overflow=False)
                for _ in range(frames_per_check)
            ]
            raw = np.frombuffer(b"".join(frames), dtype=np.int16)
            if self._channels > 1:
                raw = raw.reshape(-1, self._channels).mean(axis=1)
            audio_data = raw.astype(np.float32) / 32768.0
            has_speech = np.sqrt(np.mean(audio_data ** 2)) > self.silence_threshold

            if has_speech:
                audio_buffer.append(audio_data)
                if not is_speaking:
                    is_speaking = True
                    print("🎤 检测到语音...", end="", flush=True)
                last_speech_time = time.time()
            elif is_speaking:
                if time.time() - last_speech_time >= self.silence_timeout:
                    print(" ⏸️ 识别中...")
                    if audio_buffer:
                        text = self._recognize(np.concatenate(audio_buffer))
                        print(f"📝 {text}" if text else "📝 (无内容)")
                        return text
                    return ""
                else:
                    audio_buffer.append(audio_data)

    # ── 内部方法 ──────────────────────────────────────────────

    def _open_stream(self):
        """打开 WASAPI 回环流"""
        p = audio.PyAudio()
        wasapi_info = p.get_host_api_info_by_type(audio.paWASAPI)
        speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        print(f"🔊 默认扬声器: {speakers['name']}")

        try:
            loopbacks = list(p.get_loopback_device_info_generator())
        except AttributeError:
            loopbacks = [
                p.get_device_info_by_index(i) for i in range(p.get_device_count())
                if p.get_device_info_by_index(i)["maxInputChannels"] > 0
                and "loopback" in p.get_device_info_by_index(i)["name"].lower()
            ]

        dev = next((d for d in loopbacks if speakers["name"] in d["name"]), None)
        if not dev:
            raise RuntimeError("❌ 未找到回环设备！请启用立体声混音或安装 VB-CABLE")

        self._device_sr = int(dev["defaultSampleRate"])
        self._channels = dev["maxInputChannels"]
        print(f"✅ 回环: {dev['name']}  {self._device_sr}Hz  {self._channels}ch")

        self._pyaudio = p
        self._stream = p.open(
            format=audio.paInt16, channels=self._channels, rate=self._device_sr,
            input=True, input_device_index=dev["index"],
        )

    def _close_stream(self):
        """关闭音频流"""
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None

    @staticmethod
    def resample(data, orig_sr, target_sr):
        """重采样"""
        if orig_sr == target_sr:
            return data
        return np.interp(
            np.linspace(0, len(data), int(len(data) * target_sr / orig_sr)),
            np.arange(len(data)), data,
        ).astype(np.float32)

    def _recognize(self, audio_np):
        """Whisper 语音识别（超过 max_audio_sec 自动分段）"""
        audio_16k = self.resample(audio_np, self._device_sr, self.sample_rate)
        chunk_len = self.max_audio_sec * self.sample_rate
        texts = []
        for start in range(0, len(audio_16k), chunk_len):
            chunk = audio_16k[start:start + chunk_len]
            feat = self.processor(
                chunk, sampling_rate=self.sample_rate, return_tensors="pt"
            ).input_features.to(device=self.device, dtype=torch.float16)
            with torch.no_grad():
                ids = self.model.generate(
                    feat, language="zh",
                    task="transcribe",
                    max_new_tokens=444,
                    forced_decoder_ids=self._zh_decoder_ids,
                )
            texts.append(self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip())
        return "".join(texts)


if __name__ == "__main__":
    asr = WhisperRealtimeASR()
    asr.start()

    try:
        for i in range(3):
            print(f"\n--- 第 {i + 1} 次识别 ---")
            text = asr.recognize_once(timeout=30.0)
            print(f"结果: {text}")
    except KeyboardInterrupt:
        pass
    finally:
        asr.stop()
