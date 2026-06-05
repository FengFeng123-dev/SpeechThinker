import subprocess
import threading
import base64
import time
from collections import deque
from enum import Enum

import cv2
import mss
import numpy as np
from PIL import Image


class StreamType(Enum):
    SCREEN = "screen"
    WEBCAM = "webcam"
    PHONE_USB = "phone_usb"


class VideoStream:
    def __init__(self, scale=0.5, queue_size=3, webcam_index=0, fps=30,
                 phone_usb_url="http://127.0.0.1:4747/video",
                 phone_usb_mode="droidcam"):
        """
        :param scale: 帧等比例压缩比例 (0~1)
        :param queue_size: 队列最大长度
        :param webcam_index: 摄像头索引
        :param fps: 抽帧目标帧率
        :param phone_usb_url: 手机USB摄像头流的URL
        :param phone_usb_mode: 手机USB摄像头模式 (droidcam/ipcam/scrcpy)
        """
        self.scale = scale
        self.queue_size = queue_size
        self.webcam_index = webcam_index
        self.fps = fps
        self.phone_usb_url = phone_usb_url
        self.phone_usb_mode = phone_usb_mode

        self._queue = deque(maxlen=queue_size)
        self._current_type = None
        self._cap = None
        self._sct = None
        self._scrcpy_proc = None
        self._adb_forwarded = False
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

    # ── 公开接口 ──────────────────────────────────────────────

    def switch(self, stream_type: StreamType):
        """切换视频流渠道，关闭旧渠道，开启新渠道"""
        with self._lock:
            self._close_current()
            self._current_type = stream_type
            if stream_type == StreamType.SCREEN:
                self._sct = mss.mss()
            elif stream_type == StreamType.WEBCAM:
                self._cap = cv2.VideoCapture(self.webcam_index)
                if not self._cap.isOpened():
                    raise RuntimeError(f"无法打开摄像头 (index={self.webcam_index})")
            elif stream_type == StreamType.PHONE_USB:
                self._start_phone_usb()

    def start(self, stream_type: StreamType):
        """启动子线程，持续抽帧入队"""
        if self._running:
            return
        self.switch(stream_type)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """停止子线程并释放资源"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        with self._lock:
            self._close_current()

    def get_frame(self):
        """从队列取出最新帧 (base64 字符串)，队列为空返回 None"""
        with self._lock:
            if self._queue:
                return self._queue.pop()
            return None

    def get_latest_frame(self):
        """获取队列中最新帧并清空队列"""
        with self._lock:
            if self._queue:
                return self._queue.pop()
            return None

    def get_all_frames(self):
        """获取队列中所有帧的 base64 列表并清空队列"""
        with self._lock:
            frames = list(self._queue)
            self._queue.clear()
            return frames

    @property
    def current_type(self):
        return self._current_type

    @property
    def queue_len(self):
        return len(self._queue)

    # ── 内部方法 ──────────────────────────────────────────────

    def _close_current(self):
        """关闭当前流渠道，释放资源"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._sct is not None:
            self._sct.close()
            self._sct = None
        if self._scrcpy_proc is not None:
            try:
                self._scrcpy_proc.terminate()
                self._scrcpy_proc.wait(timeout=3)
            except Exception:
                self._scrcpy_proc.kill()
            self._scrcpy_proc = None
        if self._adb_forwarded:
            try:
                subprocess.run(["adb", "forward", "--remove", "tcp:8080"],
                               capture_output=True, timeout=5)
            except Exception:
                pass
            self._adb_forwarded = False
        self._current_type = None

    def _capture_frame(self):
        """从当前活跃渠道抓取一帧"""
        if self._current_type == StreamType.SCREEN:
            monitor = self._sct.monitors[0]
            img = self._sct.grab(monitor)
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame
        elif self._current_type == StreamType.WEBCAM:
            ret, frame = self._cap.read()
            if not ret:
                return None
            return frame
        elif self._current_type == StreamType.PHONE_USB:
            if self._cap is not None:
                ret, frame = self._cap.read()
                if not ret:
                    return None
                return frame
        return None

    def _start_phone_usb(self):
        """启动手机USB摄像头视频流"""
        if self.phone_usb_mode == "droidcam":
            print(f"[PhoneUSB] DroidCam模式，连接 {self.phone_usb_url}")
            self._cap = cv2.VideoCapture(self.phone_usb_url)
            if not self._cap.isOpened():
                raise RuntimeError(
                    f"无法连接DroidCam视频流 ({self.phone_usb_url})\n"
                    "请确保：\n"
                    "  1. 手机已安装DroidCam客户端并启动\n"
                    "  2. 手机通过USB连接到电脑\n"
                    "  3. DroidCam PC客户端已运行"
                )

        elif self.phone_usb_mode == "ipcam":
            import re
            url = self.phone_usb_url
            host_match = re.search(r'://([^/:]+)', url)
            is_localhost = False
            if host_match:
                host = host_match.group(1).lower()
                if host in ('127.0.0.1', 'localhost', '::1', '[::1]'):
                    is_localhost = True

            if is_localhost:
                print("[PhoneUSB] IP摄像头模式 (ADB端口转发)...")
                try:
                    result = subprocess.run(
                        ["adb", "forward", "tcp:8080", "tcp:8080"],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode != 0:
                        raise RuntimeError(
                            f"ADB端口转发失败: {result.stderr}\n"
                            "请确保：\n"
                            "  1. 已安装ADB并加入PATH\n"
                            "  2. 手机已开启USB调试\n"
                            "  3. 手机已通过USB连接"
                        )
                    self._adb_forwarded = True
                except FileNotFoundError:
                    raise RuntimeError(
                        "未找到adb命令，请安装Android SDK Platform-Tools并加入PATH\n"
                        "或者使用WiFi直连：将phone_usb_url设置为手机的局域网IP地址"
                    )
            else:
                print("[PhoneUSB] IP摄像头模式 (WiFi直连)...")

            print(f"[PhoneUSB] 连接IP摄像头 {url}")
            self._cap = cv2.VideoCapture(url)
            if not self._cap.isOpened():
                raise RuntimeError(
                    f"无法连接IP摄像头视频流 ({url})\n"
                    "请确保：\n"
                    "  1. 手机和电脑在同一局域网\n"
                    "  2. 手机已安装DroidCam/IP Webcam等APP并启动服务器\n"
                    "  3. URL格式正确，例如 http://192.168.x.x:4747/video"
                )

        elif self.phone_usb_mode == "scrcpy":
            print("[PhoneUSB] scrcpy模式，启动scrcpy...")
            try:
                self._scrcpy_proc = subprocess.Popen(
                    ["scrcpy", "--no-display", "--v4l2-sink=/dev/video0"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                time.sleep(2)
                self._cap = cv2.VideoCapture(0)
                if not self._cap.isOpened():
                    raise RuntimeError(
                        "无法通过scrcpy打开v4l2设备\n"
                        "请确保：\n"
                        "  1. 已安装scrcpy\n"
                        "  2. 系统支持v4l2loopback (Linux)\n"
                        "  3. 手机已通过USB连接并授权调试"
                    )
            except FileNotFoundError:
                raise RuntimeError(
                    "未找到scrcpy命令，请安装scrcpy: https://github.com/Genymobile/scrcpy"
                )
        else:
            raise ValueError(
                f"不支持的手机USB模式: {self.phone_usb_mode}\n"
                "可选模式: droidcam, ipcam, scrcpy"
            )

    def _compress_frame(self, frame):
        """对帧进行等比例像素压缩"""
        if self.scale >= 1.0:
            return frame
        h, w = frame.shape[:2]
        new_w = int(w * self.scale)
        new_h = int(h * self.scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _encode_frame(self, frame):
        """将帧编码为 JPEG 再 base64"""
        _, buf = cv2.imencode(".jpg", frame)
        return base64.b64encode(buf).decode("utf-8")

    def _enqueue(self, data):
        """入队：队列满时自动淘汰最旧帧"""
        with self._lock:
            self._queue.append(data)

    def _loop(self):
        """子线程主循环：持续抽帧 → 压缩 → 编码 → 入队"""
        interval = 1.0 / self.fps
        while self._running:
            t0 = time.monotonic()
            with self._lock:
                stream_type = self._current_type

            if stream_type is None:
                time.sleep(0.1)
                continue

            try:
                frame = self._capture_frame()
                if frame is not None:
                    frame = self._compress_frame(frame)
                    encoded = self._encode_frame(frame)
                    self._enqueue(encoded)
            except Exception as e:
                print(f"[VideoStream] 采集帧异常: {e}")

            elapsed = time.monotonic() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ── 上下文管理器 ──────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()

    def preview(self, stream_type: StreamType):
        """实时预览当前视频流，支持键盘交互切换"""
        self.start(stream_type)
        print(f"预览 {stream_type.value} 流  scale={self.scale}  fps={self.fps}")
        print("按 'q' 退出 | 's' 循环切换渠道 (screen → webcam → phone_usb)")

        stream_cycle = [StreamType.SCREEN, StreamType.WEBCAM, StreamType.PHONE_USB]

        try:
            while True:
                frame_b64 = self.get_frame()
                if frame_b64 is None:
                    time.sleep(0.01)
                    continue

                img_bytes = base64.b64decode(frame_b64)
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if frame is not None:
                    h, w = frame.shape[:2]
                    info = (f"{self.current_type.value} | {w}x{h} | "
                            f"scale={self.scale:.2f} | queue={self.queue_len}")
                    cv2.putText(frame, info, (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.imshow("VideoStream Preview", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    idx = stream_cycle.index(self.current_type) if self.current_type in stream_cycle else 0
                    new = stream_cycle[(idx + 1) % len(stream_cycle)]
                    print(f"切换到 {new.value}")
                    try:
                        self.switch(new)
                    except RuntimeError as e:
                        print(f"切换失败: {e}")
                        for offset in range(1, len(stream_cycle)):
                            fallback = stream_cycle[(idx + 1 + offset) % len(stream_cycle)]
                            try:
                                self.switch(fallback)
                                print(f"回退到 {fallback.value}")
                                break
                            except RuntimeError:
                                continue
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    vs = VideoStream(scale=0.5, fps=30,
                     phone_usb_url="http://192.168.0.101:4747/video",
                     phone_usb_mode="ipcam")
    vs.preview(StreamType.PHONE_USB)
