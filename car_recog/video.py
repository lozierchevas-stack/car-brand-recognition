"""视频流读取 (生产者): 抽帧 + 防卡顿队列策略.

设计要点 (防卡顿):
- 生产者-消费者多线程模型: 读取线程只负责解码入队, 推理线程只负责消费;
- 抽帧机制: 每 stride 帧提取一次, 缓解边缘设备算力瓶颈;
- 队列满时丢弃最旧帧 (put_latest), 保证消费者拿到的永远是最新画面,
  避免 RTSP 网络抖动导致的累积延迟;
- 本地文件读到结尾 / 摄像头断流恢复失败时, 投递 None 哨兵通知消费者优雅退出。
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import NamedTuple

import cv2
import numpy as np

logger = logging.getLogger("car-recog.video")


class FrameItem(NamedTuple):
    frame: np.ndarray
    index: int
    timestamp: float


class VideoSource(threading.Thread):
    """视频生产者线程. source: 0=默认摄像头 / RTSP URL / 本地视频文件路径."""

    def __init__(self, source, stride: int = 5, queue_size: int = 4,
                 max_read_fails: int = 60, name: str = "video-producer"):
        super().__init__(daemon=True, name=name)
        self.source = source
        self.stride = max(1, int(stride))
        self.queue: queue.Queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self.max_read_fails = int(max_read_fails)
        self._stop = threading.Event()
        self.frames_read = 0
        self.frames_enqueued = 0
        self.dropped = 0

    def _is_live(self) -> bool:
        return isinstance(self.source, int) or str(self.source).lower().startswith(
            ("rtsp://", "rtmp://", "http://", "https://"))

    def _open(self):
        src = self.source
        # 数字字符串 (如 CLI 传入的 "0") 归一化为摄像头索引,
        # 否则会被 OpenCV 当作文件路径打开失败
        if isinstance(src, str) and src.isdigit():
            src = int(src)
            self.source = src
        # 摄像头按后端回退: 显式 DSHOW -> 显式 MSMF -> 默认。
        # 默认路径在生产者线程 (非主线程) 中会永久阻塞 (OpenCV 5 实测),
        # 显式指定后端不受影响; RTSP/文件走默认后端即可
        backends = [None]
        if isinstance(src, int):
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]
        cap = None
        for be in backends:
            cap = cv2.VideoCapture(src, be) if be is not None else cv2.VideoCapture(src)
            if cap.isOpened():
                break
            cap.release()
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            return None
        if isinstance(src, str) and src.lower().startswith(("rtsp://", "rtmp://")):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 降低 RTSP 解码缓冲, 减少延迟
        return cap

    def _file_fps(self, cap) -> float:
        """本地文件按原始帧率回放 (模拟真实采集节奏), 实时源返回 0 表示不节流."""
        if self._is_live():
            return 0.0
        try:
            return float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        except cv2.error:
            return 0.0

    def _put_latest(self, item) -> None:
        """队满丢最旧, 保证消费者拿到最新帧."""
        while not self._stop.is_set():
            try:
                self.queue.put_nowait(item)
                self.frames_enqueued += 1
                return
            except queue.Full:
                try:
                    self.queue.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass

    def run(self) -> None:
        retry = 0
        while not self._stop.is_set():
            cap = self._open()
            if cap is None:
                if not self._is_live():
                    logger.error("无法打开视频源: %s", self.source)
                    break
                retry += 1
                delay = min(2 * retry, 30)
                logger.warning("打开视频源失败(第 %d 次), %ds 后重试...", retry, delay)
                if self._stop.wait(delay):
                    break
                continue
            retry = 0
            fails = 0
            fps = self._file_fps(cap)
            frame_delay = 1.0 / fps if fps > 0 else 0.0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    fails += 1
                    if fails >= self.max_read_fails:
                        if not self._is_live():
                            logger.info("视频源读取结束 (本地文件已播完)")
                            self._put_latest(None)  # 结束哨兵
                            cap.release()
                            return
                        logger.warning("连续 %d 帧读取失败, 尝试重连视频源...", fails)
                        break
                    time.sleep(0.02)
                    continue
                fails = 0
                self.frames_read += 1
                if self.frames_read % self.stride == 0:
                    self._put_latest(FrameItem(frame, self.frames_read, time.time()))
                if frame_delay > 0:
                    time.sleep(frame_delay)
            cap.release()
            if self._stop.wait(2.0):
                break
        # 兜底哨兵 (打开失败/被停止等情况)
        if not self._stop.is_set():
            try:
                self.queue.put_nowait(None)
            except queue.Full:
                pass

    def stop(self) -> None:
        self._stop.set()
