"""通用工具: 日志 / 随机种子 / 设备选择 / 图像预处理 / FPS 统计."""
from __future__ import annotations

import collections
import datetime
import logging
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ImageNet 统计量, 训练与推理共用, 保证预处理一致性
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _torch():
    """torch 按需导入: 数据管线等多进程工具不需要 torch,
    避免每个子进程付出数秒的 torch 加载开销 (Windows spawn 下尤其明显)."""
    import torch
    return torch


def setup_logger(name: str = "car-recog", level: int = logging.INFO,
                 log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    # handler 挂到命名空间根 (如 "car-recog") 而非叶子 logger:
    # 兄弟模块 (car-recog.ota 等) 用裸 getLogger 创建, 依赖 propagate 到根,
    # 若根没有 handler 其日志会被静默丢弃.
    base = logging.getLogger(name.split(".", 1)[0])
    if not base.handlers:
        base.setLevel(level)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        base.addHandler(sh)
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            base.addHandler(fh)
    return logger


def seed_everything(seed: int) -> None:
    torch = _torch()
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preference: str = "auto"):
    torch = _torch()
    if preference in ("cpu", "cuda"):
        return torch.device(preference)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2 归一化, 兼容 (N,d) 与 (d,) 两种形状; 零向量保持不变."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        return x / norm if norm > 0 else x
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def preprocess_frame(frame_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    """BGR 帧 -> (1,3,size,size) float32 归一化张量, 与训练预处理保持一致.

    说明: 输入 ROI 已经过边界框裁剪, 采用直接缩放 (允许轻微形变),
    这是细粒度车辆识别任务的标准做法。
    """
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    return np.ascontiguousarray(img)[None]


def now_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class FPSMeter:
    """滑动窗口 FPS 统计."""

    def __init__(self, window: int = 50):
        self._times: collections.deque = collections.deque(maxlen=window)

    def tick(self) -> float | None:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) < 2:
            return None
        return (len(self._times) - 1) / (now - self._times[0])
