"""推理后端: ONNX Runtime / TensorRT 引擎的统一嵌入接口.

create_embedder() 按配置与运行环境自动选择后端:
  TensorRT engine (存在且依赖可用) -> ONNX Runtime (CUDA -> CPU 降级)。
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .utils import l2_normalize, preprocess_frame

logger = logging.getLogger("car-recog.inference")


class OnnxEmbedder:
    """ONNX Runtime 特征提取器, 接口: embed_batch(list_of_bgr_frames) -> (N,d) 归一化特征."""

    def __init__(self, onnx_path: str | Path, providers=None):
        import onnxruntime as ort
        available = ort.get_available_providers()
        if providers is None:
            providers = [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider",
                                     "CPUExecutionProvider") if p in available]
        self.sess = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [o.name for o in self.sess.get_outputs()]
        input_shape = self.sess.get_inputs()[0].shape  # (N,3,H,W), N 可为动态
        if len(input_shape) < 4 or not isinstance(input_shape[2], int):
            raise ValueError(f"ONNX 输入形状异常: {input_shape}")
        self.size = int(input_shape[2])
        self.active_providers = self.sess.get_providers()
        logger.info("ONNX Runtime 执行后端: %s", self.active_providers)

    def embed_batch(self, frames_bgr) -> np.ndarray:
        blob = np.stack([preprocess_frame(f, self.size)[0] for f in frames_bgr])
        blob = np.ascontiguousarray(blob, dtype=np.float32)
        out = self.sess.run(self.output_names, {self.input_name: blob})[0]
        return l2_normalize(np.asarray(out, dtype=np.float32))


def create_embedder(cfg, onnx_override: str | None = None):
    """按配置选择 TensorRT engine / ONNX 后端, 失败时给出清晰降级日志."""
    engine = cfg.edge.engine or ""
    if not engine and Path(cfg.trt.engine_path).is_file():
        engine = cfg.trt.engine_path
    if engine and Path(engine).is_file():
        try:
            from .trt import TRTEngineRunner
            logger.info("使用 TensorRT 引擎: %s", engine)
            return TRTEngineRunner(engine)
        except Exception as e:
            logger.warning("TensorRT 引擎加载失败 (%s), 回退 ONNX Runtime", e)
    onnx_path = onnx_override or cfg.edge.embedder or cfg.export.onnx_path
    if not Path(onnx_path).is_file():
        raise FileNotFoundError(
            f"未找到推理模型: {onnx_path}\n请先运行: python train_export.py export "
            f"--checkpoint artifacts/checkpoints/best.pth")
    return OnnxEmbedder(onnx_path)


def embed_image_paths(embedder, image_paths, chunk: int = 64) -> np.ndarray:
    """批量嵌入磁盘图像路径 (供 build_gallery 使用), 分块避免内存峰值."""
    vectors = []
    for i in range(0, len(image_paths), chunk):
        batch = []
        for p in image_paths[i:i + chunk]:
            frame = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"读取图像失败: {p}")
            batch.append(frame)
        vectors.append(embedder.embed_batch(batch))
    return np.vstack(vectors)
