#!/usr/bin/env python3
"""ONNX -> TensorRT (TRT) 引擎构建与基准测试 (推理加速部署).

用法:
  python export_trt.py --onnx artifacts/models/embedding.onnx --fp16
  python export_trt.py --onnx artifacts/models/embedding.onnx --benchmark --engine xxx.engine

依赖 (可选): pip install tensorrt pycuda
替代方案: 让 onnxruntime 走 TensorrtExecutionProvider (无需手写 TRT Runtime 调用),
          OnnxEmbedder 已按优先级自动探测该后端。
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from car_recog import utils
from car_recog.trt import build_engine

logger = utils.setup_logger("car-recog.trt-export")


def benchmark(engine_path: Path, frames: int = 128, warmup: int = 10) -> None:
    from car_recog.trt import TRTEngineRunner
    runner = TRTEngineRunner(engine_path)
    size = runner.size
    # 合成帧 (真实部署中替换为摄像头帧)
    synth = [np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
             for _ in range(warmup + frames)]
    for f in synth[:warmup]:
        runner.embed_batch([f])
    t0 = time.perf_counter()
    for f in synth[warmup:]:
        runner.embed_batch([f])
    dt = time.perf_counter() - t0
    logger.info("基准: %d 帧 / %.2f s | 吞吐 %.1f FPS | 平均延迟 %.2f ms",
                frames, dt, frames / dt, dt / frames * 1000)


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX -> TensorRT 引擎构建")
    parser.add_argument("--onnx", default="artifacts/models/embedding.onnx")
    parser.add_argument("--engine", default="artifacts/models/embedding.engine")
    parser.add_argument("--fp16", action="store_true", default=True, help="启用 FP16 (默认)")
    parser.add_argument("--fp32", action="store_true", help="使用 FP32 精度")
    parser.add_argument("--workspace-gb", type=float, default=2.0)
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=1)
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--benchmark", action="store_true", help="构建完成后运行基准测试")
    args = parser.parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        raise SystemExit(f"ONNX 模型不存在: {onnx_path}, 请先运行: python train_export.py export")
    engine_path = Path(args.engine)
    build_engine(onnx_path, engine_path, fp16=not args.fp32,
                 workspace_gb=args.workspace_gb, min_batch=args.min_batch,
                 opt_batch=args.opt_batch, max_batch=args.max_batch)
    print(f"\n引擎已就绪: {engine_path}")
    print("在 configs/default.yaml 中设置 edge.engine 指向该文件, 边缘端将优先使用 TensorRT 推理。")
    if args.benchmark:
        benchmark(engine_path)


if __name__ == "__main__":
    main()
