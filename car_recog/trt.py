"""TensorRT 引擎构建与推理执行 (可选依赖: tensorrt / pycuda).

- build_engine(): ONNX -> TensorRT engine, 支持 FP16 与动态 Batch 优化配置;
- TRTEngineRunner: 提供与 OnnxEmbedder 一致的 embed_batch() 接口,
  供 edge_iot_inference.py / build_gallery.py 无缝切换。

安装: pip install tensorrt pycuda  (Windows 需先安装对应版本的 NVIDIA TensorRT SDK 与 CUDA)
注: 若不想手写 TRT Runtime 调用, 也可让 onnxruntime 使用 TensorrtExecutionProvider,
    OnnxEmbedder 已按优先级自动探测该后端。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .utils import l2_normalize, preprocess_frame

logger = logging.getLogger("car-recog.trt")


def _import_trt():
    """导入 TensorRT: 优先官方 tensorrt 顶层包; Windows 上 NVIDIA 只发布
    tensorrt-cu12-bindings + tensorrt-cu12-libs 轮子 (顶层包缺位), 回退 bindings."""
    try:
        import tensorrt as trt
    except ImportError:
        try:
            import tensorrt_bindings as trt
        except ImportError:
            raise SystemExit(
                "未安装 TensorRT: pip install --extra-index-url https://pypi.nvidia.com "
                "tensorrt-cu12-bindings tensorrt-cu12-libs (或安装 NVIDIA TensorRT SDK)")
    return trt


def build_engine(onnx_path: str | Path, engine_path: str | Path, fp16: bool = True,
                 workspace_gb: float = 2, min_batch: int = 1, opt_batch: int = 1,
                 max_batch: int = 16) -> Path:
    """将 ONNX 模型编译为 TensorRT engine 并落盘.

    FP16 双路径:
      - TRT <= 10: BuilderFlag.FP16 (构建期由 TRT 选低精度实现);
      - TRT >= 11: 精度标志已移除 (强类型网络), 需离线把 ONNX 图转 FP16
        (图 I/O 保持 FP32) 后再构建, 引擎精度由图的 dtype 决定。
    """
    trt = _import_trt()
    onnx_path, engine_path = Path(onnx_path), Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    # 外部数据格式 (.onnx + .onnx.data) 下 TRT 按 CWD 解析权重路径, 先转单文件再解析
    work_path, tmp_path = onnx_path, None
    if Path(str(onnx_path) + ".data").is_file():
        import onnx as _onnx
        tmp_path = engine_path.parent / (onnx_path.stem + "_selfcontained.onnx")
        model = _onnx.load(onnx_path, load_external_data=True)
        _onnx.save_model(model, tmp_path, save_as_external_data=False)
        work_path = tmp_path
        logger.info("外部权重已并入单文件 ONNX: %s", tmp_path)
    legacy_fp16 = hasattr(trt.BuilderFlag, "FP16")
    try:
        if fp16 and not legacy_fp16:
            import onnx as _onnx
            from onnxconverter_common import float16 as _f16
            model = _onnx.load(work_path, load_external_data=False)
            model = _f16.convert_float_to_float16(model, keep_io_types=True)
            fp16_path = work_path.with_name(work_path.stem + "_fp16.onnx")
            _onnx.save_model(model, fp16_path, save_as_external_data=False)
            work_path = fp16_path
            logger.info("TRT >= 11: ONNX 图已离线转 FP16 (I/O 保持 FP32): %s", fp16_path)
        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        # TRT >= 10 网络默认显式批, 该标志在 TRT 11 中已移除 (仅旧版兼容)
        flag = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", 0)
        network = builder.create_network(1 << int(flag))
        parser = trt.OnnxParser(network, trt_logger)
        with open(work_path, "rb") as f:
            if not parser.parse(f.read()):
                errors = "\n".join(str(parser.get_error(i))
                                   for i in range(parser.num_errors))
                raise RuntimeError(f"ONNX 解析失败:\n{errors}")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
        input_name = network.get_input(0).name
        profile = builder.create_optimization_profile()
        profile.set_shape(input_name,
                          (min_batch, 3, 224, 224), (opt_batch, 3, 224, 224), (max_batch, 3, 224, 224))
        config.add_optimization_profile(profile)
        if fp16 and legacy_fp16:
            if getattr(builder, "platform_has_fast_fp16", True):
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("已启用 FP16 精度 (TRT <= 10 builder 标志)")
            else:
                logger.warning("当前 GPU 不支持 FP16, 使用 FP32 构建")

        logger.info("开始构建 TensorRT 引擎 (首次构建可能需要数分钟)...")
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT 引擎构建失败")
        with open(engine_path, "wb") as f:
            f.write(bytes(serialized))
    finally:
        for p in (tmp_path, work_path if work_path != onnx_path else None):
            if p is not None:
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
    logger.info("TensorRT 引擎已保存: %s (%.1f MB)", engine_path,
                engine_path.stat().st_size / (1 << 20))
    return engine_path


class TRTEngineRunner:
    """TensorRT 引擎推理器 (静态 Batch=1 引擎), 与 OnnxEmbedder 同接口.

    CUDA 侧使用官方 cuda-python 绑定 (pycuda 无 Windows 轮子).
    """

    def __init__(self, engine_path: str | Path):
        try:
            trt = _import_trt()
            from cuda.bindings import runtime as rt
        except SystemExit:
            raise
        except ImportError:
            raise SystemExit(
                "TensorRT 推理需要 tensorrt + cuda-python: "
                "pip install --extra-index-url https://pypi.nvidia.com "
                "tensorrt-cu12-bindings tensorrt-cu12-libs && pip install cuda-python")
        self._rt = rt
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"引擎反序列化失败: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        in_shape = self.engine.get_tensor_shape(self.input_name)
        out_shape = self.engine.get_tensor_shape(self.output_name)
        self.size = int(in_shape[2])
        self.out_dim = int(out_shape[1])
        # 动态批引擎 (Batch=-1): 取优化档 max batch 作为缓冲上限, 运行时 set_input_shape
        self._dynamic = in_shape[0] < 0
        self._last_n = 0
        if self._dynamic:
            if hasattr(self.engine, "get_tensor_profile_shape"):  # TRT >= 11: (name, profile)
                max_shape = self.engine.get_tensor_profile_shape(self.input_name, 0)[2]
            else:  # TRT <= 10: (profile, name)
                max_shape = self.engine.get_profile_shape(0, self.input_name)[2]
            in_shape = [max_shape[0], in_shape[1], in_shape[2], in_shape[3]]
            out_shape = [max_shape[0], out_shape[1]]
        elif in_shape[0] != 1:
            raise RuntimeError(f"引擎 Batch={in_shape[0]}, 本运行器仅支持 Batch=1 或动态批引擎")
        self._max_n = int(in_shape[0])
        # cuda-python 12.x 新式 API: 输出参数以 (err, value) 元组返回
        _, self.d_input = rt.cudaMalloc(int(np.prod(in_shape)) * 4)
        _, self.d_output = rt.cudaMalloc(int(np.prod(out_shape)) * 4)
        self.context.set_tensor_address(self.input_name, self.d_input)
        self.context.set_tensor_address(self.output_name, self.d_output)
        _, self._stream = rt.cudaStreamCreate()

    def _check(self, ret) -> None:
        err = ret[0] if isinstance(ret, tuple) else ret
        if err:
            msg = self._rt.cudaGetErrorString(err)
            if isinstance(msg, bytes):
                msg = msg.decode(errors="replace")
            raise RuntimeError(f"CUDA 调用失败 ({err}): {msg}")

    def embed_batch(self, frames_bgr) -> np.ndarray:
        rt = self._rt
        kinds = rt.cudaMemcpyKind
        outs = []
        # 超过引擎 max batch 时自动分块 (避免 exceedProfile 失败)
        for i in range(0, len(frames_bgr), self._max_n):
            chunk = frames_bgr[i:i + self._max_n]
            n = len(chunk)
            if self._dynamic and n != self._last_n:
                self.context.set_input_shape(self.input_name, (n, 3, self.size, self.size))
                self._last_n = n
            blob = np.stack([preprocess_frame(f, self.size)[0] for f in chunk])  # (n,3,H,W)
            self._check(rt.cudaMemcpyAsync(self.d_input, blob, blob.nbytes,
                                           kinds.cudaMemcpyHostToDevice, self._stream))
            if hasattr(self.context, "execute_async_v3"):  # TRT >= 8.5
                ok = self.context.execute_async_v3(self._stream)
            else:
                ok = self.context.execute_async_v2(self._stream)
            if not ok:
                raise RuntimeError("TensorRT 执行失败")
            out = np.empty((n, self.out_dim), dtype=np.float32)
            self._check(rt.cudaMemcpyAsync(out, self.d_output, out.nbytes,
                                           kinds.cudaMemcpyDeviceToHost, self._stream))
            self._check(rt.cudaStreamSynchronize(self._stream))
            outs.append(out)
        return l2_normalize(np.vstack(outs) if len(outs) > 1 else outs[0])

    def __del__(self):
        try:
            self._check(self._rt.cudaStreamSynchronize(self._stream))
        except Exception:
            pass
