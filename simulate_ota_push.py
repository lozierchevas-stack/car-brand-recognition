#!/usr/bin/env python3
"""OTA 模拟器: 扮演云端, 向边缘端下发"新品牌"增量特征向量 (零停机扩容演示).

模式:
  1) 真实图像: python simulate_ota_push.py --onnx embedding.onnx --images-dir 新车图片 --brand "Porsche_911"
  2) 随机演示 (无数据): python simulate_ota_push.py --onnx embedding.onnx --brand "NewBrand_X" --random
  3) MQTT 推流: 追加 --mqtt-host 127.0.0.1 --mqtt-topic car-recog/ota

生成 *.npz 增量包到 --out-dir (默认 data/ota_inbox), 边缘端 OtaManager 自动发现并
热加载, 全程不停机、不重训、不重启进程。
"""
from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

import numpy as np

from car_recog import utils
from car_recog.inference import OnnxEmbedder, embed_image_paths
from car_recog.ota import build_ota_pack

logger = utils.setup_logger("car-recog.ota-sim")


def main() -> None:
    parser = argparse.ArgumentParser(description="OTA 云端模拟: 下发新品牌增量特征向量")
    parser.add_argument("--onnx", default="", help="嵌入模型 (真实图像模式需要)")
    parser.add_argument("--images-dir", default="", help="新品牌图片目录")
    parser.add_argument("--brand", default="NewBrand", help="新品牌名称")
    parser.add_argument("--n", type=int, default=16, help="生成/采样向量数")
    parser.add_argument("--random", action="store_true", help="无数据随机演示模式")
    parser.add_argument("--out-dir", default="data/ota_inbox", help="增量包输出目录")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--mqtt-host", default="", help="直接推流到 MQTT Broker")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--mqtt-topic", default="car-recog/ota")
    args = parser.parse_args()

    rng = np.random.default_rng(42)

    if args.images_dir:
        if not args.onnx:
            raise SystemExit("真实图像模式需要 --onnx 嵌入模型")
        image_dir = Path(args.images_dir)
        paths = sorted(p for p in image_dir.iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
        if not paths:
            raise SystemExit(f"{image_dir} 下没有图像")
        embedder = OnnxEmbedder(args.onnx)
        sampled = [paths[i % len(paths)] for i in range(args.n)]
        vectors = embed_image_paths(embedder, sampled)
        if len(paths) < args.n:  # 图像不足时用轻微抖动扩充, 模拟同一品牌的外观变化
            extra = args.n - len(paths)
            vectors = np.vstack([vectors,
                                 vectors[:extra] + rng.normal(0, 0.03,
                                                              (extra, vectors.shape[1]))])
    else:
        if not args.onnx:
            raise SystemExit("随机演示模式也需要 --onnx (用于确定特征维度); "
                             "或提供 --images-dir 使用真实图像")
        # 模拟一个"未见品牌"的特征簇: 随机中心 + 类内抖动
        import onnxruntime as ort
        sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
        dim = int(sess.get_outputs()[0].shape[1])
        center = rng.normal(0, 1, (1, dim)).astype(np.float32)
        vectors = center + rng.normal(0, 0.35, (args.n, dim)).astype(np.float32)
        vectors = utils.l2_normalize(vectors)

    logger.info("生成 OTA 增量包: 品牌=%s, 向量=%d 条, 维度=%d",
                args.brand, len(vectors), vectors.shape[1])

    if args.mqtt_host:
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            raise SystemExit("MQTT 推流需要 paho-mqtt: pip install paho-mqtt")
        payload = {
            "brand": args.brand,
            "version": args.version,
            "dim": int(vectors.shape[1]),
            "vectors_b64": base64.b64encode(
                vectors.astype(np.float32).tobytes()).decode(),
        }
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.connect(args.mqtt_host, args.mqtt_port, keepalive=60)
        client.loop_start()
        client.publish(args.mqtt_topic, json.dumps(payload), qos=1)
        time.sleep(1.0)
        client.loop_stop()
        client.disconnect()
        logger.info("已推流到 %s:%s topic=%s", args.mqtt_host, args.mqtt_port, args.mqtt_topic)

    pack_path = Path(args.out_dir) / f"ota_{args.brand}_v{args.version}.npz"
    pack = build_ota_pack(vectors, args.brand, pack_path, version=args.version)
    logger.info("增量包已生成: %s (边缘端 OtaManager 将自动热加载)", pack)


if __name__ == "__main__":
    main()
