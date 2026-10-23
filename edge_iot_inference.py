#!/usr/bin/env python3
"""边缘端主控逻辑 (核心交付 3): 多线程抽帧 -> Visual RAG 双重阈值检索 -> MQTT 硬件联动 -> 难例采集.

线程模型 (生产者-消费者):
  VideoSource 线程      —— 摄像头/RTSP/视频文件读取, 每 stride 帧入队一次 (队列满丢最旧, 防卡顿)
  主线程 (消费者)        —— 推理 + Faiss 检索 + 双重阈值决策 + 渲染
  MqttBridge 后台线程    —— 识别结果上报 / 云端命令订阅 / 断线重连补发
  OtaManager 线程       —— 轮询 ota_dir 增量包, 新品牌特征零停机注入
  HardCaseMiner 线程    —— 难例图像异步落盘 (JSONL 元数据, 可直接灌入 Hive/Flink)

业务闭环:
  识别成功      -> MQTT 上报 (OneNET 数据点) + 白名单品牌联动硬件 (如智能道闸抬杆)
  拒识/临界置信 -> HardCaseMiner 采集难例 (主动学习), 回流云端供下一轮模型进化

用法:
  python edge_iot_inference.py --source 0                      # 摄像头
  python edge_iot_inference.py --source rtsp://ip/stream       # RTSP 网络流
  python edge_iot_inference.py --source demo.mp4 --no-mqtt     # 本地视频 (无云)
"""
from __future__ import annotations

import argparse
import json
import logging
import queue
import socket
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from car_recog import utils
from car_recog.config import load_config
from car_recog.detector import YoloV8VehicleDetector
from car_recog.hardware import build_hardware_controller
from car_recog.inference import create_embedder
from car_recog.mqtt import MqttBridge
from car_recog.ota import OtaManager
from car_recog.retrieval import FaissRetriever
from car_recog.video import VideoSource

logger = utils.setup_logger("car-recog.edge")


# ======================================================================
# 难例采集 (Active Learning)
# ======================================================================

class HardCaseMiner:
    """异步难例采集器: 拒识帧/临界置信帧的 ROI 落盘 + JSONL 元数据流.

    JSONL 每行一条记录, 与 Hive/Flink 数仓直接兼容:
    {"device_id","brand_guess","similarity","decision","hard_reason","timestamp","image_path"}
    """

    def __init__(self, out_dir: str | Path, publish_fn=None, max_queue: int = 256,
                 jpeg_quality: int = 92):
        self.out_dir = Path(out_dir)
        self.publish_fn = publish_fn
        self.jpeg_quality = jpeg_quality
        self.queue: queue.Queue = queue.Queue(maxsize=max(1, max_queue))
        self.saved = 0
        self.dropped = 0
        self._stop = False
        self._thread = None

    def start(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run,
                                        daemon=True, name="hardcase-writer")
        self._thread.start()

    def submit(self, crop_bgr: np.ndarray, meta: dict) -> None:
        """异步提交难例 (主推理线程不做任何磁盘 IO)."""
        try:
            self.queue.put_nowait((crop_bgr, meta))
        except queue.Full:
            self.dropped += 1
            logger.warning("难例队列已满, 丢弃 1 帧 (累计 %d)", self.dropped)

    def _run(self) -> None:
        meta_path = self.out_dir / "hard_cases.jsonl"
        while not self._stop:
            try:
                crop, meta = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                day = datetime.now().strftime("%Y-%m-%d")
                img_dir = self.out_dir / day
                img_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                fname = f"{ts}_{meta.get('brand', 'UNKNOWN')}.jpg"
                if not cv2.imwrite(str(img_dir / fname), crop,
                                   [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]):
                    logger.error("难例落盘失败: %s", fname)
                    continue
                meta = {**meta, "image_path": str((img_dir / fname).relative_to(self.out_dir)),
                        "saved_at": utils.now_str()}
                with open(meta_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(meta, ensure_ascii=False) + "\n")
                self.saved += 1
                if self.publish_fn:
                    try:
                        self.publish_fn(meta)
                    except Exception as e:
                        logger.warning("难例上报失败: %s", e)
            except Exception as e:
                logger.error("难例处理异常: %s", e)

    def close(self) -> None:
        self._stop = True
        if self._thread:
            self._thread.join(timeout=3.0)


# ======================================================================
# 边缘端应用
# ======================================================================

class EdgeApp:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.embedder = create_embedder(cfg, onnx_override=args.onnx)
        self.retriever = self._load_retriever()

        # 可选: YOLOv8 车辆检测 (先检测 ROI 再识别)
        self.detector = None
        det_cfg = cfg.edge.detection
        if det_cfg.enabled and Path(det_cfg.model_path).is_file():
            self.detector = YoloV8VehicleDetector(
                det_cfg.model_path, conf_thres=det_cfg.conf_thres,
                iou_thres=det_cfg.iou_thres, classes=list(det_cfg.classes))
        elif det_cfg.enabled:
            logger.warning("检测模型不存在 (%s), 回退整帧识别模式", det_cfg.model_path)

        # 硬件执行机构 (仿真 / 串口)
        self.hardware = build_hardware_controller(cfg)

        # MQTT 云平台桥 (结果上报 + 命令订阅 + OTA 订阅)
        self.mqtt = None
        if cfg.mqtt.enabled and not args.no_mqtt:
            self.mqtt = MqttBridge(cfg, on_command=self.on_cloud_command,
                                   on_ota=self.on_ota_payload)

        # OTA 热更新线程 (注入后回写索引目录, 重启不丢)
        self.ota = None
        if cfg.ota.enabled:
            index_dir = Path(self.args.index_dir or cfg.retrieval.index_dir)
            self.ota = OtaManager(self.retriever, cfg.paths.ota_dir,
                                  poll_interval=cfg.ota.poll_interval_s,
                                  on_ack=self.on_ota_ack,
                                  save_dir=index_dir if (index_dir / "index.faiss").is_file() else None)

        # 难例采集器
        self.miner = None
        if cfg.hard_case.enabled:
            publish_fn = None
            if cfg.hard_case.publish_mqtt and self.mqtt:
                publish_fn = self.mqtt.publish_hardcase
            self.miner = HardCaseMiner(cfg.paths.hard_case_dir, publish_fn=publish_fn,
                                       max_queue=cfg.hard_case.max_queue,
                                       jpeg_quality=cfg.hard_case.jpeg_quality)

        self.device_id = (cfg.mqtt.onenet.device_name or socket.gethostname())
        self.whitelist = set(cfg.edge.whitelist or [])
        self.gate_cooldown = float(cfg.edge.gate_cooldown_s or 5.0)
        self._last_gate_ts = 0.0
        self.stats = {"frames": 0, "accepted": 0, "rejected": 0,
                      "brands": Counter()}
        self.display = bool(cfg.edge.display) and not args.no_display
        self.topk = int(cfg.retrieval.topk or 5)

    # ---------- 检索库加载 ----------

    def _load_retriever(self) -> FaissRetriever:
        r_cfg = self.cfg.retrieval
        index_dir = Path(self.args.index_dir or r_cfg.index_dir)
        thresholds = {"min_sim": r_cfg.min_sim, "margin": r_cfg.margin,
                      "critical_band": r_cfg.critical_band}
        if (index_dir / "index.faiss").is_file():
            ret = FaissRetriever.load(index_dir, **thresholds)
            logger.info("Faiss 知识库已加载: %s (v%d, %d 条向量, %d 个品牌)",
                        index_dir, ret.version, ret.size, len(ret.brands()))
            return ret
        # 回退: 从训练检查点携带的类中心 gallery 直接构建索引
        ckpt_path = Path(self.args.checkpoint or
                         self.cfg.paths.artifacts_dir + "/checkpoints/best.pth")
        if ckpt_path.is_file():
            import torch
            gallery = torch.load(ckpt_path, map_location="cpu",
                                 weights_only=False)["gallery"]
            ret = FaissRetriever(r_cfg.dim, **thresholds)
            ret.add_vectors(gallery["vectors"], gallery["brands"], source="checkpoint")
            logger.info("已从检查点 gallery 构建索引 (%d 条向量)", ret.size)
            return ret
        raise SystemExit(
            f"未找到 Faiss 索引: {index_dir}\n请先运行: python build_gallery.py "
            f"--onnx {self.cfg.edge.embedder}")

    # ---------- 云端命令 / OTA 回调 ----------

    def on_cloud_command(self, cmd: dict, reply) -> None:
        """处理云端下发指令 (OneNET cmd/request), 联动底层硬件执行机构."""
        action = str(cmd.get("action") or cmd.get("cmd") or "")
        logger.info("收到云端指令: %s", cmd)
        try:
            if action == "gate":
                value = str(cmd.get("value", "toggle"))
                ok = self.hardware.gate_open(int(cmd.get("duration_ms", 3000))) \
                    if value == "open" else self.hardware.gate_close()
                reply(200, "gate ok" if ok else "gate failed")
            elif action == "ota" and self.ota:
                count = self.ota.ingest_mqtt_payload(cmd)
                reply(200, f"ota injected {count}")
            elif action == "ping":
                reply(200, "pong")
            else:
                reply(400, f"unknown action: {action}")
        except Exception as e:
            logger.error("指令执行异常: %s", e)
            reply(500, str(e))

    def on_ota_payload(self, payload: dict) -> None:
        if self.ota:
            self.ota.ingest_mqtt_payload(payload)

    def on_ota_ack(self, brands, count, version) -> None:
        if self.mqtt:
            self.mqtt.publish_result({"ota_ack": f"v{version}",
                                      "ota_brand": ",".join(sorted(set(brands)))})

    # ---------- 推理与决策 ----------

    def process_frame(self, frame: np.ndarray) -> list:
        """整帧/多 ROI 特征提取 + 双重阈值检索, 返回 [(box|None, MatchResult), ...]."""
        if self.detector is not None:
            boxes = self.detector.detect(frame)
            if not boxes:
                return []
            rois = [frame[y1:y2, x1:x2] for x1, y1, x2, y2, _ in boxes]
            embeds = self.embedder.embed_batch(rois)
            return list(zip(boxes, [self.retriever.match(e, self.topk) for e in embeds]))
        embeds = self.embedder.embed_batch([frame])
        return [(None, self.retriever.match(embeds[0], self.topk))]

    def handle_decision(self, frame: np.ndarray, box, m, frame_idx: int, ts: float) -> None:
        """决策分发: 接受 -> 上报+硬件联动; 拒识/临界 -> 难例采集."""
        brand = m.brand or "UNKNOWN"
        result = {
            "device_id": self.device_id,
            "brand": brand,
            "confidence": round(m.confidence, 4),
            "similarity": round(m.similarity, 4),
            "decision": "accept" if m.accepted else "reject",
            "reason": m.reason,
            "timestamp": ts,
            "frame_idx": frame_idx,
        }
        if m.accepted:
            self.stats["accepted"] += 1
            self.stats["brands"][brand] += 1
            if self.mqtt:
                self.mqtt.publish_result({**result, "confidence": m.confidence})
            # 白名单品牌 -> 联动硬件 (如智能道闸抬杆)
            if brand in self.whitelist and \
                    time.time() - self._last_gate_ts >= self.gate_cooldown:
                if self.hardware.gate_open(int(self.cfg.edge.gate_duration_ms or 3000)):
                    self._last_gate_ts = time.time()
                    logger.info("白名单品牌 [%s] 命中, 已联动道闸抬杆", brand)
        else:
            self.stats["rejected"] += 1

        # 难例采集: 直接拒识 或 落在临界置信带 (主动学习数据源)
        if self.miner and (not m.accepted or self.retriever.is_critical(m.similarity)):
            crop = frame
            if box is not None:
                x1, y1, x2, y2 = box[:4]
                crop = frame[y1:y2, x1:x2]
            self.miner.submit(crop, {**result,
                                     "hard_reason": m.reason if not m.accepted else "critical_band"})

    # ---------- 渲染 ----------

    def draw_overlay(self, frame: np.ndarray, detections: list, fps) -> np.ndarray:
        for box, m in detections:
            x1, y1, x2, y2 = box[:4] if box else (0, 0, frame.shape[1], frame.shape[0])
            if m.accepted:
                color, label = (0, 200, 0), f"{m.brand} {m.confidence:.2f}"
            elif m.reason == "ambiguous":
                color, label = (0, 165, 255), f"AMBIG {m.brand} {m.similarity:.2f}"
            else:
                color, label = (0, 0, 255), f"UNKNOWN {m.similarity:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(16, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"FPS {fps:.1f}" if fps else "FPS --",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame

    # ---------- 主循环 ----------

    def run(self) -> None:
        producer = VideoSource(self.args.source or self.cfg.edge.source,
                               stride=self.args.stride or self.cfg.edge.stride,
                               queue_size=self.cfg.edge.queue_size)
        producer.start()
        if self.mqtt:
            self.mqtt.start()
        if self.ota:
            self.ota.start()
        if self.miner:
            self.miner.start()
        fps_meter = utils.FPSMeter()
        t_start = time.time()
        logger.info("边缘端主控已启动 (source=%s, stride=%d, 品牌数=%d, 阈值: min_sim=%.2f margin=%.2f)",
                    producer.source, producer.stride,
                    len(self.retriever.brands()), self.retriever.min_sim, self.retriever.margin)

        try:
            while True:
                try:
                    item = producer.queue.get(timeout=0.5)
                except queue.Empty:
                    if not producer.is_alive():
                        break
                    continue
                if item is None:  # 视频源结束哨兵
                    logger.info("视频源已结束, 退出主循环")
                    break
                frame, frame_idx, ts = item
                detections = self.process_frame(frame)
                for box, m in detections:
                    self.handle_decision(frame, box, m, frame_idx, ts)
                self.stats["frames"] += 1
                fps = fps_meter.tick()

                if self.display:
                    try:
                        canvas = self.draw_overlay(frame, detections, fps)
                        cv2.imshow("Car Brand Recognition - Edge", canvas)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            logger.info("用户按下 q, 退出")
                            break
                    except cv2.error as e:
                        logger.warning("显示异常 (%s), 关闭渲染继续运行", e)
                        self.display = False

                if self.stats["frames"] % 100 == 0:
                    logger.info(
                        "运行统计 | 帧 %d | 接受 %d / 拒识 %d | 难例 %d | "
                        "FPS %s | 队列 %d/%d | 索引 v%d",
                        self.stats["frames"], self.stats["accepted"], self.stats["rejected"],
                        self.miner.saved if self.miner else 0,
                        f"{fps:.1f}" if fps else "--",
                        producer.queue.qsize(), producer.queue.maxsize,
                        self.retriever.version)
        except KeyboardInterrupt:
            logger.info("收到中断信号, 安全退出中...")
        finally:
            # 清理阶段屏蔽二次 Ctrl+C, 避免退出流程被打断产生 traceback
            import signal as _signal
            old_handler = _signal.signal(_signal.SIGINT, _signal.SIG_IGN)
            try:
                producer.stop()
                if self.mqtt:
                    self.mqtt.stop()
                if self.ota:
                    self.ota.stop()
                if self.miner:
                    self.miner.close()
                self.hardware.close()
            finally:
                _signal.signal(_signal.SIGINT, old_handler)
            try:
                cv2.destroyAllWindows()
            except cv2.error:  # headless 版 OpenCV 无 highgui, 忽略
                pass

            elapsed = time.time() - t_start
            logger.info("运行时长 %.1f s | 处理帧 %d | 接受 %d / 拒识 %d | "
                        "难例采集 %d (丢弃 %d) | 生产者丢帧 %d | OTA 注入 %d",
                        elapsed, self.stats["frames"], self.stats["accepted"],
                        self.stats["rejected"],
                        self.miner.saved if self.miner else 0,
                        self.miner.dropped if self.miner else 0,
                        producer.dropped,
                        self.ota.total_injected if self.ota else 0)
            if self.stats["brands"]:
                top = self.stats["brands"].most_common(5)
                logger.info("识别品牌 Top5: %s",
                            ", ".join(f"{b}x{c}" for b, c in top))


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="边缘端主控: 抽帧 + Visual RAG 检索 + MQTT 联动 + 难例采集")
    parser.add_argument("--config", default=None, help="配置文件 (默认 configs/default.yaml)")
    parser.add_argument("--source", default=None, help="视频源: 0/RTSP URL/视频文件路径")
    parser.add_argument("--onnx", default=None, help="覆盖 ONNX 嵌入模型路径")
    parser.add_argument("--index-dir", default=None, help="覆盖 Faiss 索引目录")
    parser.add_argument("--checkpoint", default="", help="索引缺失时从训练检查点 gallery 构建")
    parser.add_argument("--stride", type=int, default=None, help="抽帧间隔 (默认 5)")
    parser.add_argument("--no-display", action="store_true", help="无界面模式 (无头服务器)")
    parser.add_argument("--no-mqtt", action="store_true", help="禁用 MQTT 云通信")
    parser.add_argument("--log-level", default=None, help="日志级别 DEBUG/INFO/WARNING")
    args = parser.parse_args()

    cfg = load_config(args.config)
    level = getattr(logging, (args.log_level or cfg.project.log_level or "INFO").upper())
    logger.setLevel(level)
    utils.seed_everything(cfg.project.seed)

    app = EdgeApp(cfg, args)
    app.run()


if __name__ == "__main__":
    main()
