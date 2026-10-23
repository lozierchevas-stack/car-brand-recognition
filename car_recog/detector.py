"""可选模块: YOLOv8(ONNX) 车辆检测 —— 先检测车辆 ROI, 再送品牌识别.

模型导出 (ultralytics): yolo export model=yolov8n.pt format=onnx imgsz=640 opset=17
输出张量形状 (1, 4+80, 8400), 本模块手工完成解码与 NMS,
边缘端运行零 ultralytics 依赖 (仅 onnxruntime)。

配置 edge.detection.enabled=false 时跳过本模块, 直接对整帧提取特征
(适用于已对准车位的固定机位场景)。
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger("car-recog.detector")

# COCO 车辆相关类别
COCO_VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}


class YoloV8VehicleDetector:
    def __init__(self, onnx_path: str, conf_thres: float = 0.35, iou_thres: float = 0.45,
                 classes=(2, 5, 7), imgsz: int = 640):
        import onnxruntime as ort
        available = ort.get_available_providers()
        providers = [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider",
                                 "CPUExecutionProvider") if p in available]
        self.sess = ort.InferenceSession(str(onnx_path), providers=providers)
        self.in_name = self.sess.get_inputs()[0].name
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = set(classes)
        self.imgsz = imgsz
        logger.info("YOLOv8 车辆检测器已加载: %s (后端 %s)", onnx_path, providers)

    def detect(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """返回 [(x1,y1,x2,y2,conf), ...] 原图坐标系."""
        img, ratio, pad = self._letterbox(frame_bgr, self.imgsz)
        blob = img.transpose(2, 0, 1)[None]  # (1,3,640,640)
        pred = self.sess.run(None, {self.in_name: blob})[0]  # (1, 4+nc, 8400)
        pred = np.squeeze(pred, 0).T  # (8400, 4+nc)
        boxes_xywh = pred[:, :4]
        scores = pred[:, 4:]

        cls_ids = scores.argmax(axis=1)
        confs = scores.max(axis=1)
        keep = np.where((confs >= self.conf_thres) &
                        np.isin(cls_ids, list(self.classes)))[0]
        if len(keep) == 0:
            return []

        boxes = boxes_xywh[keep]
        xyxy = np.stack([
            boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
            boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], axis=1)
        xyxy = self._unletterbox(xyxy, ratio, pad)

        order = confs[keep].argsort()[::-1]
        picks = self._nms(xyxy[order], confs[keep][order], self.iou_thres)
        h, w = frame_bgr.shape[:2]
        results = []
        for p in picks:
            x1, y1, x2, y2 = xyxy[order][p]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if x2 - x1 >= 8 and y2 - y1 >= 8:
                results.append((x1, y1, x2, y2, float(confs[keep][order][p])))
        return results

    def _letterbox(self, img: np.ndarray, size: int):
        """等比缩放 + 灰边填充 (与 ultralytics 导出模型对齐)."""
        h, w = img.shape[:2]
        r = min(size / h, size / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        return canvas.astype(np.float32) / 255.0, r, (pad_x, pad_y)

    @staticmethod
    def _unletterbox(xyxy: np.ndarray, ratio: float, pad: tuple) -> np.ndarray:
        return (xyxy - np.array([pad[0], pad[1], pad[0], pad[1]])) / ratio

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
        x1, y1 = boxes[:, 0], boxes[:, 1]
        x2, y2 = boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order_idx = scores.argsort()[::-1]
        keep = []
        while order_idx.size > 0:
            i = order_idx[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order_idx[1:]])
            yy1 = np.maximum(y1[i], y1[order_idx[1:]])
            xx2 = np.minimum(x2[i], x2[order_idx[1:]])
            yy2 = np.minimum(y2[i], y2[order_idx[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order_idx[1:]] - inter + 1e-6)
            order_idx = order_idx[1:][iou <= iou_thres]
        return keep
