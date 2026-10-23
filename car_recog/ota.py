"""OTA 向量热更新: 新品牌特征零停机注入.

通道 (可同时启用):
  1) 本地目录轮询: 云端/上位机把 *.npz 增量包放入 ota_dir, 边缘端自动发现并加载;
  2) MQTT 推送: 订阅 ota 主题, 载荷 {"brand":..., "vectors_b64":..., "version":...}。

包格式 (npz): vectors (N,dim) float32 / labels (N,) str / version int / created_at str

"零停机"保证: 注入仅调用 FaissRetriever.add_vectors() (RLock 保护),
不重建索引、不重训模型、不重启进程, 新品牌即可参与下一帧检索。
"""
from __future__ import annotations

import base64
import json
import logging
import threading
from pathlib import Path

import numpy as np

from .utils import now_str

logger = logging.getLogger("car-recog.ota")


class OtaManager(threading.Thread):
    def __init__(self, retriever, ota_dir: str | Path, poll_interval: float = 5.0,
                 on_ack=None, save_dir: str | Path | None = None):
        super().__init__(daemon=True, name="ota-manager")
        self.retriever = retriever
        self.ota_dir = Path(ota_dir)
        self.poll_interval = poll_interval
        self.on_ack = on_ack  # (brand, count, version) -> None
        self.save_dir = Path(save_dir) if save_dir else None  # 注入后回写索引, 断电/重启不丢
        self._stop = threading.Event()
        self._processed: dict[str, tuple] = {}  # 文件 -> (mtime, size), 防重复加载
        self.total_injected = 0

    def _persist(self) -> None:
        """把注入后的索引持久化到磁盘, 保证边缘端重启后 OTA 向量不丢失."""
        if self.save_dir is None:
            return
        try:
            self.retriever.save(self.save_dir)
            logger.info("[OTA] 索引已持久化到 %s (v%d, %d 条向量)",
                        self.save_dir, self.retriever.version, self.retriever.size)
        except Exception as e:
            logger.error("OTA 索引持久化失败: %s", e)

    def run(self) -> None:
        self.ota_dir.mkdir(parents=True, exist_ok=True)
        logger.info("OTA 热更新线程已启动, 监听目录: %s", self.ota_dir)
        while not self._stop.wait(self.poll_interval):
            try:
                self.scan_dir()
            except Exception as e:
                logger.error("OTA 目录扫描异常: %s", e)

    def scan_dir(self) -> None:
        for f in sorted(self.ota_dir.glob("*.npz")):
            stat = (f.stat().st_mtime, f.stat().st_size)
            if self._processed.get(str(f)) == stat:
                continue
            try:
                count, version, brands = self.ingest_pack(f)
                self.total_injected += count
                logger.info("[OTA] 已热更新 %d 条向量 (版本 v%d), 品牌: %s, 进程未重启",
                            count, version, ", ".join(sorted(set(brands))))
                self._persist()
                # 归档已处理增量包, 避免边缘端重启后重复注入
                archive = self.ota_dir / "processed"
                archive.mkdir(exist_ok=True)
                f.replace(archive / f.name)
                if self.on_ack:
                    self.on_ack(brands, count, version)
            except Exception as e:
                logger.error("OTA 包加载失败 %s: %s", f.name, e)
            self._processed[str(f)] = stat

    def ingest_pack(self, path: str | Path) -> tuple[int, int, list[str]]:
        data = np.load(path, allow_pickle=False)
        vectors = np.asarray(data["vectors"], dtype=np.float32)
        labels = [str(x) for x in data["labels"]]
        version = int(data["version"]) if "version" in data else None
        ids, ver = self.retriever.add_vectors(vectors, labels, source="ota")
        return len(ids), ver, labels

    def ingest_mqtt_payload(self, payload: dict) -> int:
        """处理 MQTT 推送的 OTA 包: {"brand","vectors_b64","version"}(可含 "dim")."""
        brand = str(payload.get("brand", "ota_unknown"))
        raw = base64.b64decode(payload["vectors_b64"])
        dim = int(payload.get("dim", self.retriever.dim))
        vectors = np.frombuffer(raw, dtype=np.float32).reshape(-1, dim)
        labels = [brand] * len(vectors)
        ids, ver = self.retriever.add_vectors(vectors, labels, source="ota")
        self.total_injected += len(ids)
        logger.info("[OTA] MQTT 增量包已加载: %s x%d (版本 v%d)", brand, len(ids), ver)
        self._persist()
        if self.on_ack:
            self.on_ack(labels, len(ids), ver)
        return len(ids)

    def stop(self) -> None:
        self._stop.set()


def build_ota_pack(vectors: np.ndarray, brand: str, out_path: str | Path,
                   version: int = 1) -> Path:
    """构建一个 OTA 增量包 (供云端/上位机侧工具使用, 见 simulate_ota_push.py)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.array([brand] * len(vectors))
    np.savez(out_path, vectors=np.asarray(vectors, dtype=np.float32),
             labels=labels, version=int(version), created_at=now_str())
    return out_path
