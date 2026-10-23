"""Visual RAG 检索核心: Faiss 向量知识库 + 双重阈值拒识 + OTA 热更新.

检索语义: 特征经 L2 归一化后, Faiss 内积 (IndexFlatIP) 等价于余弦相似度。

双重阈值拒识 (解决模型过度自信问题):
  1) top1 相似度 < min_sim          -> 特征空间中没有任何已知品牌足够接近
                                       -> 判为 "未知品牌/非目标车辆";
  2) top1 品牌与其余最佳品牌的相似度差 < margin (跨品牌间隔)
                                     -> 样本落在两个品牌决策边界附近, 判别不充分
                                       -> 判为 "模棱两可"。
任一触发即拒绝输出, 并将该帧转交难例采集 (数据飞轮入口)。

OTA: add_vectors() 只追加不改动, IndexIDMap2 保证 search 后可安全追加;
配合 RLock 实现"不停机、不重训"的秒级新品牌注册。
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np

from .utils import l2_normalize, now_str


@dataclass
class MatchResult:
    accepted: bool
    brand: str | None
    similarity: float
    confidence: float
    second_sim: float
    margin_ok: bool
    reason: str  # ok / low_similarity / ambiguous / empty_index
    topk: list = field(default_factory=list)  # [(brand, sim), ...]


def _sim_to_confidence(sim: float, min_sim: float) -> float:
    """相似度 -> 置信度的单调映射 (工程启发式):
    低于拒识阈值记 0; 从阈值到 1.0 线性映射到 [0.05, 1.0]."""
    if sim < min_sim:
        return 0.0
    return float(min(1.0, 0.05 + 0.95 * (sim - min_sim) / max(1e-6, 1.0 - min_sim)))


class FaissRetriever:
    """线程安全的内积检索器 (IndexIDMap2 + IndexFlatIP), 支持 OTA 增量写入与持久化."""

    def __init__(self, dim: int, min_sim: float = 0.55, margin: float = 0.10,
                 critical_band: float = 0.05):
        self.dim = dim
        self.min_sim = min_sim
        self.margin = margin
        self.critical_band = critical_band
        self._lock = threading.RLock()
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self._next_id = 0
        self._meta: dict[int, dict] = {}
        self._vectors: list[tuple[int, np.ndarray]] = []  # 保留向量以支持 compact 重建
        self.version = 0

    @property
    def size(self) -> int:
        return len(self._meta)

    def brands(self) -> list[str]:
        with self._lock:
            return sorted({m["brand"] for m in self._meta.values()})

    def add_vectors(self, vectors: np.ndarray, brands, source: str = "ota"):
        """追加特征向量 (自动 L2 归一化), 返回 (新增id列表, 新版本号)."""
        vectors = l2_normalize(np.asarray(vectors, dtype=np.float32))
        brands = [str(b) for b in brands]
        if len(vectors) != len(brands):
            raise ValueError(f"向量数 {len(vectors)} 与品牌标签数 {len(brands)} 不一致")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"特征维度 {vectors.shape[1]} 与索引维度 {self.dim} 不一致")
        with self._lock:
            self.version += 1
            ids = np.arange(self._next_id, self._next_id + len(vectors), dtype=np.int64)
            self._index.add_with_ids(vectors, ids)
            for j, (i, b) in enumerate(zip(ids.tolist(), brands)):
                self._meta[i] = {"brand": b, "source": source,
                                 "version": self.version, "added_at": now_str()}
                self._vectors.append((i, vectors[j].copy()))
            self._next_id += len(vectors)
        return ids.tolist(), self.version

    def search(self, query: np.ndarray, k: int = 5):
        """单条查询 -> (相似度数组, id数组). 空库返回空数组."""
        q = l2_normalize(query)
        if q.ndim == 1:
            q = q[None]
        with self._lock:
            if self.size == 0:
                return np.array([], np.float32), np.array([], np.int64)
            sims, ids = self._index.search(q, min(k, self.size))
        return sims[0], ids[0]

    def match(self, query: np.ndarray, k: int = 5) -> MatchResult:
        """双重阈值决策: 接受 / 低相似度拒识 / 模棱两可拒识.

        margin 语义: top1 所属品牌 vs 其他最佳品牌的相似度差 (跨品牌间隔)。
        知识库中每个品牌有多条向量, 同品牌样本天然占据 top 前若干位且彼此
        相似度接近 —— 若直接取 top1 与 top2 之差会把正确识别误判为模棱两可,
        因此 margin 只跨品牌计算。内部检索宽度取 max(k, 8) 以保证间隔可靠。
        """
        sims, ids = self.search(query, max(k, 8))
        if len(sims) == 0:
            return MatchResult(False, None, 0.0, 0.0, 0.0, False, "empty_index", [])
        top1 = float(sims[0])
        top1_brand = self._meta[int(ids[0])]["brand"] if int(ids[0]) in self._meta else None
        topk = [(self._meta[int(ids[i])]["brand"], float(sims[i]))
                for i in range(len(sims)) if int(ids[i]) in self._meta]
        conf = _sim_to_confidence(top1, self.min_sim)
        if top1 < self.min_sim:
            return MatchResult(False, topk[0][0] if topk else None, top1, conf,
                               float(sims[1]) if len(sims) > 1 else -1.0, False,
                               "low_similarity", topk)
        # 跨品牌间隔: 其余品牌中的最佳相似度
        best_other = -1.0
        for i in range(1, len(sims)):
            iid = int(ids[i])
            if iid in self._meta and self._meta[iid]["brand"] != top1_brand:
                best_other = float(sims[i])
                break
        margin_ok = (best_other < 0) or (top1 - best_other) >= self.margin
        if not margin_ok:
            return MatchResult(False, topk[0][0] if topk else None, top1, conf,
                               best_other, False, "ambiguous", topk)
        return MatchResult(True, topk[0][0], top1, conf, best_other, True, "ok", topk)

    def is_critical(self, similarity: float) -> bool:
        """是否处于临界置信带 (min_sim <= sim < min_sim + critical_band),
        临界样本是主动学习 (Active Learning) 的高价值难例."""
        return self.min_sim <= similarity < self.min_sim + self.critical_band

    def compact(self) -> None:
        """用保留的向量重建索引 (IndexFlatIP 不支持删除, 移除过期条目时使用)."""
        with self._lock:
            fresh = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))
            for i, vec in self._vectors:
                fresh.add_with_ids(vec[None], np.array([i], dtype=np.int64))
            self._index = fresh

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with self._lock:
            faiss.write_index(self._index, str(directory / "index.faiss"))
            meta = {"dim": self.dim, "version": self.version, "next_id": self._next_id,
                    "min_sim": self.min_sim, "margin": self.margin,
                    "critical_band": self.critical_band,
                    "entries": {str(i): m for i, m in self._meta.items()}}
            with open(directory / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory: str | Path, **threshold_overrides) -> "FaissRetriever":
        directory = Path(directory)
        with open(directory / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        ret = cls(dim=meta["dim"],
                  min_sim=threshold_overrides.get("min_sim", meta.get("min_sim", 0.55)),
                  margin=threshold_overrides.get("margin", meta.get("margin", 0.10)),
                  critical_band=threshold_overrides.get("critical_band",
                                                        meta.get("critical_band", 0.05)))
        with ret._lock:
            ret._index = faiss.read_index(str(directory / "index.faiss"))
            ret._next_id = int(meta["next_id"])
            ret.version = int(meta["version"])
            ret._meta = {int(k): v for k, v in meta["entries"].items()}
            # 从索引重建向量缓存 (IndexFlatIP 支持 reconstruct), 以保持 compact 能力
            for i in ret._meta:
                try:
                    ret._vectors.append((i, ret._index.reconstruct(i).copy()))
                except Exception:
                    pass
        return ret
