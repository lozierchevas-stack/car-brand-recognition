"""Faiss 双重阈值检索与 OTA 热更新的单元测试 (合成数据).

运行: python -m pytest tests/test_retrieval.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from car_recog.retrieval import FaissRetriever  # noqa: E402
from car_recog.utils import l2_normalize  # noqa: E402

DIM = 32


def _clusters(rng, centers: list[np.ndarray], n: int = 8, sigma: float = 0.05):
    """围绕每个中心生成类内高斯簇 (L2 归一化)."""
    vecs, labels = [], []
    for brand, c in enumerate(centers):
        for _ in range(n):
            vecs.append(c + rng.normal(0, sigma, DIM))
            labels.append(f"Brand{brand}")
    return l2_normalize(np.vstack(vecs)), labels


@pytest.fixture
def centers():
    rng = np.random.default_rng(0)
    cs = l2_normalize(rng.normal(0, 1, (3, DIM)))
    return rng, cs  # 随机 3 个中心, 两两夹角接近 90 度


def test_accept_in_class(centers):
    rng, cs = centers
    ret = FaissRetriever(dim=DIM, min_sim=0.6, margin=0.1)
    vecs, labels = _clusters(rng, cs)
    ret.add_vectors(vecs, labels, source="gallery")
    q = l2_normalize(cs[1] + rng.normal(0, 0.03, DIM))
    m = ret.match(q)
    assert m.accepted and m.brand == "Brand1"
    assert m.similarity > 0.9
    assert m.reason == "ok"


def test_reject_low_similarity(centers):
    """远离所有类中心的查询 -> 未知品牌/非目标车辆."""
    rng, cs = centers
    ret = FaissRetriever(dim=DIM, min_sim=0.6, margin=0.1)
    vecs, labels = _clusters(rng, cs)
    ret.add_vectors(vecs, labels, source="gallery")
    q = l2_normalize(rng.normal(0, 1, DIM))  # 与各中心相似度期望 ~0
    m = ret.match(q)
    assert not m.accepted
    assert m.reason == "low_similarity"
    assert m.confidence == 0.0


def test_reject_ambiguous(centers):
    """两中心连线中点 -> top1 与 top2 相似度几乎相同 -> 模棱两可拒识."""
    rng, cs = centers
    ret = FaissRetriever(dim=DIM, min_sim=0.5, margin=0.1)
    vecs, labels = _clusters(rng, cs)
    ret.add_vectors(vecs, labels, source="gallery")
    q = l2_normalize(cs[0] + cs[1])  # 中点, 距两中心相似度相等
    m = ret.match(q)
    assert not m.accepted
    assert m.reason == "ambiguous"
    assert abs(m.similarity - m.second_sim) < 0.1


def test_ota_hot_update(centers):
    """新品牌向量追加后立即可检索 (零停机), 且版本号递增."""
    rng, cs = centers
    ret = FaissRetriever(dim=DIM, min_sim=0.6, margin=0.1)
    vecs, labels = _clusters(rng, cs)
    ret.add_vectors(vecs, labels, source="gallery")
    v0 = ret.version
    # OTA 注入新品牌
    new_center = l2_normalize(rng.normal(0, 1, (1, DIM)))
    new_vecs, _ = _clusters(rng, [new_center[0]], n=4)
    ids, v1 = ret.add_vectors(new_vecs, ["NewBrand"] * 4, source="ota")
    assert v1 == v0 + 1
    assert "NewBrand" in ret.brands()
    q = l2_normalize(new_center[0] + rng.normal(0, 0.03, DIM))
    m = ret.match(q)
    assert m.accepted and m.brand == "NewBrand"


def test_save_load_roundtrip(centers, tmp_path):
    rng, cs = centers
    ret = FaissRetriever(dim=DIM, min_sim=0.6, margin=0.1)
    vecs, labels = _clusters(rng, cs)
    ret.add_vectors(vecs, labels, source="gallery")
    ret.save(tmp_path)
    ret2 = FaissRetriever.load(tmp_path)
    q = l2_normalize(cs[2] + rng.normal(0, 0.03, DIM))
    m1, m2 = ret.match(q), ret2.match(q)
    assert m1.accepted == m2.accepted
    assert m1.brand == m2.brand
    assert abs(m1.similarity - m2.similarity) < 1e-6
    assert ret2.version == ret.version


def test_critical_band(centers):
    rng, cs = centers
    ret = FaissRetriever(dim=DIM, min_sim=0.55, margin=0.1, critical_band=0.05)
    vecs, labels = _clusters(rng, cs)
    ret.add_vectors(vecs, labels, source="gallery")
    assert ret.is_critical(0.56)
    assert ret.is_critical(0.549) is False
    assert ret.is_critical(0.601) is False
