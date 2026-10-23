#!/usr/bin/env python3
"""构建 Faiss 视觉向量知识库 (Visual RAG 索引).

两种模式:
  1) 在线提取 (推荐): 用 ONNX/TensorRT 模型对裁剪数据集逐品牌提取特征;
     python build_gallery.py --onnx artifacts/models/embedding.onnx --n-per-class 8
  2) 检查点直用 (秒级): 直接复用 train_export.py 保存在检查点里的类中心 gallery;
     python build_gallery.py --checkpoint artifacts/checkpoints/best.pth

输出: artifacts/retrieval/index.faiss + meta.json (双重阈值/版本号/品牌元数据),
     边缘端 FaissRetriever.load() 直接加载。
"""
from __future__ import annotations

import argparse
import random

import torch

from car_recog import utils
from car_recog.config import load_config
from car_recog.dataset import CarBrandDataset
from car_recog.inference import OnnxEmbedder, embed_image_paths
from car_recog.retrieval import FaissRetriever

logger = utils.setup_logger("car-recog.gallery")


def build_from_checkpoint(ckpt_path: str) -> tuple:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    gallery = ckpt["gallery"]
    return gallery["vectors"], gallery["brands"]


def build_from_data(embedder, cropped_root: str, split: str, n_per_class: int, seed: int) -> tuple:
    ds = CarBrandDataset(cropped_root, split, transform=None)
    per_brand: dict[str, list[str]] = {}
    for path, label in ds.samples:
        per_brand.setdefault(ds.brands[label], []).append(path)
    rng = random.Random(seed)
    paths, brands = [], []
    for brand, files in sorted(per_brand.items()):
        k = min(n_per_class, len(files))
        for p in rng.sample(files, k):
            paths.append(p)
            brands.append(brand)
    logger.info("开始嵌入 %d 张图像 (%d 个品牌)...", len(paths), len(per_brand))
    vectors = embed_image_paths(embedder, paths)
    return vectors, brands


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 Faiss 视觉向量知识库")
    parser.add_argument("--config", default=None)
    parser.add_argument("--onnx", default="", help="ONNX 嵌入模型路径 (在线提取模式)")
    parser.add_argument("--checkpoint", default="", help="训练检查点路径 (类中心直用模式)")
    parser.add_argument("--cropped-root", default="", help="裁剪数据根目录 (默认取配置)")
    parser.add_argument("--split", default="train", help="用于构建库的划分 (train/val/test)")
    parser.add_argument("--n-per-class", type=int, default=8, help="每品牌采样图像数")
    parser.add_argument("--out", default="", help="索引输出目录 (默认取配置)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    utils.seed_everything(cfg.project.seed)
    cropped_root = args.cropped_root or cfg.paths.cropped_root

    if args.checkpoint and not args.onnx:
        logger.info("模式: 检查点类中心直用 (%s)", args.checkpoint)
        vectors, brands = build_from_checkpoint(args.checkpoint)
    else:
        if not args.onnx:
            raise SystemExit("请提供 --onnx (在线提取) 或 --checkpoint (类中心直用)")
        embedder = OnnxEmbedder(args.onnx)
        vectors, brands = build_from_data(embedder, cropped_root, args.split,
                                          args.n_per_class, cfg.project.seed)

    retriever = FaissRetriever(cfg.retrieval.dim,
                               min_sim=cfg.retrieval.min_sim,
                               margin=cfg.retrieval.margin,
                               critical_band=cfg.retrieval.critical_band)
    retriever.add_vectors(vectors, brands, source="gallery")
    out_dir = args.out or cfg.retrieval.index_dir
    retriever.save(out_dir)

    logger.info("知识库构建完成: %s | %d 条向量 / %d 个品牌 / 版本 v%d",
                out_dir, retriever.size, len(retriever.brands()), retriever.version)
    logger.info("边缘端启动: python edge_iot_inference.py --index-dir %s", out_dir)


if __name__ == "__main__":
    main()
