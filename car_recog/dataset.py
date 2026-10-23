"""数据集与 P-K Sampler.

P-K Sampling: 每个 Batch 随机抽取 P 个品牌, 每个品牌抽取 K 张图 (batch_size = P*K)。
相比完全随机采样, 它保证每个 batch 内同类样本成对出现, 使 ArcFace 在每个
优化步都能看到充分的类内/类间结构, 是 ReID/细粒度检索的工业标配。
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

from .transforms import get_train_transforms, get_val_transforms

_IMG_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


class CarBrandDataset(Dataset):
    """目录结构: root/{split}/{brand}/{image}, 类名即品牌名 (按字母序映射标签)."""

    def __init__(self, root: str | Path, split: str = "train", transform=None):
        self.root = Path(root)
        self.split = split
        split_dir = self.root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"数据目录不存在: {split_dir}\n请先运行: python data_pipeline.py download && python data_pipeline.py crop")
        self.brands = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
        self.brand_to_label = {b: i for i, b in enumerate(self.brands)}
        self.samples: list[tuple[str, int]] = []
        for brand in self.brands:
            for p in sorted((split_dir / brand).iterdir()):
                if p.suffix.lower() in _IMG_SUFFIXES:
                    self.samples.append((str(p), self.brand_to_label[brand]))
        if not self.samples:
            raise FileNotFoundError(f"{split_dir} 下没有找到任何图像, 请先运行数据管线")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def labels(self) -> list[int]:
        return [s[1] for s in self.samples]

    def label_to_brand(self) -> dict[int, str]:
        return {i: b for b, i in self.brand_to_label.items()}


class PKSampler(Sampler):
    """P-K 批量采样器: 每批 P 个类 x K 张图, 类内样本不足 K 时带重复采样."""

    def __init__(self, labels, p: int, k: int, num_batches: int | None = None, seed: int | None = None):
        self.p = p
        self.k = k
        self.class_to_idx: dict[int, list[int]] = defaultdict(list)
        for i, lb in enumerate(labels):
            self.class_to_idx[lb].append(i)
        self.classes = sorted(c for c, idx in self.class_to_idx.items() if len(idx) >= k)
        if len(self.classes) < p:
            raise ValueError(
                f"可用于 P-K 采样的类别数 {len(self.classes)} < P={p}, "
                f"请调小 train.p 或扩充每类样本数 (至少 K={k} 张)")
        total = sum(len(v) for v in self.class_to_idx.values())
        self.num_batches = num_batches if num_batches is not None else max(1, total // (p * k))
        self.rng = random.Random(seed)

    def __iter__(self):
        for _ in range(self.num_batches):
            batch: list[int] = []
            for cls in self.rng.sample(self.classes, self.p):
                pool = self.class_to_idx[cls]
                batch.extend(pool if len(pool) == self.k else self.rng.sample(pool, self.k))
            self.rng.shuffle(batch)
            yield from batch

    def __len__(self) -> int:
        return self.num_batches * self.p * self.k


def make_loaders(cfg):
    """按配置构建 训练(P-K采样)/验证 DataLoader, 并返回品牌名列表."""
    root = Path(cfg.paths.cropped_root)
    train_ds = CarBrandDataset(root, "train",
                               get_train_transforms(cfg.dataset.image_size, cfg.train.aug))
    val_split = "val" if (root / "val").is_dir() else "test"
    val_ds = CarBrandDataset(root, val_split,
                             get_val_transforms(cfg.dataset.image_size))

    p, k = cfg.train.p, cfg.train.k
    sampler = PKSampler(train_ds.labels(), p, k, seed=cfg.project.seed)
    train_loader = DataLoader(train_ds, batch_size=p * k, sampler=sampler,
                              num_workers=cfg.dataset.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=max(32, p * k), shuffle=False,
                            num_workers=cfg.dataset.num_workers, pin_memory=True)
    return train_loader, val_loader, train_ds.brands
