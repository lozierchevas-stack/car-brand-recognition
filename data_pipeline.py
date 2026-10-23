#!/usr/bin/env python3
"""数据加载与预处理管线 (核心交付 1): 合规数据集下载 -> .mat 标注解析 -> GT 边界框精准裁剪 -> 规范化归档.

设计要点:
- 下载: 仅使用官方直链 (Stanford Cars Dataset), 支持断点续传, 严禁使用网络爬虫;
- 解析: 读取数据集自带的结构化标注文件 (.mat, 兼容 scipy/h5py 两种读取路径),
  提取图片绝对路径 / 类别标签 / 精确边界框 (Bounding Box) 坐标;
- 裁剪: 基于 Ground-Truth 边界框用 OpenCV 精准裁剪车辆主体 ROI, 多进程并行,
  按 {split}/{brand}/{image} 规范化归档, 并输出 label_map.json 与 manifest.csv;
- 兼容: 额外提供 VMMRdb (csv 标注) 的尽力解析入口, 需自行从官方渠道获取数据。

用法:
  python data_pipeline.py download                         # 下载 Stanford Cars (~2GB)
  python data_pipeline.py crop --val-frac 0.1              # 解析+裁剪+归档 (10% 带框验证集)
  python data_pipeline.py crop --dataset vmmrdb --csv xxx.csv --images-dir xxx  # VMMRdb 模式

说明: Stanford 官方 test 集标注不含边界框, test 划分按整图归档 (仅用于最终评测);
     需要带边界框的验证集时请使用 --val-frac 从训练集切分。
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import multiprocessing as mp
import os
import random
import re
import shutil
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import tqdm

try:  # 项目内使用 car_recog 的日志工具; 单独拷贝本文件时降级为 stdlib
    from car_recog.utils import setup_logger as _setup_logger
except ImportError:
    def _setup_logger(name, level, log_file=None):
        logging.basicConfig(level=level,
                            format="%(asctime)s | %(levelname)-7s | %(message)s")
        return logging.getLogger(name)

logger = _setup_logger("car-recog.data", logging.INFO)

# Stanford Cars 官方直链 (学术数据集, 遵守其使用条款)
# 注: 官方服务器当前已失效 (https 404 / http 500, 参见 pytorch/vision#7545),
# 下载失败时自动回退到下方 GitHub 完整镜像。
STANFORD_URLS = {
    "cars_train.tgz": "https://ai.stanford.edu/~jkrause/car196/cars_train.tgz",
    "cars_test.tgz": "https://ai.stanford.edu/~jkrause/car196/cars_test.tgz",
    "car_devkit.tgz": "https://ai.stanford.edu/~jkrause/car196/car_devkit.tgz",
}

# GitHub 完整镜像 (第三方镜像, 含 cars_train/cars_test 图片与 devkit .mat 边界框标注)。
# 候选源按顺序尝试: 国内加速代理优先 (codeload 直连在国内常被限速至 ~50kB/s),
# 失效时自动回退 codeload 直连; 可按需增删候选。
GITHUB_MIRROR = {
    "urls": [
        "https://gh-proxy.com/https://github.com/jhpohovey/StanfordCars-Dataset/archive/refs/heads/main.tar.gz",
        "https://codeload.github.com/jhpohovey/StanfordCars-Dataset/tar.gz/refs/heads/main",
    ],
    "filename": "stanford_cars_github_mirror.tgz",
    "done_marker": ".github_mirror.done",
}

# HuggingFace parquet 镜像 (国内可通过环境变量 HF_ENDPOINT=https://hf-mirror.com 加速,
# resolve 下载支持 Range 断点续传)。图片以 parquet 行序存储, 行序与 .mat 标注顺序
# 一致 (torchvision StanfordCars 按 .mat 顺序构建), 提取时做全量标签序列校验。
HF_MIRROR = {
    "repos": {
        "train": "Multimodal-Fatima/StanfordCars_train",
        "test": "Multimodal-Fatima/StanfordCars_test",
    },
    "done_marker": ".hf_parquet.done",
}

# .mat 标注文件 (来自 GitHub raw, 均为小文件, 供 hf-parquet 模式补全 devkit)
RAW_DEVKIT_FILES = {
    "cars_meta.mat": ("devkit/cars_meta.mat", "devkit"),
    "cars_train_annos.mat": ("devkit/cars_train_annos.mat", "devkit"),
    "cars_test_annos_withlabels.mat": ("cars_test_annos_withlabels.mat", "devkit"),
}
RAW_BASE = "https://raw.githubusercontent.com/jhpohovey/StanfordCars-Dataset/main/stanford_cars"


# ======================================================================
# 1. 下载与解压 (合规: 官方直链 + 断点续传, 无任何爬虫逻辑)
# ======================================================================

def _fetch_url(url: str, timeout: int = 30) -> requests.Response:
    """先尝试 https, 失败回退 http (斯坦福服务器两种均可用)."""
    variants = [url]
    if url.startswith("https://"):
        variants.append(url.replace("https://", "http://", 1))
    last_err = "unknown"
    for u in variants:
        try:
            resp = requests.get(u, stream=True, timeout=timeout)
            if resp.status_code == 200:
                return resp
            last_err = f"HTTP {resp.status_code}"
            resp.close()
        except requests.RequestException as e:
            last_err = str(e)
    raise RuntimeError(f"下载失败 {url}: {last_err}")


def download(url: str, dest: Path, resume: bool = True) -> Path:
    """流式下载, 支持断点续传 (Range), 带进度条与完整性校验."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    probe = _fetch_url(url)
    total = int(probe.headers.get("Content-Length", 0))
    probe.close()
    if existing and total and existing >= total:
        logger.info("已存在且完整, 跳过: %s", dest.name)
        return dest
    headers = {}
    mode, initial = "wb", 0
    if existing and resume and total:
        headers["Range"] = f"bytes={existing}-"
        mode, initial = "ab", existing
    resp = requests.get(url, stream=True, headers=headers, timeout=60)
    if resp.status_code not in (200, 206):
        resp.close()
        raise RuntimeError(f"下载失败 {url}: HTTP {resp.status_code}")
    if mode == "ab" and resp.status_code == 200:
        # 服务器忽略 Range 返回全量 200: 追加会损坏文件, 必须重头下载
        logger.warning("服务器不支持断点续传 (Range 被忽略), 重新从头下载")
        mode, initial = "wb", 0
    with open(dest, mode) as f:
        with tqdm.tqdm(total=total or None, initial=initial, desc=dest.name,
                       unit="B", unit_scale=True, unit_divisor=1024) as pbar:
            for chunk in resp.iter_content(1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))
    resp.close()
    if total and dest.stat().st_size != total:
        raise RuntimeError(f"文件不完整: {dest} ({dest.stat().st_size}/{total})")
    return dest


def parallel_download(url: str, dest: Path, segments: int = 16, timeout: int = 60) -> Path:
    """多线程分段下载 (要求服务器支持 Range 且返回 Content-Length).

    单连接大文件下载容易被限速/中断, 分段并行可显著提升吞吐;
    每个分段独立落盘为 .part{i}, 重跑时自动跳过已完成分段 (分段级断点续传)。
    """
    import concurrent.futures as cf

    probe = requests.get(url, stream=True, timeout=timeout,
                         headers={"Range": "bytes=0-0"})
    if probe.status_code != 206:
        probe.close()
        raise RuntimeError(f"服务器不支持 Range 分段下载: HTTP {probe.status_code}")
    total = int(probe.headers.get("Content-Range", "/0").split("/")[-1])
    probe.close()
    dest.parent.mkdir(parents=True, exist_ok=True)
    segments = max(1, min(segments, max(1, total // (1 << 20))))
    part_size = math.ceil(total / segments)

    def fetch_seg(idx: int, start: int, end: int) -> None:
        part = dest.parent / f"{dest.name}.part{idx:02d}"
        if part.is_file() and part.stat().st_size == end - start + 1:
            return  # 已完成分段, 跳过
        tmp = part.with_suffix(part.suffix + ".tmp")
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=timeout,
                                 headers={"Range": f"bytes={start}-{end}"},
                                 stream=True)
                if r.status_code != 206:
                    r.close()
                    raise RuntimeError(f"HTTP {r.status_code}")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        if chunk:
                            f.write(chunk)
                r.close()
                if tmp.stat().st_size != end - start + 1:
                    raise RuntimeError("分段长度不完整")
                tmp.replace(part)
                return
            except Exception as e:
                logger.warning("分段 %d 下载失败 (%s), 重试 %d/3", idx, e, attempt + 1)
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"分段 {idx} 三次重试后仍失败")

    specs = [(i, i * part_size, min(total, (i + 1) * part_size) - 1)
             for i in range(segments)]
    with tqdm.tqdm(total=segments, desc=f"分段下载 {dest.name}") as pbar:
        with cf.ThreadPoolExecutor(max_workers=segments) as pool:
            futures = {pool.submit(fetch_seg, *s): s for s in specs}
            for fut in cf.as_completed(futures):
                fut.result()  # 抛出异常则终止
                pbar.update(1)
    # 拼接分段
    with open(dest, "wb") as out:
        for i in range(segments):
            part = dest.parent / f"{dest.name}.part{i:02d}"
            with open(part, "rb") as f:
                shutil.copyfileobj(f, out)
            part.unlink()
    if dest.stat().st_size != total:
        raise RuntimeError(f"拼接后大小校验失败: {dest.stat().st_size}/{total}")
    return dest


def extract_tgz(tgz_path: Path, dest_dir: Path) -> None:
    """安全解压 tgz (校验归档成员路径, 防路径穿越)."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tgz_path, "r:gz") as tar:
        root = dest_dir.resolve()
        for m in tar.getmembers():
            target = (dest_dir / m.name).resolve()
            if not str(target).startswith(str(root)):
                raise RuntimeError(f"非法归档成员 (路径穿越): {m.name}")
        logger.info("解压 %s -> %s ...", tgz_path.name, dest_dir)
        tar.extractall(dest_dir)


def _download_official(raw_root: Path, resume: bool) -> None:
    for name, url in STANFORD_URLS.items():
        tgz = raw_root / name
        download(url, tgz, resume=resume)
        extract_tgz(tgz, raw_root)


def _reorganize_mirror(raw_root: Path) -> None:
    """GitHub 镜像包解压后为 {repo}-main/stanford_cars/{cars_train,cars_test,devkit,...},
    重组为管线约定的 data/raw/{cars_train,cars_test,devkit} 布局, 并把
    仓库根目录的 test 标签文件并入 devkit/ (官方 devkit 不含该文件)。"""
    extracted = next(raw_root.glob("StanfordCars-Dataset-*/stanford_cars"), None)
    if extracted is None:
        raise RuntimeError("镜像包布局异常: 未找到 stanford_cars 目录")
    for sub in ("cars_train", "cars_test", "devkit"):
        src, dst = extracted / sub, raw_root / sub
        if not src.is_dir():
            raise RuntimeError(f"镜像包缺少 {sub}")
        if dst.exists() and any(dst.iterdir()):
            logger.info("已存在且非空, 跳过重组: %s", dst)
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
    labels = extracted / "cars_test_annos_withlabels.mat"
    if labels.is_file() and not (raw_root / "devkit" / labels.name).exists():
        shutil.move(str(labels), str(raw_root / "devkit" / labels.name))
    shutil.rmtree(extracted.parent, ignore_errors=True)
    logger.info("镜像数据已重组为 raw/{cars_train, cars_test, devkit} 布局")


def _ensure_devkit(raw_root: Path) -> None:
    """从 GitHub raw 补齐 devkit 小文件 (cars_meta / train_annos / test_annos 含边界框)."""
    devkit = raw_root / "devkit"
    devkit.mkdir(parents=True, exist_ok=True)
    for fname, (rel, _) in RAW_DEVKIT_FILES.items():
        dst = devkit / fname
        if dst.is_file() and dst.stat().st_size > 0:
            continue
        logger.info("下载标注文件 %s", fname)
        download(f"{RAW_BASE}/{rel}", dst, resume=True)


def _extract_hf_parquet(raw_root: Path, cache: Path) -> None:
    """提取 HF parquet 镜像: 按行序写出图片, 并全量校验与 .mat 标注的顺序一致性.

    边界框裁剪依赖 parquet 行序 == .mat 标注顺序 (parquet 无原始文件名列),
    顺序错位会导致裁剪错位, 因此提取后逐行比对类别标签序列, 不一致即中止。
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit("提取 HF parquet 镜像需要 pyarrow: pip install pyarrow")
    _ensure_devkit(raw_root)
    devkit = raw_root / "devkit"
    meta = _load_mat(devkit / "cars_meta.mat")
    class_names = [str(_scalar(c)) for c in np.atleast_1d(meta["class_names"])]
    anno_files = {"train": devkit / "cars_train_annos.mat",
                  "test": devkit / "cars_test_annos_withlabels.mat"}
    for split in ("train", "test"):
        out_dir = raw_root / f"cars_{split}"
        out_dir.mkdir(parents=True, exist_ok=True)
        parquet_labels, gidx = [], 0
        for f in sorted(cache.glob(f"{split}_*.parquet")):
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=128):
                for row in batch.to_pylist():
                    img = row["image"]
                    data = img["bytes"] if isinstance(img, dict) else bytes(img)
                    (out_dir / f"{gidx + 1:05d}.jpg").write_bytes(data)
                    parquet_labels.append(int(row["label"]))
                    gidx += 1
        annos = _load_mat(anno_files[split])
        mat_labels = [int(_scalar(a["class"])) - 1
                      for a in np.atleast_1d(annos.get("annotations"))]
        if len(mat_labels) != len(parquet_labels):
            raise RuntimeError(
                f"{split}: parquet 行数 {len(parquet_labels)} 与 .mat 标注数 "
                f"{len(mat_labels)} 不一致, 边界框映射不可靠, 请更换数据源")
        mismatch = sum(1 for a, b in zip(mat_labels, parquet_labels) if a != b)
        if mismatch:
            raise RuntimeError(
                f"{split}: parquet 与 .mat 有 {mismatch} 处标签顺序不一致, "
                f"边界框映射不可靠, 请更换数据源")
        logger.info("%s: 提取 %d 张图像, parquet 与 .mat 标签序列全量校验通过 (%d 类)",
                    split, gidx, len(class_names))


def _download_hf_parquet(raw_root: Path, resume: bool) -> None:
    """从 HuggingFace parquet 镜像下载并提取 (支持断点续传)."""
    marker = raw_root / HF_MIRROR["done_marker"]
    if marker.is_file():
        logger.info("HF parquet 镜像数据已就绪, 跳过下载")
        return
    api = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
    cache = raw_root / "_hf_parquet"
    cache.mkdir(parents=True, exist_ok=True)
    for split, repo in HF_MIRROR["repos"].items():
        resp = _fetch_url(f"{api}/api/datasets/{repo}")
        info = resp.json()
        resp.close()
        parquet_files = [s["rfilename"] for s in info.get("siblings", [])
                         if s["rfilename"].endswith(".parquet")]
        if not parquet_files:
            raise RuntimeError(f"{repo} 未找到 parquet 文件")
        for fname in parquet_files:
            dst = cache / f"{split}_{Path(fname).name}"
            if dst.is_file() and dst.stat().st_size > 0:
                continue
            url = f"{api}/datasets/{repo}/resolve/main/{fname}"
            logger.info("分段并行下载 %s/%s", repo, fname)
            try:
                parallel_download(url, dst, segments=16)
            except RuntimeError as e:
                logger.warning("分段下载失败 (%s), 回退普通下载", e)
                download(url, dst, resume=resume)
    _extract_hf_parquet(raw_root, cache)
    shutil.rmtree(cache, ignore_errors=True)
    marker.write_text("ok", encoding="utf-8")
    logger.info("HF parquet 镜像提取完成 (标记文件 %s)", marker.name)


def _download_github_mirror(raw_root: Path, resume: bool) -> None:
    """GitHub 完整镜像: 按候选源顺序下载整包 tarball (约 1.9GB) 并重组布局."""
    marker = raw_root / GITHUB_MIRROR["done_marker"]
    if marker.is_file():
        logger.info("GitHub 镜像数据已就绪 (存在 %s), 跳过下载", marker.name)
        return
    tgz = raw_root / GITHUB_MIRROR["filename"]
    last_err = None
    for url in GITHUB_MIRROR["urls"]:
        try:
            logger.info("下载 GitHub 完整镜像 (约 1.9GB, 中断后重跑可续传): %s", url)
            download(url, tgz, resume=resume)
            break
        except RuntimeError as e:
            last_err = e
            logger.warning("镜像源失败 (%s), 尝试下一候选...", e)
    else:
        raise RuntimeError(f"所有镜像源均不可用: {last_err}")
    extract_tgz(tgz, raw_root)
    _reorganize_mirror(raw_root)
    marker.write_text("ok", encoding="utf-8")
    logger.info("镜像下载+重组完成 (标记文件 %s)", marker.name)


def download_stanford(raw_root: Path, resume: bool = True, source: str = "auto") -> None:
    """下载 Stanford Cars: 官方直链优先, 依次回退 HF parquet 镜像 / GitHub 完整镜像."""
    raw_root.mkdir(parents=True, exist_ok=True)
    chain = {"auto": ["stanford", "hf-parquet", "github-mirror"]}.get(source, [source])
    for src in chain:
        try:
            if src == "stanford":
                _download_official(raw_root, resume)
            elif src == "hf-parquet":
                _download_hf_parquet(raw_root, resume)
            elif src == "github-mirror":
                _download_github_mirror(raw_root, resume)
            else:
                raise RuntimeError(f"未知数据源: {src}")
            return
        except RuntimeError as e:
            logger.warning("数据源 [%s] 失败: %s", src, e)
    raise RuntimeError(f"所有数据源均不可用: {chain}")


# ======================================================================
# 2. 标注解析 (.mat / .csv)
# ======================================================================

def _load_mat(path: Path) -> dict:
    """读取 .mat: 优先 scipy (v5), 兼容 h5py (v7.3)."""
    try:
        from scipy.io import loadmat
    except ImportError:
        raise SystemExit("解析 .mat 需要 scipy: pip install scipy")
    try:
        return loadmat(str(path), squeeze_me=True)
    except NotImplementedError:
        try:
            import h5py
        except ImportError:
            raise SystemExit(f"{path.name} 为 MATLAB v7.3 格式, 需要 h5py: pip install h5py\n"
                             f"或在 MATLAB 中将其另存为 v5 格式 (-v7) 后重试")
        with h5py.File(path, "r") as f:
            out = {}
            for key in f.keys():
                out[key] = f[key][()]
            return out


def _scalar(v):
    """将 .mat 中的标量包装 (0-d ndarray / bytes) 规整为 Python 标量."""
    if isinstance(v, np.ndarray):
        v = v.reshape(-1)[0]
    if isinstance(v, bytes):
        return v.decode("utf-8", "ignore")
    return v


def parse_stanford_annos(anno_path: Path, meta_path: Path, image_dir: Path) -> tuple[list[dict], list[str]]:
    """解析 Stanford Cars 标注: 提取绝对路径/类别标签/边界框.

    cars_train_annos.mat 字段: fname, class, bbox_x1/y1/x2/y2 (1-based 像素坐标);
    cars_test_annos_withlabels.mat 仅含 fname 与 class (无边界框, 整图归档)。
    """
    meta = _load_mat(meta_path)
    class_names = [str(_scalar(c)) for c in np.atleast_1d(meta["class_names"])]
    annos = _load_mat(anno_path)
    records = []
    items = np.atleast_1d(annos.get("annotations"))
    for item in items:
        fname = str(_scalar(item["fname"]))
        cls = int(_scalar(item["class"])) - 1  # 类别 1..N -> 0..N-1
        rec = {"image_path": str(image_dir / fname), "brand": class_names[cls],
               "label": cls, "bbox": None}
        try:  # test 集无 bbox 字段
            rec["bbox"] = tuple(int(round(float(_scalar(item[k]))))
                                for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"))
        except (KeyError, ValueError):
            pass
        records.append(rec)
    return records, class_names


def parse_vmmrdb_csv(csv_path: Path, image_root: Path) -> tuple[list[dict], list[str]]:
    """VMMRdb 标注 CSV 的尽力解析 (列顺序不固定, 按表头推断).

    典型列: [图片路径, bbox(如 "[x1 y1 x2 y2]" 或 "[x1, y1, x2, y2]"), 品牌标签, ...]
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        hdr = [h.strip().lower() for h in header]
        path_col = next((i for i, h in enumerate(hdr) if any(k in h for k in ("path", "file", "image"))), 0)
        bbox_col = next((i for i, h in enumerate(hdr) if any(k in h for k in ("bbox", "box"))), None)
        label_col = next((i for i, h in enumerate(hdr) if any(k in h for k in ("label", "make", "brand"))), None)
        if label_col is None:
            label_col = len(hdr) - 1
        records = []
        for row in reader:
            if not row:
                continue
            brand = row[label_col].strip()
            bbox = None
            if bbox_col is not None and bbox_col < len(row):
                nums = re.findall(r"-?\d+\.?\d*", row[bbox_col])
                if len(nums) >= 4:
                    bbox = tuple(int(round(float(x))) for x in nums[:4])
            records.append({"image_path": str(image_root / row[path_col].strip()),
                            "brand": brand, "label": -1, "bbox": bbox})
    brands = sorted({r["brand"] for r in records})
    for r in records:
        r["label"] = brands.index(r["brand"])
    return records, brands


# ======================================================================
# 3. GT 边界框精准裁剪
# ======================================================================

_ILLEGAL_CHARS = re.compile(r'[<>:"/\\|?*]')


def sanitize_name(name: str) -> str:
    return _ILLEGAL_CHARS.sub("_", name).strip()


def crop_roi(src: str, dst: Path, bbox, margin_frac: float = 0.0,
             quality: int = 95, min_size: int = 24) -> tuple[bool, str]:
    """按边界框精准裁剪 ROI 并落盘. bbox 为 None 时整图归档 (官方 test 集).

    Stanford 标注为 1-based 像素坐标, 裁剪前统一减 1 转 0-based 并夹取到图像边界。
    """
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        return False, "图像读取失败"
    h, w = img.shape[:2]
    if bbox is None:
        roi = img
    else:
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1 - 1), max(0, y1 - 1)
        x2, y2 = min(w, x2), min(h, y2)
        if margin_frac > 0:  # 可选: 按比例外扩, 保留更多车身上下文
            bw, bh = x2 - x1, y2 - y1
            dx, dy = int(bw * margin_frac), int(bh * margin_frac)
            x1, y1 = max(0, x1 - dx), max(0, y1 - dy)
            x2, y2 = min(w, x2 + dx), min(h, y2 + dy)
        if x2 - x1 < min_size or y2 - y1 < min_size:
            return False, f"ROI 过小 ({x2 - x1}x{y2 - y1})"
        roi = img[y1:y2, x1:x2]
    dst.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dst), roi, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return (True, "") if ok else (False, "写入失败")


def _crop_worker(task):
    """多进程裁剪 worker (模块级函数, 兼容 Windows spawn)."""
    idx, rec, out_root, margin_frac, quality, overwrite = task
    brand_dir = sanitize_name(rec["brand"])
    dst = Path(out_root) / brand_dir / Path(rec["image_path"]).name
    if dst.exists() and not overwrite:
        return idx, "skipped", ""
    ok, err = crop_roi(rec["image_path"], dst, rec["bbox"],
                       margin_frac=margin_frac, quality=quality)
    return idx, ("ok" if ok else "fail"), err


def crop_split(records: list[dict], out_root: Path, split: str, workers: int,
               margin_frac: float, quality: int, limit: int, overwrite: bool) -> dict:
    """多进程裁剪单个划分, 返回统计与成功记录索引集合."""
    tasks = [(i, r, out_root / split, margin_frac, quality, overwrite)
             for i, r in enumerate(records)]
    if limit > 0:
        tasks = tasks[:limit]
    stats = {"total": len(tasks), "ok": 0, "skipped": 0, "fail": 0, "failures": []}
    ok_idx = set()
    workers = max(1, min(workers, mp.cpu_count()))
    pbar = tqdm.tqdm(total=len(tasks), desc=f"裁剪 [{split}]")
    results = []
    if workers == 1 or len(tasks) < workers * 2:
        for r in map(_crop_worker, tasks):
            results.append(r)
            pbar.update(1)
    else:
        # 注意: imap_unordered 的迭代必须发生在 Pool 上下文内完成消费,
        # 否则 Windows 下退出 with 后池被终止, 迭代器未消费完会死锁
        with mp.Pool(processes=workers) as pool:
            for r in pool.imap_unordered(_crop_worker, tasks, chunksize=32):
                results.append(r)
                pbar.update(1)
    pbar.close()
    for idx, status, err in results:
        if status == "ok":
            stats["ok"] += 1
            ok_idx.add(idx)
        elif status == "skipped":
            stats["skipped"] += 1
            ok_idx.add(idx)
        else:
            stats["fail"] += 1
            if len(stats["failures"]) < 100:
                stats["failures"].append(f"{records[idx]['image_path']}: {err}")
    return stats, ok_idx


def split_val_from_train(cropped_root: Path, val_fraction: float, seed: int) -> None:
    """按品牌分层从 train 切分 val (保持带框 ROI 的验证集)."""
    train_dir = cropped_root / "train"
    val_dir = cropped_root / "val"
    rng = random.Random(seed)
    total = 0
    for brand_dir in sorted(train_dir.iterdir()):
        files = sorted(brand_dir.iterdir())
        n_val = max(1, int(round(len(files) * val_fraction)))
        if n_val >= len(files):
            n_val = max(1, len(files) // 4)
        picks = rng.sample(files, n_val)
        for p in picks:
            dst = val_dir / brand_dir.name / p.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dst))
            total += 1
    logger.info("已切分验证集: %d 张 -> %s", total, val_dir)


def write_metadata(cropped_root: Path, records: list[dict], ok_idx: set,
                   split_of: dict, class_names: list[str]) -> None:
    """输出 label_map.json 与 manifest.csv (下游训练/Hive 数仓的数据契约)."""
    label_map = {"num_classes": len(class_names), "class_names": class_names}
    with open(cropped_root / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    with open(cropped_root / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "brand", "label", "relative_path",
                         "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"])
        for i, rec in enumerate(records):
            if i not in ok_idx:
                continue
            bbox = rec["bbox"] or ("", "", "", "")
            rel = Path(rec["image_path"]).name
            writer.writerow([split_of[i], rec["brand"], rec["label"],
                             f"{split_of[i]}/{sanitize_name(rec['brand'])}/{rel}", *bbox])
    logger.info("元数据已输出: %s", cropped_root / "manifest.csv")


# ======================================================================
# CLI
# ======================================================================

def cmd_download(args) -> None:
    raw_root = Path(args.root) / "raw"
    logger.info("开始下载 Stanford Cars Dataset -> %s (source=%s)", raw_root, args.source)
    t0 = time.time()
    download_stanford(raw_root, resume=not args.no_resume, source=args.source)
    logger.info("下载+解压完成, 耗时 %.1f min", (time.time() - t0) / 60)


def cmd_crop(args) -> None:
    cropped_root = Path(args.root) / "cropped"
    cropped_root.mkdir(parents=True, exist_ok=True)
    raw_root = Path(args.root) / "raw"

    if args.dataset == "stanford-cars":
        train_img_dir = raw_root / "cars_train"
        devkit = raw_root / "devkit"
        for p in (train_img_dir, devkit / "cars_train_annos.mat", devkit / "cars_meta.mat"):
            if not p.exists():
                raise SystemExit(f"缺少 {p}, 请先运行: python data_pipeline.py download")
        train_records, class_names = parse_stanford_annos(
            devkit / "cars_train_annos.mat", devkit / "cars_meta.mat", train_img_dir)
        test_records = []
        test_annos = devkit / "cars_test_annos_withlabels.mat"
        if test_annos.exists():
            test_records, _ = parse_stanford_annos(
                test_annos, devkit / "cars_meta.mat", raw_root / "cars_test")
            logger.info("注意: 官方 test 集标注不含边界框, 将整图归档用于评测")
    elif args.dataset == "vmmrdb":
        if not args.vmmrdb_csv or not args.vmmrdb_images:
            raise SystemExit("VMMRdb 模式需要 --vmmrdb-csv 与 --vmmrdb-images")
        records, class_names = parse_vmmrdb_csv(
            Path(args.vmmrdb_csv), Path(args.vmmrdb_images))
        train_records, test_records = records, []
        logger.info("VMMRdb 无官方划分, 全部归档为 train (可用 --val-frac 切分)")
    else:
        raise SystemExit(f"未知数据集: {args.dataset}")

    logger.info("train 记录: %d, test 记录: %d, 类别数: %d",
                len(train_records), len(test_records), len(class_names))

    split_of = {}
    stats_train, ok_train = crop_split(train_records, cropped_root, "train",
                                       args.workers, args.margin, args.quality,
                                       args.limit, args.overwrite)
    for i in ok_train:
        split_of[i] = "train"
    stats_test, ok_test = {}, set()
    if test_records:
        stats_test, ok_test = crop_split(test_records, cropped_root, "test",
                                         args.workers, args.margin, args.quality,
                                         args.limit, args.overwrite)
        offset = len(train_records)
        for i in ok_test:
            split_of[i + offset] = "test"

    if 0 < args.val_frac < 1 and ok_train:
        split_val_from_train(cropped_root, args.val_frac, args.seed)
        # 切分后文件被移动到 val/, 按实际落盘位置重新判断 manifest 归属
        for i in ok_train:
            rec = train_records[i]
            rel = Path(f"{sanitize_name(rec['brand'])}/{Path(rec['image_path']).name}")
            split_of[i] = "val" if (cropped_root / "val" / rel).exists() else "train"

    all_records = train_records + test_records
    ok_idx = ok_train | {i + len(train_records) for i in ok_test}
    write_metadata(cropped_root, all_records, ok_idx, split_of, class_names)

    # 汇总报告
    report = {"train": stats_train, "test": stats_test}
    with open(cropped_root / "crop_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    for split, s in report.items():
        if not s:
            continue
        logger.info("[%s] 成功 %d / 跳过 %d / 失败 %d",
                    split, s["ok"], s["skipped"], s["fail"])
        if s["fail"]:
            logger.warning("前若干失败样例: %s", s["failures"][:5])


def main() -> None:
    parser = argparse.ArgumentParser(description="汽车品牌识别 - 数据加载与预处理管线")
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="下载并解压 Stanford Cars Dataset")
    p_dl.add_argument("--root", default="data", help="数据根目录 (默认 data)")
    p_dl.add_argument("--source", default="auto",
                      choices=["auto", "stanford", "hf-parquet", "github-mirror"],
                      help="下载源: auto=官方优先, 自动回退 HF parquet / GitHub 镜像 (默认)")
    p_dl.add_argument("--no-resume", action="store_true", help="禁用断点续传")

    p_crop = sub.add_parser("crop", help="解析标注并按 GT 边界框精准裁剪归档")
    p_crop.add_argument("--root", default="data", help="数据根目录 (默认 data)")
    p_crop.add_argument("--dataset", default="stanford-cars",
                        choices=["stanford-cars", "vmmrdb"])
    p_crop.add_argument("--vmmrdb-csv", default="", help="VMMRdb 标注 CSV 路径")
    p_crop.add_argument("--vmmrdb-images", default="", help="VMMRdb 图片目录")
    p_crop.add_argument("--margin", type=float, default=0.0,
                        help="边界框外扩比例 (0 表示严格按 GT 裁剪, 默认 0)")
    p_crop.add_argument("--val-frac", type=float, default=0.0,
                        help="从训练集按品牌分层切分验证集的比例 (0~1, 默认 0)")
    p_crop.add_argument("--workers", type=int, default=min(8, mp.cpu_count()),
                        help="裁剪并行进程数")
    p_crop.add_argument("--quality", type=int, default=95, help="JPEG 保存质量")
    p_crop.add_argument("--limit", type=int, default=0, help="仅处理前 N 张 (冒烟测试)")
    p_crop.add_argument("--overwrite", action="store_true", help="覆盖已存在输出")
    p_crop.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.command == "download":
        cmd_download(args)
    else:
        cmd_crop(args)


if __name__ == "__main__":
    main()
